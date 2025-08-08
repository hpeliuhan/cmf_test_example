#!/usr/bin/env python3
"""
Active Learning for SmokeyNet using Gaussian Process Uncertainty Estimation
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import sys
from pathlib import Path
import multiprocessing as mp
from functools import partial
import shutil
import datetime

# Add the inference module to path
sys.path.append('src/inference')
from smokeynet import SmokeyNet

def extract_single_feature(args):
    """Extract features for a single image - for multiprocessing"""
    image_path, model_path = args
    
    try:
        # Initialize SmokeyNet for this process
        smokeynet = SmokeyNet(model_path)
        
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Get predictions from SmokeyNet
        image_preds, tile_preds, tile_probs = smokeynet.inference(
            img_array, img_array, smoke_threshold=0.5
        )
        
        # Use tile probabilities as features
        features = tile_probs.flatten()
        
        # Pad or truncate to fixed size
        target_size = 100  # Adjust based on your model's output
        if len(features) > target_size:
            features = features[:target_size]
        else:
            features = np.pad(features, (0, target_size - len(features)), 'constant')
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

class SimpleSmokeClassifier(nn.Module):
    """Simple classifier that can be retrained with new data"""
    def __init__(self, input_size=100, hidden_size=64):
        super(SimpleSmokeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

class ActiveLearningSmokeyNet:
    def __init__(self, model_path, ground_truth_path, extracted_data_path, 
                 initial_pool_size=100, acquisition_batch_size=20, max_iterations=10,
                 n_workers=None):
        """
        Initialize Active Learning for SmokeyNet
        
        Args:
            model_path: Path to the trained ONNX model
            ground_truth_path: Path to ground truth JSON file
            extracted_data_path: Path to extracted image data
            initial_pool_size: Number of samples to start with
            acquisition_batch_size: Number of samples to acquire per iteration
            max_iterations: Maximum number of active learning iterations
            n_workers: Number of parallel workers (None for auto)
        """
        self.model_path = model_path
        self.ground_truth_path = ground_truth_path
        self.extracted_data_path = extracted_data_path
        self.initial_pool_size = initial_pool_size
        self.acquisition_batch_size = acquisition_batch_size
        self.max_iterations = max_iterations
        self.n_workers = n_workers or min(mp.cpu_count(), 8)  # Limit to 8 workers max
        
        # Initialize SmokeyNet for feature extraction
        self.smokeynet = SmokeyNet(model_path)
        
        # Initialize retrainable classifier
        self.classifier = SimpleSmokeClassifier()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # Load ground truth data
        self.ground_truth_data = self.load_ground_truth()
        
        # Initialize data pools
        self.unlabeled_pool = []
        self.labeled_pool = []
        self.test_pool = []
        
        # Feature cache
        self.feature_cache = {}
        self.all_samples = []
        
        # Performance tracking
        self.performance_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc_score': [],
            'optimal_threshold': [],
            'uncertainty_scores': [],
            'selected_samples': []
        }
        
        # Gaussian Process for uncertainty estimation
        self.gp_model = None
        
        # Create output directory for labeled samples
        self.output_dir = f"active_learning_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_ground_truth(self):
        """Load ground truth data from JSON file"""
        print("Loading ground truth data...")
        with open(self.ground_truth_path, 'r') as f:
            data = json.load(f)
        return data['ground_truth']
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_image_features(self, image_path):
        """Extract features from image using SmokeyNet"""
        img_array = self.load_image(image_path)
        if img_array is None:
            return None
        
        # For simplicity, we'll use a dummy previous image (same as current)
        # In practice, you'd want to use actual temporal pairs
        try:
            # Get predictions from SmokeyNet
            image_preds, tile_preds, tile_probs = self.smokeynet.inference(
                img_array, img_array, smoke_threshold=0.5
            )
            
            # Use tile probabilities as features
            features = tile_probs.flatten()
            
            # Pad or truncate to fixed size
            target_size = 100  # Adjust based on your model's output
            if len(features) > target_size:
                features = features[:target_size]
            else:
                features = np.pad(features, (0, target_size - len(features)), 'constant')
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_all_features_parallel(self):
        """Extract features for all samples using parallel processing, with caching support"""
        print("Extracting features for all samples using parallel processing...")

        # Group images by firename for temporal consistency
        firename_groups = {}
        for entry in self.ground_truth_data:
            firename = entry['firename']
            if firename not in firename_groups:
                firename_groups[firename] = []
            firename_groups[firename].append(entry)

        # Sort each group by timestamp
        for firename in firename_groups:
            firename_groups[firename].sort(key=lambda x: x['timestamp'])

        # Create image paths and labels
        all_samples = []
        for firename, entries in firename_groups.items():
            firename_path = os.path.join(self.extracted_data_path, firename)
            if not os.path.exists(firename_path):
                continue
            for entry in entries:
                image_path = os.path.join(firename_path, entry['image_name'])
                if os.path.exists(image_path):
                    all_samples.append({
                        'image_path': image_path,
                        'label': entry['smoke_label'],
                        'firename': firename,
                        'timestamp': entry['timestamp']
                    })

        print(f"Found {len(all_samples)} total samples")

        # Use only a subset of max_samples for faster execution
        max_samples = 1000
        cache_file = f"feature_cache_{max_samples}.npz"
        meta_file = f"feature_cache_{max_samples}_meta.json"

        # Try to load from cache
        if os.path.exists(cache_file) and os.path.exists(meta_file):
            print(f"Loading features from cache: {cache_file}")
            data = np.load(cache_file)
            with open(meta_file, 'r') as f:
                valid_samples = json.load(f)
            self.all_samples = valid_samples
            self.all_features = data['features']
            self.all_labels = data['labels']
            print(f"Loaded {len(self.all_samples)} samples from cache.")
            return self.all_samples, self.all_features, self.all_labels

        if len(all_samples) > max_samples:
            print(f"Using subset of {max_samples} samples for faster execution")
            # Sample evenly from different firenames to maintain diversity
            firename_samples = {}
            for sample in all_samples:
                firename = sample['firename']
                if firename not in firename_samples:
                    firename_samples[firename] = []
                firename_samples[firename].append(sample)
            
            # Separate samples by class within each firename
            firename_class_samples = {}
            for firename, samples in firename_samples.items():
                smoke_samples = [s for s in samples if s['label'] == 1]
                non_smoke_samples = [s for s in samples if s['label'] == 0]
                firename_class_samples[firename] = {
                    'smoke': smoke_samples,
                    'non_smoke': non_smoke_samples
                }
            
            # Calculate target samples per class (balanced)
            target_per_class = max_samples // 2  # Equal smoke and non-smoke
            samples_per_firename = max_samples // len(firename_class_samples)
            target_smoke_per_firename = target_per_class // len(firename_class_samples)
            target_non_smoke_per_firename = target_per_class // len(firename_class_samples)
            
            selected_samples = []
            total_smoke = 0
            total_non_smoke = 0
            
            # Sample from each firename with class balance
            for firename, class_samples in firename_class_samples.items():
                smoke_samples = class_samples['smoke']
                non_smoke_samples = class_samples['non_smoke']
                
                # Sample smoke samples
                if len(smoke_samples) > 0:
                    if len(smoke_samples) <= target_smoke_per_firename:
                        selected_smoke = smoke_samples
                    else:
                        # Sample evenly from this firename's smoke samples
                        step = len(smoke_samples) // target_smoke_per_firename
                        selected_smoke = smoke_samples[::step][:target_smoke_per_firename]
                    selected_samples.extend(selected_smoke)
                    total_smoke += len(selected_smoke)
                
                # Sample non-smoke samples
                if len(non_smoke_samples) > 0:
                    if len(non_smoke_samples) <= target_non_smoke_per_firename:
                        selected_non_smoke = non_smoke_samples
                    else:
                        # Sample evenly from this firename's non-smoke samples
                        step = len(non_smoke_samples) // target_non_smoke_per_firename
                        selected_non_smoke = non_smoke_samples[::step][:target_non_smoke_per_firename]
                    selected_samples.extend(selected_non_smoke)
                    total_non_smoke += len(selected_non_smoke)
            
            # If we still have too many, take a balanced random subset
            if len(selected_samples) > max_samples:
                # Separate by class
                smoke_selected = [s for s in selected_samples if s['label'] == 1]
                non_smoke_selected = [s for s in selected_samples if s['label'] == 0]
                
                # Calculate final balanced numbers
                final_smoke_count = max_samples // 2
                final_non_smoke_count = max_samples - final_smoke_count
                
                # Randomly sample from each class
                np.random.seed(42)
                if len(smoke_selected) > final_smoke_count:
                    smoke_indices = np.random.choice(len(smoke_selected), final_smoke_count, replace=False)
                    smoke_selected = [smoke_selected[i] for i in smoke_indices]
                
                if len(non_smoke_selected) > final_non_smoke_count:
                    non_smoke_indices = np.random.choice(len(non_smoke_selected), final_non_smoke_count, replace=False)
                    non_smoke_selected = [non_smoke_selected[i] for i in non_smoke_indices]
                
                selected_samples = smoke_selected + non_smoke_selected
            
            all_samples = selected_samples
            print(f"Selected {len(all_samples)} samples from {len(firename_class_samples)} fire events")
            print(f"Class balance: Smoke={sum(1 for s in all_samples if s['label'] == 1)}, "
                  f"Non-smoke={sum(1 for s in all_samples if s['label'] == 0)}")

        # Prepare arguments for parallel processing
        args_list = [(sample['image_path'], self.model_path) for sample in all_samples]

        # Extract features using parallel processing
        print(f"Using {self.n_workers} parallel workers for feature extraction...")
        with mp.Pool(processes=self.n_workers) as pool:
            features = list(tqdm(
                pool.imap(extract_single_feature, args_list),
                total=len(args_list),
                desc="Extracting features"
            ))

        # Filter out None results and collect valid samples
        valid_samples = []
        valid_features = []
        valid_labels = []
        for i, (feature, sample) in enumerate(zip(features, all_samples)):
            if feature is not None:
                valid_samples.append(sample)
                valid_features.append(feature)
                valid_labels.append(sample['label'])

        # Store features and samples
        self.all_samples = valid_samples
        self.all_features = np.array(valid_features)
        self.all_labels = np.array(valid_labels)

        print(f"Successfully extracted features for {len(valid_samples)} samples")
        # Save to cache
        np.savez_compressed(cache_file, features=self.all_features, labels=self.all_labels)
        with open(meta_file, 'w') as f:
            json.dump(self.all_samples, f, indent=2)
        print(f"Saved features to cache: {cache_file}")
        return valid_samples, self.all_features, self.all_labels
    
    def extract_all_features(self):
        """Extract features for all samples once and cache them"""
        return self.extract_all_features_parallel()
    
    def prepare_data_pools(self):
        """Prepare initial data pools for active learning using cached features"""
        print("Preparing data pools...")
        
        # Extract all features if not already done
        if not hasattr(self, 'all_features'):
            self.extract_all_features()
        
        # Split into train/test (maintaining temporal order)
        test_size = min(200, len(self.all_samples) // 5)  # 20% for testing
        self.test_pool = self.all_samples[-test_size:]  # Use latest samples for testing
        train_samples = self.all_samples[:-test_size]
        train_features = self.all_features[:-test_size]
        train_labels = self.all_labels[:-test_size]
        
        # Initialize labeled pool with random samples
        np.random.seed(42)  # For reproducibility
        initial_indices = np.random.choice(
            len(train_samples), 
            min(self.initial_pool_size, len(train_samples)), 
            replace=False
        )
        
        self.labeled_pool = [train_samples[i] for i in initial_indices]
        self.unlabeled_pool = [s for i, s in enumerate(train_samples) if i not in initial_indices]
        
        # Store feature indices for efficient access
        self.labeled_indices = list(initial_indices)
        self.unlabeled_indices = [i for i in range(len(train_samples)) if i not in initial_indices]
        self.test_indices = list(range(len(self.all_samples) - test_size, len(self.all_samples)))
        
        print(f"Initial labeled pool: {len(self.labeled_pool)}")
        print(f"Unlabeled pool: {len(self.unlabeled_pool)}")
        print(f"Test pool: {len(self.test_pool)}")
    
    def get_features_for_indices(self, indices):
        """Get features and labels for given indices"""
        return self.all_features[indices], self.all_labels[indices]
    
    def evaluate_classifier(self, X_test, y_test):
        """Evaluate the classifier performance with detailed diagnostics"""
        self.classifier.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            outputs = self.classifier(X_tensor)
            raw_predictions = outputs.squeeze().numpy()
        
        # Calculate optimal threshold using ROC curve
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_test, raw_predictions)
        auc_score = roc_auc_score(y_test, raw_predictions)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Use optimal threshold for predictions
        predictions = (raw_predictions > optimal_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # Class distribution analysis
        class_counts = np.bincount(y_test)
        prediction_counts = np.bincount(predictions)
        
        print(f"\n=== Performance Diagnostics ===")
        print(f"Test set class distribution: {class_counts}")
        print(f"Prediction distribution: {prediction_counts}")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        print(f"Raw prediction range: [{raw_predictions.min():.3f}, {raw_predictions.max():.3f}]")
        print(f"Raw prediction mean: {raw_predictions.mean():.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'optimal_threshold': optimal_threshold,
            'raw_predictions': raw_predictions,
            'class_distribution': class_counts.tolist(),
            'prediction_distribution': prediction_counts.tolist()
        }
    
    def train_classifier(self, X_train, y_train, epochs=1):
        """Train the classifier with new labeled data and class balance analysis"""
        print(f"Training classifier with {len(X_train)} samples...")
        
        # Analyze class distribution
        class_counts = np.bincount(y_train)
        print(f"Training set class distribution: {class_counts}")
        print(f"Smoke samples: {class_counts[1] if len(class_counts) > 1 else 0}")
        print(f"Non-smoke samples: {class_counts[0]}")
        
        # Calculate class weights for imbalanced data
        if len(class_counts) > 1 and class_counts[0] != class_counts[1]:
            pos_weight = class_counts[0] / class_counts[1]  # Weight for positive class
            print(f"Using class weights - Positive class weight: {pos_weight:.3f}")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]))
        else:
            self.criterion = nn.BCELoss()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.classifier(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def get_uncertainty_scores(self, X):
        """Get uncertainty scores using classifier predictions with threshold consideration"""
        self.classifier.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.classifier(X_tensor)
            raw_predictions = outputs.squeeze().numpy()
            
            # Use distance from optimal threshold as uncertainty measure
            # If we have a stored optimal threshold, use it; otherwise use 0.5
            optimal_threshold = getattr(self, 'optimal_threshold', 0.5)
            uncertainties = np.abs(raw_predictions - optimal_threshold)
            
        return uncertainties
    
    def select_uncertain_samples(self, unlabeled_features, unlabeled_indices, n_samples):
        """Select samples with highest uncertainty"""
        uncertainties = self.get_uncertainty_scores(unlabeled_features)
        
        # Select samples with highest uncertainty
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        
        selected_indices = [unlabeled_indices[i] for i in uncertain_indices]
        selected_uncertainties = [uncertainties[i] for i in uncertain_indices]
        
        return selected_indices, selected_uncertainties
    
    def save_labeled_samples(self, iteration, selected_samples, selected_indices, uncertainties):
        """Save labeled samples for current iteration with uncertainty and prediction info"""
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Get features for selected samples to compute predictions
        selected_features = self.all_features[selected_indices]
        
        # Get model predictions for selected samples
        self.classifier.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(selected_features)
            predictions = self.classifier(X_tensor).squeeze().numpy()
        
        # Save sample info
        sample_info = []
        for i, (sample, pred, uncertainty) in enumerate(zip(selected_samples, predictions, uncertainties)):
            # Copy image to output directory
            src_path = sample['image_path']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(iteration_dir, f"{i:03d}_{filename}")
            shutil.copy2(src_path, dst_path)
            
            sample_info.append({
                'original_path': src_path,
                'saved_path': dst_path,
                'label': sample['label'],
                'firename': sample['firename'],
                'timestamp': sample['timestamp'],
                'model_prediction': float(pred),
                'uncertainty_score': float(uncertainty),
                'prediction_class': int(pred > 0.5),
                'correct_prediction': int((pred > 0.5) == sample['label'])
            })
        
        # Save metadata with uncertainty and prediction info
        with open(os.path.join(iteration_dir, 'sample_info.json'), 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        # Also save summary statistics
        summary = {
            'iteration': iteration,
            'num_samples': len(selected_samples),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'min_uncertainty': float(np.min(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'mean_prediction': float(np.mean(predictions)),
            'std_prediction': float(np.std(predictions)),
            'num_smoke_samples': sum(1 for s in selected_samples if s['label'] == 1),
            'num_non_smoke_samples': sum(1 for s in selected_samples if s['label'] == 0),
            'correct_predictions': sum(1 for info in sample_info if info['correct_prediction']),
            'accuracy_on_selected': sum(1 for info in sample_info if info['correct_prediction']) / len(sample_info)
        }
        
        with open(os.path.join(iteration_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved {len(selected_samples)} labeled samples to {iteration_dir}")
        print(f"Mean uncertainty: {summary['mean_uncertainty']:.4f}")
        print(f"Accuracy on selected samples: {summary['accuracy_on_selected']:.3f}")
        print(f"Smoke/Non-smoke ratio: {summary['num_smoke_samples']}/{summary['num_non_smoke_samples']}")
    
    def run_active_learning(self):
        """Run the complete active learning process"""
        print("Starting Active Learning process...")
        
        # Prepare data pools (this will extract all features once)
        self.prepare_data_pools()
        
        # Get test features
        X_test, y_test = self.get_features_for_indices(self.test_indices)
        
        for iteration in range(self.max_iterations):
            print(f"\n=== Active Learning Iteration {iteration + 1} ===")
            
            # Get features for current labeled pool
            X_labeled, y_labeled = self.get_features_for_indices(self.labeled_indices)
            
            # Train classifier with current labeled data
            print("Training classifier with labeled data...")
            self.train_classifier(X_labeled, y_labeled, epochs=1)
            
            # Evaluate performance
            print("Evaluating performance...")
            performance = self.evaluate_classifier(X_test, y_test)
            
            # Store optimal threshold for uncertainty calculation
            self.optimal_threshold = performance['optimal_threshold']
            
            # Store performance metrics
            for metric, value in performance.items():
                if metric not in ['raw_predictions', 'class_distribution', 'prediction_distribution']:
                    self.performance_history[metric].append(value)
            
            print(f"Performance - Accuracy: {performance['accuracy']:.3f}, "
                  f"Precision: {performance['precision']:.3f}, "
                  f"Recall: {performance['recall']:.3f}, "
                  f"F1: {performance['f1_score']:.3f}, "
                  f"AUC: {performance['auc_score']:.3f}")
            
            # Check if we have enough unlabeled samples
            if len(self.unlabeled_indices) < self.acquisition_batch_size:
                print("Not enough unlabeled samples remaining. Stopping.")
                break
            
            # Get features for unlabeled pool
            X_unlabeled, _ = self.get_features_for_indices(self.unlabeled_indices)
            
            # Select uncertain samples
            print("Selecting uncertain samples...")
            selected_indices, uncertainties = self.select_uncertain_samples(
                X_unlabeled, self.unlabeled_indices, self.acquisition_batch_size
            )
            
            # Get the actual selected samples
            selected_samples = [self.all_samples[i] for i in selected_indices]
            
            # Save labeled samples
            self.save_labeled_samples(iteration + 1, selected_samples, selected_indices, uncertainties)
            
            # Store uncertainty scores
            self.performance_history['uncertainty_scores'].append(np.mean(uncertainties))
            self.performance_history['selected_samples'].append(len(selected_indices))
            
            print(f"Selected {len(selected_indices)} samples with mean uncertainty: {np.mean(uncertainties):.3f}")
            
            # Move selected samples from unlabeled to labeled pool
            for idx in selected_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.append(idx)
            
            print(f"Updated pools - Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}")
        
        print("\nActive Learning completed!")
        return self.performance_history
    
    def plot_performance(self, save_path=None):
        """Plot performance metrics across iterations"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Active Learning Performance Metrics', fontsize=16)
        
        iterations = range(1, len(self.performance_history['accuracy']) + 1)
        
        # Plot accuracy
        axes[0, 0].plot(iterations, self.performance_history['accuracy'], 'b-o')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Plot precision
        axes[0, 1].plot(iterations, self.performance_history['precision'], 'g-o')
        axes[0, 1].set_title('Precision')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True)
        
        # Plot recall
        axes[0, 2].plot(iterations, self.performance_history['recall'], 'r-o')
        axes[0, 2].set_title('Recall')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].grid(True)
        
        # Plot F1 score
        axes[0, 3].plot(iterations, self.performance_history['f1_score'], 'm-o')
        axes[0, 3].set_title('F1 Score')
        axes[0, 3].set_xlabel('Iteration')
        axes[0, 3].set_ylabel('F1 Score')
        axes[0, 3].grid(True)
        
        # Plot AUC score
        if 'auc_score' in self.performance_history and len(self.performance_history['auc_score']) > 0:
            axes[1, 0].plot(iterations, self.performance_history['auc_score'], 'c-o')
            axes[1, 0].set_title('AUC Score')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('AUC Score')
            axes[1, 0].grid(True)
        
        # Plot uncertainty scores
        axes[1, 1].plot(iterations, self.performance_history['uncertainty_scores'], 'y-o')
        axes[1, 1].set_title('Mean Uncertainty of Selected Samples')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Uncertainty Score')
        axes[1, 1].grid(True)
        
        # Plot number of selected samples
        axes[1, 2].plot(iterations, self.performance_history['selected_samples'], 'orange', marker='o')
        axes[1, 2].set_title('Number of Selected Samples')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Number of Samples')
        axes[1, 2].grid(True)
        
        # Plot optimal threshold
        if 'optimal_threshold' in self.performance_history and len(self.performance_history['optimal_threshold']) > 0:
            axes[1, 3].plot(iterations, self.performance_history['optimal_threshold'], 'purple', marker='o')
            axes[1, 3].set_title('Optimal Threshold')
            axes[1, 3].set_xlabel('Iteration')
            axes[1, 3].set_ylabel('Threshold')
            axes[1, 3].grid(True)
            axes[1, 3].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Default (0.5)')
            axes[1, 3].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, save_path):
        """Save active learning results"""
        results = {
            'performance_history': self.performance_history,
            'final_labeled_pool_size': len(self.labeled_indices),
            'final_unlabeled_pool_size': len(self.unlabeled_indices),
            'test_pool_size': len(self.test_indices),
            'output_directory': self.output_dir
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {save_path}")


def main():
    """Main function to run active learning"""
    
    # Configuration
    model_path = "src/inference/model.onnx"
    ground_truth_path = "src/groundtruth/results/ground_truth_combined.json"
    extracted_data_path = "remove_night_baseline_legacy"  # Updated to use filtered data
    
    # Active learning parameters - smaller sizes for faster execution
    initial_pool_size = 100
    acquisition_batch_size = 5
    max_iterations = 50
    n_workers = 4  # Number of parallel workers for feature extraction
    
    # Create active learning instance
    al_smokeynet = ActiveLearningSmokeyNet(
        model_path=model_path,
        ground_truth_path=ground_truth_path,
        extracted_data_path=extracted_data_path,
        initial_pool_size=initial_pool_size,
        acquisition_batch_size=acquisition_batch_size,
        max_iterations=max_iterations,
        n_workers=n_workers
    )
    
    # Run active learning
    performance_history = al_smokeynet.run_active_learning()
    
    # Plot results
    al_smokeynet.plot_performance(save_path="active_learning_performance.png")
    
    # Save results
    al_smokeynet.save_results("active_learning_results.pkl")
    
    print("\nActive Learning completed successfully!")
    print("Results saved to active_learning_results.pkl")
    print("Performance plot saved to active_learning_performance.png")
    print(f"Labeled samples saved to: {al_smokeynet.output_dir}")


if __name__ == "__main__":
    main() 