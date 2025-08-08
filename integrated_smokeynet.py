#!/usr/bin/env python3
"""
Integrated SmokeyNet with Active Learning
Converts ONNX model to PyTorch and adds trainable classifier
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
import pickle
from tqdm import tqdm
import sys
from pathlib import Path
import multiprocessing as mp
from functools import partial
import shutil
import datetime
import onnx
import onnxruntime

# Add the inference module to path
sys.path.append('src/inference')
from smokeynet import SmokeyNet

class IntegratedSmokeyNet(nn.Module):
    """Integrated SmokeyNet with trainable classifier head and uncertainty estimation"""
    
    def __init__(self, onnx_model_path, input_size=45, hidden_size=64):  # Changed to 45 for 5x9 tiles
        super(IntegratedSmokeyNet, self).__init__()
        
        # Initialize SmokeyNet for feature extraction (for inference)
        self.smokeynet_inference = SmokeyNet(onnx_model_path)
        
        # Add trainable classifier head with dropout for uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),  # Higher dropout for better uncertainty estimation
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Higher dropout for better uncertainty estimation
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Enable dropout during inference for uncertainty estimation
        self.mc_dropout_enabled = False
        self.n_mc_samples = 5  # PERFORMANCE: Reduced from 10 to 5 for speed
        
        # Store dropout layers for easier access
        self.dropout_layers = []
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                self.dropout_layers.append(module)
    
    def extract_features(self, current_img, previous_img):
        """Extract features using SmokeyNet inference"""
        # Use the original SmokeyNet for feature extraction
        image_preds, tile_preds, tile_probs = self.smokeynet_inference.inference(
            current_img, previous_img, smoke_threshold=0.5
        )
        
        # Use tile probabilities as features (should be 45 for 5x9 grid)
        features = tile_probs.flatten()
        
        # Ensure we have exactly 45 features (5x9 grid)
        if len(features) != 45:
            print(f"Warning: Expected 45 features, got {len(features)}. Padding/truncating...")
            if len(features) > 45:
                features = features[:45]
            else:
                features = np.pad(features, (0, 45 - len(features)), 'constant')
        
        return torch.FloatTensor(features)
    
    def enable_mc_dropout(self, enabled=True):
        """Enable/disable Monte Carlo dropout for uncertainty estimation"""
        self.mc_dropout_enabled = enabled
        for dropout_layer in self.dropout_layers:
            if enabled:
                dropout_layer.train()  # Enable dropout during inference
            else:
                dropout_layer.eval()   # Disable dropout during normal inference
    
    def predict_with_uncertainty(self, current_img, previous_img, n_samples=5):  # PERFORMANCE: Reduced default
        """Get prediction with uncertainty using Monte Carlo Dropout"""
        # Enable MC dropout
        self.enable_mc_dropout(True)
        
        features = self.extract_features(current_img, previous_img)
        predictions = []
        
        with torch.no_grad():
            for i in range(n_samples):
                output = self.classifier(features)
                pred = output.item()
                predictions.append(pred)
                # PERFORMANCE: Remove debug printing for speed
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions)
        
        # Compute uncertainty using multiple methods
        std_uncertainty = np.std(predictions)  # Standard deviation as uncertainty
        
        # Also compute entropy-based uncertainty
        # Convert predictions to probabilities and compute entropy
        probs = np.array(predictions)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        probs = np.clip(probs, epsilon, 1 - epsilon)
        entropy = -np.mean(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
        
        # Use the maximum of std and entropy for more robust uncertainty
        uncertainty = max(std_uncertainty, entropy)
        
        # Disable MC dropout
        self.enable_mc_dropout(False)
        
        # PERFORMANCE: Remove excessive debug printing
        # Only print if uncertainty is unusually high or low
        if uncertainty > 0.3 or uncertainty < 0.01:
            print(f"  Unusual uncertainty: {uncertainty:.4f} (pred: {mean_prediction:.4f})")
        
        return mean_prediction, uncertainty
    
    def get_smoke_localization(self, current_img, previous_img, smoke_threshold=0.5):
        """Get smoke localization information with proper 5x9 tile grid"""
        # Get SmokeyNet predictions
        image_preds, tile_preds, tile_probs = self.smokeynet_inference.inference(
            current_img, previous_img, smoke_threshold
        )
        
        # Ensure we have the correct tile grid (5x9 = 45 tiles)
        tile_probs_flat = tile_probs.flatten()
        if len(tile_probs_flat) != 45:
            print(f"Warning: Expected 45 tiles, got {len(tile_probs_flat)}. Adjusting...")
            if len(tile_probs_flat) > 45:
                tile_probs_flat = tile_probs_flat[:45]
            else:
                tile_probs_flat = np.pad(tile_probs_flat, (0, 45 - len(tile_probs_flat)), 'constant')
        
        # Reshape to 5x9 grid
        tile_probs_grid = tile_probs_flat.reshape(5, 9)
        
        # Count smoke tiles
        smoke_tiles = np.sum(tile_preds[0])
        
        # Get smoke tile indices and probabilities
        smoke_tile_indices = np.where(tile_preds[0] == 1)[0]
        smoke_tile_probabilities = tile_probs[0][smoke_tile_indices]
        
        # Create smoke locations with only index and probability
        smoke_locations = []
        for idx, prob in zip(smoke_tile_indices, smoke_tile_probabilities):
            confidence = 'high' if prob > 0.8 else 'medium' if prob > 0.6 else 'low'
            smoke_locations.append({
                'tile_index': int(idx),
                'probability': float(prob),
                'confidence': confidence
            })
        
        return {
            'smoke_tiles': int(smoke_tiles),
            'image_has_smoke': bool(image_preds[0]),
            'tile_probabilities': tile_probs_flat.tolist(),
            'tile_grid_shape': [5, 9],  # 5x9 grid
            'smoke_locations': smoke_locations
        }
    
    def create_smoke_visualization(self, src_path, smoke_localization, annotated_path):
        """Create and save an annotated image with 5x9 tile grid and probabilities overlay."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.open(src_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            width, height = img.size

            # Get tile probabilities for 5x9 grid
            tile_probs = smoke_localization.get('tile_probabilities', [])
            grid_shape = smoke_localization.get('tile_grid_shape', [5, 9])
            grid_rows, grid_cols = grid_shape
            
            if len(tile_probs) != grid_rows * grid_cols:
                print(f"Warning: Expected {grid_rows * grid_cols} tiles, got {len(tile_probs)}")
                # Pad or truncate to correct size
                if len(tile_probs) > grid_rows * grid_cols:
                    tile_probs = tile_probs[:grid_rows * grid_cols]
                else:
                    tile_probs = tile_probs + [0.0] * (grid_rows * grid_cols - len(tile_probs))
            
            tile_w = width // grid_cols
            tile_h = height // grid_rows

            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(10, tile_h//8))
            except Exception:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(10, tile_h//8))
                except Exception:
                    font = ImageFont.load_default()

            for idx, prob in enumerate(tile_probs):
                # Convert to float if it's not already
                try:
                    prob = float(prob)
                except (ValueError, TypeError):
                    prob = 0.0
                
                row = idx // grid_cols
                col = idx % grid_cols
                x0 = col * tile_w
                y0 = row * tile_h
                x1 = x0 + tile_w
                y1 = y0 + tile_h
                
                # Draw rectangle with color based on probability
                if prob > 0.7:
                    outline_color = "red"
                elif prob > 0.5:
                    outline_color = "orange"
                else:
                    outline_color = "green"
                
                draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=2)
                
                # Draw probability text
                try:
                    text = f"{prob:.2f}"
                    # Use getbbox instead of textsize (deprecated)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    text_x = x0 + (tile_w - text_w) // 2
                    text_y = y0 + (tile_h - text_h) // 2
                    draw.text((text_x, text_y), text, fill="yellow", font=font)
                except Exception as e:
                    print(f"Error drawing text for tile {idx}: {e}")

            # Add overall image information
            image_has_smoke = smoke_localization.get('image_has_smoke', False)
            smoke_tiles = smoke_localization.get('smoke_tiles', 0)
            info_text = f"Smoke: {image_has_smoke}, Tiles: {smoke_tiles}, Grid: {grid_rows}x{grid_cols}"
            draw.text((10, 10), info_text, fill="white", font=ImageFont.load_default())

            img.save(annotated_path)
            print(f"Successfully created annotated image: {annotated_path}")
            return True
        except Exception as e:
            print(f"Error creating annotated image for {src_path}: {e}")
            return False
    
    def forward(self, current_img, previous_img):
        """Forward pass through integrated model"""
        # Extract features
        features = self.extract_features(current_img, previous_img)
        
        # Pass through classifier
        output = self.classifier(features)
        
        return output
    
    def predict_tile_level(self, current_img, previous_img):
        """Get tile-level predictions (like original SmokeyNet)"""
        image_preds, tile_preds, tile_probs = self.smokeynet_inference.inference(
            current_img, previous_img, smoke_threshold=0.5
        )
        return image_preds, tile_preds, tile_probs
    
    def predict_image_level(self, current_img, previous_img):
        """Get image-level prediction from integrated model"""
        self.eval()
        with torch.no_grad():
            output = self.forward(current_img, previous_img)
            return output.item()

class ActiveLearningIntegratedSmokeyNet:
    def __init__(self, model_path, ground_truth_path, extracted_data_path, 
                 initial_pool_size=100, acquisition_batch_size=20, max_iterations=10,
                 n_workers=None, device='cpu'):
        """
        Initialize Active Learning for Integrated SmokeyNet
        """
        self.model_path = model_path
        self.ground_truth_path = ground_truth_path
        self.extracted_data_path = extracted_data_path
        self.initial_pool_size = initial_pool_size
        self.acquisition_batch_size = acquisition_batch_size
        self.max_iterations = max_iterations
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.device = device
        
        # Initialize integrated model
        self.model = IntegratedSmokeyNet(model_path).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
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
            'selected_samples': [],
            'tile_accuracy': [],
            'image_accuracy': [],
            'iteration_time': [],  # Track time per iteration
            'labeled_pool_size': [],  # Track pool sizes
            'unlabeled_pool_size': []
        }
        
        # Gaussian Process for uncertainty estimation
        self.gp_model = None
        
        # Create output directory
        self.output_dir = f"integrated_active_learning_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Active Learning initialized on device: {self.device}")
    
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
        cache_file = f"integrated_feature_cache_{max_samples}.npz"
        meta_file = f"integrated_feature_cache_{max_samples}_meta.json"
        
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
                        step = len(smoke_samples) // target_smoke_per_firename
                        selected_smoke = smoke_samples[::step][:target_smoke_per_firename]
                    selected_samples.extend(selected_smoke)
                    total_smoke += len(selected_smoke)
                
                # Sample non-smoke samples
                if len(non_smoke_samples) > 0:
                    if len(non_smoke_samples) <= target_non_smoke_per_firename:
                        selected_non_smoke = non_smoke_samples
                    else:
                        step = len(non_smoke_samples) // target_non_smoke_per_firename
                        selected_non_smoke = non_smoke_samples[::step][:target_non_smoke_per_firename]
                    selected_non_smoke.extend(selected_non_smoke)
                    total_non_smoke += len(selected_non_smoke)
            
            # If we still have too many, take a balanced random subset
            if len(selected_samples) > max_samples:
                smoke_selected = [s for s in selected_samples if s['label'] == 1]
                non_smoke_selected = [s for s in selected_samples if s['label'] == 0]
                
                final_smoke_count = max_samples // 2
                final_non_smoke_count = max_samples - final_smoke_count
                
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
        
        # Extract features using the integrated model
        features = []
        labels = []
        valid_samples = []
        
        for sample in tqdm(all_samples, desc="Extracting features"):
            current_img = self.load_image(sample['image_path'])
            if current_img is not None:
                # For simplicity, use same image as previous (in practice, get actual previous)
                previous_img = current_img
                
                try:
                    # Extract features using integrated model
                    feature = self.model.extract_features(current_img, previous_img)
                    features.append(feature.numpy())
                    labels.append(sample['label'])
                    valid_samples.append(sample)
                except Exception as e:
                    print(f"Error extracting features from {sample['image_path']}: {e}")
        
        # Store features and samples
        self.all_samples = valid_samples
        self.all_features = np.array(features)
        self.all_labels = np.array(labels)
        
        print(f"Successfully extracted features for {len(valid_samples)} samples")
        
        # Save to cache
        np.savez_compressed(cache_file, features=self.all_features, labels=self.all_labels)
        with open(meta_file, 'w') as f:
            json.dump(self.all_samples, f, indent=2)
        print(f"Saved features to cache: {cache_file}")
        
        return valid_samples, self.all_features, self.all_labels
    
    def prepare_data_pools(self):
        """Prepare initial data pools for active learning using cached features"""
        print("Preparing data pools...")
        
        # Extract all features if not already done
        if not hasattr(self, 'all_features'):
            self.extract_all_features_parallel()
        
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
        """Get indices and labels for given indices (for end-to-end training)"""
        # For end-to-end training, we return indices and labels directly
        # The actual feature extraction happens during training
        return indices, self.all_labels[indices]
    
    def train_model(self, X_train, y_train, epochs=1):
        """Train the entire integrated model (feature extractor + classifier) end-to-end with new labeled data."""
        print(f"Training integrated model (end-to-end) with {len(X_train)} samples...")

        # Analyze class distribution
        class_counts = np.bincount(y_train)
        print(f"Training set class distribution: {class_counts}")

        # Calculate class weights for imbalanced data
        if len(class_counts) > 1 and class_counts[0] != class_counts[1]:
            pos_weight = class_counts[0] / class_counts[1]
            print(f"Using class weights - Positive class weight: {pos_weight:.3f}")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).to(self.device))
        else:
            self.criterion = nn.BCELoss()

        from PIL import Image as PILImage
        TARGET_SIZE = (224, 224)  # (width, height)
        # Prepare dataset: need to load images for each sample
        X_imgs = []
        for idx in range(len(X_train)):
            sample = self.labeled_pool[idx] if idx < len(self.labeled_pool) else None
            if sample is not None:
                img = self.load_image(sample['image_path'])
                if img is not None:
                    # Resize image to target size (width, height)
                    pil_img = PILImage.fromarray(img)
                    pil_img = pil_img.resize(TARGET_SIZE, PILImage.BILINEAR)
                    img = np.array(pil_img)
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    X_imgs.append(img)
                else:
                    X_imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                X_imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
        X_imgs = np.stack(X_imgs)
        X_imgs_tensor = torch.FloatTensor(X_imgs).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        # For now, use previous_img = current_img (can be improved for temporal)
        dataset = TensorDataset(X_imgs_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Update optimizer to all params
        for epoch in range(epochs):
            total_loss = 0
            for batch_imgs, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = []
                for img in batch_imgs:
                    arr = img.cpu().numpy()
                    # If shape is (3, 224, 224), transpose to (224, 224, 3)
                    if arr.shape[0] == 3 and arr.shape[-1] != 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    arr = np.squeeze(arr)
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                    if arr.shape != (224, 224, 3):
                        print(f"Warning: image shape before model call: {arr.shape}, dtype: {arr.dtype}")
                    out = self.model(arr, arr)
                    outputs.append(out)
                outputs = torch.stack(outputs)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def evaluate_model(self, X_test, y_test):
        """Evaluate the integrated model performance with end-to-end inference"""
        self.model.eval()
        
        # Process test samples end-to-end
        predictions = []
        for i, sample_idx in enumerate(X_test):  # X_test contains sample indices
            sample = self.all_samples[sample_idx]
            current_img = self.load_image(sample['image_path'])
            if current_img is not None:
                # Use same image as previous for simplicity
                previous_img = current_img
                
                # Get prediction using the integrated model
                with torch.no_grad():
                    output = self.model(current_img, previous_img)
                    predictions.append(output.item())
            else:
                predictions.append(0.0)  # Default prediction for failed images
        
        predictions = np.array(predictions)
        
        # Calculate optimal threshold using ROC curve
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        auc_score = roc_auc_score(y_test, predictions)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Use optimal threshold for predictions
        binary_predictions = (predictions > optimal_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, binary_predictions)
        precision = precision_score(y_test, binary_predictions, zero_division=0)
        recall = recall_score(y_test, binary_predictions, zero_division=0)
        f1 = f1_score(y_test, binary_predictions, zero_division=0)
        
        # Class distribution analysis
        class_counts = np.bincount(y_test)
        prediction_counts = np.bincount(binary_predictions)
        
        print(f"\n=== Performance Diagnostics ===")
        print(f"Test set class distribution: {class_counts}")
        print(f"Prediction distribution: {prediction_counts}")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        print(f"Raw prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"Raw prediction mean: {predictions.mean():.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'optimal_threshold': optimal_threshold,
            'raw_predictions': predictions,
            'class_distribution': class_counts.tolist(),
            'prediction_distribution': prediction_counts.tolist()
        }
    
    def train_gaussian_process(self, X, y):
        """Train Gaussian Process model for uncertainty estimation"""
        # Define kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        
        # Train GP
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        self.gp_model.fit(X, y)
    
    def get_uncertainty_scores(self, X):
        """Get uncertainty scores using Monte Carlo Dropout instead of Gaussian Process"""
        # Use Monte Carlo Dropout for uncertainty estimation
        uncertainties = []
        
        for i, sample_idx in enumerate(X):  # X contains sample indices
            sample = self.all_samples[sample_idx]
            current_img = self.load_image(sample['image_path'])
            if current_img is not None:
                # Use same image as previous for simplicity
                previous_img = current_img
                
                # Get prediction with uncertainty using Monte Carlo Dropout
                mean_pred, uncertainty = self.model.predict_with_uncertainty(
                    current_img, previous_img, n_samples=5
                )
                uncertainties.append(uncertainty)
            else:
                uncertainties.append(1.0)  # High uncertainty for failed images
        
        return np.array(uncertainties)
    
    def select_uncertain_samples(self, unlabeled_features, unlabeled_indices, n_samples):
        """Select samples with highest uncertainty using Monte Carlo Dropout"""
        # Get uncertainty scores using Monte Carlo Dropout
        uncertainties = self.get_uncertainty_scores(unlabeled_indices)
        
        # Select samples with highest uncertainty
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        
        selected_indices = [unlabeled_indices[i] for i in uncertain_indices]
        selected_uncertainties = [uncertainties[i] for i in uncertain_indices]
        
        print(f"Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
        print(f"Selected uncertainties: {[f'{u:.4f}' for u in selected_uncertainties]}")
        
        return selected_indices, selected_uncertainties
    
    def save_labeled_samples(self, iteration, selected_samples, selected_indices, uncertainties):
        """Save labeled samples for current iteration with Monte Carlo Dropout uncertainty and annotated images."""
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        annotated_dir = os.path.join(iteration_dir, "annotated_images")
        os.makedirs(annotated_dir, exist_ok=True)

        # Save sample info
        sample_info = []
        for i, (sample, uncertainty) in enumerate(zip(selected_samples, uncertainties)):
            # Copy image to output directory
            src_path = sample['image_path']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(iteration_dir, f"{i:03d}_{filename}")
            shutil.copy2(src_path, dst_path)

            # Save annotated image with 5x9 tile grid
            annotated_path = os.path.join(annotated_dir, f"{i:03d}_annotated_{filename}")
            # Load current and previous image (for now, use same image for both)
            current_img = self.load_image(src_path)
            previous_img = current_img
            smoke_localization = self.model.get_smoke_localization(current_img, previous_img)
            success = self.model.create_smoke_visualization(src_path, smoke_localization, annotated_path)
            if not success:
                print(f"Failed to create annotated image: {os.path.basename(annotated_path)}")

            # Get Monte Carlo Dropout prediction and uncertainty
            if current_img is not None:
                mean_pred, mc_uncertainty = self.model.predict_with_uncertainty(
                    current_img, previous_img, n_samples=5
                )
            else:
                mean_pred, mc_uncertainty = 0.0, 1.0

            # Extract tile information for detailed analysis
            tile_probabilities = smoke_localization.get('tile_probabilities', [])
            smoke_locations = smoke_localization.get('smoke_locations', [])
            tile_grid_shape = smoke_localization.get('tile_grid_shape', [5, 9])
            
            # Calculate tile statistics
            if tile_probabilities:
                tile_probs_array = np.array(tile_probabilities)
                tile_stats = {
                    'mean_probability': float(np.mean(tile_probs_array)),
                    'max_probability': float(np.max(tile_probs_array)),
                    'min_probability': float(np.min(tile_probs_array)),
                    'std_probability': float(np.std(tile_probs_array)),
                    'num_high_prob_tiles': int(np.sum(tile_probs_array > 0.7)),
                    'num_medium_prob_tiles': int(np.sum((tile_probs_array > 0.5) & (tile_probs_array <= 0.7))),
                    'num_low_prob_tiles': int(np.sum(tile_probs_array <= 0.5)),
                    'tile_grid_shape': tile_grid_shape
                }
            else:
                tile_stats = {
                    'mean_probability': 0.0,
                    'max_probability': 0.0,
                    'min_probability': 0.0,
                    'std_probability': 0.0,
                    'num_high_prob_tiles': 0,
                    'num_medium_prob_tiles': 0,
                    'num_low_prob_tiles': 0,
                    'tile_grid_shape': tile_grid_shape
                }

            sample_info.append({
                'original_path': src_path,
                'saved_path': dst_path,
                'annotated_path': annotated_path,
                'label': sample['label'],
                'firename': sample['firename'],
                'timestamp': sample['timestamp'],
                'model_prediction': float(mean_pred),
                'mc_uncertainty': float(mc_uncertainty),
                'selection_uncertainty': float(uncertainty),  # Uncertainty used for selection
                'prediction_class': int(mean_pred > 0.5),
                'correct_prediction': int((mean_pred > 0.5) == sample['label']),
                'smoke_localization': {
                    'image_has_smoke': smoke_localization.get('image_has_smoke', False),
                    'smoke_tiles': smoke_localization.get('smoke_tiles', 0),
                    'tile_probabilities': tile_probabilities,
                    'tile_grid_shape': tile_grid_shape,
                    'smoke_locations': smoke_locations,
                    'tile_statistics': tile_stats
                }
            })

        # Save metadata with uncertainty and prediction info
        with open(os.path.join(iteration_dir, 'sample_info.json'), 'w') as f:
            json.dump(sample_info, f, indent=2)

        # Also save summary statistics
        summary = {
            'iteration': iteration,
            'num_samples': len(selected_samples),
            'mean_mc_uncertainty': float(np.mean([s['mc_uncertainty'] for s in sample_info])),
            'std_mc_uncertainty': float(np.std([s['mc_uncertainty'] for s in sample_info])),
            'min_mc_uncertainty': float(np.min([s['mc_uncertainty'] for s in sample_info])),
            'max_mc_uncertainty': float(np.max([s['mc_uncertainty'] for s in sample_info])),
            'mean_prediction': float(np.mean([s['model_prediction'] for s in sample_info])),
            'std_prediction': float(np.std([s['model_prediction'] for s in sample_info])),
            'num_smoke_samples': sum(1 for s in selected_samples if s['label'] == 1),
            'num_non_smoke_samples': sum(1 for s in selected_samples if s['label'] == 0),
            'correct_predictions': sum(1 for info in sample_info if info['correct_prediction']),
            'accuracy_on_selected': sum(1 for info in sample_info if info['correct_prediction']) / len(sample_info),
            'tile_grid_shape': tile_grid_shape
        }

        with open(os.path.join(iteration_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved {len(selected_samples)} labeled samples to {iteration_dir}")
        print(f"Mean MC uncertainty: {summary['mean_mc_uncertainty']:.4f}")
        print(f"MC uncertainty range: [{summary['min_mc_uncertainty']:.4f}, {summary['max_mc_uncertainty']:.4f}]")
        print(f"Accuracy on selected samples: {summary['accuracy_on_selected']:.3f}")
        print(f"Smoke/Non-smoke ratio: {summary['num_smoke_samples']}/{summary['num_non_smoke_samples']}")
        print(f"Tile grid: {tile_grid_shape[0]}x{tile_grid_shape[1]} = {tile_grid_shape[0] * tile_grid_shape[1]} tiles")
    
    def run_active_learning(self):
        """Run the complete active learning process with Monte Carlo Dropout uncertainty"""
        print("Starting Integrated Active Learning process with Monte Carlo Dropout...")
        
        # Prepare data pools
        self.prepare_data_pools()
        
        # Get test features
        X_test, y_test = self.get_features_for_indices(self.test_indices)
        
        # Create CSV file for performance logging
        import csv
        import time
        csv_path = os.path.join(self.output_dir, 'performance_log.csv')
        csv_headers = ['iteration', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 
                      'optimal_threshold', 'mean_uncertainty', 'selected_samples', 
                      'labeled_pool_size', 'unlabeled_pool_size', 'iteration_time_sec']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
        
        print(f"Performance logging to: {csv_path}")
        print(f"{'='*80}")
        print(f"{'Iteration':<10} {'Accuracy':<9} {'Precision':<9} {'Recall':<9} {'F1':<9} {'AUC':<9} {'Uncertainty':<11} {'Time(s)':<8} {'Pools':<15}")
        print(f"{'='*80}")
        
        for iteration in range(self.max_iterations):
            iteration_start_time = time.time()
            print(f"\n=== Active Learning Iteration {iteration + 1} ===")
            
            # Get features for current labeled pool
            X_labeled, y_labeled = self.get_features_for_indices(self.labeled_indices)
            
            # Train integrated model with current labeled data (end-to-end training)
            print("Training integrated model with labeled data...")
            self.train_model(X_labeled, y_labeled, epochs=1)
            
            # Evaluate performance
            print("Evaluating performance...")
            performance = self.evaluate_model(X_test, y_test)
            
            # Store optimal threshold for uncertainty calculation
            self.optimal_threshold = performance['optimal_threshold']
            
            # Store performance metrics
            for metric, value in performance.items():
                if metric not in ['raw_predictions', 'class_distribution', 'prediction_distribution']:
                    self.performance_history[metric].append(value)
            
            # Check if we have enough unlabeled samples
            if len(self.unlabeled_indices) < self.acquisition_batch_size:
                print("Not enough unlabeled samples remaining. Stopping.")
                break
            
            # Select uncertain samples using Monte Carlo Dropout
            print("Selecting uncertain samples using Monte Carlo Dropout...")
            selected_indices, uncertainties = self.select_uncertain_samples(
                None, self.unlabeled_indices, self.acquisition_batch_size  # No need for features
            )
            
            # Get the actual selected samples
            selected_samples = [self.all_samples[i] for i in selected_indices]
            
            # Save labeled samples with detailed uncertainty information
            self.save_labeled_samples(iteration + 1, selected_samples, selected_indices, uncertainties)
            
            # Calculate iteration time
            iteration_time = time.time() - iteration_start_time
            
            # Store additional metrics
            mean_uncertainty = np.mean(uncertainties)
            self.performance_history['uncertainty_scores'].append(mean_uncertainty)
            self.performance_history['selected_samples'].append(len(selected_indices))
            self.performance_history['iteration_time'].append(iteration_time)
            self.performance_history['labeled_pool_size'].append(len(self.labeled_indices))
            self.performance_history['unlabeled_pool_size'].append(len(self.unlabeled_indices))
            
            # Enhanced console output
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} SUMMARY:")
            print(f"{'='*60}")
            print(f"Performance Metrics:")
            print(f"  • Accuracy:     {performance['accuracy']:.4f}")
            print(f"  • Precision:    {performance['precision']:.4f}")
            print(f"  • Recall:       {performance['recall']:.4f}")
            print(f"  • F1 Score:     {performance['f1_score']:.4f}")
            print(f"  • AUC Score:    {performance['auc_score']:.4f}")
            print(f"  • Threshold:    {performance['optimal_threshold']:.4f}")
            print(f"Active Learning Metrics:")
            print(f"  • Mean Uncertainty: {mean_uncertainty:.4f}")
            print(f"  • Selected Samples: {len(selected_indices)}")
            print(f"  • Iteration Time:   {iteration_time:.1f}s")
            print(f"Pool Status:")
            print(f"  • Labeled Pool:     {len(self.labeled_indices)} samples")
            print(f"  • Unlabeled Pool:   {len(self.unlabeled_indices)} samples")
            print(f"  • Test Pool:        {len(self.test_indices)} samples")
            
            # Compact summary line for tracking
            pools_info = f"{len(self.labeled_indices)}/{len(self.unlabeled_indices)}"
            print(f"{iteration+1:<10} {performance['accuracy']:<9.4f} {performance['precision']:<9.4f} {performance['recall']:<9.4f} {performance['f1_score']:<9.4f} {performance['auc_score']:<9.4f} {mean_uncertainty:<11.4f} {iteration_time:<8.1f} {pools_info:<15}")
            
            # Save to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    iteration + 1,
                    performance['accuracy'],
                    performance['precision'], 
                    performance['recall'],
                    performance['f1_score'],
                    performance['auc_score'],
                    performance['optimal_threshold'],
                    mean_uncertainty,
                    len(selected_indices),
                    len(self.labeled_indices),
                    len(self.unlabeled_indices),
                    iteration_time
                ])
            
            # Move selected samples from unlabeled to labeled pool
            for idx in selected_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.append(idx)
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY:")
        print(f"{'='*80}")
        total_time = sum(self.performance_history['iteration_time'])
        final_accuracy = self.performance_history['accuracy'][-1] if self.performance_history['accuracy'] else 0
        final_f1 = self.performance_history['f1_score'][-1] if self.performance_history['f1_score'] else 0
        print(f"• Total Iterations: {len(self.performance_history['accuracy'])}")
        print(f"• Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"• Average Time per Iteration: {total_time/len(self.performance_history['accuracy']):.1f}s")
        print(f"• Final Accuracy: {final_accuracy:.4f}")
        print(f"• Final F1 Score: {final_f1:.4f}")
        print(f"• Final Labeled Pool: {len(self.labeled_indices)} samples")
        print(f"• Performance Log: {csv_path}")
        print(f"• Output Directory: {self.output_dir}")
        
        print("\nIntegrated Active Learning completed!")
        return self.performance_history
    
    def plot_performance(self, save_path=None):
        """Plot performance metrics across iterations"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Integrated Active Learning Performance Metrics', fontsize=16)
        
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
    """Main function to run integrated active learning"""
    
    # Configuration
    model_path = "src/inference/model.onnx"
    ground_truth_path = "src/groundtruth/results/ground_truth_combined.json"
    extracted_data_path = "remove_night_baseline_legacy"
    
    # Active learning parameters
    initial_pool_size = 100
    acquisition_batch_size = 5
    max_iterations = 50
    n_workers = 4
    
    # Create integrated active learning instance
    al_smokeynet = ActiveLearningIntegratedSmokeyNet(
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
    al_smokeynet.plot_performance(save_path="integrated_active_learning_performance.png")
    
    # Save results
    al_smokeynet.save_results("integrated_active_learning_results.pkl")
    
    print("\nIntegrated Active Learning completed successfully!")
    print("Results saved to integrated_active_learning_results.pkl")
    print("Performance plot saved to integrated_active_learning_performance.png")
    print(f"Labeled samples saved to: {al_smokeynet.output_dir}")


if __name__ == "__main__":
    main() 