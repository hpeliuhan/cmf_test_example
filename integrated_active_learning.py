#!/usr/bin/env python3
"""
Integrated Active Learning for SmokeyNet with Enhanced Training Visualization
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import argparse
from PIL import Image

from integrated_smokeynet import ActiveLearningIntegratedSmokeyNet

class EnhancedActiveLearning(ActiveLearningIntegratedSmokeyNet):
    """Enhanced Active Learning with detailed training visualization and logging"""
    
    def __init__(self, *args, **kwargs):
        # Extract new parameters from kwargs before passing to parent
        self.detailed_logging = kwargs.pop('detailed_logging', True)
        self.device = kwargs.pop('device', 'cpu')
        self.epochs_per_iteration = kwargs.pop('epochs_per_iteration', 1)
        
        super().__init__(*args, **kwargs)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Add image cache for performance
        self.image_cache = {}
        self.max_cache_size = 1000  # Limit cache size
        
        self.training_history = {
            'loss_per_epoch': [],
            'weight_norms': [],
            'gradient_norms': [],
            'learning_rates': [],
            'class_weights': []
        }
        
        print(f"Enhanced Active Learning initialized on device: {self.device}")
    
    def load_and_cache_image(self, image_path):
        """Load image with caching for better performance"""
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Cache if we haven't exceeded limit
            if len(self.image_cache) < self.max_cache_size:
                self.image_cache[image_path] = img_array
                
            return img_array
        except Exception as e:
            if self.detailed_logging:
                print(f"Error loading image {image_path}: {e}")
            return None
    
    def train_model(self, X_train, y_train, epochs=None):
        """Enhanced training with detailed logging of weight updates and cost function"""
        if epochs is None:
            epochs = self.epochs_per_iteration
            
        print(f"\n=== Enhanced Training with {len(X_train)} samples for {epochs} epochs on {self.device} ===")
        
        if self.detailed_logging:
            print("Initial model state:")
            self._log_model_state()
        
        # Analyze class distribution
        class_counts = np.bincount(y_train)
        print(f"Training set class distribution: {class_counts}")
        
        # Calculate class weights for imbalanced data
        if len(class_counts) > 1 and class_counts[0] != class_counts[1]:
            pos_weight = class_counts[0] / class_counts[1]
            print(f"Using class weights - Positive class weight: {pos_weight:.3f}")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).to(self.device))
            self.training_history['class_weights'].append(pos_weight)
        else:
            self.criterion = nn.BCELoss()
            self.training_history['class_weights'].append(1.0)
        
        from PIL import Image as PILImage
        TARGET_SIZE = (224, 224)
        
        # PERFORMANCE OPTIMIZATION: Batch process images
        print("Preprocessing images in batch...")
        X_imgs = []
        valid_indices = []
        
        for idx in range(len(X_train)):
            sample = self.labeled_pool[idx] if idx < len(self.labeled_pool) else None
            if sample is not None:
                # Use cached image loading
                img = self.load_and_cache_image(sample['image_path'])
                if img is not None:
                    pil_img = PILImage.fromarray(img)
                    pil_img = pil_img.resize(TARGET_SIZE, PILImage.BILINEAR)
                    img = np.array(pil_img)
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    X_imgs.append(img)
                    valid_indices.append(idx)
                else:
                    # Skip invalid images instead of using zeros
                    continue
            else:
                continue
        
        if len(X_imgs) == 0:
            print("No valid images found for training!")
            return
            
        # Update labels to match valid images
        y_train_valid = y_train[valid_indices] if len(valid_indices) < len(y_train) else y_train
        
        X_imgs = np.stack(X_imgs)
        X_imgs_tensor = torch.FloatTensor(X_imgs).to(self.device)
        y_tensor = torch.FloatTensor(y_train_valid).unsqueeze(1).to(self.device)
        
        # PERFORMANCE OPTIMIZATION: Larger batch size and more workers
        batch_size = min(32, len(X_imgs))  # Increase batch size for better GPU utilization
        dataset = TensorDataset(X_imgs_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True if self.device == 'cuda' else False)
        
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        epoch_losses = []
        epoch_weight_norms = []
        epoch_gradient_norms = []
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_imgs, batch_y in dataloader:
                batch_imgs = batch_imgs.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                # PERFORMANCE OPTIMIZATION: Batch process instead of individual images
                batch_outputs = []
                for img in batch_imgs:
                    # Keep tensor operations on GPU
                    arr = img.cpu().numpy()
                    if arr.shape[0] == 3 and arr.shape[-1] != 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    arr = np.squeeze(arr)
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                    out = self.model(arr, arr)
                    batch_outputs.append(out)
                
                outputs = torch.stack(batch_outputs)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Log gradient norms less frequently for performance
                if self.detailed_logging and batch_count == 0 and epoch == 0:  # Only first batch of first epoch
                    grad_norm = self._calculate_gradient_norm()
                    epoch_gradient_norms.append(grad_norm)
                
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            
            # Calculate weight norms
            weight_norm = self._calculate_weight_norm()
            epoch_weight_norms.append(weight_norm)
            
            # Reduce logging frequency for performance
            if self.detailed_logging and epochs > 1 and (epoch + 1) % max(1, epochs // 2) == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Weight norm: {weight_norm:.4f}")
                if epoch_gradient_norms and len(epoch_gradient_norms) > epoch:
                    print(f"  Gradient norm: {epoch_gradient_norms[epoch]:.4f}")
                print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Store learning rate
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        # Store training history
        self.training_history['loss_per_epoch'].append(epoch_losses)
        self.training_history['weight_norms'].append(epoch_weight_norms)
        self.training_history['gradient_norms'].append(epoch_gradient_norms)
        
        if self.detailed_logging:
            print("\nFinal model state:")
            self._log_model_state()
            # Only plot training curves if more than 1 epoch
            if epochs > 1:
                self._plot_training_curves(epochs)
    
    def _log_model_state(self):
        """Log detailed model state information"""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
                print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
                if param.grad is not None:
                    print(f"    grad_norm: {param.grad.norm().item():.4f}")
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def _calculate_weight_norm(self):
        """Calculate L2 norm of all model weights"""
        total_norm = 0
        for param in self.model.parameters():
            if param.data is not None:
                param_norm = param.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _calculate_gradient_norm(self):
        """Calculate L2 norm of all gradients"""
        total_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _plot_training_curves(self, epochs):
        """Plot training curves for the current iteration"""
        if not self.training_history['loss_per_epoch']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - Iteration {len(self.training_history["loss_per_epoch"])}', fontsize=16)
        
        # Plot loss
        loss_curve = self.training_history['loss_per_epoch'][-1]
        axes[0, 0].plot(range(1, len(loss_curve) + 1), loss_curve, 'b-o')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Plot weight norms
        if self.training_history['weight_norms']:
            weight_curve = self.training_history['weight_norms'][-1]
            axes[0, 1].plot(range(1, len(weight_curve) + 1), weight_curve, 'g-o')
            axes[0, 1].set_title('Weight Norms')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('L2 Norm')
            axes[0, 1].grid(True)
        
        # Plot gradient norms
        if self.training_history['gradient_norms']:
            grad_curve = self.training_history['gradient_norms'][-1]
            axes[1, 0].plot(range(1, len(grad_curve) + 1), grad_curve, 'r-o')
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('L2 Norm')
            axes[1, 0].grid(True)
        
        # Plot learning rates
        if self.training_history['learning_rates']:
            lr_curve = self.training_history['learning_rates'][-epochs:]
            axes[1, 1].plot(range(1, len(lr_curve) + 1), lr_curve, 'm-o')
            axes[1, 1].set_title('Learning Rates')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save training curves
        training_curves_path = os.path.join(self.output_dir, f"training_curves_iter_{len(self.training_history['loss_per_epoch'])}.png")
        plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {training_curves_path}")
        plt.close()
    
    def save_detailed_results(self, save_path):
        """Save detailed results including training history"""
        results = {
            'performance_history': self.performance_history,
            'training_history': self.training_history,
            'final_labeled_pool_size': len(self.labeled_indices),
            'final_unlabeled_pool_size': len(self.unlabeled_indices),
            'test_pool_size': len(self.test_indices),
            'output_directory': self.output_dir,
            'model_architecture': str(self.model),
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        }
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Detailed results saved to {save_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Integrated Active Learning for SmokeyNet')
    parser.add_argument('--model_path', default="src/inference/model.onnx", help='Path to ONNX model')
    parser.add_argument('--ground_truth_path', default="src/groundtruth/results/ground_truth_combined.json", help='Path to ground truth JSON')
    parser.add_argument('--extracted_data_path', default="remove_night_baseline_legacy", help='Path to extracted data')
    parser.add_argument('--initial_pool_size', type=int, default=100, help='Initial labeled pool size')
    parser.add_argument('--acquisition_batch_size', type=int, default=5, help='Number of samples to acquire per iteration')
    parser.add_argument('--max_iterations', type=int, default=50, help='Maximum number of active learning iterations')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for parallel processing')
    parser.add_argument('--detailed_logging', action='store_true', default=True, help='Enable detailed logging and visualization')
    parser.add_argument('--test_mode', action='store_true', default=False, help='Run in test mode with smaller datasets')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of exploration runs to perform')
    parser.add_argument('--epochs_per_iteration', type=int, default=1, help='Number of epochs per training iteration')
    parser.add_argument('--fast_mode', action='store_true', default=False, help='Enable fast mode with minimal settings for quick testing')
    
    # Parse arguments, but make them optional by providing defaults
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    if args.fast_mode:
        print("Running in FAST MODE with minimal settings")
        args.initial_pool_size = 20
        args.acquisition_batch_size = 3
        args.max_iterations = 5
        args.epochs_per_iteration = 1
        args.detailed_logging = False  # Reduce logging for speed
    elif args.test_mode:
        print("Running in TEST MODE with reduced dataset sizes")
        args.initial_pool_size = 10
        args.acquisition_batch_size = 2
        args.max_iterations = 3
        args.epochs_per_iteration = 1  # Keep at 1 for speed
    
    # PERFORMANCE: Add fast mode for quicker testing
    if args.max_iterations > 10:
        print("Large iteration count detected - consider using --test_mode for faster testing")
    
    # PERFORMANCE: Optimize for speed if on CPU
    if device == 'cpu':
        print("CPU detected - applying CPU optimizations")
        torch.set_num_threads(min(8, torch.get_num_threads()))  # Limit CPU threads
    
    print("=== Integrated Active Learning Configuration ===")
    print(f"Model path: {args.model_path}")
    print(f"Ground truth path: {args.ground_truth_path}")
    print(f"Extracted data path: {args.extracted_data_path}")
    print(f"Initial pool size: {args.initial_pool_size}")
    print(f"Acquisition batch size: {args.acquisition_batch_size}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Epochs per iteration: {args.epochs_per_iteration}")
    print(f"Number of workers: {args.n_workers}")
    print(f"Detailed logging: {args.detailed_logging}")
    print(f"Test mode: {args.test_mode}")
    print(f"Device: {device}")
    print(f"Number of runs: {args.num_runs}")
    print("=" * 50)
    
    # Run multiple explorations
    all_results = []
    
    for run_idx in range(args.num_runs):
        print(f"\n{'='*60}")
        print(f"STARTING RUN {run_idx + 1} of {args.num_runs}")
        print(f"{'='*60}")
        
        # Create enhanced active learning instance
        al_smokeynet = EnhancedActiveLearning(
            model_path=args.model_path,
            ground_truth_path=args.ground_truth_path,
            extracted_data_path=args.extracted_data_path,
            initial_pool_size=args.initial_pool_size,
            acquisition_batch_size=args.acquisition_batch_size,
            max_iterations=args.max_iterations,
            n_workers=args.n_workers,
            detailed_logging=args.detailed_logging,
            device=device,
            epochs_per_iteration=args.epochs_per_iteration
        )
        
        # Run active learning
        print(f"\nStarting Enhanced Active Learning process (Run {run_idx + 1})...")
        performance_history = al_smokeynet.run_active_learning()
        
        # Save results for this run
        run_suffix = f"_run_{run_idx + 1}" if args.num_runs > 1 else ""
        
        # Plot results
        al_smokeynet.plot_performance(save_path=f"integrated_active_learning_performance{run_suffix}.png")
        
        # Save detailed results
        al_smokeynet.save_detailed_results(f"integrated_active_learning_detailed_results{run_suffix}.pkl")
        
        # Save standard results
        al_smokeynet.save_results(f"integrated_active_learning_results{run_suffix}.pkl")
        
        # Store results for comparison
        all_results.append({
            'run': run_idx + 1,
            'performance_history': performance_history,
            'output_dir': al_smokeynet.output_dir,
            'final_accuracy': performance_history['accuracy'][-1] if performance_history['accuracy'] else 0.0,
            'final_f1': performance_history['f1_score'][-1] if performance_history['f1_score'] else 0.0
        })
        
        print(f"\n=== Run {run_idx + 1} completed! ===")
        print(f"Final accuracy: {all_results[-1]['final_accuracy']:.3f}")
        print(f"Final F1 score: {all_results[-1]['final_f1']:.3f}")
        print(f"Output directory: {al_smokeynet.output_dir}")
    
    # Summary of all runs
    if args.num_runs > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY OF ALL {args.num_runs} RUNS")
        print(f"{'='*60}")
        
        accuracies = [r['final_accuracy'] for r in all_results]
        f1_scores = [r['final_f1'] for r in all_results]
        
        print(f"Final Accuracies: {[f'{acc:.3f}' for acc in accuracies]}")
        print(f"Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Best Accuracy: {np.max(accuracies):.3f} (Run {np.argmax(accuracies) + 1})")
        
        print(f"\nFinal F1 Scores: {[f'{f1:.3f}' for f1 in f1_scores]}")
        print(f"Mean F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        print(f"Best F1 Score: {np.max(f1_scores):.3f} (Run {np.argmax(f1_scores) + 1})")
        
        # Save combined results
        combined_results = {
            'all_runs': all_results,
            'summary': {
                'num_runs': args.num_runs,
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'best_accuracy': float(np.max(accuracies)),
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'best_f1': float(np.max(f1_scores))
            }
        }
        
        import pickle
        with open('integrated_active_learning_combined_results.pkl', 'wb') as f:
            pickle.dump(combined_results, f)
        
        print(f"\nCombined results saved to: integrated_active_learning_combined_results.pkl")
    
    print(f"\n{'='*60}")
    print("ALL RUNS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 