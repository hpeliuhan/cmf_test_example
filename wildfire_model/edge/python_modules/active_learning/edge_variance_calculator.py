"""
Edge Variance Calculator for Active Learning
Computes variances on extracted frames and augmentations, sends to fine-tune server
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from PIL import Image


class EdgeVarianceCalculator:
    def __init__(self, model, fine_tune_server_url: str, logger=None):
        """
        Initialize Edge Variance Calculator
        
        Args:
            model: Loaded TensorFlow model for inference
            fine_tune_server_url: URL of the fine-tune server
            logger: Logger instance
        """
        self.model = model
        self.fine_tune_server_url = fine_tune_server_url
        self.logger = logger or logging.getLogger(__name__)
        
        # Create data generator for augmentations
        self.augmentation_generator = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            rescale=1./255
        )

    def extract_frames_with_augmentations(self, video_path: str, output_dir: str, 
                                        frame_interval: int = 10) -> str:
        """
        Extract frames from video and create augmentations
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames and augmentations
            frame_interval: Extract every nth frame
            
        Returns:
            str: Path to directory containing frames and augmentations
        """
        self.logger.info(f"Extracting frames with augmentations from {video_path}")
        
        # Create output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(output_dir, f"{video_name}_augmented")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video file: {video_path}")
        
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Resize frame to target size
                    frame_resized = cv2.resize(frame, (224, 224))
                    
                    # Save original frame
                    orig_path = os.path.join(frames_dir, f"{video_name}_frame_{saved_count:04d}_orig.jpg")
                    cv2.imwrite(orig_path, frame_resized)
                    
                    # Create augmentations using OpenCV
                    augmentations = self._create_augmentations_opencv(frame_resized)
                    
                    # Save augmentations
                    for i, aug_frame in enumerate(augmentations):
                        aug_path = os.path.join(frames_dir, f"{video_name}_frame_{saved_count:04d}_aug{i}.jpg")
                        cv2.imwrite(aug_path, aug_frame)
                    
                    saved_count += 1
                
                frame_count += 1
            
        finally:
            cap.release()
        
        # Calculate total images (original + augmentations)
        total_images = saved_count * 5  # 1 original + 4 augmentations
        self.logger.info(f"Extracted {total_images} images (including augmentations) to {frames_dir}")
        return frames_dir
    
    def _create_augmentations_opencv(self, frame: np.ndarray) -> List[np.ndarray]:
        """Create augmentations using OpenCV operations"""
        augmentations = []
        
        # Horizontal flip
        aug1 = cv2.flip(frame, 1)
        augmentations.append(aug1)
        
        # Brightness adjustment
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)  # Increase brightness
        aug2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmentations.append(aug2)
        
        # Slight rotation
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
        aug3 = cv2.warpAffine(frame, M, (cols, rows))
        augmentations.append(aug3)
        
        # Gaussian blur
        aug4 = cv2.GaussianBlur(frame, (5, 5), 0)
        augmentations.append(aug4)
        
        return augmentations

    def preprocess_image(self, img_path: str, size=(224, 224)) -> np.ndarray:
        """Preprocess image for model input"""
        img = image.load_img(img_path, target_size=size)
        arr = image.img_to_array(img) / 255.0
        return np.expand_dims(arr, 0)

    def get_image_files(self, directory: str, exts=("jpg", "png")) -> List[str]:
        """Get all image files from directory"""
        files = []
        for ext in exts:
            pattern = os.path.join(directory, f"*.{ext}")
            files.extend(glob(pattern))
        return files

    def group_augmentations(self, image_files: List[str]) -> Dict[str, List[str]]:
        """Group augmented images by their base name"""
        groups = {}
        for path in image_files:
            name = os.path.basename(path)
            base = name.split("_aug")[0].split("_orig")[0]
            if base not in groups:
                groups[base] = []
            groups[base].append(path)
        
        # Only keep groups with multiple images (original + augmentations)
        valid_groups = {base: paths for base, paths in groups.items() if len(paths) > 1}
        self.logger.info(f"Grouped {len(image_files)} files into {len(valid_groups)} augmentation sets")
        return valid_groups

    def compute_variances(self, frames_dir: str) -> pd.DataFrame:
        """
        Compute prediction variances for augmented frame groups
        
        Args:
            frames_dir: Directory containing frames and augmentations
            
        Returns:
            pandas.DataFrame: Variance data for each frame group
        """
        self.logger.info(f"Computing variances for frames in {frames_dir}")
        
        if not self.model:
            raise Exception("No model loaded for variance computation")
        
        # Get all image files and group by base name
        files = self.get_image_files(frames_dir)
        groups = self.group_augmentations(files)
        
        records = []
        for base, paths in groups.items():
            try:
                # Get predictions for all augmentations
                predictions = []
                for path in paths:
                    pred = self.model.predict(self.preprocess_image(path), verbose=0)
                    # Handle both binary and multi-class outputs
                    if pred.shape[-1] == 1:
                        pred_value = float(pred.flatten()[0])
                    else:
                        pred_value = float(np.max(pred.flatten()))
                    predictions.append(pred_value)
                
                # Calculate variance metrics
                mean_pred = np.mean(predictions)
                var_pred = np.var(predictions)
                std_pred = np.std(predictions)
                
                records.append({
                    'base': base,
                    'paths': paths,
                    'mean_pred': mean_pred,
                    'var_pred': var_pred,
                    'std_pred': std_pred,
                    'predictions': predictions,
                    'num_augmentations': len(predictions)
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to compute variance for {base}: {e}")
                continue
        
        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values('var_pred', ascending=False).reset_index(drop=True)
        
        self.logger.info(f"Computed variances for {len(df)} frame groups")
        return df

    def send_variances_to_server(self, variance_data: pd.DataFrame, job_id: str, 
                                video_path: str) -> Dict[str, Any]:
        """
        Send variance data to fine-tune server
        
        Args:
            variance_data: DataFrame with variance information
            job_id: Job identifier
            video_path: Original video path
            
        Returns:
            dict: Response from fine-tune server
        """
        try:
            # Prepare data for transmission (convert pandas to serializable format)
            variance_records = []
            for _, row in variance_data.iterrows():
                record = {
                    'base': row['base'],
                    'mean_pred': float(row['mean_pred']),
                    'var_pred': float(row['var_pred']),
                    'std_pred': float(row['std_pred']),
                    'predictions': [float(p) for p in row['predictions']],
                    'num_augmentations': int(row['num_augmentations']),
                    'paths': row['paths']  # Include file paths for reference
                }
                variance_records.append(record)
            
            payload = {
                'job_id': job_id,
                'video_path': video_path,
                'variance_data': variance_records,
                'total_groups': len(variance_records),
                'model_info': {
                    'input_shape': str(self.model.input_shape),
                    'output_shape': str(self.model.output_shape)
                }
            }
            
            self.logger.info(f"Sending {len(variance_records)} variance records to fine-tune server")
            
            # Send to fine-tune server
            response = requests.post(
                f"{self.fine_tune_server_url}/submit_variances",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Successfully sent variance data to server: {result.get('message', 'OK')}")
                return result
            else:
                error_msg = f"Server error {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"Failed to send variance data to server: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def process_video_for_active_learning(self, video_path: str, job_id: str, 
                                        output_dir: str = "./active_learning_frames") -> Dict[str, Any]:
        """
        Complete pipeline: extract frames, compute variances, send to server
        
        Args:
            video_path: Path to video file
            job_id: Job identifier
            output_dir: Directory for temporary frame storage
            
        Returns:
            dict: Results from the active learning pipeline
        """
        try:
            # Step 1: Extract frames with augmentations
            frames_dir = self.extract_frames_with_augmentations(
                video_path, output_dir, frame_interval=10
            )
            
            # Step 2: Compute variances
            variance_data = self.compute_variances(frames_dir)
            
            if len(variance_data) == 0:
                return {
                    'success': False,
                    'error': 'No variance data computed - no valid frame groups found'
                }
            
            # Step 3: Send to fine-tune server
            server_response = self.send_variances_to_server(variance_data, job_id, video_path)
            
            return {
                'success': True,
                'frames_dir': frames_dir,
                'total_frame_groups': len(variance_data),
                'high_variance_samples': variance_data.head(10).to_dict('records'),
                'server_response': server_response
            }
            
        except Exception as e:
            error_msg = f"Active learning pipeline failed: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
