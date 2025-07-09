"""
Inference Engine for video analysis and frame processing
"""

import os
import cv2
import numpy as np
import pandas as pd
import glob
from PIL import Image
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional


class InferenceEngine:
    def __init__(self, frames_folder: str, logger):
        self.frames_folder = frames_folder
        self.logger = logger
        self.image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}
        
        # Ensure frames folder exists
        os.makedirs(frames_folder, exist_ok=True)
    
    def get_image_files(self, test_dir: str, exclude_bases=None) -> List[str]:
        """Get all image files from directory, excluding specified base names"""
        if exclude_bases is None:
            exclude_bases = set()
        
        image_files = []
        for ext in self.image_extensions:
            pattern = os.path.join(test_dir, f"*.{ext}")
            image_files.extend(glob.glob(pattern, recursive=False))
            # Also check uppercase
            pattern = os.path.join(test_dir, f"*.{ext.upper()}")
            image_files.extend(glob.glob(pattern, recursive=False))
        
        # Filter out excluded files
        filtered_files = []
        for f in image_files:
            base_name = os.path.splitext(os.path.basename(f))[0]
            if base_name not in exclude_bases:
                filtered_files.append(f)
        
        self.logger.info(f"Found {len(filtered_files)} image files in {test_dir}")
        return sorted(filtered_files)
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """Preprocess single image for model input"""
        try:
            # Load image using PIL
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(target_size)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def run_inference(self, model, test_dir: str, exclude_bases=None, job_id=None, log_callback=None) -> pd.DataFrame:
        """
        Run inference on all images in a directory
        
        Args:
            model: TensorFlow model for inference
            test_dir: Directory containing images
            exclude_bases: Set of base filenames to exclude
            job_id: Job ID for logging
            log_callback: Function to call for logging (should accept level, message, job_id, category)
        
        Returns:
            DataFrame with inference results
        """
        def log(level, message, category="INFERENCE"):
            if log_callback:
                log_callback(level, message, job_id, category)
            else:
                self.logger.info(f"[{category}] {message}")
        
        log("INFO", f"🎯 Running inference on {test_dir} (excluding {len(exclude_bases) if exclude_bases else 0} files)")
        
        files = self.get_image_files(test_dir, exclude_bases=exclude_bases)
        log("INFO", f"📁 Found {len(files)} image files to process")
        
        if len(files) == 0:
            log("WARNING", "No image files found in directory")
            return pd.DataFrame()
        
        results = []
        out_units = model.output_shape[-1]
        log("INFO", f"🎛️  Model output units: {out_units}")
        
        # Log inference method based on output units
        if out_units == 1:
            log("INFO", "🔢 Using binary classification (threshold = 0.5)")
        else:
            log("INFO", f"🔢 Using multi-class classification ({out_units} classes)")
        
        processed_count = 0
        error_count = 0
        detection_count = 0
        start_time = datetime.now()
        
        for i, f in enumerate(files):
            try:
                # Log progress every 10 files or for small batches
                if i % max(1, len(files) // 10) == 0:
                    progress = (i / len(files)) * 100
                    log("INFO", f"🔄 Processing image {i+1}/{len(files)} ({progress:.1f}%): {os.path.basename(f)}")
                
                # Preprocess image
                preprocessed = self.preprocess_image(f)
                if preprocessed is None:
                    log("ERROR", f"Failed to preprocess image: {os.path.basename(f)}")
                    error_count += 1
                    continue
                
                # Run model prediction
                probs = model.predict(preprocessed, verbose=0).flatten()
                
                if out_units == 1:
                    prob = float(probs[0])
                    pred = int(prob > 0.5)
                    confidence = prob if pred == 1 else (1.0 - prob)
                    
                    # For debugging: log some predictions
                    if i < 5:  # Log first 5 predictions
                        log("DEBUG", f"   Binary prediction for {os.path.basename(f)}: raw_prob={prob:.3f}, pred={pred}, confidence={confidence:.3f}")
                else:
                    pred = int(np.argmax(probs))
                    prob = float(probs[pred])
                    confidence = prob
                    
                    # For debugging: log some predictions
                    if i < 5:  # Log first 5 predictions
                        all_probs = [f'{p:.3f}' for p in probs]
                        log("DEBUG", f"   Multi-class prediction for {os.path.basename(f)}: pred={pred}, prob={prob:.3f}, all_probs=[{', '.join(all_probs)}]")
                
                results.append({
                    'image': os.path.basename(f), 
                    'pred': pred, 
                    'prob': prob,
                    'confidence': confidence,
                    'image_path': f
                })
                
                processed_count += 1
                
                # Log high-confidence wildfire detections immediately
                if pred == 1:
                    detection_count += 1
                    if confidence > 0.7:
                        log("WARNING", f"🔥 HIGH CONFIDENCE WILDFIRE: {os.path.basename(f)} (confidence: {confidence:.3f})")
                    else:
                        log("INFO", f"🟡 Fire detection: {os.path.basename(f)} (confidence: {confidence:.3f})")
                
            except Exception as e:
                error_count += 1
                log("ERROR", f"❌ Error processing image {os.path.basename(f)}: {e}")
                # Add failed result
                results.append({
                    'image': os.path.basename(f), 
                    'pred': -1, 
                    'prob': 0.0,
                    'confidence': 0.0,
                    'image_path': f,
                    'error': str(e)
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate timing
        end_time = datetime.now()
        inference_duration = (end_time - start_time).total_seconds()
        
        # Log final statistics
        successful_predictions = len(df[df['pred'] != -1])
        
        log("SUCCESS", f"✅ Inference completed:")
        log("INFO", f"   • Total files: {len(files)}")
        log("INFO", f"   • Successfully processed: {successful_predictions}")
        log("INFO", f"   • Errors: {error_count}")
        log("INFO", f"   • Fire detections: {detection_count}")
        log("INFO", f"   • Detection rate: {(detection_count/successful_predictions)*100:.2f}%" if successful_predictions > 0 else "   • Detection rate: 0%")
        log("INFO", f"   • Total time: {inference_duration:.2f} seconds")
        log("INFO", f"   • Average per image: {inference_duration/len(files):.3f} seconds" if len(files) > 0 else "   • Average per image: 0 seconds")
        
        # Log confidence statistics for detections
        if detection_count > 0:
            detections = df[df['pred'] == 1]
            if len(detections) > 0:
                avg_conf = detections['confidence'].mean()
                max_conf = detections['confidence'].max()
                min_conf = detections['confidence'].min()
                log("INFO", f"   • Detection confidence - Avg: {avg_conf:.3f}, Max: {max_conf:.3f}, Min: {min_conf:.3f}")
        
        return df
    
    def extract_frames_from_video(self, video_path: str, output_dir: str, frame_interval: int = 30, job_id=None, log_callback=None) -> Tuple[str, List[str]]:
        """Extract frames from video for inference with detailed logging"""
        def log(level, message, category="FRAMES"):
            if log_callback:
                log_callback(level, message, job_id, category)
            else:
                self.logger.info(f"[{category}] {message}")
        
        try:
            # Create output directory for this video
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            log("INFO", f"📹 Extracting frames from video: {os.path.basename(video_path)}")
            log("INFO", f"📁 Output directory: {video_frames_dir}")
            log("INFO", f"⏱️  Frame interval: every {frame_interval} frames")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            log("INFO", f"📊 Video properties:")
            log("INFO", f"   • Total frames: {total_frames}")
            log("INFO", f"   • FPS: {fps:.2f}")
            log("INFO", f"   • Duration: {duration:.2f} seconds")
            log("INFO", f"   • Expected extracted frames: {total_frames // frame_interval}")
            
            frame_count = 0
            saved_frames = 0
            extracted_files = []
            start_time = datetime.now()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save every nth frame
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frame_filename = f"frame_{frame_count:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    
                    # Save frame
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        extracted_files.append(frame_path)
                        saved_frames += 1
                        
                        # Log progress every 20 frames
                        if saved_frames % 20 == 0:
                            progress = (frame_count / total_frames) * 100
                            log("INFO", f"🔄 Extracted {saved_frames} frames ({progress:.1f}% complete)")
                    else:
                        log("WARNING", f"⚠️ Failed to save frame at {timestamp:.2f}s")
                
                frame_count += 1
            
            cap.release()
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            log("SUCCESS", f"✅ Frame extraction completed:")
            log("INFO", f"   • Total frames processed: {frame_count}")
            log("INFO", f"   • Frames saved: {saved_frames}")
            log("INFO", f"   • Extraction time: {extraction_time:.2f} seconds")
            log("INFO", f"   • Average rate: {saved_frames/extraction_time:.2f} frames/second" if extraction_time > 0 else "   • Average rate: N/A")
            log("INFO", f"   • Directory: {video_frames_dir}")
            
            return video_frames_dir, extracted_files
            
        except Exception as e:
            log("ERROR", f"❌ Frame extraction failed: {e}")
            raise
    
    def process_inference_results(self, df: pd.DataFrame, video_path: str, job_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process the inference DataFrame into video analysis results"""
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Analyze results
        wildfire_detections = df[df['pred'] == 1] if 'pred' in df.columns else pd.DataFrame()
        
        # Extract timestamps from frame names
        detections_list = []
        for _, row in wildfire_detections.iterrows():
            try:
                # Extract timestamp from filename like "frame_000030_t1.00s.jpg"
                filename = row['image']
                if '_t' in filename and 's.jpg' in filename:
                    timestamp_str = filename.split('_t')[1].split('s.jpg')[0]
                    timestamp = float(timestamp_str)
                else:
                    # Fallback: estimate from frame number
                    frame_num = int(filename.split('_')[1]) if '_' in filename else 0
                    timestamp = frame_num / fps
                
                detection = {
                    'frame_file': filename,
                    'timestamp': timestamp,
                    'fire_confidence': float(row.get('confidence', row.get('prob', 0))),
                    'prediction': int(row['pred']),
                    'alert_level': 'HIGH' if row.get('confidence', row.get('prob', 0)) > 0.8 else 'MEDIUM',
                    'image_path': row.get('image_path', '')
                }
                detections_list.append(detection)
                
            except Exception as e:
                self.logger.error(f"Error processing detection row: {e}")
        
        # Calculate summary statistics
        confidence_scores = df['confidence'].tolist() if 'confidence' in df.columns else df['prob'].tolist() if 'prob' in df.columns else []
        
        results = {
            'job_id': job_id,
            'video_path': video_path,
            'analysis_method': 'frame_extraction_inference',
            'analysis_timestamp': datetime.now().isoformat(),
            'model_info': model_info,
            'total_frames_analyzed': len(df),
            'original_video_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'detections': detections_list,
            'summary': {
                'wildfire_detected': len(wildfire_detections) > 0,
                'total_detections': len(wildfire_detections),
                'confidence_scores': confidence_scores,
                'detection_timestamps': [d['timestamp'] for d in detections_list],
                'high_risk_frames': len([d for d in detections_list if d['fire_confidence'] > 0.8]),
                'average_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                'max_confidence': float(np.max(confidence_scores)) if confidence_scores else 0.0,
                'risk_percentage': (len(wildfire_detections) / len(df)) * 100 if len(df) > 0 else 0.0
            },
            'inference_dataframe': df.to_dict('records')  # Include full DataFrame data
        }
        
        self.logger.info(f"Processed results: {len(wildfire_detections)} detections out of {len(df)} frames")
        
        return results
