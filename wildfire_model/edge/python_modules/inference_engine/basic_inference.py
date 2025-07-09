import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import uuid
import threading
import time

class BasicVideoInference:
    def __init__(self, model=None, frames_output_dir="./video_frames", logger=None):
        """
        Initialize basic video inference engine that works with Flask app's model system
        
        Args:
            model: Pre-loaded TensorFlow model (from Flask app)
            frames_output_dir: Directory to save extracted frames
            logger: Logger instance
        """
        self.model = model
        self.frames_output_dir = frames_output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.target_size = None
        
        # Create frames directory
        os.makedirs(frames_output_dir, exist_ok=True)
        
        # Auto-detect input size if model is provided
        if self.model:
            self.detect_input_size()
        
        self.logger.info(f"BasicVideoInference initialized with model: {'✅ Loaded' if model else '❌ None'}")
        self.logger.info(f"Frames will be saved to: {frames_output_dir}")
    
    def set_model(self, model):
        """Set or update the model"""
        self.model = model
        if model:
            self.detect_input_size()
        self.logger.info(f"Model updated: {'✅ Loaded' if model else '❌ None'}")
    
    def detect_input_size(self):
        """Auto-detect the required input size from the model"""
        if not self.model:
            self.target_size = (224, 224)
            return
            
        try:
            input_shape = self.model.input_shape
            self.logger.info(f"Detecting input size from model shape: {input_shape}")
            
            # Extract height and width from input shape
            # Expected format: (None, height, width, channels)
            if len(input_shape) == 4:
                height = input_shape[1]
                width = input_shape[2]
                
                if height is not None and width is not None:
                    self.target_size = (width, height)  # PIL expects (width, height)
                    self.logger.info(f"Auto-detected target size: {self.target_size}")
                else:
                    self.logger.warning("Model input shape has None dimensions, using default 224x224")
                    self.target_size = (224, 224)
            else:
                self.logger.warning(f"Unexpected input shape format: {input_shape}")
                self.target_size = (224, 224)
                
        except Exception as e:
            self.logger.error(f"Failed to detect input size: {e}")
            self.target_size = (224, 224)
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array: Preprocessed image ready for inference
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def extract_frames(self, video_path, frame_interval=30, max_frames=None, job_id=None):
        """
        Extract frames from video for inference
        
        Args:
            video_path: Path to the video file
            frame_interval: Extract every nth frame (default: 30 = every 30 frames)
            max_frames: Maximum number of frames to extract (None = no limit)
            job_id: Job ID for logging
            
        Returns:
            tuple: (extracted_files, video_frames_dir)
        """
        try:
            self.logger.info(f"📹 Starting frame extraction from: {os.path.basename(video_path)}")
            
            # Create video-specific output directory
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_dir = os.path.join(self.frames_output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            self.logger.info(f"📊 Video properties:")
            self.logger.info(f"   • Total frames: {total_frames}")
            self.logger.info(f"   • FPS: {fps:.2f}")
            self.logger.info(f"   • Duration: {duration:.2f} seconds")
            self.logger.info(f"   • Frame interval: every {frame_interval} frames")
            
            expected_frames = min(total_frames // frame_interval, max_frames) if max_frames else total_frames // frame_interval
            self.logger.info(f"   • Expected extracted frames: {expected_frames}")
            
            # Extract frames
            frame_count = 0
            saved_frames = 0
            extracted_files = []
            
            # Progress bar
            pbar = tqdm(total=expected_frames, desc="Extracting frames", unit="frame")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we should extract this frame
                if frame_count % frame_interval == 0:
                    # Check max frames limit
                    if max_frames and saved_frames >= max_frames:
                        break
                    
                    # Calculate timestamp
                    timestamp = frame_count / fps
                    
                    # Generate filename
                    frame_filename = f"frame_{frame_count:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    
                    # Save frame
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        extracted_files.append(frame_path)
                        saved_frames += 1
                        pbar.update(1)
                    else:
                        self.logger.warning(f"Failed to save frame at {timestamp:.2f}s")
                
                frame_count += 1
            
            cap.release()
            pbar.close()
            
            self.logger.info(f"✅ Frame extraction completed:")
            self.logger.info(f"   • Frames processed: {frame_count}")
            self.logger.info(f"   • Frames saved: {saved_frames}")
            self.logger.info(f"   • Output directory: {video_frames_dir}")
            
            return extracted_files, video_frames_dir
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            raise
    
    def extract_frames_by_time(self, video_path, time_interval=2.0, max_frames=None, job_id=None):
        """
        Extract frames from video based on time intervals
        
        Args:
            video_path: Path to the video file
            time_interval: Extract frame every N seconds (default: 2.0 seconds)
            max_frames: Maximum number of frames to extract
            job_id: Job ID for logging
            
        Returns:
            tuple: (extracted_files, video_frames_dir)
        """
        try:
            self.logger.info(f"📹 Starting time-based frame extraction from: {os.path.basename(video_path)}")
            
            # Create video-specific output directory
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_dir = os.path.join(self.frames_output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame interval based on time
            frame_interval = int(fps * time_interval)
            expected_frames = min(int(duration / time_interval), max_frames) if max_frames else int(duration / time_interval)
            
            self.logger.info(f"📊 Video properties:")
            self.logger.info(f"   • Total frames: {total_frames}")
            self.logger.info(f"   • FPS: {fps:.2f}")
            self.logger.info(f"   • Duration: {duration:.2f} seconds")
            self.logger.info(f"   • Time interval: {time_interval}s")
            self.logger.info(f"   • Frame interval: {frame_interval} frames")
            self.logger.info(f"   • Expected extracted frames: {expected_frames}")
            
            # Extract frames
            frame_count = 0
            saved_frames = 0
            extracted_files = []
            
            # Progress bar
            pbar = tqdm(total=expected_frames, desc="Extracting frames", unit="frame")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we should extract this frame (based on time interval)
                if frame_count % frame_interval == 0:
                    # Check max frames limit
                    if max_frames and saved_frames >= max_frames:
                        break
                    
                    # Calculate timestamp
                    timestamp = frame_count / fps
                    
                    # Generate filename
                    frame_filename = f"frame_{frame_count:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    
                    # Save frame
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        extracted_files.append(frame_path)
                        saved_frames += 1
                        pbar.update(1)
                        self.logger.debug(f"Saved frame at {timestamp:.2f}s")
                    else:
                        self.logger.warning(f"Failed to save frame at {timestamp:.2f}s")
                
                frame_count += 1
            
            cap.release()
            pbar.close()
            
            self.logger.info(f"✅ Time-based frame extraction completed:")
            self.logger.info(f"   • Frames processed: {frame_count}")
            self.logger.info(f"   • Frames saved: {saved_frames}")
            self.logger.info(f"   • Output directory: {video_frames_dir}")
            
            return extracted_files, video_frames_dir
            
        except Exception as e:
            self.logger.error(f"Time-based frame extraction failed: {e}")
            raise

    def run_inference_on_frames(self, frame_files, job_id=None):
        """
        Run inference on extracted frames
        
        Args:
            frame_files: List of frame file paths
            job_id: Job ID for logging
            
        Returns:
            list: List of inference results
        """
        if not self.model:
            raise Exception("No model loaded. Please set a model first.")
            
        try:
            self.logger.info(f"🔄 Running inference on {len(frame_files)} frames")
            
            results = []
            fire_detections = 0
            
            # Progress bar
            pbar = tqdm(frame_files, desc="Processing frames", unit="frame")
            
            for frame_path in pbar:
                try:
                    # Extract timestamp from filename
                    filename = os.path.basename(frame_path)
                    timestamp = 0.0
                    
                    if '_t' in filename and 's.jpg' in filename:
                        timestamp_str = filename.split('_t')[1].split('s.jpg')[0]
                        timestamp = float(timestamp_str)
                    
                    # Preprocess image
                    img_array = self.preprocess_image(frame_path)
                    if img_array is None:
                        continue
                    
                    # Run inference
                    predictions = self.model.predict(img_array, verbose=0)
                    probs = predictions[0]  # Remove batch dimension
                    
                    # Determine output format based on model output
                    if len(probs) == 1:
                        # Binary classification with single output (sigmoid)
                        fire_prob = float(probs[0])
                        no_fire_prob = 1.0 - fire_prob
                        predicted_class = 1 if fire_prob > 0.5 else 0
                        confidence = fire_prob if predicted_class == 1 else no_fire_prob
                    else:
                        # Multi-class classification (softmax)
                        predicted_class = int(np.argmax(probs))
                        confidence = float(probs[predicted_class])
                        no_fire_prob = float(probs[0]) if len(probs) > 0 else 0.0
                        fire_prob = float(probs[1]) if len(probs) > 1 else 0.0
                    
                    # Create result
                    result = {
                        'image_path': frame_path,
                        'frame_filename': filename,
                        'timestamp': timestamp,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'fire_detected': predicted_class == 1,
                        'probabilities': {
                            'no_fire': no_fire_prob,
                            'fire': fire_prob
                        }
                    }
                    
                    results.append(result)
                    
                    # Count fire detections
                    if result['fire_detected']:
                        fire_detections += 1
                        pbar.set_postfix({"Fire detections": fire_detections})
                    
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_path}: {e}")
                    results.append({
                        'frame_path': frame_path,
                        'frame_filename': os.path.basename(frame_path),
                        'timestamp': timestamp,
                        'error': str(e),
                        'fire_detected': False
                    })
            
            pbar.close()
            
            self.logger.info(f"✅ Inference completed:")
            self.logger.info(f"   • Total frames processed: {len(results)}")
            self.logger.info(f"   • Fire detections: {fire_detections}")
            self.logger.info(f"   • Detection rate: {(fire_detections/len(results))*100:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Frame inference failed: {e}")
            raise
    
    def analyze_results(self, results):
        """
        Analyze inference results and generate summary
        
        Args:
            results: List of inference results
            
        Returns:
            dict: Analysis summary
        """
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(results)
            
            # Filter out error results
            valid_results = df[~df['fire_detected'].isna()]
            fire_detections = valid_results[valid_results['fire_detected'] == True]
            
            # Calculate statistics
            total_frames = len(valid_results)
            fire_count = len(fire_detections)
            detection_rate = (fire_count / total_frames * 100) if total_frames > 0 else 0
            
            summary = {
                'total_frames_analyzed': total_frames,
                'fire_detections': fire_count,
                'detection_rate_percent': detection_rate,
                'fire_detected_overall': fire_count > 0,
                'error_count': len(df) - total_frames
            }
            
            if fire_count > 0:
                summary.update({
                    'average_fire_confidence': fire_detections['confidence'].mean(),
                    'max_fire_confidence': fire_detections['confidence'].max(),
                    'min_fire_confidence': fire_detections['confidence'].min(),
                    'fire_detection_timestamps': fire_detections['timestamp'].tolist(),
                    'first_detection_time': fire_detections['timestamp'].min(),
                    'last_detection_time': fire_detections['timestamp'].max()
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Results analysis failed: {e}")
            return {'error': str(e)}
    
    def process_video(self, video_path, frame_interval=30, max_frames=None, job_id=None):
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every nth frame
            max_frames: Maximum frames to process
            job_id: Job ID for tracking
            
        Returns:
            dict: Complete analysis results
        """
        try:
            self.logger.info(f"🎬 Starting basic video inference: {os.path.basename(video_path)}")
            
            if not self.model:
                raise Exception("No model loaded. Please ensure a model is available.")
            
            # Extract frames
            frame_files, frames_dir = self.extract_frames(
                video_path, frame_interval, max_frames, job_id
            )
            
            if not frame_files:
                raise Exception("No frames were extracted from the video")
            
            # Run inference on frames
            results = self.run_inference_on_frames(frame_files, job_id)
            
            # Analyze results
            summary = self.analyze_results(results)
            
            # Create final results structure
            final_results = {
                'job_id': job_id,
                'video_path': video_path,
                'analysis_method': 'basic_inference',
                'model_info': {
                    'input_shape': str(self.model.input_shape),
                    'output_shape': str(self.model.output_shape),
                    'target_size': self.target_size,
                    'parameters': self.model.count_params()
                },
                'frames_directory': frames_dir,
                'total_frames_analyzed': len(results),
                'summary': summary,
                'detections': [r for r in results if r.get('fire_detected', False)],
                'detailed_results': results
            }
            
            self.logger.info(f"✅ Basic video inference completed")
            self.logger.info(f"   • Fire detected: {'YES' if summary['fire_detected_overall'] else 'NO'}")
            self.logger.info(f"   • Detection rate: {summary['detection_rate_percent']:.2f}%")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            raise

    def process_video_by_time(self, video_path, time_interval=2.0, max_frames=50, job_id=None):
        """
        Complete video processing pipeline using time-based frame extraction
        
        Args:
            video_path: Path to video file
            time_interval: Extract frame every N seconds (default: 2.0 seconds)
            max_frames: Maximum frames to process (default: 50 for good coverage)
            job_id: Job ID for tracking
            
        Returns:
            dict: Complete analysis results
        """
        try:
            self.logger.info(f"🎬 Starting time-based video inference: {os.path.basename(video_path)}")
            
            if not self.model:
                raise Exception("No model loaded. Please ensure a model is available.")
            
            # Extract frames based on time intervals
            frame_files, frames_dir = self.extract_frames_by_time(
                video_path, time_interval, max_frames, job_id
            )
            
            if not frame_files:
                raise Exception("No frames were extracted from the video")
            
            # Run inference on frames
            results = self.run_inference_on_frames(frame_files, job_id)
            
            # Analyze results
            summary = self.analyze_results(results)
            
            # Create final results structure
            final_results = {
                'job_id': job_id,
                'video_path': video_path,
                'analysis_method': 'time_based_inference',
                'extraction_params': {
                    'time_interval': time_interval,
                    'max_frames': max_frames
                },
                'model_info': {
                    'input_shape': str(self.model.input_shape),
                    'output_shape': str(self.model.output_shape),
                    'target_size': self.target_size,
                    'parameters': self.model.count_params()
                },
                'frames_directory': frames_dir,
                'total_frames_analyzed': len(results),
                'summary': summary,
                'detections': [r for r in results if r.get('fire_detected', False)],
                'detailed_results': results
            }
            
            self.logger.info(f"✅ Time-based video inference completed")
            self.logger.info(f"   • Fire detected: {'YES' if summary['fire_detected_overall'] else 'NO'}")
            self.logger.info(f"   • Detection rate: {summary['detection_rate_percent']:.2f}%")
            self.logger.info(f"   • Frames analyzed: {len(results)}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Time-based video processing failed: {e}")
            raise

# Standalone usage functions
def create_basic_inference_engine(model_path):
    """Create a basic inference engine with a model from file"""
    model = tf.keras.models.load_model(model_path)
    return BasicVideoInference(model)

def process_video_file(video_path, model_path, output_dir="./basic_inference_results"):
    """Process a video file with basic inference"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Create inference engine
        inference_engine = BasicVideoInference(model)
        
        # Process video
        results = inference_engine.process_video(video_path)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{video_name}_basic_inference_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV results
        csv_path = os.path.join(output_dir, f"{video_name}_basic_inference_{timestamp}.csv")
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Processing complete:")
        print(f"   • JSON results: {json_path}")
        print(f"   • CSV results: {csv_path}")
        print(f"   • Fire detected: {'YES' if results['summary']['fire_detected_overall'] else 'NO'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return None

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python basic_inference.py <model_path> <video_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    # Process video
    results = process_video_file(video_path, model_path)
    
    if results:
        print("🎉 Basic inference completed successfully!")
    else:
        print("❌ Basic inference failed!")