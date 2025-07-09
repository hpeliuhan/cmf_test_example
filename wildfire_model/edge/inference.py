import os
import cv2
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class VideoInferenceProcessor:
    """Handles video processing and wildfire detection inference"""
    
    def __init__(self, 
                 model_path: str,
                 temp_dir: str = "temp_frames",
                 confidence_threshold: float = 0.5):
        """
        Initialize the video inference processor
        
        Args:
            model_path: Path to the trained model
            temp_dir: Directory for temporary frame storage
            confidence_threshold: Threshold for fire detection
        """
        self.model_path = model_path
        self.temp_dir = temp_dir
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained wildfire detection model"""
        try:
            tf.keras.backend.clear_session()
            self.model = load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_frames_from_video(self, 
                                video_path: str, 
                                frame_interval: int = 30) -> List[str]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to the video file
            frame_interval: Extract every Nth frame
            
        Returns:
            List of extracted frame file paths
        """
        logger.info(f"Extracting frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        # Generate unique video ID for frame naming
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_id = f"{video_name}_{timestamp}"
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    frame_filename = f"{video_id}_frame_{saved_count:04d}.jpg"
                    frame_path = os.path.join(self.temp_dir, frame_filename)
                    
                    # Resize frame if too large (optional optimization)
                    height, width = frame.shape[:2]
                    if width > 1920:  # Resize large frames
                        scale = 1920 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_count += 1
                
                frame_count += 1
                
        finally:
            cap.release()
        
        logger.info(f"Extracted {saved_count} frames from video")
        return frame_paths
    
    def preprocess_image(self, img_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Args:
            img_path: Path to image file
            target_size: Target size for model input
            
        Returns:
            Preprocessed image array
        """
        try:
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {e}")
            raise
    
    def predict_frame(self, frame_path: str) -> Dict[str, Any]:
        """
        Run inference on a single frame
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(frame_path)
            
            # Run inference
            prediction = self.model.predict(processed_img, verbose=0)
            
            # Handle different model output formats
            if len(prediction.shape) > 1 and prediction.shape[1] == 1:
                # Binary classification (sigmoid output)
                confidence = float(prediction[0][0])
                fire_detected = confidence > self.confidence_threshold
            else:
                # Multi-class classification (softmax output)
                confidence = float(np.max(prediction))
                predicted_class = int(np.argmax(prediction))
                fire_detected = predicted_class == 1  # Assuming class 1 is fire
            
            return {
                'frame_path': frame_path,
                'fire_detected': fire_detected,
                'confidence': confidence,
                'prediction_raw': prediction.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error predicting frame {frame_path}: {e}")
            return {
                'frame_path': frame_path,
                'fire_detected': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def process_video(self, 
                     video_path: str, 
                     results_dir: str,
                     frame_interval: int = 30) -> Dict[str, Any]:
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to input video
            results_dir: Directory to save results
            frame_interval: Frame extraction interval
            
        Returns:
            Complete processing results
        """
        logger.info(f"Starting video processing for {video_path}")
        start_time = datetime.now()
        
        try:
            # 1. Extract frames
            frame_paths = self.extract_frames_from_video(video_path, frame_interval)
            
            if not frame_paths:
                raise ValueError("No frames extracted from video")
            
            # 2. Run inference on all frames
            logger.info(f"Running inference on {len(frame_paths)} frames")
            frame_results = []
            fire_detections = 0
            
            for i, frame_path in enumerate(frame_paths):
                result = self.predict_frame(frame_path)
                frame_results.append(result)
                
                if result.get('fire_detected', False):
                    fire_detections += 1
                
                # Log progress every 10 frames
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(frame_results)} frames")
            
            # 3. Aggregate results
            total_frames = len(frame_results)
            fire_percentage = (fire_detections / total_frames) * 100 if total_frames > 0 else 0
            
            # Determine overall video classification
            video_has_fire = fire_percentage > 10  # If >10% of frames have fire
            
            # Get video metadata
            video_name = os.path.basename(video_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 4. Compile final results
            results = {
                'video_info': {
                    'filename': video_name,
                    'total_frames_analyzed': total_frames,
                    'frame_interval': frame_interval,
                    'processing_time_seconds': processing_time,
                    'timestamp': start_time.isoformat()
                },
                'detection_summary': {
                    'fire_detected_overall': video_has_fire,
                    'frames_with_fire': fire_detections,
                    'fire_percentage': round(fire_percentage, 2),
                    'confidence_threshold': self.confidence_threshold
                },
                'frame_results': frame_results,
                'model_info': {
                    'model_path': self.model_path,
                    'input_shape': str(self.model.input_shape),
                    'output_shape': str(self.model.output_shape)
                }
            }
            
            # 5. Save results
            os.makedirs(results_dir, exist_ok=True)
            
            # Save JSON results
            results_filename = f"{os.path.splitext(video_name)[0]}_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            results_path = os.path.join(results_dir, results_filename)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save summary text file
            summary_filename = f"{os.path.splitext(video_name)[0]}_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
            summary_path = os.path.join(results_dir, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write(f"Wildfire Detection Results\n")
                f.write(f"========================\n\n")
                f.write(f"Video: {video_name}\n")
                f.write(f"Processed: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Processing Time: {processing_time:.2f} seconds\n")
                f.write(f"Total Frames Analyzed: {total_frames}\n")
                f.write(f"Frames with Fire: {fire_detections}\n")
                f.write(f"Fire Percentage: {fire_percentage:.2f}%\n")
                f.write(f"Overall Fire Detected: {'YES' if video_has_fire else 'NO'}\n")
            
            # 6. Cleanup temporary frames
            self.cleanup_temp_frames(frame_paths)
            
            logger.info(f"Video processing completed successfully")
            logger.info(f"Results saved to: {results_path}")
            
            return {
                'success': True,
                'results_file': results_path,
                'summary_file': summary_path,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_path': video_path
            }
    
    def cleanup_temp_frames(self, frame_paths: List[str]):
        """Clean up temporary frame files"""
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp frame {frame_path}: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']


def run_inference_cli():
    """Command line interface for video inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wildfire Detection Video Inference")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default="../best_model.h5", help="Path to model file")
    parser.add_argument("--output", default="results", help="Output directory for results")
    parser.add_argument("--interval", type=int, default=30, help="Frame extraction interval")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoInferenceProcessor(
        model_path=args.model,
        confidence_threshold=args.threshold
    )
    
    # Process video
    result = processor.process_video(
        video_path=args.video,
        results_dir=args.output,
        frame_interval=args.interval
    )
    
    if result['success']:
        print(f"Processing completed successfully")
        print(f"Results: {result['results_file']}")
        print(f"Summary: {result['summary_file']}")
    else:
        print(f"Processing failed: {result['error']}")


if __name__ == "__main__":
    run_inference_cli()