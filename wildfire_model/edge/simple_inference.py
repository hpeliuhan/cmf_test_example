import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import logging
from datetime import datetime
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleInference:
    def __init__(self, model_path, target_size=None):
        """
        Initialize the simple inference engine
        
        Args:
            model_path: Path to the trained model (.h5 file)
            target_size: Target size for image preprocessing (width, height). If None, will auto-detect from model
        """
        self.model_path = model_path
        self.target_size = target_size
        self.model = None
        self.class_names = ['no_fire', 'fire']  # Adjust based on your model
        
        # Load model and detect input size
        self.load_model()
        
        # Auto-detect target size from model if not provided
        if self.target_size is None:
            self.detect_input_size()
    
    def load_model(self):
        """Load the TensorFlow model"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_input_size(self):
        """Auto-detect the required input size from the model"""
        try:
            input_shape = self.model.input_shape
            logger.info(f"Detecting input size from model shape: {input_shape}")
            
            # Extract height and width from input shape
            # Expected format: (None, height, width, channels)
            if len(input_shape) == 4:
                height = input_shape[1]
                width = input_shape[2]
                
                if height is not None and width is not None:
                    self.target_size = (width, height)  # PIL expects (width, height)
                    logger.info(f"Auto-detected target size: {self.target_size}")
                else:
                    logger.warning("Model input shape has None dimensions, using default 224x224")
                    self.target_size = (224, 224)
            else:
                logger.warning(f"Unexpected input shape format: {input_shape}")
                self.target_size = (224, 224)
                
        except Exception as e:
            logger.error(f"Failed to detect input size: {e}")
            logger.info("Using default target size: (224, 224)")
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
            logger.info(f"Preprocessing image: {image_path}")
            logger.info(f"Target size: {self.target_size}")
            
            # Load image
            image = Image.open(image_path)
            logger.info(f"Original image size: {image.size}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info(f"Converted image from {image.mode} to RGB")
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to: {image.size}")
            
            # Convert to numpy array
            img_array = np.array(image)
            logger.info(f"Numpy array shape: {img_array.shape}")
            
            # Normalize pixel values to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            logger.info(f"Final preprocessed shape: {img_array.shape}")
            logger.info(f"Expected model input shape: {self.model.input_shape}")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def run_inference(self, image_path):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Inference results containing probabilities, confidence, and predicted class
        """
        try:
            logger.info(f"Running inference on: {image_path}")
            
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Verify shapes match
            expected_shape = self.model.input_shape
            actual_shape = img_array.shape
            
            if expected_shape[1:] != actual_shape[1:]:
                raise ValueError(f"Shape mismatch: expected {expected_shape}, got {actual_shape}")
            
            # Run inference
            logger.info("Running model prediction...")
            predictions = self.model.predict(img_array, verbose=0)
            logger.info(f"Model prediction completed. Output shape: {predictions.shape}")
            
            # Get probabilities
            probs = predictions[0]  # Remove batch dimension
            logger.info(f"Probabilities: {probs}")
            
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
            
            # Create result dictionary
            result = {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'probabilities': {
                    'no_fire': no_fire_prob,
                    'fire': fire_prob
                },
                'predicted_class': predicted_class,
                'predicted_class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'fire_detected': predicted_class == 1,
                'model_info': {
                    'model_path': self.model_path,
                    'input_shape': str(self.model.input_shape),
                    'output_shape': str(self.model.output_shape),
                    'target_size': self.target_size
                }
            }
            
            logger.info(f"Inference completed:")
            logger.info(f"  - Predicted class: {result['predicted_class_name']}")
            logger.info(f"  - Confidence: {result['confidence']:.4f}")
            logger.info(f"  - Fire detected: {result['fire_detected']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def save_results(self, results, output_path):
        """
        Save inference results to a JSON file
        
        Args:
            results: Inference results dictionary
            output_path: Path to save the results file
        """
        try:
            logger.info(f"Saving results to: {output_path}")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save results as JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

def main():
    """Main function to run inference on a single image"""
    
    # Configuration
    MODEL_PATH = "./best_model.h5"  # Update this path
    IMAGE_PATH = "./frame_0052.jpg"  # Update this path
    RESULTS_DIR = "./inference_results"
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.info("Please update MODEL_PATH with the correct path to your model file")
        return
    
    if not os.path.exists(IMAGE_PATH):
        logger.error(f"Image file not found: {IMAGE_PATH}")
        logger.info("Please update IMAGE_PATH with the correct path to your image file")
        return
    
    try:
        # Initialize inference engine (target_size will be auto-detected)
        inference_engine = SimpleInference(MODEL_PATH)
        
        # Run inference
        results = inference_engine.run_inference(IMAGE_PATH)
        
        # Generate output filename
        image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{image_name}_inference_{timestamp}.json"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        
        # Save results
        inference_engine.save_results(results, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 INFERENCE SUMMARY")
        print("="*60)
        print(f"📸 Image: {IMAGE_PATH}")
        print(f"🤖 Model: {MODEL_PATH}")
        print(f"📐 Input size: {inference_engine.target_size}")
        print(f"🔍 Prediction: {results['predicted_class_name']}")
        print(f"📊 Confidence: {results['confidence']:.4f}")
        print(f"🔥 Fire detected: {'YES' if results['fire_detected'] else 'NO'}")
        print(f"📈 Probabilities:")
        print(f"   • No fire: {results['probabilities']['no_fire']:.4f}")
        print(f"   • Fire: {results['probabilities']['fire']:.4f}")
        print(f"💾 Results saved to: {output_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return

if __name__ == "__main__":
    main()