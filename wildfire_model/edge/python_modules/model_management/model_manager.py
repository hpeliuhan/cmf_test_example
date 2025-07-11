"""
Model Manager for downloading, validating, and loading ML models
"""

import os
import hashlib
import json
import requests
import tensorflow as tf
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


def safe_count_params(model_or_weights):
    """Safely count parameters handling different TensorFlow versions"""
    try:
        if hasattr(model_or_weights, 'count_params'):
            return model_or_weights.count_params()
        elif isinstance(model_or_weights, list):
            # For weight lists, use numpy
            total = 0
            for w in model_or_weights:
                if hasattr(w, 'shape'):
                    total += np.prod(w.shape)
            return int(total)
        else:
            return 'Unknown'
    except Exception:
        return 'Unknown'


class ModelManager:
    def __init__(self, model_cache_dir: str, logger):
        self.model_cache_dir = model_cache_dir
        self.logger = logger
        self.model = None
        self.model_url = None
        self.current_model_path = None  # Track the path of the currently loaded model
        self.model_download_status = {}
        self.current_model_info = {}
        
        # Log cache directory for debugging
        self.logger.info(f"ModelManager initialized with cache dir: {model_cache_dir}")
        self.logger.info(f"Absolute cache dir path: {os.path.abspath(model_cache_dir)}")
        
        # Ensure model cache directory exists
        os.makedirs(model_cache_dir, exist_ok=True)
        self.logger.info(f"Model cache directory created/verified: {model_cache_dir}")
        
        # Log directory permissions and contents for debugging
        try:
            stat_info = os.stat(model_cache_dir)
            self.logger.info(f"Cache dir permissions: {oct(stat_info.st_mode)[-3:]}")
            contents = os.listdir(model_cache_dir)
            self.logger.info(f"Cache dir contents: {len(contents)} items")
            if contents:
                h5_files = [f for f in contents if f.endswith('.h5')]
                if h5_files:
                    self.logger.info(f"Existing .h5 files: {h5_files}")
        except Exception as e:
            self.logger.warning(f"Could not get cache dir info: {e}")
    
    def download_model_from_github(self, url: str, destination_path: str) -> Dict[str, Any]:
        """Download model from GitHub URL"""
        try:
            self.logger.info(f"Starting download from: {url}")
            
            # Convert blob URL to raw URL if needed
            if '/blob/' in url:
                raw_url = url.replace('/blob/', '/raw/')
                self.logger.info(f"Converted to raw URL: {raw_url}")
            else:
                raw_url = url
            
            # Download the file
            response = requests.get(raw_url, stream=True, timeout=60)
            
            self.logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Ensure destination directory exists
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                
                # Download with progress tracking
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(destination_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Log progress for large files
                            if total_size > 0 and downloaded_size % (1024 * 1024) == 0:  # Every MB
                                progress = (downloaded_size / total_size) * 100
                                self.logger.info(f"Download progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)")
                
                final_size = os.path.getsize(destination_path)
                self.logger.info(f"✅ Model downloaded successfully: {destination_path} ({final_size} bytes)")
                
                return {
                    'success': True,
                    'filepath': destination_path,
                    'file_size': final_size,
                    'downloaded': True,
                    'message': 'Download completed successfully'
                }
            else:
                error_msg = f"Failed to download. Status code: {response.status_code}"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'http_error'
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error downloading from {url}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'network_error'
            }
        except Exception as e:
            error_msg = f"Error downloading from {url}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'general_error'
            }
    
    def validate_model_file(self, filepath: str) -> Dict[str, Any]:
        """Validate downloaded model file"""
        try:
            if not os.path.exists(filepath):
                return {
                    'valid': False,
                    'error': 'File does not exist'
                }
            
            file_size = os.path.getsize(filepath)
            file_ext = os.path.splitext(filepath)[1].lower()
            
            self.logger.info(f"Validating model: {filepath} ({file_size} bytes, {file_ext})")
            
            # Check file size
            if file_size < 1024:  # Less than 1KB
                return {
                    'valid': False,
                    'error': 'File too small to be a valid model'
                }
            
            # Check if it's an HTML page (common GitHub mistake)
            with open(filepath, 'rb') as f:
                first_bytes = f.read(100)
                if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                    return {
                        'valid': False,
                        'error': 'File appears to be HTML page, not a model file. Check if URL is correct.'
                    }
            
            # Check file extension
            valid_extensions = ['.h5', '.pb', '.onnx', '.tflite', '.pkl', '.pth', '.pt']
            if file_ext not in valid_extensions:
                return {
                    'valid': False,
                    'error': f'Invalid model file extension: {file_ext}. Expected: {valid_extensions}'
                }
            
            # Try to load with TensorFlow if it's a .h5 file
            if file_ext == '.h5':
                try:
                    test_model = tf.keras.models.load_model(filepath)
                    model_info = {
                        'input_shape': str(test_model.input_shape) if hasattr(test_model, 'input_shape') else 'Unknown',
                        'output_shape': str(test_model.output_shape) if hasattr(test_model, 'output_shape') else 'Unknown',
                        'parameters': test_model.count_params() if hasattr(test_model, 'count_params') else 'Unknown'
                    }
                    self.logger.info(f"✅ Model validation successful: {model_info}")
                    return {
                        'valid': True,
                        'file_size': file_size,
                        'file_extension': file_ext,
                        'model_info': model_info,
                        'message': 'Model file is valid'
                    }
                except Exception as e:
                    return {
                        'valid': False,
                        'error': f'Failed to load TensorFlow model: {str(e)}'
                    }
            
            # For other formats, just check basic validity
            return {
                'valid': True,
                'file_size': file_size,
                'file_extension': file_ext,
                'message': 'Model file appears valid (basic validation)'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def create_dummy_model(self):
        """Create a dummy model for testing"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            self.logger.info("Dummy model created successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to create dummy model: {e}")
            return None
    
    def load_initial_model(self) -> bool:
        """Load initial model (dummy or from local file)"""
        # Try to load from local file first
        local_model_path = os.path.join(self.model_cache_dir, 'wildfire_model.h5')
        
        try:
            if os.path.exists(local_model_path):
                self.model = tf.keras.models.load_model(local_model_path)
                self.logger.info(f"Model loaded from local file: {local_model_path}")
                return True
        except Exception as e:
            self.logger.warning(f"Failed to load local model: {e}")
        
        # Create dummy model as fallback
        self.model = self.create_dummy_model()
        return self.model is not None
    
    def get_model_detailed_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if self.model is None:
            return {
                'loaded': False,
                'error': 'No model loaded'
            }
        
        try:
            # Get model architecture hash for verification
            model_config = self.model.get_config() if hasattr(self.model, 'get_config') else {}
            config_str = json.dumps(model_config, sort_keys=True) if model_config else ""
            architecture_hash = hashlib.md5(config_str.encode()).hexdigest()[:8] if config_str else "unknown"
            
            # Get model weights hash for verification
            try:
                weights = self.model.get_weights()
                weights_str = str([w.shape for w in weights])
                weights_hash = hashlib.md5(weights_str.encode()).hexdigest()[:8]
            except:
                weights_hash = "unknown"
            
            model_info = {
                'loaded': True,
                'source_url': self.model_url,
                'source_type': 'downloaded' if self.model_download_status.get('filepath') else 'dummy',
                'file_path': self.model_download_status.get('filepath', 'N/A'),
                'architecture': {
                    'input_shape': str(self.model.input_shape),
                    'output_shape': str(self.model.output_shape),
                    'output_units': self.model.output_shape[-1],
                    'total_parameters': self.model.count_params(),
                    'trainable_parameters': self.model.count_params(),
                    'layer_count': len(self.model.layers),
                    'architecture_hash': architecture_hash,
                    'weights_hash': weights_hash
                },
                'layers': [],
                'compilation_info': {},
                'file_info': {}
            }
            
            # Get layer details
            for i, layer in enumerate(self.model.layers):
                layer_info = {
                    'index': i,
                    'name': layer.name,
                    'type': type(layer).__name__,
                    'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                    'param_count': layer.count_params() if hasattr(layer, 'count_params') else 0
                }
                model_info['layers'].append(layer_info)
            
            # Get compilation info
            try:
                if hasattr(self.model, 'optimizer') and self.model.optimizer:
                    model_info['compilation_info']['optimizer'] = type(self.model.optimizer).__name__
                if hasattr(self.model, 'loss') and self.model.loss:
                    model_info['compilation_info']['loss'] = str(self.model.loss)
                if hasattr(self.model, 'metrics') and self.model.metrics:
                    model_info['compilation_info']['metrics'] = [str(m) for m in self.model.metrics]
            except:
                model_info['compilation_info'] = {'note': 'Compilation info not available'}
            
            # Get file info if available
            if self.model_download_status.get('filepath') and os.path.exists(self.model_download_status['filepath']):
                file_path = self.model_download_status['filepath']
                file_stats = os.stat(file_path)
                model_info['file_info'] = {
                    'file_size': file_stats.st_size,
                    'file_size_mb': round(file_stats.st_size / 1024 / 1024, 2),
                    'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'file_exists': True
                }
            else:
                model_info['file_info'] = {'file_exists': False}
            
            # Update global model info
            self.current_model_info = model_info
            
            return model_info
            
        except Exception as e:
            error_info = {
                'loaded': True,
                'error': f'Error getting model info: {str(e)}',
                'basic_info': {
                    'input_shape': str(self.model.input_shape) if hasattr(self.model, 'input_shape') else 'unknown',
                    'output_shape': str(self.model.output_shape) if hasattr(self.model, 'output_shape') else 'unknown'
                }
            }
            self.current_model_info = error_info
            return error_info
    
    def get_current_model(self):
        """Get the currently loaded model"""
        return self.model

    def get_model_info(self):
        """Get information about the current model"""
        if self.model is None:
            return {
                'loaded': False,
                'error': 'No model loaded'
            }
        
        try:
            return {
                'loaded': True,
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape),
                'parameters': self.model.count_params(),
                'model_type': type(self.model).__name__
            }
        except Exception as e:
            return {
                'loaded': True,
                'error': f'Error getting model info: {str(e)}'
            }

    def is_model_loaded(self):
        """Check if a model is currently loaded"""
        return self.model is not None

    def load_model_from_file(self, model_path):
        """Load model from a local file"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            self.logger.info(f"Loading model from file: {model_path}")
            self.logger.info(f"Model file exists: {os.path.exists(model_path)}")
            self.logger.info(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
            
            model = tf.keras.models.load_model(model_path)
            
            if model is not None:
                self.model = model
                self.model_url = f"file://{model_path}"
                self.current_model_path = model_path  # Track the current model path
                
                # Log model characteristics for debugging
                params = safe_count_params(model)
                self.logger.info(f"Model loaded successfully from file")
                self.logger.info(f"Model type: {type(model).__name__}")
                self.logger.info(f"Model parameters: {params}")
                self.logger.info(f"Model input shape: {getattr(model, 'input_shape', 'Unknown')}")
                self.logger.info(f"Model output shape: {getattr(model, 'output_shape', 'Unknown')}")
                
                return True
            else:
                self.logger.error("Failed to load model from file")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model from file: {e}")
            return False

    def load_model_from_url(self, github_url, force_download=False):
        """Load model from GitHub URL"""
        try:
            self.logger.info(f"Loading model from URL: {github_url}")
            
            # Determine filename from URL
            filename = os.path.basename(github_url.split('/')[-1])
            if not filename or '.' not in filename:
                filename = 'downloaded_model.h5'
                
            # Set destination path
            destination_path = os.path.join(self.model_cache_dir, filename)
            
            # Check if model already exists and force_download is False
            if os.path.exists(destination_path) and not force_download:
                self.logger.info(f"Model already exists at {destination_path}, using cached version")
                return self.load_model_from_file(destination_path)
            
            # Download the model
            download_result = self.download_model_from_github(github_url, destination_path)
            
            if download_result.get('success') and os.path.exists(destination_path):
                # Load the downloaded model
                success = self.load_model_from_file(destination_path)
                if success:
                    self.model_url = github_url
                return success
            else:
                self.logger.error("Failed to download model from URL")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model from URL: {e}")
            return False

    def load_initial_model(self):
        """Load initial model - can be customized based on requirements"""
        try:
            # Try to load from a default location
            default_model_path = os.path.join(self.model_cache_dir, 'best_model.h5')
            
            if os.path.exists(default_model_path):
                return self.load_model_from_file(default_model_path)
            else:
                self.logger.info("No initial model found, will load model when requested")
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading initial model: {e}")
            return False
