import os
import sys
import pandas as pd

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔧 Starting Flask app initialization...")

try:
    from flask import Flask, request, jsonify, render_template_string, send_file
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Flask import error: {e}")
    sys.exit(1)

try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import error: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully: {tf.__version__}")
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    sys.exit(1)

try:
    import requests
    print("✅ Requests imported successfully")
except ImportError as e:
    print(f"❌ Requests import error: {e}")
    sys.exit(1)

# Try to import custom modules from correct paths
try:
    from python_modules.model_management.model_manager import ModelManager
    from python_modules.inference_engine.inference_engine import InferenceEngine
    print("✅ Custom modules imported successfully")
except ImportError as e:
    print(f"❌ Custom module import error: {e}")
    print("Make sure python_modules/model_management/model_manager.py and python_modules/inference_engine/inference_engine.py exist")
    sys.exit(1)

# Continue with other imports
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import json
import threading
from queue import Queue
from collections import deque
import time
import logging
from PIL import Image
import glob
import hashlib

print("✅ All imports successful")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration from environment variables with defaults
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './videos')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', './results')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models')
FRAMES_FOLDER = os.getenv('FRAMES_FOLDER', './frames')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 100 * 1024 * 1024))
GITHUB_DOWNLOAD_TIMEOUT = int(os.getenv('GITHUB_DOWNLOAD_TIMEOUT', 300))
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}

print(f"📁 Configuration:")
print(f"  - Upload folder: {UPLOAD_FOLDER}")
print(f"  - Results folder: {RESULTS_FOLDER}")
print(f"  - Model cache: {MODEL_CACHE_DIR}")
print(f"  - Frames folder: {FRAMES_FOLDER}")
print(f"  - Max file size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")

# Create directories
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("✅ Directories created successfully")
except Exception as e:
    print(f"❌ Error creating directories: {e}")
    sys.exit(1)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize our modular components
model_manager = ModelManager(MODEL_CACHE_DIR, logger)
inference_engine = InferenceEngine(FRAMES_FOLDER, logger)

# Global variables for Flask app state
model = None
inference_queue = Queue()
inference_status = {}
system_logs = deque(maxlen=1000)
detailed_logs = deque(maxlen=5000)
active_jobs = {}
model_url = None
model_download_status = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'last_updated': datetime.now().isoformat()
}
current_inference_logs = {}

# Import BasicVideoInference
try:
    from python_modules.inference_engine.basic_inference import BasicVideoInference
    print("✅ BasicVideoInference imported successfully")
except ImportError as e:
    print(f"❌ BasicVideoInference import error: {e}")
    BasicVideoInference = None

def log_message(level, message, job_id=None, category='SYSTEM'):
    """Enhanced log message function with categories and detailed tracking"""
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message,
        'job_id': job_id,
        'category': category,
        'thread_name': threading.current_thread().name
    }
    
    # Add to system logs (short)
    system_logs.append(log_entry)
    
    # Add to detailed logs (longer retention)
    detailed_logs.append(log_entry)
    
    # If it's an inference job, also store in job-specific logs
    if job_id and job_id in active_jobs:
        if 'logs' not in active_jobs[job_id]:
            active_jobs[job_id]['logs'] = []
        active_jobs[job_id]['logs'].append(log_entry)
    
    # Also log to standard logger
    getattr(logger, level.lower(), logger.info)(f"[{category}] {message}")

def get_model_basic_info():
    """Get basic model information"""
    if model is None:
        return {'loaded': False}
    
    try:
        return {
            'loaded': True,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'parameters': model.count_params()
        }
    except Exception as e:
        return {
            'loaded': True,
            'error': f'Error getting model info: {str(e)}'
        }

def get_model_detailed_info():
    """Get detailed model information"""
    if model is None:
        return {
            'loaded': False,
            'error': 'No model loaded'
        }
    
    try:
        # Get basic info from model manager
        basic_info = model_manager.get_model_info()
        
        # Add additional detailed information
        detailed_info = {
            'loaded': True,
            'model_type': type(model).__name__,
            'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'Unknown',
            'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else 'Unknown',
            'total_params': model.count_params() if hasattr(model, 'count_params') else 'Unknown',
            'trainable_params': sum([tf.keras.utils.count_params(w) for w in model.trainable_weights]) if hasattr(model, 'trainable_weights') else 'Unknown',
            'non_trainable_params': sum([tf.keras.utils.count_params(w) for w in model.non_trainable_weights]) if hasattr(model, 'non_trainable_weights') else 'Unknown',
            'layers': len(model.layers) if hasattr(model, 'layers') else 'Unknown',
            'model_size_mb': get_model_size_mb(),
            'model_path': getattr(model_manager, 'current_model_path', 'Unknown'),
            'loaded_at': datetime.now().isoformat()
        }
        
        return detailed_info
        
    except Exception as e:
        return {
            'loaded': True,
            'error': f'Error getting detailed model info: {str(e)}'
        }

def get_model_size_mb():
    """Get approximate model size in MB"""
    try:
        if model is None:
            return 0
        
        # Calculate approximate size based on parameters
        total_params = model.count_params()
        # Assuming float32 (4 bytes per parameter)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
        
    except Exception as e:
        return 0

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_video_with_basic_inference(video_path, job_id):
    """Analyze video using BasicVideoInference"""
    try:
        log_message("INFO", f"Starting video analysis with BasicVideoInference", job_id)
        
        if model is None:
            raise Exception("No model loaded")
        
        if BasicVideoInference is None:
            raise Exception("BasicVideoInference not available")
        
        # Update job status
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['current_stage'] = 'Starting inference'
        active_jobs[job_id]['progress'] = 10
        
        # Create BasicVideoInference instance
        frames_dir = os.path.join(FRAMES_FOLDER, job_id)
        basic_inference = BasicVideoInference(
            model=model,
            frames_output_dir=frames_dir,
            logger=logger
        )
        
        # Run inference with frame-based extraction for comprehensive analysis
        # Extract every 5th frame with increased frame limit for more coverage
        results = basic_inference.process_video(
            video_path, 
            frame_interval=5,   # Extract every 5th frame instead of 30th
            max_frames=200,     # Increase to 200 frames for better coverage
            job_id=job_id
        )
        
        # Update job status
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['current_stage'] = 'Analysis complete'
        active_jobs[job_id]['progress'] = 100
        active_jobs[job_id]['results'] = results
        active_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        log_message("SUCCESS", f"Video analysis completed successfully", job_id)
        return results
        
    except Exception as e:
        error_msg = f"Error in video analysis: {str(e)}"
        log_message("ERROR", error_msg, job_id)
        
        # Update job status
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['current_stage'] = 'Failed'
        active_jobs[job_id]['error'] = error_msg
        active_jobs[job_id]['failed_at'] = datetime.now().isoformat()
        
        raise e

def inference_worker():
    """Background worker for processing inference jobs"""
    log_message("INFO", "Inference worker started")
    
    while True:
        try:
            if not inference_queue.empty():
                job_data = inference_queue.get()
                job_id = job_data['job_id']
                video_path = job_data['video_path']
                
                log_message("INFO", f"Processing job {job_id}", job_id)
                
                # Run inference
                analyze_video_with_basic_inference(video_path, job_id)
                
                # Mark task as done
                inference_queue.task_done()
                
            else:
                time.sleep(1)
                
        except Exception as e:
            log_message("ERROR", f"Inference worker error: {str(e)}")
            time.sleep(5)

# [Keep all your existing model loading functions...]

def download_model_from_github(url, destination_path):
    """Download model from GitHub URL"""
    try:
        log_message("INFO", f"Starting download from: {url}", None, "DOWNLOAD")
        
        # Convert blob URL to raw URL if needed
        if '/blob/' in url:
            raw_url = url.replace('/blob/', '/raw/')
            log_message("INFO", f"Converted to raw URL: {raw_url}", None, "DOWNLOAD")
        else:
            raw_url = url
        
        # Download the file
        response = requests.get(raw_url, stream=True, timeout=60)
        
        log_message("INFO", f"Response status: {response.status_code}", None, "DOWNLOAD")
        
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
                            log_message("INFO", f"Download progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", None, "DOWNLOAD")
            
            final_size = os.path.getsize(destination_path)
            log_message("SUCCESS", f"✅ Model downloaded successfully: {destination_path} ({final_size} bytes)", None, "DOWNLOAD")
            
            return {
                'success': True,
                'filepath': destination_path,
                'file_size': final_size,
                'downloaded': True,
                'message': 'Download completed successfully'
            }
        else:
            error_msg = f"Failed to download. Status code: {response.status_code}"
            log_message("ERROR", error_msg, None, "DOWNLOAD")
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'http_error'
            }
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error downloading from {url}: {str(e)}"
        log_message("ERROR", error_msg, None, "DOWNLOAD")
        return {
            'success': False,
            'error': error_msg,
            'error_type': 'network_error'
        }
    except Exception as e:
        error_msg = f"Error downloading from {url}: {str(e)}"
        log_message("ERROR", error_msg, None, "DOWNLOAD")
        return {
            'success': False,
            'error': error_msg,
            'error_type': 'general_error'
        }

def validate_model_file(filepath):
    """Validate downloaded model file"""
    try:
        if not os.path.exists(filepath):
            return {
                'valid': False,
                'error': 'File does not exist'
            }
        
        file_size = os.path.getsize(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        log_message("INFO", f"Validating model: {filepath} ({file_size} bytes, {file_ext})", None, "VALIDATION")
        
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
                    'parameters': model.count_params() if hasattr(test_model, 'count_params') else 'Unknown'
                }
                log_message("SUCCESS", f"✅ Model validation successful: {model_info}", None, "VALIDATION")
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

def load_model_from_url(github_url, force_download=False):
    """Load model from GitHub URL"""
    global model, model_url, model_download_status
    
    model_url = github_url
    
    try:
        log_message("INFO", f"Loading model from GitHub: {github_url}", None, "MODEL_LOAD")
        
        # Update download status
        model_download_status = {
            'status': 'downloading',
            'url': github_url,
            'progress': 0,
            'message': 'Starting download...'
        }
        
        # Determine filename from URL
        filename = os.path.basename(github_url.split('/')[-1])
        if not filename or '.' not in filename:
            filename = 'downloaded_model.h5'
        
        model_path = os.path.join(MODEL_CACHE_DIR, filename)
        
        # Check if file already exists and force_download is False
        if os.path.exists(model_path) and not force_download:
            log_message("INFO", f"Model file already exists: {model_path}", None, "MODEL_LOAD")
            download_result = {
                'success': True,
                'filepath': model_path,
                'file_size': os.path.getsize(model_path),
                'downloaded': False,
                'message': 'Using existing file'
            }
        else:
            # Download model using the integrated function
            download_result = download_model_from_github(github_url, model_path)
        
        if not download_result['success']:
            model_download_status = {
                'status': 'error',
                'error': download_result['error']
            }
            log_message("ERROR", f"Model download failed: {download_result['error']}", None, "MODEL_LOAD")
            return False
        
        model_path = download_result['filepath']
        
        # Validate model
        log_message("INFO", "Validating downloaded model...", None, "MODEL_LOAD")
        validation_result = validate_model_file(model_path)
        
        if not validation_result['valid']:
            model_download_status = {
                'status': 'error',
                'error': f"Model validation failed: {validation_result['error']}"
            }
            log_message("ERROR", f"Model validation failed: {validation_result['error']}", None, "MODEL_LOAD")
            return False
        
        # Load model
        log_message("INFO", "Loading validated model...", None, "MODEL_LOAD")
        model = tf.keras.models.load_model(model_path)
        
        # Log detailed model information
        log_model_details(model, {
            'source': github_url,
            'filepath': model_path,
            'file_size': download_result['file_size'],
            'downloaded': download_result['downloaded']
        })
        
        model_download_status = {
            'status': 'completed',
            'url': github_url,
            'filepath': model_path,
            'file_size': download_result['file_size'],
            'downloaded': download_result['downloaded'],
            'validation': validation_result,
            'message': 'Model loaded successfully'
        }
        
        log_message("SUCCESS", f"Model loaded successfully from {github_url}", None, "MODEL_LOAD")
        return True
        
    except Exception as e:
        error_msg = f"Error loading model from URL: {str(e)}"
        model_download_status = {
            'status': 'error',
            'error': error_msg
        }
        log_message("ERROR", error_msg, None, "MODEL_LOAD")
        return False

def create_dummy_model():
    """Create a dummy model for testing"""
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        log_message("INFO", "Dummy model created successfully", None, "MODEL")
        return model
    except Exception as e:
        log_message("ERROR", f"Failed to create dummy model: {e}", None, "MODEL")
        return None

def load_initial_model():
    """Load initial model (dummy or from local file)"""
    global model
    
    # Try to load from local file first
    local_model_path = os.path.join(MODEL_CACHE_DIR, 'wildfire_model.h5')
    
    try:
        if os.path.exists(local_model_path):
            model = tf.keras.models.load_model(local_model_path)
            log_message("INFO", f"Model loaded from local file: {local_model_path}", None, "MODEL")
            log_model_details(model, {'source': local_model_path})
            return True
    except Exception as e:
        log_message("WARNING", f"Failed to load local model: {e}", None, "MODEL")
    
    # Create dummy model as fallback
    model = create_dummy_model()
    if model:
        log_model_details(model, {'source': 'dummy_model'})
    return model is not None

def log_model_details(model, info):
    """Log model details for debugging"""
    try:
        if model is None:
            log_message("INFO", "Model is None")
            return
        
        log_message("INFO", f"Model loaded: {info}")
        log_message("INFO", f"Model type: {type(model).__name__}")
        
        if hasattr(model, 'input_shape'):
            log_message("INFO", f"Input shape: {model.input_shape}")
        if hasattr(model, 'output_shape'):
            log_message("INFO", f"Output shape: {model.output_shape}")
        if hasattr(model, 'count_params'):
            log_message("INFO", f"Parameters: {model.count_params()}")
            
    except Exception as e:
        log_message("ERROR", f"Error logging model details: {e}")

# Flask Routes

@app.route('/logs')
def logs_page():
    """Comprehensive logging display page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔍 Wildfire Detection - Live Logs</title>
        <style>
            body { 
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; 
                margin: 0; 
                padding: 20px; 
                background: #1a1a1a; 
                color: #e0e0e0; 
            }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { 
                background: #2d2d2d; 
                padding: 20px; 
                border-radius: 8px; 
                margin-bottom: 20px;
                border-left: 4px solid #ff6b35;
            }
            .controls { 
                background: #2d2d2d; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 20px;
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            .log-container { 
                background: #1e1e1e; 
                border: 1px solid #444; 
                border-radius: 8px; 
                height: 600px; 
                overflow-y: auto; 
                padding: 15px;
                font-size: 13px;
                line-height: 1.4;
            }
            .log-entry { 
                margin-bottom: 8px; 
                padding: 6px 8px; 
                border-radius: 4px;
                border-left: 3px solid #555;
                background: rgba(255,255,255,0.02);
            }
            .log-entry:hover { background: rgba(255,255,255,0.05); }
            
            /* Log level colors */
            .ERROR { border-left-color: #ff4757; background: rgba(255,71,87,0.1); }
            .WARNING { border-left-color: #ffa502; background: rgba(255,165,2,0.1); }
            .SUCCESS { border-left-color: #2ed573; background: rgba(46,213,115,0.1); }
            .INFO { border-left-color: #3742fa; background: rgba(55,66,250,0.1); }
            
            /* Category colors */
            .SYSTEM { border-left-color: #747d8c; }
            .MODEL { border-left-color: #5f27cd; }
            .INFERENCE { border-left-color: #00d2d3; }
            .FRAMES { border-left-color: #ff9ff3; }
            .ANALYSIS { border-left-color: #54a0ff; }
            .PREDICTION { border-left-color: #ff6348; }
            .DOWNLOAD { border-left-color: #2ed573; }
            .WORKER { border-left-color: #ff7675; }
            .PROCESSING { border-left-color: #a29bfe; }
            .STATS { border-left-color: #fd79a8; }
            .DETECTION { border-left-color: #e84393; background: rgba(232,67,147,0.15); }
            .SUMMARY { border-left-color: #00cec9; background: rgba(0,206,201,0.1); }
            
            .timestamp { color: #888; font-size: 11px; }
            .level { 
                font-weight: bold; 
                padding: 2px 6px; 
                border-radius: 3px; 
                font-size: 10px;
                margin-right: 8px;
            }
            .category { 
                color: #74b9ff; 
                font-weight: bold; 
                margin-right: 8px;
                font-size: 11px;
            }
            .job-id { 
                color: #fd79a8; 
                font-family: monospace; 
                margin-right: 8px;
                font-size: 11px;
            }
            .message { color: #e0e0e0; }
            
            .btn { 
                background: #74b9ff; 
                color: white; 
                border: none; 
                padding: 8px 15px; 
                border-radius: 4px; 
                cursor: pointer;
                font-size: 12px;
            }
            .btn:hover { background: #0984e3; }
            .btn.active { background: #00b894; }
            
            select, input { 
                background: #2d2d2d; 
                color: #e0e0e0; 
                border: 1px solid #555; 
                padding: 6px 10px; 
                border-radius: 4px;
                font-size: 12px;
            }
            
            .stats { 
                display: flex; 
                gap: 20px; 
                margin-bottom: 15px;
                flex-wrap: wrap;
            }
            .stat-item { 
                background: #2d2d2d; 
                padding: 10px 15px; 
                border-radius: 6px; 
                text-align: center;
                min-width: 80px;
            }
            .stat-value { 
                font-size: 18px; 
                font-weight: bold; 
                color: #74b9ff; 
            }
            .stat-label { 
                font-size: 11px; 
                color: #888; 
                text-transform: uppercase;
            }
            
            .model-info {
                background: #2d2d2d;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #5f27cd;
            }
            
            .auto-scroll {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #74b9ff;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 16px;
                width: 50px;
                height: 50px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Wildfire Detection - Live Inference Logs</h1>
                <p>Real-time monitoring of model loading, inference execution, and analysis results</p>
            </div>
            
            <div class="model-info">
                <h3>🤖 Current Model Information</h3>
                <div id="modelInfo">Loading model information...</div>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="totalLogs">0</div>
                    <div class="stat-label">Total Logs</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="activeJobs">0</div>
                    <div class="stat-label">Active Jobs</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="errorCount">0</div>
                    <div class="stat-label">Errors</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="detectionCount">0</div>
                    <div class="stat-label">Detections</div>
                </div>
            </div>
            
            <div class="controls">
                <button id="autoRefresh" class="btn active" onclick="toggleAutoRefresh()">🔄 Auto Refresh</button>
                <button class="btn" onclick="clearLogs()">🗑️ Clear Display</button>
                <button class="btn" onclick="refreshLogs()">↻ Refresh Now</button>
                
                <select id="levelFilter" onchange="applyFilters()">
                    <option value="">All Levels</option>
                    <option value="ERROR">Errors Only</option>
                    <option value="WARNING">Warnings</option>
                    <option value="SUCCESS">Success</option>
                    <option value="INFO">Info</option>
                </select>
                
                <select id="categoryFilter" onchange="applyFilters()">
                    <option value="">All Categories</option>
                    <option value="MODEL">Model</option>
                    <option value="INFERENCE">Inference</option>
                    <option value="FRAMES">Frame Processing</option>
                    <option value="ANALYSIS">Analysis</option>
                    <option value="DETECTION">Detections</option>
                    <option value="DOWNLOAD">Downloads</option>
                    <option value="WORKER">Worker</option>
                    <option value="SYSTEM">System</option>
                </select>
                
                <input type="text" id="jobFilter" placeholder="Filter by Job ID" oninput="applyFilters()" />
                <input type="text" id="searchFilter" placeholder="Search message..." oninput="applyFilters()" />
                
                <span style="margin-left: auto; color: #888; font-size: 12px;">
                    Last Updated: <span id="lastUpdate">Never</span>
                </span>
            </div>
            
            <div class="log-container" id="logContainer">
                <div style="text-align: center; color: #888; padding: 50px;">
                    Loading logs...
                </div>
            </div>
        </div>
        
        <button class="auto-scroll" id="scrollBtn" onclick="scrollToBottom()" title="Scroll to bottom">⬇️</button>
        
        <script>
            let autoRefreshEnabled = true;
            let refreshInterval;
            let allLogs = [];
            let filteredLogs = [];
            
            function toggleAutoRefresh() {
                autoRefreshEnabled = !autoRefreshEnabled;
                const btn = document.getElementById('autoRefresh');
                
                if (autoRefreshEnabled) {
                    btn.textContent = '🔄 Auto Refresh';
                    btn.classList.add('active');
                    startAutoRefresh();
                } else {
                    btn.textContent = '⏸️ Paused';
                    btn.classList.remove('active');
                    stopAutoRefresh();
                }
            }
            
            function startAutoRefresh() {
                if (refreshInterval) clearInterval(refreshInterval);
                refreshInterval = setInterval(refreshLogs, 2000);
                refreshLogs(); // Initial load
            }
            
            function stopAutoRefresh() {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                    refreshInterval = null;
                }
            }
            
            async function refreshLogs() {
                try {
                    const response = await fetch('/api/logs');
                    const data = await response.json();
                    
                    if (data.success) {
                        allLogs = data.logs;
                        updateStats(data.stats);
                        updateModelInfo(data.model_info);
                        applyFilters();
                        
                        document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                    }
                } catch (error) {
                    console.error('Error fetching logs:', error);
                }
            }
            
            function updateStats(stats) {
                document.getElementById('totalLogs').textContent = stats.total_logs;
                document.getElementById('activeJobs').textContent = stats.active_jobs;
                document.getElementById('errorCount').textContent = stats.error_count;
                document.getElementById('detectionCount').textContent = stats.detection_count;
            }
            
            function updateModelInfo(modelInfo) {
                let html = '';
                if (modelInfo.model_loaded) {
                    html = `
                        <strong>Status:</strong> ✅ Loaded<br>
                        <strong>Source:</strong> ${modelInfo.source || 'Unknown'}<br>
                        <strong>Input Shape:</strong> ${modelInfo.input_shape || 'Unknown'}<br>
                        <strong>Output Shape:</strong> ${modelInfo.output_shape || 'Unknown'}<br>
                        <strong>Parameters:</strong> ${(modelInfo.parameters || 0).toLocaleString()}<br>
                        <strong>Output Units:</strong> ${modelInfo.output_units || 'Unknown'}
                    `;
                } else {
                    html = '<strong>Status:</strong> ❌ No model loaded';
                }
                document.getElementById('modelInfo').innerHTML = html;
            }
            
            function applyFilters() {
                const levelFilter = document.getElementById('levelFilter').value;
                const categoryFilter = document.getElementById('categoryFilter').value;
                const jobFilter = document.getElementById('jobFilter').value.toLowerCase();
                const searchFilter = document.getElementById('searchFilter').value.toLowerCase();
                
                filteredLogs = allLogs.filter(log => {
                    if (levelFilter && log.level !== levelFilter) return false;
                    if (categoryFilter && log.category !== categoryFilter) return false;
                    if (jobFilter && (!log.job_id || !log.job_id.toLowerCase().includes(jobFilter))) return false;
                    if (searchFilter && !log.message.toLowerCase().includes(searchFilter)) return false;
                    return true;
                });
                
                displayLogs();
            }
            
            function displayLogs() {
                const container = document.getElementById('logContainer');
                const shouldScrollToBottom = isScrolledToBottom();
                
                if (filteredLogs.length === 0) {
                    container.innerHTML = '<div style="text-align: center; color: #888; padding: 50px;">No logs match current filters</div>';
                    return;
                }
                
                let html = '';
                filteredLogs.slice(-1000).forEach(log => { // Show last 1000 logs
                    const timestamp = new Date(log.timestamp).toLocaleTimeString();
                    const jobId = log.job_id ? `[${log.job_id.substring(0, 8)}]` : '';
                    
                    html += `
                        <div class="log-entry ${log.level} ${log.category}">
                            <span class="timestamp">${timestamp}</span>
                            <span class="level">${log.level}</span>
                            <span class="category">${log.category}</span>
                            ${jobId ? `<span class="job-id">${jobId}</span>` : ''}
                            <span class="message">${escapeHtml(log.message)}</span>
                        </div>
                    `;
                });
                
                container.innerHTML = html;
                
                if (shouldScrollToBottom) {
                    scrollToBottom();
                }
            }
            
            function isScrolledToBottom() {
                const container = document.getElementById('logContainer');
                return container.scrollTop + container.clientHeight >= container.scrollHeight - 10;
            }
            
            function scrollToBottom() {
                const container = document.getElementById('logContainer');
                container.scrollTop = container.scrollHeight;
            }
            
            function clearLogs() {
                document.getElementById('logContainer').innerHTML = '<div style="text-align: center; color: #888; padding: 50px;">Logs cleared (refresh to reload)</div>';
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            // Start auto-refresh on page load
            startAutoRefresh();
            
            // Update scroll button visibility
            document.getElementById('logContainer').addEventListener('scroll', function() {
                const scrollBtn = document.getElementById('scrollBtn');
                if (isScrolledToBottom()) {
                    scrollBtn.style.opacity = '0.3';
                } else {
                    scrollBtn.style.opacity = '1';
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/api/logs')
def api_logs():
    """API endpoint to get logs with statistics"""
    try:
        # Get recent logs
        logs_list = list(detailed_logs)
        
        # Calculate statistics
        total_logs = len(logs_list)
        error_count = len([log for log in logs_list if log['level'] == 'ERROR'])
        detection_count = len([log for log in logs_list if log['category'] == 'DETECTION'])
        active_jobs_count = len([job for job in active_jobs.values() if job.get('status') == 'processing'])
        
        # Get current model info
        model_info = {
            'model_loaded': model is not None,
            'source': model_url or 'local/dummy',
            'input_shape': str(model.input_shape) if model else None,
            'output_shape': str(model.output_shape) if model else None,
            'parameters': model.count_params() if model else None,
            'output_units': model.output_shape[-1] if model else None
        };
        
        return jsonify({
            'success': True,
            'logs': logs_list,
            'stats': {
                'total_logs': total_logs,
                'error_count': error_count,
                'detection_count': detection_count,
                'active_jobs': active_jobs_count
            },
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/job_status/<job_id>')
def job_status(job_id):
    """Get status of analysis job"""
    if job_id not in active_jobs:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    job_info = active_jobs[job_id].copy()
    

    job_info['queue_size'] = inference_queue.qsize()
    
    return jsonify({
        'success': True,
        'job': job_info
    })

@app.route('/job_results/<job_id>')
def job_results(job_id):
    """Get results of completed analysis job"""
    if job_id not in active_jobs:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    job = active_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({
            'success': False,
            'error': f'Job not completed yet. Current status: {job["status"]}'
        }), 400
    
    if 'results' not in job:
        return jsonify({
            'success': False,
            'error': 'Results not available'
        }), 404
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'results': job['results']
    })

@app.route('/download_results/<job_id>')
def download_results(job_id):
    """Download results file"""
    if job_id not in active_jobs:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    job = active_jobs[job_id]
    
    if 'results_file' not in job or not os.path.exists(job['results_file']):
        return jsonify({
            'success': False,
            'error': 'Results file not available'
        }), 404
    
    return send_file(
        job['results_file'],
        as_attachment=True,
        download_name=f"wildfire_analysis_{job_id}.json"
    )

@app.route('/download_csv/<job_id>')
def download_csv(job_id):
    """Download inference results as CSV"""
    if job_id not in active_jobs:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    job = active_jobs[job_id]
    
    if 'csv_file' not in job or not os.path.exists(job['csv_file']):
        return jsonify({
            'success': False,
            'error': 'CSV file not available'
        }), 404
    
    return send_file(
        job['csv_file'],
        as_attachment=True,
        download_name=f"inference_results_{job_id}.csv"
    )

@app.route('/active_jobs')
def active_jobs_list():
    """Get list of all active jobs"""
    jobs_summary = []
    
    for job_id, job in active_jobs.items():
        summary = {
            'job_id': job_id,
            'filename': job.get('filename', 'Unknown'),
            'status': job.get('status', 'Unknown'),
            'progress': job.get('progress', 0),
            'created_at': job.get('created_at', 'Unknown'),
            'analysis_method': job.get('analysis_method', 'Unknown'),
            'frames_extracted': job.get('frames_extracted', 0)
        }
        
        if job.get('status') == 'completed' and 'results' in job:
            summary['wildfire_detected'] = job['results']['summary'].get('wildfire_detected', False)
            summary['total_detections'] = job['results']['summary'].get('total_detections', 0)
            summary['frames_analyzed'] = job['results'].get('total_frames_analyzed', 0)
        summary['risk_level'] = 'LOW'
        if job.get('status') == 'completed' and 'results' in job:
            if job['results']['summary'].get('wildfire_detected', False):
                summary['risk_level'] = 'HIGH'
            elif job['results']['summary'].get('average_confidence', 0) > 0.7:
                summary['risk_level'] = 'MEDIUM'
        
        jobs_summary.append(summary)
    
    return jsonify({
        'success': True,
        'jobs': jobs_summary
    })

# Add these routes after the existing imports and before the main block

# Health check endpoint
@app.route('/')
@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model is not None,
            'model_info': get_model_basic_info(),
            'active_jobs': len(active_jobs),
            'queue_size': inference_queue.qsize()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Upload model endpoint
@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Upload a model file"""
    try:
        if 'model' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No model file provided'
            }), 400
        
        file = request.files['model']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save uploaded model
        filename = secure_filename(file.filename)
        model_path = os.path.join(MODEL_CACHE_DIR, filename)
        file.save(model_path)
        
        # Load the uploaded model
        success = model_manager.load_model_from_file(model_path)
        
        if success:
            global model
            model = model_manager.get_current_model()
            return jsonify({
                'success': True,
                'message': 'Model uploaded and loaded successfully',
                'model_path': model_path,
                'model_info': get_model_basic_info()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load uploaded model'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Load model from URL endpoint
@app.route('/load_model', methods=['POST'])
def load_model():
    """Load model from GitHub URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'No URL provided'
            }), 400
        
        github_url = data['url']
        force_download = data.get('force_download', False)
        
        # Load model using model manager
        success = model_manager.load_model_from_url(github_url, force_download)
        
        if success:
            global model
            model = model_manager.get_current_model()
            return jsonify({
                'success': True,
                'message': 'Model loaded successfully',
                'model_info': get_model_basic_info(),
                'url': github_url
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load model from URL'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Get current model status
@app.route('/get_current_model_status')
def get_current_model_status():
    """Get current model status and information"""
    try:
        if model is None:
            return jsonify({
                'model_loaded': False,
                'error': 'No model loaded'
            })
        
        model_info = get_model_detailed_info();
        return jsonify({
            'model_loaded': True,
            'model_info': model_info,
            'download_status': model_download_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'model_loaded': False,
            'error': str(e)
        }), 500

# Upload and analyze video endpoint
@app.route('/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    """Upload video and start analysis"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'No model loaded. Please load a model first.'
            }), 400
        
        # Check if file is in request
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file format. Allowed: {ALLOWED_EXTENSIONS}'
            }), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{job_id[:8]}_{filename}"
        video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(video_path);
        
        # Create job record
        active_jobs[job_id] = {
            'job_id': job_id,
            'filename': filename,
            'video_path': video_path,
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(video_path),
            'current_stage': 'Queued'
        }
        
        # Add to inference queue
        inference_queue.put({
            'job_id': job_id,
            'video_path': video_path
        })
        
        log_message("INFO", f"Video uploaded and queued for analysis: {filename}", job_id)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video uploaded successfully and queued for analysis',
            'filename': filename,
            'queue_position': inference_queue.qsize()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Get all jobs endpoint
@app.route('/jobs')
def get_all_jobs():
    """Get status of all jobs"""
    try:
        return jsonify({
            'success': True,
            'jobs': list(active_jobs.values()),
            'total_jobs': len(active_jobs),
            'queue_size': inference_queue.qsize(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Main execution block
if __name__ == '__main__':
    # Start the inference worker in a separate thread
    worker_thread = threading.Thread(target=inference_worker, daemon=True)
    worker_thread.start()
    
    # Load initial model
    try:
        model_manager.load_initial_model()
        model = model_manager.get_current_model()
        if model:
            log_message("SUCCESS", "Initial model loaded successfully")
        else:
            log_message("INFO", "No initial model found, will load when requested")
    except Exception as e:
        log_message("ERROR", f"Failed to load initial model: {e}")
    
    # Start Flask app
    log_message("INFO", "Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
