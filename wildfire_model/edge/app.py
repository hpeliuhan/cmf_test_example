import os
import sys

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

# Continue with other imports
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import json
import threading
from queue import Queue
from collections import deque
import time

print("✅ All imports successful")

app = Flask(__name__)

# Configuration from environment variables with defaults
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './videos')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', './results')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 100 * 1024 * 1024))
GITHUB_DOWNLOAD_TIMEOUT = int(os.getenv('GITHUB_DOWNLOAD_TIMEOUT', 300))
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}

print(f"📁 Configuration:")
print(f"  - Upload folder: {UPLOAD_FOLDER}")
print(f"  - Results folder: {RESULTS_FOLDER}")
print(f"  - Model cache: {MODEL_CACHE_DIR}")
print(f"  - Max file size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")

# Create directories
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    print("✅ Directories created successfully")
except Exception as e:
    print(f"❌ Error creating directories: {e}")
    sys.exit(1)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables
model = None
inference_queue = Queue()
inference_status = {}
system_logs = deque(maxlen=1000)
active_jobs = {}
model_url = None
model_download_status = {}

def log_message(level, message, job_id=None):
    """Add log message to system logs"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message,
        'job_id': job_id
    }
    system_logs.append(log_entry)
    print(f"[{level}] {message}")

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame, target_size=(224, 224)):
    """Preprocess frame for model input"""
    try:
        # Resize frame
        resized = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_frame = np.expand_dims(normalized, axis=0)
        
        return batch_frame
    except Exception as e:
        log_message("ERROR", f"Error preprocessing frame: {e}")
        return None

def analyze_video(video_path, job_id):
    """Analyze video for wildfire detection"""
    global model
    
    try:
        log_message("INFO", f"Starting video analysis: {video_path}", job_id)
        
        if model is None:
            raise Exception("No model loaded")
        
        # Update job status
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 0
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        log_message("INFO", f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s", job_id)
        
        # Analysis results
        results = {
            'job_id': job_id,
            'video_path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'detections': [],
            'summary': {
                'wildfire_detected': False,
                'confidence_scores': [],
                'detection_timestamps': [],
                'high_risk_frames': 0
            }
        }
        
        frame_count = 0
        detection_threshold = 0.7  # Confidence threshold for wildfire detection
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = (frame_count / total_frames) * 100
            active_jobs[job_id]['progress'] = progress
            
            # Process every 30th frame (to speed up analysis)
            if frame_count % 30 == 0:
                timestamp = frame_count / fps
                
                # Preprocess frame
                processed_frame = preprocess_frame(frame)
                if processed_frame is None:
                    continue
                
                # Run inference
                try:
                    predictions = model.predict(processed_frame, verbose=0)
                    
                    # Assuming binary classification: [no_fire, fire]
                    if len(predictions[0]) >= 2:
                        fire_confidence = float(predictions[0][1])
                        no_fire_confidence = float(predictions[0][0])
                    else:
                        fire_confidence = float(predictions[0][0])
                        no_fire_confidence = 1.0 - fire_confidence
                    
                    results['summary']['confidence_scores'].append(fire_confidence)
                    
                    # Check if wildfire detected
                    if fire_confidence > detection_threshold:
                        results['summary']['wildfire_detected'] = True
                        results['summary']['detection_timestamps'].append(timestamp)
                        results['summary']['high_risk_frames'] += 1
                        
                        detection = {
                            'frame_number': frame_count,
                            'timestamp': timestamp,
                            'fire_confidence': fire_confidence,
                            'no_fire_confidence': no_fire_confidence,
                            'alert_level': 'HIGH' if fire_confidence > 0.9 else 'MEDIUM'
                        }
                        results['detections'].append(detection)
                        
                        log_message("WARNING", f"Wildfire detected at {timestamp:.2f}s (confidence: {fire_confidence:.3f})", job_id)
                
                except Exception as e:
                    log_message("ERROR", f"Error during inference at frame {frame_count}: {e}", job_id)
                    continue
            
            # Log progress every 10%
            if frame_count % (total_frames // 10) == 0:
                log_message("INFO", f"Analysis progress: {progress:.1f}%", job_id)
        
        cap.release()
        
        # Calculate final statistics
        if results['summary']['confidence_scores']:
            avg_confidence = np.mean(results['summary']['confidence_scores'])
            max_confidence = np.max(results['summary']['confidence_scores'])
            results['summary']['average_confidence'] = float(avg_confidence)
            results['summary']['max_confidence'] = float(max_confidence)
            results['summary']['total_detections'] = len(results['detections'])
            results['summary']['risk_percentage'] = (results['summary']['high_risk_frames'] / (total_frames // 30)) * 100
        
        # Save results
        results_filename = f"analysis_{job_id}.json"
        results_path = os.path.join(RESULTS_FOLDER, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update job status
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['progress'] = 100
        active_jobs[job_id]['results'] = results
        active_jobs[job_id]['results_file'] = results_path
        
        log_message("SUCCESS", f"Video analysis completed. Results saved to: {results_path}", job_id)
        
        return results
        
    except Exception as e:
        error_msg = f"Video analysis failed: {str(e)}"
        log_message("ERROR", error_msg, job_id)
        
        # Update job status
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['error'] = error_msg
        
        return None

def inference_worker():
    """Background worker for processing inference queue"""
    log_message("INFO", "Inference worker started")
    
    while True:
        try:
            if not inference_queue.empty():
                job_data = inference_queue.get()
                job_id = job_data['job_id']
                video_path = job_data['video_path']
                
                log_message("INFO", f"Processing job {job_id}", job_id)
                
                # Analyze video
                results = analyze_video(video_path, job_id)
                
                inference_queue.task_done()
            else:
                time.sleep(1)
                
        except Exception as e:
            log_message("ERROR", f"Inference worker error: {e}")
            time.sleep(5)

def download_model_from_github(url, destination_path):
    """
    Download model from GitHub URL (integrated from test_github_url.py)
    
    Args:
        url (str): GitHub URL (blob or raw)
        destination_path (str): Local path to save the model
    
    Returns:
        dict: Download result with success status and details
    """
    try:
        log_message("INFO", f"Starting download from: {url}")
        
        # Convert blob URL to raw URL if needed
        if '/blob/' in url:
            raw_url = url.replace('/blob/', '/raw/')
            log_message("INFO", f"Converted to raw URL: {raw_url}")
        else:
            raw_url = url
        
        # Download the file
        response = requests.get(raw_url, stream=True, timeout=60)
        
        log_message("INFO", f"Response status: {response.status_code}")
        
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
                            log_message("INFO", f"Download progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)")
            
            final_size = os.path.getsize(destination_path)
            log_message("SUCCESS", f"✅ Model downloaded successfully: {destination_path} ({final_size} bytes)")
            
            return {
                'success': True,
                'filepath': destination_path,
                'file_size': final_size,
                'downloaded': True,
                'message': 'Download completed successfully'
            }
        else:
            error_msg = f"Failed to download. Status code: {response.status_code}"
            log_message("ERROR", error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'http_error'
            }
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error downloading from {url}: {str(e)}"
        log_message("ERROR", error_msg)
        return {
            'success': False,
            'error': error_msg,
            'error_type': 'network_error'
        }
    except Exception as e:
        error_msg = f"Error downloading from {url}: {str(e)}"
        log_message("ERROR", error_msg)
        return {
            'success': False,
            'error': error_msg,
            'error_type': 'general_error'
        }

def validate_model_file(filepath):
    """
    Validate that the downloaded file is a valid model
    
    Args:
        filepath (str): Path to the model file
    
    Returns:
        dict: Validation result
    """
    try:
        if not os.path.exists(filepath):
            return {
                'valid': False,
                'error': 'File does not exist'
            }
        
        file_size = os.path.getsize(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        log_message("INFO", f"Validating model: {filepath} ({file_size} bytes, {file_ext})")
        
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
                log_message("SUCCESS", f"✅ Model validation successful: {model_info}")
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
        log_message("INFO", f"Loading model from GitHub: {github_url}")
        
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
            log_message("INFO", f"Model file already exists: {model_path}")
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
            log_message("ERROR", f"Model download failed: {download_result['error']}")
            return False
        
        model_path = download_result['filepath']
        
        # Validate model
        log_message("INFO", "Validating downloaded model...")
        validation_result = validate_model_file(model_path)
        
        if not validation_result['valid']:
            model_download_status = {
                'status': 'error',
                'error': f"Model validation failed: {validation_result['error']}"
            }
            log_message("ERROR", f"Model validation failed: {validation_result['error']}")
            return False
        
        # Load model
        log_message("INFO", "Loading validated model...")
        model = tf.keras.models.load_model(model_path)
        
        model_download_status = {
            'status': 'completed',
            'url': github_url,
            'filepath': model_path,
            'file_size': download_result['file_size'],
            'downloaded': download_result['downloaded'],
            'validation': validation_result,
            'message': 'Model loaded successfully'
        }
        
        log_message("SUCCESS", f"Model loaded successfully from {github_url}")
        return True
        
    except Exception as e:
        error_msg = f"Error loading model from URL: {str(e)}"
        model_download_status = {
            'status': 'error',
            'error': error_msg
        }
        log_message("ERROR", error_msg)
        return False

def create_dummy_model():
    """Create a dummy model for testing"""
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        log_message("INFO", "Dummy model created successfully")
        return model
    except Exception as e:
        log_message("ERROR", f"Failed to create dummy model: {e}")
        return None

def load_initial_model():
    """Load initial model (dummy or from local file)"""
    global model
    
    # Try to load from local file first
    local_model_path = os.path.join(MODEL_CACHE_DIR, 'wildfire_model.h5')
    
    try:
        if os.path.exists(local_model_path):
            model = tf.keras.models.load_model(local_model_path)
            log_message("INFO", f"Model loaded from local file: {local_model_path}")
            return True
    except Exception as e:
        log_message("WARNING", f"Failed to load local model: {e}")
    
    # Create dummy model as fallback
    model = create_dummy_model()
    return model is not None

# Flask Routes - Video Analysis
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
        
        file.save(video_path)
        
        # Create job record
        active_jobs[job_id] = {
            'job_id': job_id,
            'filename': filename,
            'video_path': video_path,
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(video_path)
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

@app.route('/job_status/<job_id>')
def job_status(job_id):
    """Get status of analysis job"""
    if job_id not in active_jobs:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    job_info = active_jobs[job_id].copy()
    
    # Add queue information
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
            'created_at': job.get('created_at', 'Unknown')
        }
        
        if job.get('status') == 'completed' and 'results' in job:
            summary['wildfire_detected'] = job['results']['summary'].get('wildfire_detected', False)
            summary['total_detections'] = job['results']['summary'].get('total_detections', 0)
        
        jobs_summary.append(summary)
    
    return jsonify({
        'success': True,
        'queue_size': inference_queue.qsize(),
        'total_jobs': len(active_jobs),
        'jobs': jobs_summary
    })

# Flask Routes - Model Management
@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Load model from GitHub URL"""
    try:
        data = request.get_json()
        
        if not data or 'github_url' not in data:
            return jsonify({
                'success': False,
                'error': 'GitHub URL is required'
            }), 400
        
        github_url = data['github_url']
        force_download = data.get('force_download', False)
        
        # Validate URL
        if 'github.com' not in github_url:
            return jsonify({
                'success': False,
                'error': 'Invalid GitHub URL'
            }), 400
        
        # Start model loading in background thread
        def load_model_background():
            load_model_from_url(github_url, force_download)
        
        thread = threading.Thread(target=load_model_background)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model loading started',
            'github_url': github_url
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model_status')
def model_status():
    """Get current model status"""
    global model, model_download_status, model_url
    
    status = {
        'model_loaded': model is not None,
        'model_url': model_url,
        'download_status': model_download_status
    }
    
    if model is not None:
        try:
            status['model_info'] = {
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'parameters': model.count_params()
            }
        except:
            status['model_info'] = 'Info not available'
    
    return jsonify(status)

@app.route('/current_model_info')
def current_model_info():
    """Get information about the currently loaded model"""
    global model, model_url, model_download_status
    
    if model is None:
        return jsonify({
            'model_loaded': False,
            'message': 'No model currently loaded'
        })
    
    try:
        model_info = {
            'model_loaded': True,
            'source_url': model_url,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_parameters': model.count_params(),
            'layer_count': len(model.layers),
            'model_summary': []
        }
        
        # Get layer information
        for i, layer in enumerate(model.layers):
            layer_info = {
                'layer_index': i,
                'layer_name': layer.name,
                'layer_type': type(layer).__name__,
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
            }
            model_info['model_summary'].append(layer_info)
        
        return jsonify(model_info)
        
    except Exception as e:
        return jsonify({
            'model_loaded': True,
            'error': f'Error getting model info: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Enhanced health check with diagnostics"""
    health_info = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'queue_size': inference_queue.qsize(),
        'active_jobs': len([job for job in active_jobs.values() if job.get('status') == 'processing']),
        'total_logs': len(system_logs),
        'directories': {},
        'dependencies': {},
        'system_info': {}
    }
    
    # Check directories
    for name, path in [('upload', UPLOAD_FOLDER), ('results', RESULTS_FOLDER), ('models', MODEL_CACHE_DIR)]:
        health_info['directories'][name] = {
            'path': path,
            'exists': os.path.exists(path),
            'writable': os.access(path, os.W_OK) if os.path.exists(path) else False
        }
    
    # Check dependencies
    try:
        health_info['dependencies']['tensorflow'] = tf.__version__
    except:
        health_info['dependencies']['tensorflow'] = 'Not available'
    
    try:
        import cv2
        health_info['dependencies']['opencv'] = cv2.__version__
    except:
        health_info['dependencies']['opencv'] = 'Not available'
    
    # System info
    health_info['system_info'] = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'user_id': os.getuid() if hasattr(os, 'getuid') else 'unknown'
    }
    
    return jsonify(health_info)

@app.route('/')
def home():
    """Enhanced home page with video upload and model management"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔥 Wildfire Detection System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1000px; margin: 0 auto; }
            .card { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #0056b3; }
            .btn-danger { background: #dc3545; }
            .btn-danger:hover { background: #c82333; }
            input[type="url"], input[type="file"] { width: 100%; padding: 8px; margin: 5px 0; }
            .status { margin: 10px 0; padding: 10px; border-radius: 3px; }
            .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .info { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
            .progress { background: #f8f9fa; border: 1px solid #dee2e6; height: 20px; border-radius: 3px; }
            .progress-bar { background: #007bff; height: 100%; border-radius: 3px; transition: width 0.3s; }
            .job-item { border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 3px; }
            .two-column { display: flex; gap: 20px; }
            .column { flex: 1; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔥 Wildfire Detection System</h1>
            
            <div class="card">
                <h3>📊 System Status</h3>
                <p><strong>Status:</strong> <span id="systemStatus">Loading...</span></p>
                <p><strong>Model Loaded:</strong> <span id="modelLoaded">Checking...</span></p>
                <p><strong>Model Source:</strong> <span id="modelSource">Unknown</span></p>
                <p><strong>Queue Size:</strong> <span id="queueSize">0</span></p>
                <p><strong>Active Jobs:</strong> <span id="activeJobs">0</span></p>
            </div>
            
            <div class="two-column">
                <div class="column">
                    <div class="card">
                        <h3>🤖 Model Management</h3>
                        <div id="modelStatus" class="status info">Loading model status...</div>
                        
                        <h4>Load New Model from GitHub</h4>
                        <input type="url" id="githubUrl" placeholder="https://github.com/user/repo/blob/main/model.h5">
                        <br>
                        <label>
                            <input type="checkbox" id="forceDownload"> Force re-download
                        </label>
                        <br><br>
                        <button onclick="loadModel()" class="btn">📥 Load Model</button>
                        <button onclick="checkStatus()" class="btn">🔄 Refresh Status</button>
                    </div>
                </div>
                
                <div class="column">
                    <div class="card">
                        <h3>📹 Video Analysis</h3>
                        <div id="uploadStatus" class="status info">Ready to upload video</div>
                        
                        <h4>Upload Video for Analysis</h4>
                        <input type="file" id="videoFile" accept=".mp4,.avi,.mov,.mkv,.webm,.flv,.wmv">
                        <br><br>
                        <button onclick="uploadVideo()" class="btn">🎬 Upload & Analyze</button>
                        <button onclick="refreshJobs()" class="btn">🔄 Refresh Jobs</button>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>📋 Analysis Jobs</h3>
                <div id="jobsList">Loading jobs...</div>
            </div>
            
            <div class="card">
                <h3>🔗 API Endpoints</h3>
                <ul>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/model_status">Model Status (JSON)</a></li>
                    <li><a href="/current_model_info">Detailed Model Info (JSON)</a></li>
                    <li><a href="/active_jobs">Active Jobs (JSON)</a></li>
                </ul>
            </div>
        </div>
        
        <script>
            let statusInterval;
            
            async function checkStatus() {
                try {
                    // Get system health
                    const healthResponse = await fetch('/health');
                    const health = await healthResponse.json();
                    
                    document.getElementById('systemStatus').textContent = health.status;
                    document.getElementById('modelLoaded').textContent = health.model_loaded ? 'Yes ✅' : 'No ❌';
                    document.getElementById('queueSize').textContent = health.queue_size;
                    document.getElementById('activeJobs').textContent = health.active_jobs;
                    
                    // Get model status
                    const modelResponse = await fetch('/model_status');
                    const modelStatus = await modelResponse.json();
                    
                    document.getElementById('modelSource').textContent = modelStatus.model_url || 'None';
                    
                    let statusHtml = '';
                    if (modelStatus.download_status && Object.keys(modelStatus.download_status).length > 0) {
                        const ds = modelStatus.download_status;
                        statusHtml = `
                            <strong>Download Status:</strong> ${ds.status}<br>
                            ${ds.message ? `<strong>Message:</strong> ${ds.message}<br>` : ''}
                            ${ds.error ? `<strong>Error:</strong> ${ds.error}<br>` : ''}
                            ${ds.file_size ? `<strong>File Size:</strong> ${(ds.file_size / 1024 / 1024).toFixed(2)} MB<br>` : ''}
                        `;
                        
                        const statusDiv = document.getElementById('modelStatus');
                        statusDiv.innerHTML = statusHtml;
                        statusDiv.className = 'status ' + (ds.status === 'completed' ? 'success' : ds.status === 'error' ? 'error' : 'info');
                    }
                } catch (error) {
                    console.error('Error checking status:', error);
                }
            }
            
            async function loadModel() {
                const githubUrl = document.getElementById('githubUrl').value;
                const forceDownload = document.getElementById('forceDownload').checked;
                
                if (!githubUrl) {
                    alert('Please enter a GitHub URL');
                    return;
                }
                
                try {
                    const response = await fetch('/load_model', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            github_url: githubUrl,
                            force_download: forceDownload
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        alert('Model loading started! Status will update automatically.');
                        startStatusPolling();
                    } else {
                        alert('Failed to start model loading: ' + result.error);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function uploadVideo() {
                const fileInput = document.getElementById('videoFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a video file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('video', file);
                
                const uploadStatus = document.getElementById('uploadStatus');
                uploadStatus.innerHTML = 'Uploading video...';
                uploadStatus.className = 'status info';
                
                try {
                    const response = await fetch('/upload_and_analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        uploadStatus.innerHTML = `Video uploaded successfully! Job ID: ${result.job_id}`;
                        uploadStatus.className = 'status success';
                        refreshJobs();
                        startJobPolling();
                    } else {
                        uploadStatus.innerHTML = 'Upload failed: ' + result.error;
                        uploadStatus.className = 'status error';
                    }
                } catch (error) {
                    uploadStatus.innerHTML = 'Upload error: ' + error.message;
                    uploadStatus.className = 'status error';
                }
            }
            
            async function refreshJobs() {
                try {
                    const response = await fetch('/active_jobs');
                    const result = await response.json();
                    
                    if (result.success) {
                        let jobsHtml = '';
                        
                        if (result.jobs.length === 0) {
                            jobsHtml = '<p>No active jobs</p>';
                        } else {
                            result.jobs.forEach(job => {
                                const statusClass = job.status === 'completed' ? 'success' : 
                                                  job.status === 'failed' ? 'error' : 
                                                  job.status === 'processing' ? 'warning' : 'info';
                                
                                jobsHtml += `
                                    <div class="job-item ${statusClass}">
                                        <strong>Job ${job.job_id.substring(0, 8)}</strong> - ${job.filename}<br>
                                        Status: ${job.status} | Progress: ${job.progress}%<br>
                                        ${job.status === 'processing' ? `
                                            <div class="progress">
                                                <div class="progress-bar" style="width: ${job.progress}%"></div>
                                            </div>
                                        ` : ''}
                                        ${job.status === 'completed' ? `
                                            <strong>Wildfire Detected:</strong> ${job.wildfire_detected ? 'Yes ⚠️' : 'No ✅'} | 
                                            <strong>Detections:</strong> ${job.total_detections || 0}<br>
                                            <button onclick="downloadResults('${job.job_id}')" class="btn">📄 Download Results</button>
                                            <button onclick="viewResults('${job.job_id}')" class="btn">👁️ View Results</button>
                                        ` : ''}
                                    </div>
                                `;
                            });
                        }
                        
                        document.getElementById('jobsList').innerHTML = jobsHtml;
                    }
                } catch (error) {
                    console.error('Error refreshing jobs:', error);
                }
            }
            
            async function downloadResults(jobId) {
                window.open(`/download_results/${jobId}`, '_blank');
            }
            
            async function viewResults(jobId) {
                try {
                    const response = await fetch(`/job_results/${jobId}`);
                    const result = await response.json();
                    
                    if (result.success) {
                        const resultsWindow = window.open('', '_blank');
                        resultsWindow.document.write(`
                            <html>
                                <head><title>Analysis Results - ${jobId}</title></head>
                                <body>
                                    <h1>Wildfire Detection Results</h1>
                                    <pre>${JSON.stringify(result.results, null, 2)}</pre>
                                </body>
                            </html>
                        `);
                    } else {
                        alert('Failed to get results: ' + result.error);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            function startStatusPolling() {
                if (statusInterval) clearInterval(statusInterval);
                statusInterval = setInterval(checkStatus, 2000);
            }
            
            function startJobPolling() {
                const jobInterval = setInterval(async () => {
                    await refreshJobs();
                    
                    // Check if all jobs are completed
                    const response = await fetch('/active_jobs');
                    const result = await response.json();
                    
                    const hasActiveJobs = result.jobs.some(job => 
                        job.status === 'processing' || job.status === 'queued'
                    );
                    
                    if (!hasActiveJobs) {
                        clearInterval(jobInterval);
                    }
                }, 3000);
            }
            
            // Load initial status and jobs
            checkStatus();
            refreshJobs();
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🔧 Starting Flask application...")
    
    # Load initial model
    if load_initial_model():
        log_message("SUCCESS", "Initial model loaded successfully")
    else:
        log_message("ERROR", "Failed to load initial model")
        sys.exit(1)
    
    # Start inference worker thread
    worker_thread = threading.Thread(target=inference_worker, daemon=True)
    worker_thread.start()
    
    log_message("INFO", "🔥 Wildfire Detection Server Starting...")
    log_message("INFO", f"📹 Server will be available at: http://0.0.0.0:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        log_message("ERROR", f"Failed to start Flask app: {e}")
        sys.exit(1)
