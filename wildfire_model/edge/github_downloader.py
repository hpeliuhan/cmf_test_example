# github_downloader.py
import os
import requests
import hashlib
from urllib.parse import urlparse
from datetime import datetime
import tempfile
import shutil

def log_download(message, level="INFO"):
    """Log download messages"""
    print(f"[{level}] {message}")

def get_filename_from_url(url):
    """Extract filename from GitHub URL"""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # Handle GitHub raw URLs
    if 'github.com' in url and '/blob/' in url:
        # Convert blob URL to raw URL
        url = url.replace('/blob/', '/raw/')
    
    return filename, url

def calculate_file_hash(filepath):
    """Calculate MD5 hash of file for verification"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file_from_github(url, destination_dir, filename=None, force_download=False):
    """
    Download a file from GitHub
    
    Args:
        url (str): GitHub URL (can be blob or raw URL)
        destination_dir (str): Directory to save the file
        filename (str, optional): Custom filename, if None uses URL filename
        force_download (bool): Force download even if file exists
    
    Returns:
        dict: Download result with status, filepath, and metadata
    """
    try:
        # Ensure destination directory exists
        os.makedirs(destination_dir, exist_ok=True)
        
        # Get filename and convert URL if needed
        if filename is None:
            filename, download_url = get_filename_from_url(url)
        else:
            download_url = url.replace('/blob/', '/raw/') if '/blob/' in url else url
        
        filepath = os.path.join(destination_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath) and not force_download:
            file_size = os.path.getsize(filepath)
            log_download(f"File already exists: {filepath} ({file_size} bytes)")
            return {
                'success': True,
                'filepath': filepath,
                'filename': filename,
                'file_size': file_size,
                'downloaded': False,
                'message': 'File already exists'
            }
        
        log_download(f"Starting download from: {download_url}")
        
        # Download file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Get file size from headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Download to temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filepath = temp_file.name
            downloaded_size = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Log progress for large files
                    if total_size > 0 and downloaded_size % (1024 * 1024) == 0:  # Every MB
                        progress = (downloaded_size / total_size) * 100
                        log_download(f"Download progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)")
        
        # Verify download
        if total_size > 0 and downloaded_size != total_size:
            os.unlink(temp_filepath)
            raise Exception(f"Download incomplete: {downloaded_size}/{total_size} bytes")
        
        # Move temp file to final location
        shutil.move(temp_filepath, filepath)
        
        # Calculate file hash for verification
        file_hash = calculate_file_hash(filepath)
        
        log_download(f"✅ Download completed: {filepath} ({downloaded_size} bytes, MD5: {file_hash[:8]}...)")
        
        return {
            'success': True,
            'filepath': filepath,
            'filename': filename,
            'file_size': downloaded_size,
            'file_hash': file_hash,
            'downloaded': True,
            'download_time': datetime.now().isoformat(),
            'message': 'Download completed successfully'
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error downloading from {url}: {str(e)}"
        log_download(error_msg, "ERROR")
        return {
            'success': False,
            'error': error_msg,
            'error_type': 'network_error'
        }
    
    except Exception as e:
        error_msg = f"Error downloading from {url}: {str(e)}"
        log_download(error_msg, "ERROR")
        return {
            'success': False,
            'error': error_msg,
            'error_type': 'general_error'
        }

def download_model_from_github(github_url, models_dir='./models', model_name=None):
    """
    Download a model file from GitHub
    
    Args:
        github_url (str): GitHub URL to the model file
        models_dir (str): Directory to save models
        model_name (str, optional): Custom model name
    
    Returns:
        dict: Download result
    """
    log_download(f"🔄 Downloading model from GitHub: {github_url}")
    
    # Validate GitHub URL
    if 'github.com' not in github_url:
        return {
            'success': False,
            'error': 'Invalid GitHub URL',
            'error_type': 'invalid_url'
        }
    
    # Download the model
    result = download_file_from_github(
        url=github_url,
        destination_dir=models_dir,
        filename=model_name
    )
    
    if result['success']:
        log_download(f"🎉 Model download successful: {result['filepath']}")
    else:
        log_download(f"❌ Model download failed: {result['error']}", "ERROR")
    
    return result

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
        
        # Check file size (models should be reasonable size)
        if file_size < 1024:  # Less than 1KB
            return {
                'valid': False,
                'error': 'File too small to be a valid model'
            }
        
        # Check file extension
        valid_extensions = ['.h5', '.pb', '.onnx', '.tflite', '.pkl', '.pth', '.pt']
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext not in valid_extensions:
            return {
                'valid': False,
                'error': f'Invalid model file extension: {file_ext}. Expected: {valid_extensions}'
            }
        
        # Try to load with TensorFlow if it's a .h5 file
        if file_ext == '.h5':
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(filepath)
                model_info = {
                    'input_shape': model.input_shape if hasattr(model, 'input_shape') else 'Unknown',
                    'output_shape': model.output_shape if hasattr(model, 'output_shape') else 'Unknown',
                    'parameters': model.count_params() if hasattr(model, 'count_params') else 'Unknown'
                }
                log_download(f"✅ Model validation successful: {model_info}")
            except Exception as e:
                return {
                    'valid': False,
                    'error': f'Failed to load TensorFlow model: {str(e)}'
                }
        
        return {
            'valid': True,
            'file_size': file_size,
            'file_extension': file_ext,
            'message': 'Model file is valid'
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f'Validation error: {str(e)}'
        }

# Test function
def test_github_download():
    """Test the GitHub download functionality"""
    # Test with a small file from GitHub
    test_url = "https://github.com/hpeliuhan/cmf_test_example/blob/wildfire_model/model.tflite"
    result = download_file_from_github(test_url, './test_downloads')
    
    print("Test Result:", result)
    
    # Clean up test file
    if result['success'] and os.path.exists(result['filepath']):
        os.remove(result['filepath'])
        os.rmdir('./test_downloads')

if __name__ == "__main__":
    test_github_download()