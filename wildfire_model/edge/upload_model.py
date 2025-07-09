#!/usr/bin/env python3
# upload_model.py

import requests
import json
import time
import argparse
import sys
from urllib.parse import urlparse

class ModelUploader:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        
    def check_server_health(self):
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Server is running and healthy")
                return True
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def validate_url(self, url):
        """Validate if URL is a proper GitHub URL"""
        parsed = urlparse(url)
        
        if not parsed.scheme:
            print("❌ URL must include protocol (http:// or https://)")
            return False
            
        if 'github.com' not in parsed.netloc:
            print("⚠️  Warning: URL is not from GitHub")
            
        # Check for common model file extensions
        valid_extensions = ['.h5', '.pb', '.onnx', '.tflite', '.pkl', '.pth', '.pt']
        if not any(url.lower().endswith(ext) for ext in valid_extensions):
            print(f"⚠️  Warning: URL doesn't end with common model extensions: {valid_extensions}")
            
        return True
    
    def upload_model(self, github_url, force_download=False):
        """Upload model from GitHub URL"""
        print(f"🚀 Starting model upload from: {github_url}")
        
        # Validate URL
        if not self.validate_url(github_url):
            return False
        
        # Check server health
        if not self.check_server_health():
            return False
        
        # Send upload request
        payload = {
            "github_url": github_url,
            "force_download": force_download
        }
        
        try:
            print("📤 Sending upload request...")
            response = requests.post(
                f"{self.base_url}/load_model",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ Upload request successful!")
                    print(f"   Message: {result.get('message')}")
                    return True
                else:
                    print(f"❌ Upload failed: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return False
    
    def monitor_progress(self, timeout=300):
        """Monitor the model loading progress"""
        print("📊 Monitoring progress...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/model_status", timeout=10)
                if response.status_code == 200:
                    status = response.json()
                    download_status = status.get('download_status', {})
                    
                    if download_status:
                        current_status = download_status.get('status', 'unknown')
                        print(f"   Status: {current_status}")
                        
                        if current_status == 'completed':
                            print("🎉 Model loading completed successfully!")
                            self.print_model_info(status)
                            return True
                        elif current_status == 'error':
                            error_msg = download_status.get('error', 'Unknown error')
                            print(f"❌ Model loading failed: {error_msg}")
                            return False
                        elif current_status == 'downloading':
                            print("   📥 Downloading model...")
                        elif current_status == 'processing':
                            print("   🔄 Processing model...")
                
                time.sleep(5)  # Check every 5 seconds
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Error checking progress: {e}")
                time.sleep(5)
        
        print("⏰ Timeout reached while monitoring progress")
        return False
    
    def print_model_info(self, status):
        """Print detailed model information"""
        print("\n📋 MODEL INFORMATION:")
        print(f"   Model Loaded: {'✅' if status.get('model_loaded') else '❌'}")
        print(f"   Source URL: {status.get('model_url', 'Unknown')}")
        
        download_status = status.get('download_status', {})
        if download_status:
            file_size = download_status.get('file_size')
            if file_size:
                print(f"   File Size: {file_size / 1024 / 1024:.2f} MB")
            
            filepath = download_status.get('filepath')
            if filepath:
                print(f"   Local Path: {filepath}")
        
        model_info = status.get('model_info', {})
        if model_info:
            print(f"   Input Shape: {model_info.get('input_shape', 'Unknown')}")
            print(f"   Output Shape: {model_info.get('output_shape', 'Unknown')}")
            parameters = model_info.get('parameters')
            if parameters:
                print(f"   Parameters: {parameters:,}")
    
    def get_current_model_status(self):
        """Get and display current model status"""
        try:
            response = requests.get(f"{self.base_url}/model_status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                self.print_model_info(status)
                return True
            else:
                print(f"❌ Failed to get model status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting model status: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Upload model from GitHub URL')
    parser.add_argument('url', help='GitHub URL of the model file')
    parser.add_argument('--server', default='http://localhost:5000', 
                       help='Server URL (default: http://localhost:5000)')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-download even if model exists')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Don\'t monitor progress after upload')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout for monitoring in seconds (default: 300)')
    parser.add_argument('--status-only', action='store_true',
                       help='Only check current model status, don\'t upload')
    
    args = parser.parse_args()
    
    uploader = ModelUploader(args.server)
    
    if args.status_only:
        print("📊 Checking current model status...")
        uploader.get_current_model_status()
        return
    
    print(f"🔥 Wildfire Model Uploader")
    print(f"   Server: {args.server}")
    print(f"   Model URL: {args.url}")
    print(f"   Force Download: {args.force}")
    print("-" * 50)
    
    # Upload model
    if uploader.upload_model(args.url, args.force):
        if not args.no_monitor:
            # Monitor progress
            if uploader.monitor_progress(args.timeout):
                print("\n✅ Model upload and loading completed successfully!")
                sys.exit(0)
            else:
                print("\n❌ Model loading failed or timed out")
                sys.exit(1)
        else:
            print("\n📤 Upload request sent. Use --status-only to check progress later.")
            sys.exit(0)
    else:
        print("\n❌ Model upload failed")
        sys.exit(1)

if __name__ == "__main__":
    main()