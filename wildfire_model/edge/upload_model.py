#!/usr/bin/env python3
# upload_model.py

import requests
import argparse
import sys
import os

def upload_model_from_github(server_url, github_url):
    """Upload model from GitHub URL"""
    try:
        # Send to the correct endpoint for GitHub URLs
        response = requests.post(
            f"{server_url}/load_model",
            json={
                "url": github_url,
                "force_download": True
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Model loaded successfully from GitHub!")
                print(f"📊 Model info: {result.get('model_info', {})}")
                return True
            else:
                print(f"❌ Failed to load model: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def upload_model_from_file(server_url, file_path):
    """Upload model from local file"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        with open(file_path, 'rb') as f:
            files = {'model': f}
            response = requests.post(
                f"{server_url}/upload_model",
                files=files,
                timeout=300
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Model uploaded successfully!")
                print(f"📊 Model info: {result.get('model_info', {})}")
                return True
            else:
                print(f"❌ Failed to upload model: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload model to WildFire detection server')
    parser.add_argument('--server', required=True, help='Server URL (e.g., http://localhost:5000)')
    parser.add_argument('model_source', help='Model source: GitHub URL or local file path')
    
    args = parser.parse_args()
    
    # Determine if it's a GitHub URL or local file
    if args.model_source.startswith('http'):
        print(f"🔄 Loading model from GitHub URL: {args.model_source}")
        success = upload_model_from_github(args.server, args.model_source)
    else:
        print(f"🔄 Uploading model from local file: {args.model_source}")
        success = upload_model_from_file(args.server, args.model_source)
    
    if success:
        print("🎉 Model operation completed successfully!")
        sys.exit(0)
    else:
        print("💥 Model operation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()