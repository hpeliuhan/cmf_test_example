import requests
import sys
import os

# === Configuration ===
UPLOAD_URL = "http://192.168.30.129:5001/upload_and_analyze"  # Replace with your actual endpoint

def upload_video(video_path):
    if not os.path.isfile(video_path):
        print(f"❌ File not found: {video_path}")
        return

    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(UPLOAD_URL, files=files)

        if response.status_code == 200:
            print(f"✅ Upload successful: {os.path.basename(video_path)}")
            print("Server response:", response.json())
        else:
            print("❌ Upload failed:", response.status_code)
            print("Server response:", response.text)

    except requests.RequestException as e:
        print(f"❌ Upload error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_video.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    upload_video(video_path)
