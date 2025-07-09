import requests
import time
import sys

def test_video_inference(video_path, server_url="http://localhost:5000"):
    """Test video inference using the Flask API"""
    
    try:
        # 1. Upload video
        print(f"📤 Uploading video: {video_path}")
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(f"{server_url}/upload_and_analyze", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.text}")
            return
        
        data = response.json()
        if not data['success']:
            print(f"❌ Upload failed: {data['error']}")
            return
        
        job_id = data['job_id']
        print(f"✅ Upload successful! Job ID: {job_id}")
        
        # 2. Poll for results
        print("🔄 Waiting for analysis to complete...")
        while True:
            response = requests.get(f"{server_url}/job_status/{job_id}")
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.text}")
                return
            
            status = response.json()
            print(f"📊 Status: {status['status']} - {status.get('current_stage', 'Processing')} ({status.get('progress', 0)}%)")
            
            if status['status'] == 'completed':
                break
            elif status['status'] == 'failed':
                print(f"❌ Analysis failed: {status.get('error', 'Unknown error')}")
                return
            
            time.sleep(5)
        
        # 3. Get results
        print("📊 Getting results...")
        response = requests.get(f"{server_url}/results/{job_id}")
        if response.status_code != 200:
            print(f"❌ Results fetch failed: {response.text}")
            return
        
        results = response.json()
        
        # Print summary
        summary = results.get('summary', {})
        print(f"\n🎯 ANALYSIS RESULTS:")
        print(f"   • Fire detected: {'YES 🔥' if summary.get('wildfire_detected') else 'NO ✅'}")
        print(f"   • Total detections: {summary.get('total_detections', 0)}")
        print(f"   • Frames analyzed: {results.get('total_frames_analyzed', 0)}")
        print(f"   • Risk percentage: {summary.get('risk_percentage', 0):.2f}%")
        print(f"   • Average confidence: {summary.get('average_confidence', 0):.3f}")
        
        # Show top detections
        detections = results.get('detections', [])
        if detections:
            print(f"\n🔝 Top detections:")
            for i, detection in enumerate(sorted(detections, key=lambda x: x['fire_confidence'], reverse=True)[:5], 1):
                print(f"   {i}. t={detection['timestamp']:.2f}s, confidence={detection['fire_confidence']:.3f}")
        
        print(f"\n💾 Full results available at: {server_url}/results/{job_id}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_test.py /path/to/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_video_inference(video_path)