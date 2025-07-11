import requests

# Upload video for active learning analysis
with open('../../fire_sample2.mp4', 'rb') as f:
    response = requests.post(
        'http://192.168.30.129:5001/analyze_for_active_learning',
        files={'video': f},
        data={'fine_tune_server_url': 'http://192.168.30.129:5002'}
    )

result = response.json()
print(f"Job ID: {result}")