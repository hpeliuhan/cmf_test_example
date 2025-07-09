import requests

# Replace with your actual raw GitHub URL
MODEL_URL = "https://github.com/hpeliuhan/cmf_test_example/blob/wildfire_model/wildfire_model/best_model.h5"
DESTINATION = "my_model.pt"

def download_model(url, dest):
    print(f"Downloading model from:\n{url}")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Model saved as: {dest}")
    else:
        print(f"❌ Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_model(MODEL_URL, DESTINATION)
