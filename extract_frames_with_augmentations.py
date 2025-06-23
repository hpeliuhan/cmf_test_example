import cv2
import os
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

# --- Configuration ---
video_files = ["fire_sample1.mp4", "fire_sample2.mp4", "fire_sample3.mp4"]
output_dir = "data/extracted_frames_with_augmentations"
frame_interval = 10  # Extract every 10th frame
resize_dim = (224, 224)  # Standard input size

# --- Augmentation transforms ---
augmentation_transforms = [
    T.RandomResizedCrop(size=resize_dim, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomHorizontalFlip(p=1.0),
    T.GaussianBlur(kernel_size=5)
]
base_transform = T.Compose([T.Resize(resize_dim), T.ToTensor()])

# --- Frame Extraction + Augmentation ---
os.makedirs(output_dir, exist_ok=True)

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error opening video file {video_file}")
        continue

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_frame_dir = output_dir  # single flat folder
    os.makedirs(video_frame_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Save original (resized)
            orig_tensor = base_transform(pil_img)
            orig_path = os.path.join(video_frame_dir, f"{video_name}_frame_{saved_count:04d}_orig.jpg")
            TF.to_pil_image(orig_tensor).save(orig_path)

            # Save 4 augmentations
            for i, aug in enumerate(augmentation_transforms):
                aug_tensor = base_transform(aug(pil_img))
                aug_path = os.path.join(video_frame_dir, f"{video_name}_frame_{saved_count:04d}_aug{i}.jpg")
                TF.to_pil_image(aug_tensor).save(aug_path)

            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Saved {saved_count * 5} images (including augments) from {video_file}.")

print("✅ All frames and augmentations saved.")
