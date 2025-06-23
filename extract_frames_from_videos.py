import cv2
import os

# Set up
video_files = ["fire_sample1.mp4", "fire_sample2.mp4", "fire_sample3.mp4"]
output_dir = "data/extracted_frames"
frame_interval = 10  # Extract every 10th frame

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error opening video file {video_file}")
        continue

    # Get video name without extension for folder naming
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_frame_dir = os.path.join(output_dir)
    os.makedirs(video_frame_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(video_frame_dir, f"{video_name}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames from {video_file}.")

print("✅ All frames extracted.")