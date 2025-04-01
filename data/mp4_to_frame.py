import os
import numpy as np
import cv2

video_folder = "/mnt/c/Users/38382/fatigue_videos/"
data_root = "/home/swj/eye_fatigue_detection/data/"
sequence_length = 300
cnn_frame_count = 300

fatigue_levels = {
    "awake": "awake",
    "mild_fatigue": "mild_fatigue",
    "moderate_fatigue": "moderate_fatigue",
    "severe_fatigue": "severe_fatigue"
}

for level in fatigue_levels.values():
    os.makedirs(os.path.join(data_root, "CNN", level), exist_ok=True)
    os.makedirs(os.path.join(data_root, "RNN", level), exist_ok=True)

video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
if not video_files:
    print("No such a video! Please make sure your path is correct!")
    exit()

for video_name in video_files:
    file_name, _ = os.path.splitext(video_name)  
    fatigue_level = None
    for level in fatigue_levels.keys():
        if level in file_name.lower():
            fatigue_level = level
            break

    if not fatigue_level:
        print(f"Cannot recognize fatigue level for {video_name}!")
        continue

    cnn_output_folder = os.path.join(data_root, "CNN", fatigue_levels[fatigue_level])
    rnn_output_folder = os.path.join(data_root, "RNN", fatigue_levels[fatigue_level])
    video_path = os.path.join(video_folder, video_name)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    step_size = max(1, total_frames // cnn_frame_count)

    frames = []
    cnn_saved = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        if frame_idx % step_size == 0 and cnn_saved < cnn_frame_count:
            cnn_frame_path = os.path.join(cnn_output_folder, f"{video_name}_frame_{cnn_saved:04d}.jpg")
            cv2.imwrite(cnn_frame_path, frame)
            cnn_saved += 1  

        frames.append(frame)
        if len(frames) == sequence_length:
            rnn_file_path = os.path.join(rnn_output_folder, f"{video_name}_seq_{frame_idx - sequence_length + 1:04d}.npy")
            np.save(rnn_file_path, np.array(frames)) 
            frames = []

        frame_idx += 1

    cap.release()
