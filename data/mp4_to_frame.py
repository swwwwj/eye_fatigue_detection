import os
import numpy as np
import cv2

video_folder = "/mnt/c/Users/38382/fatigue_videos/"
data_root = "/home/swj/eye_fatigue_detection/data/"
sequence_length = 300

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
    video_path = os.path.join(video_folder, video_name)
    fatigue_level = None
    for level in fatigue_levels.keys():
        if level in video_name.lower():
            fatigue_level = level
            break

    if not fatigue_level:
        print("cannot recognize fatigue level!")
        continue

    cnn_output_folder = os.path.join(data_root, "cnn", fatigue_levels[fatigue_level])
    rnn_output_folder = os.path.join(data_root, "rnn", fatigue_levels[fatigue_level])

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        cnn_frame_path = os.path.join(cnn_output_folder, f"{video_name}_frame_{frame_idx:04d}.jpg")
        cv2.imwrite(cnn_frame_path, frame)

        frames.append(frame)
        if len(frames) == sequence_length:
            rnn_file_path = os.path.join(rnn_output_folder, f"{video_name}_seq_{frame_idx - sequence_length + 1:04d}.npy")
            np.save(rnn_file_path, np.array(frames)) 
            frames = []

        frame_idx += 1

    cap.release()
