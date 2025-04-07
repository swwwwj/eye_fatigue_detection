import os
import numpy as np
import cv2
import dlib
import glob
import random

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

def clean_hidden_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('.'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed hidden file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

for level in fatigue_levels.values():
    cnn_path = os.path.join(data_root, "CNN", level)
    rnn_path = os.path.join(data_root, "RNN", level)
    os.makedirs(cnn_path, exist_ok=True)
    os.makedirs(rnn_path, exist_ok=True)
    clean_hidden_files(cnn_path)
    clean_hidden_files(rnn_path)

video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
if not video_files:
    print("No such a video! Please make sure your path is correct!")
    exit()

detector = dlib.get_frontal_face_detector()

for video_name in video_files:
    file_name, _ = os.path.splitext(video_name)
    fatigue_level = file_name[:-1].lower()
    
    if fatigue_level not in fatigue_levels:
        print(f"Cannot recognize fatigue level for {video_name}!")
        continue

    print(f"\nProcessing video: {video_name}")
    print(f"Fatigue level: {fatigue_level}")

    cnn_output_folder = os.path.join(data_root, "CNN", fatigue_levels[fatigue_level])
    rnn_output_folder = os.path.join(data_root, "RNN", fatigue_levels[fatigue_level])
    video_path = os.path.join(video_folder, video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_name}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    expected_rnn_sequences = total_frames // sequence_length
    print(f"Video info: {total_frames} frames, {fps} FPS")
    print(f"Expected RNN sequences: {expected_rnn_sequences}")

    cnn_files = glob.glob(os.path.join(cnn_output_folder, f"{video_name}_frame_*.jpg"))
    rnn_files = glob.glob(os.path.join(rnn_output_folder, f"{video_name}_seq_*.npy"))

    cnn_complete = len(cnn_files) >= cnn_frame_count
    rnn_complete = len(rnn_files) >= expected_rnn_sequences

    if cnn_complete and rnn_complete:
        print(f"Skipping {video_name}, complete data already exists.")
        cap.release()
        continue

    if not cnn_complete:
        print("Cleaning existing incomplete CNN data...")
        for f in cnn_files:
            os.remove(f)

    if not rnn_complete:
        print("Cleaning existing incomplete RNN data...")
        for f in rnn_files:
            os.remove(f)

    cnn_sample_interval = max(1, total_frames // cnn_frame_count)
    print(f"CNN sampling interval: {cnn_sample_interval} frames")

    frames = []
    cnn_saved = 0
    frame_idx = 0
    rnn_saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector(frame, 1)
        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            face_img = frame[y1:y2, x1:x2]
            
            try:
                face_resized = cv2.resize(face_img, (224, 224))
                
                if not cnn_complete and frame_idx % cnn_sample_interval == 0 and cnn_saved < cnn_frame_count:
                    cnn_frame_path = os.path.join(cnn_output_folder, 
                                                  f"{video_name}_frame_{cnn_saved:04d}.jpg")
                    cv2.imwrite(cnn_frame_path, face_resized)
                    cnn_saved += 1

                if not rnn_complete:
                    frames.append(face_resized)
                    if len(frames) == sequence_length:
                        rnn_file_path = os.path.join(rnn_output_folder, 
                                                     f"{video_name}_seq_{rnn_saved:04d}.npy")
                        np.save(rnn_file_path, np.array(frames))
                        frames = []
                        rnn_saved += 1

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    if not rnn_complete and len(frames) > 0:
        print("Augmenting last RNN sequence with flipping...")
        while len(frames) < sequence_length:

            choice = random.choice(frames)
            flipped = cv2.flip(choice, 1)
            frames.append(flipped)
        rnn_file_path = os.path.join(rnn_output_folder, 
                                     f"{video_name}_seq_{rnn_saved:04d}.npy")
        np.save(rnn_file_path, np.array(frames))
        rnn_saved += 1

    cap.release()
    print(f"Completed {video_name}: {cnn_saved} CNN frames, {rnn_saved} RNN sequences")

print("\nAll videos processed.")
