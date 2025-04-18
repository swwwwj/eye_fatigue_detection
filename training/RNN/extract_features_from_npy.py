import os
import numpy as np
import cv2
import dlib
from scipy.spatial import distance

data_root = "/home/swj/eye_fatigue_detection/data/"
processed_data_root = "/home/swj/eye_fatigue_detection/data/processed_features"

#加载dlib库里的预训练模型进行人脸检测
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/swj/eye_fatigue_detection/models/shape_predictor_68_face_landmarks.dat')

#EAR值，用于计算后面的参数
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#perclos为闭眼帧在总帧中的占比
def perclos(eye_aspect_ratio_sequence, threshold=0.2):
    return sum([1 for ear in eye_aspect_ratio_sequence if ear < threshold]) / len(eye_aspect_ratio_sequence)

#瞳孔运动的距离
def pupil_motion(pupil_positions):
    distances = [distance.euclidean(pupil_positions[i], pupil_positions[i-1]) for i in range(1, len(pupil_positions))]
    return np.sum(distances)

def extract_features_from_npy(npy_file):
    frames = np.load(npy_file)
    eye_aspect_ratios = []
    pupil_positions = []
    blink_frame_count = 0

    for frame_idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 1)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            left_eye = []
            right_eye = []

            for i in range(36, 42):
                left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
            for i in range(42, 48):
                right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            eye_aspect_ratios.append(ear)

            if ear < 0.2:
                blink_frame_count += 1

            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            pupil_positions.append((left_eye_center + right_eye_center) / 2)
    
    blink_rate = blink_frame_count / len(frames)
    perclos_value = perclos(eye_aspect_ratios)
    pupil_motion_distance = pupil_motion(pupil_positions)

    return blink_rate, perclos_value, pupil_motion_distance

def process_video(npy_folder, output_folder):
    for root, dirs, files in os.walk(npy_folder):
        npy_files = [f for f in files if f.endswith('.npy')]

        for npy_file in npy_files:
            npy_file_path = os.path.join(root, npy_file)
            print(f"Processing {npy_file_path}...")

            output_file_path = os.path.join(output_folder, npy_file.replace('.npy', '_features.npy'))
            if os.path.exists(output_file_path): 
                print(f"Skipping {npy_file_path}, features already exist.")
                continue

            blink_rate, perclos_value, pupil_motion_distance = extract_features_from_npy(npy_file_path)

            features = np.array([blink_rate, perclos_value, pupil_motion_distance])
            np.save(output_file_path, features)
            print(f"Saved features to {output_file_path}")

def main():
    npy_folder = "/home/swj/eye_fatigue_detection/data/RNN"
    output_folder = "/home/swj/eye_fatigue_detection/data/processed_features"
    os.makedirs(output_folder, exist_ok=True)

    process_video(npy_folder, output_folder)

if __name__ == "__main__":
    main()
