import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import sys, os

from openpose_skeleton import skeleton

def process_frame(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    pose_landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        annotated_frame = frame.copy()
        height, width, _ = frame.shape
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
    else:
        annotated_frame = frame
    return np.array(pose_landmarks), annotated_frame



def normalize_landmarks(landmarks):
    min_x, min_y, min_z = np.min(landmarks[:, :3], axis=0)
    max_x, max_y, max_z = np.max(landmarks[:, :3], axis=0)

    scale = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    normalized_landmarks = (landmarks[:, :3] - min_x) / scale

    return normalized_landmarks

def pose_similarity_cosine(landmarks1, landmarks2):
    if landmarks1.shape != landmarks2.shape:
        raise ValueError("Landmarks must have the same shape")

    selected_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    similarity = 0

    for i in selected_landmarks:
        similarity += cosine_similarity(landmarks1[i, :3].reshape(1, -1), landmarks2[i, :3].reshape(1, -1))

    return similarity / len(selected_landmarks)


def process_pair(frame1, frame2):
    landmarks1, annotated_frame1 = process_frame(frame1)
    landmarks2, annotated_frame2 = process_frame(frame2)

    if landmarks1.size > 0 and landmarks2.size > 0:
        normalized_landmarks1 = normalize_landmarks(landmarks1)
        normalized_landmarks2 = normalize_landmarks(landmarks2)
        similarity_cosine = pose_similarity_cosine(normalized_landmarks1, normalized_landmarks2)

        return frame1, frame2, similarity_cosine
    else:
        return None

video1_path = "pushups-sample_exp.mp4"
video2_path = "pushups-sample_user.mp4"

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

similarity_list = []
frame_pairs = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        result = executor.submit(process_pair, frame1, frame2)
        result = result.result()

        if result:
            frame1, frame2, similarity_cosine = result
            similarity_list.append(similarity_cosine)
            frame_pairs.append((frame1, frame2))

cap1.release()
cap2.release()

if len(similarity_list) > 0:
    similarity_array = np.array(similarity_list)
    num_top_pairs = min(5, len(similarity_array))

    top_indices = sorted(range(len(similarity_array)), key=lambda i: similarity_array[i], reverse=True)[:num_top_pairs]

    for i, index in enumerate(top_indices):
        frame1, frame2 = frame_pairs[index]
        cv2.imwrite(f"video1_frame_{i + 1}_cosine.jpg", frame1)
        cv2.imwrite(f"video2_frame_{i + 1}_cosine.jpg", frame2)

        print(f"Saved pair {i + 1} with Cosine similarity:", similarity_list[index])
else:
    print("No poses found in one or both videos.")


