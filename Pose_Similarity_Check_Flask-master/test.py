import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import euclidean

# Mediapipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

face = [0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32]
upper_body = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
lower_body = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
full_body = face + upper_body + lower_body

def normalize_landmarks(landmarks, frame_shape):
    height, width, _ = frame_shape
    normalized_landmarks = []

    for lm in landmarks:
        normalized_landmarks.append((lm.x * width, lm.y * height))

    return normalized_landmarks

def cosine_similarity(a, b):
    return 1 - cosine(a, b)
def euclidean_similarity(a, b):
    return 1 / (1 + euclidean(a, b))
def get_expert_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

def process_frame(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = []
    if results.pose_landmarks is None:
        return None

    for lm_index in full_body:
        landmarks.append(results.pose_landmarks.landmark[lm_index])

    return normalize_landmarks(landmarks, frame.shape)
def extract_expert_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    expert_frames = []

    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % fps == 0:
                landmarks = process_frame(frame, pose)
                if landmarks is not None:
                    expert_frames.append((frame, landmarks))

            frame_count += 1

    cap.release()
    return expert_frames

def compare_videos(video1, video2):
    mp_holistic = mp.solutions.holistic
    cap1 = cv2.VideoCapture(video1)

    similarities = []
    top_5 = []

    expert_frames = get_expert_frames(video2)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap1.isOpened():
            ret1, frame1 = cap1.read()

            if not ret1:
                break

            for expert_frame_index, frame2 in enumerate(expert_frames):
                norm_landmarks1 = process_frame(frame1, holistic)
                norm_landmarks2 = process_frame(frame2, holistic)

                if norm_landmarks1 is None or norm_landmarks2 is None:
                    continue

                #similarity = euclidean_similarity(np.array(norm_landmarks1).flatten(),
                #                                  np.array(norm_landmarks2).flatten())
                similarity = cosine_similarity(np.array(norm_landmarks1).flatten(),
                                               np.array(norm_landmarks2).flatten())
                similarities.append(similarity)

                if len(top_5) < 5:
                    top_5.append((similarity, frame1, frame2))
                    top_5.sort(reverse=True, key=lambda x: x[0])
                else:
                    min_similarity, min_index = min((sim, index) for index, (sim, _, _) in enumerate(top_5))
                    if similarity > min_similarity:
                        top_5[min_index] = (similarity, frame1, frame2)
                        top_5.sort(reverse=True, key=lambda x: x[0])

    cap1.release()

    plt.plot(similarities)
    plt.xlabel("Frames")
    plt.ylabel("Similarity")
    plt.show()

    return top_5
if __name__ == "__main__":
    video1 = "pushups-sample_user.mp4"
    video2 = "pushups-sample_exp.mp4"

    top_5_frames = compare_videos(video1, video2)
    connections = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (10, 11), (11, 12), (12, 13),
                   (14, 15), (15, 16), (16, 17), (18, 19)]

    for i, (similarity, frame1, frame2) in enumerate(top_5_frames):
        cv2.imwrite(f"beginner_top_{i + 1}_skeleton.png", frame1)
        cv2.imwrite(f"expert_top_{i + 1}_skeleton.png", frame2)

        print(f"Top {i + 1} similarity: {similarity}")
