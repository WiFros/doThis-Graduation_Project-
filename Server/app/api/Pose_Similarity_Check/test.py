import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor

# Mediapipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

upper_body = [11, 12, 13, 14, 15, 16, 17]
lower_body = [25, 26, 27, 28,29,30]

def normalize_landmarks(landmarks, frame_shape):
    height, width, _ = frame_shape
    normalized_landmarks = []

    for lm in landmarks:
        normalized_landmarks.append((lm.x * width, lm.y * height))

    return normalized_landmarks

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def process_frame(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = []
    if results.pose_landmarks is None:
        return None

    for lm_index in upper_body + lower_body:
        landmarks.append(results.pose_landmarks.landmark[lm_index])

    return normalize_landmarks(landmarks, frame.shape)

def compare_videos(video1, video2):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)

    similarities = []
    top_5 = []

    with ThreadPoolExecutor() as executor:
        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap1.isOpened() and cap2.isOpened():
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                if not ret1 or not ret2:
                    break

                norm_landmarks1 = executor.submit(process_frame, frame1, pose)
                norm_landmarks2 = executor.submit(process_frame, frame2, pose)

                norm_landmarks1 = norm_landmarks1.result()
                norm_landmarks2 = norm_landmarks2.result()

                if norm_landmarks1 is None or norm_landmarks2 is None:
                    continue

                similarity = cosine_similarity(np.array(norm_landmarks1).flatten(),
                                               np.array(norm_landmarks2).flatten())

                similarities.append(similarity)
                top_5.append((similarity, frame1, frame2))

                if len(top_5) > 5:
                    top_5.sort(reverse=True, key=lambda x: x[0])
                    top_5.pop()

    cap1.release()
    cap2.release()

    plt.plot(similarities)
    plt.xlabel("Frames")
    plt.ylabel("Similarity")
    plt.show()

    return top_5
def draw_landmarks_on_frame(frame, landmarks, connections):
    frame = frame.copy()
    for lm in landmarks:
        x, y = int(lm[0]), int(lm[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), thickness=-1)

    for connection in connections:
        start, end = connection
        x_start, y_start = int(landmarks[start][0]), int(landmarks[start][1])
        x_end, y_end = int(landmarks[end][0]), int(landmarks[end][1])

        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), thickness=2)

    return frame
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
