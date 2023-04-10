import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import sys, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
from openpose_skeleton import skeleton
import matplotlib.pyplot as plt

face = [0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32]
upper_body = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
lower_body = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
full_body = face + upper_body + lower_body
exercise_landmarks = {
    'squat': full_body,
    'pushup': upper_body + lower_body,
    'situp': upper_body+ lower_body,
    # 다른 운동 종류에 대한 랜드마크 인덱스를 추가하세요.
}

def plot_similarity(similarity_list):
    plt.figure(figsize=(10, 6))
    plt.plot(similarity_list, marker='o')
    plt.xlabel('Frame Pair Index')
    plt.ylabel('Similarity')
    plt.title('Pose Similarity Over Time')
    plt.grid()
    plt.show()
def process_frame(frame):# pose estimation
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



def normalize_landmarks(landmarks):# normalize landmarks
    min_x, min_y, min_z = np.min(landmarks[:, :3], axis=0)
    max_x, max_y, max_z = np.max(landmarks[:, :3], axis=0)

    scale = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    normalized_landmarks = (landmarks[:, :3] - min_x) / scale

    return normalized_landmarks

def pose_similarity_cosine(landmarks1, landmarks2, exercise):# cosine similarity
    if landmarks1.shape != landmarks2.shape:
        raise ValueError("Landmarks must have the same shape")

    selected_landmarks = exercise_landmarks[exercise]
    similarity = 0

    for i in selected_landmarks:
        similarity += cosine_similarity(landmarks1[i, :3].reshape(1, -1), landmarks2[i, :3].reshape(1, -1))

    return similarity / len(selected_landmarks)

def get_frame_similarity_list(video_path1, video_path2, exercise):
    video1_path = video_path1
    video2_path = video_path2

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    similarity_list = []

    with ThreadPoolExecutor() as executor:
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            result = executor.submit(process_pair, frame1, frame2,exercise)
            result = result.result()

            if result:
                frame1, frame2, similarity_cosine = result
                similarity_list.append(similarity_cosine)

    cap1.release()
    cap2.release()

    return similarity_list

def process_pair(frame1, frame2,exercise):# process frame
    landmarks1, annotated_frame1 = process_frame(frame1)
    landmarks2, annotated_frame2 = process_frame(frame2)

    if landmarks1.size > 0 and landmarks2.size > 0:
        normalized_landmarks1 = normalize_landmarks(landmarks1)
        normalized_landmarks2 = normalize_landmarks(landmarks2)
        similarity_cosine = pose_similarity_cosine(normalized_landmarks1, normalized_landmarks2,exercise)

        return frame1, frame2, similarity_cosine
    else:
        return None


def get_pose_similarity(video_path1, video_path2, exercise):# main function

    sk_frame1 = []
    sk_frame2 = []
    video1_path = video_path1
    video2_path = video_path2

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

            result = executor.submit(process_pair, frame1, frame2, exercise)
            result = result.result()

            if result:
                frame1, frame2, similarity_cosine = result
                similarity_list.append(similarity_cosine)
                frame_pairs.append((frame1, frame2))
    #plot_similarity(similarity_list)
    cap1.release()
    cap2.release()
    similarities = []
    if len(similarity_list) > 0:
        similarity_array = np.array(similarity_list)
        num_top_pairs = min(5, len(similarity_array))

        top_indices = sorted(range(len(similarity_array)), key=lambda i: similarity_array[i], reverse=True)[:num_top_pairs]

        for i, index in enumerate(top_indices):
            frame1, frame2 = frame_pairs[index]
            #sk_frame1.append(skeleton.draw(frame1, index))
            #sk_frame2.append(skeleton.draw(frame2, index))
            #cv2.imwrite(f"./test1_{i+1}.png", sk_frame1[i]) #test
            #cv2.imwrite(f"./test2_{i+1}.png", sk_frame2[i]) #test
            cv2.imwrite(f"video1_frame_{i + 1}_cosine.jpg", frame1)
            cv2.imwrite(f"video2_frame_{i + 1}_cosine.jpg", frame2)
            similarities.append(similarity_list[index])
    else:
        print("No poses found in one or both videos.")
    return similarities

print(get_pose_similarity('pushups-sample_exp.mp4', 'pushups-sample_user.mp4', 'squat'))