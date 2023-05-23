import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import math
import sys, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
from openpose_skeleton import skeleton
import matplotlib.pyplot as plt
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

face = [0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 30, 31, 32]
upper_body = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
lower_body = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
full_body = face + upper_body + lower_body
exercise_landmarks = {
    'Squat': full_body,
    'Pushup': upper_body + lower_body,
    'Situp': upper_body+ lower_body,
    # 다른 운동 종류에 대한 랜드마크 인덱스를 추가하세요.
}

def process_frame(frame,pose):# pose estimation
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
        # use [0][0] to convert the 2D array into a scalar value
        similarity += cosine_similarity(landmarks1[i, :3].reshape(1, -1), landmarks2[i, :3].reshape(1, -1))[0][0]

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
def resize_frame(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized
def process_pair(frame1, frame2,pose,exercise):# process frame
    landmarks1, annotated_frame1 = process_frame(frame1,pose)
    landmarks2, annotated_frame2 = process_frame(frame2,pose)

    if landmarks1.size > 0 and landmarks2.size > 0:
        normalized_landmarks1 = normalize_landmarks(landmarks1)
        normalized_landmarks2 = normalize_landmarks(landmarks2)
        similarity_cosine = pose_similarity_cosine(normalized_landmarks1, normalized_landmarks2,exercise)

        return frame1, frame2, similarity_cosine
    else:
        return None

def save_similarity_table(similarity_list, filepath):
    data = {'Frame Pair Index': list(range(1, len(similarity_list) + 1)),
            'Similarity': similarity_list}
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def plot_similarity(similarity_list, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    flattened_similarity_list = np.array(similarity_list).flatten().tolist()
    ax.plot(flattened_similarity_list, marker='o')
    ax.set_xlabel('Frame Pair Index')
    ax.set_ylabel('Similarity')
    ax.set_title('Pose Similarity Over Time')
    ax.grid()
    fig.savefig(save_path)
    plt.close(fig)

    return fig
def draw_joints(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)

    # 이미지 컬러 공간을 BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mediapipe 포즈 객체 생성
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    # 포즈를 추정하고 결과를 얻음
    results = pose.process(image_rgb)

    # 원본 이미지에 포즈 랜드마크 그리기
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Mediapipe 리소스를 종료
    pose.close()

    return image
def get_pose_similarity(video_path1, video_path2, exercise):# main function

    video1_path = video_path1
    video2_path = video_path2

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get the frame per second (fps) of the video
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    # Ensure the two videos have the same fps
    assert fps1 == fps2, "The two videos should have the same fps for comparison."
    fps = fps1  # or fps2, they are the same

    similarity_list = []
    time_similarity_list = []  # This will store the (time, similarity) pairs
    frame_pairs = []

    with mp_pose.Pose(static_image_mode = False,min_detection_confidence=0.5, model_complexity=1) as pose:
        with ThreadPoolExecutor() as executor:
            frame_index = 0  # Initialize frame index
            while cap1.isOpened() and cap2.isOpened():
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                if not ret1 or not ret2:
                    break

                result = executor.submit(process_pair, frame1, frame2,pose, exercise)
                result = result.result()

                if result:
                    frame1, frame2, similarity_cosine = result
                    similarity_list.append(similarity_cosine)
                    frame_pairs.append((frame1, frame2))

                    # Calculate time for each frame
                    time = frame_index / fps
                    time_similarity_list.append((time, similarity_cosine))

                    frame_index += 1  # Increase frame index

    cap1.release()
    cap2.release()
    similarities = []
    if len(similarity_list) > 0:
        similarity_array = np.array(similarity_list)
        num_top_pairs = min(5, len(similarity_array))

        top_indices = sorted(range(len(similarity_array)), key=lambda i: similarity_array[i], reverse=True)[:num_top_pairs]

        for i, index in enumerate(top_indices):
            frame1, frame2 = frame_pairs[index]
            cv2.imwrite(f"video1_frame_{i + 1}_cosine.jpg", frame1)
            cv2.imwrite(f"video2_frame_{i + 1}_cosine.jpg", frame2)

            # 관절 그리기
            joint_image1, joint_image2, advice = skeleton.main(frame1, frame2)
#             joint_image1 = draw_joints(f"video1_frame_{i + 1}_cosine.jpg")
#             joint_image2 = draw_joints(f"video2_frame_{i + 1}_cosine.jpg")

            # Save images with joints
            cv2.imwrite(f"video1_frame_{i + 1}_cosine_with_joints.jpg", joint_image1)
            cv2.imwrite(f"video2_frame_{i + 1}_cosine_with_joints.jpg", joint_image2)
            similarities.append(similarity_list[index])
    else:
        print("No poses found in one or both videos.")

    fig = plot_similarity(similarity_list, 'pose_similarity_plot.png')
    return similarities, frame_pairs, top_indices, fig, time_similarity_list


def main():
    similarities, frame_pairs, top_indices, fig, time_similarity_list = get_pose_similarity('output1.mp4',
                                                                                            'output2.mp4',
                                                                                            'Squat')
    #print(similarities)
    #print(top_indices)
    # Filter out pairs where similarity is less than 70
    filtered_time_similarity_list = [(time, similarity) for time, similarity in time_similarity_list if
                                     similarity * 100 >= 70]

    # Separate time and similarity into two lists
    times = [pair[0] for pair in filtered_time_similarity_list]
    similarities = [pair[1] for pair in filtered_time_similarity_list]
    for time, similarity in time_similarity_list:
        print(f"{time} : {round(similarity, 2)}")
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    window_size = 10  # Adjust this to change the amount of smoothing
    smoothed_similarities = moving_average(similarities, window_size)

    # We need to also shorten the times array so it matches the length of smoothed_similarities
    shortened_times = times[window_size - 1:]

    plt.figure(figsize=(10, 6))
    plt.plot(shortened_times, smoothed_similarities, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('Similarity')
    plt.title('Smoothed Pose Similarity Over Time (Similarity >= 70%)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
