import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine
from multiprocessing import Pool

def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_keypoints(frame, mp_pose):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(image)
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
    else:
        keypoints = [0] * 75
    return keypoints

def normalize(keypoints):
    keypoints = np.array(keypoints)
    x_coords = keypoints[0::3]
    y_coords = keypoints[1::3]
    z_coords = keypoints[2::3]
    x_coords = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
    y_coords = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))
    z_coords = (z_coords - np.min(z_coords)) / (np.max(z_coords) - np.min(z_coords))
    return np.concatenate([x_coords, y_coords, z_coords])

def cosine_similarity(frame1, frame2):
    frame1 = normalize(frame1)
    frame2 = normalize(frame2)
    similarity = 1 - cosine(frame1, frame2)
    return similarity

def compare_frames(args):
    frame1, frame2 = args
    similarity = cosine_similarity(frame1, frame2)
    return similarity

def save_frames(frames, prefix):
    for i, frame in enumerate(frames):
        file_path = f"{prefix}_{i}.jpg"
        cv2.imwrite(file_path, frame)

def save_top_frames(frames, expert_frames, mp_pose, i):
    # 멀티 프로세싱을 사용하여 두 프레임 간의 유사도를 계산합니다.
    def compare_frames_helper(frame1, frame2):
        return compare_frames((frame1, frame2))
    args = [(frames[i], expert_frames[j]) for j in range(len(expert_frames))]
    pool = Pool()
    results = pool.starmap(compare_frames_helper, args)
    # 결과에서 중복을 제거합니다.
    results = list(set(results))
    # 가장 유사도가 높은 프레임 5개를 추출합니다.
    top_indices = np.argsort(results)[::-1][:5]
    top_frames = [expert_frames[i] for i in top_indices]
    # 결과를 저장합니다.
    save_frames(top_frames, f"top_frames_{i}")
    np.save(f"similarities_{i}.npy", results[top_indices])
    # 프레임 쌍을 이미지로 저장합니다.
    for j, (frame, expert_frame) in enumerate(zip(frames[i:i+1], top_frames)):
        cv2.imwrite(f"comparison_{i}_{j}_1.jpg", frame)
        cv2.imwrite(f"comparison_{i}_{j}_2.jpg", expert_frame)

def process(video_path, expert_video_path):
    frames = read_frames(video_path)
    expert_frames = read_frames(expert_video_path)
    mp_pose = mp.solutions.pose.Pose()
    pool = Pool()
    for i in range(len(frames)):
        args = (frames, expert_frames, mp_pose, i)
        pool.apply_async(save_top_frames, args=args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    video_path = 'pushups-sample_user.mp4'
    expert_video_path = 'pushups-sample_exp.mp4'
    process(video_path, expert_video_path)
