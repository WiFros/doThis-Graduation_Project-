import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

# 이미지 파일 경로
input_image_path = "video1_frame_1_cosine.jpg"
output_image_path = "output.jpg"

# 관절을 그린 이미지 생성
joint_image = draw_joints(input_image_path)

# 결과 이미지 저장
cv2.imwrite(output_image_path, joint_image)