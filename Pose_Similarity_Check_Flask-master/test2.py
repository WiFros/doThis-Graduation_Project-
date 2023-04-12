import cv2
import mediapipe as mp
import numpy as np
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 미리 학습된 머신러닝 모델을 불러옵니다.
model = joblib.load("pose_classifier.joblib")

def extract_features(landmarks):
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features).reshape(1, -1)

    def classify_pose(landmarks):
        features = extract_features(landmarks)
        prediction = model.predict(features)
        return prediction[0]

    def main(video_path):
        cap = cv2.VideoCapture(video_path)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    position = classify_pose(results.pose_landmarks.landmark)
                    print(position)
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )

                cv2.imshow("Squat Analysis", image)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        video_path = "pushups-sample_exp.mp4"  # 여기에 동영상 파일 경로를 입력하세요.
        main(video_path)
