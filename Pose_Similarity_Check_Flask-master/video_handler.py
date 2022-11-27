import cv2
import math
import json
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_path = "Workout.mov"

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


print(fps)

video = []

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
      break

    frame = {}
    # print(cap.get(cv2.CAP_PROP_POS_MSEC))
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    frame['timestamp'] = timestamp

    # print(timestamp)
    h, w = height, width
    if h < w:
      img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
      img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
      results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      
      image_hight, image_width, _ = img.shape
      Pose = []
      for i in mp_pose.PoseLandmark:
        position = {}
        position['x'] = results.pose_landmarks.landmark[i].x
        position['y'] = results.pose_landmarks.landmark[i].y
        keypoint = {}
        keypoint['position'] = position
        keypoint['part'] = str(mp_pose.PoseLandmark(i))
        Pose.append(keypoint)
      # Pose_Land.append(str(mp_pose.PoseLandmark(i)))
      frame['Pose'] = Pose

      annotated_image = img.copy()
      for i in mp_pose.PoseLandmark:
        get_x = int(results.pose_landmarks.landmark[i].x * image_width)
        get_y = int(results.pose_landmarks.landmark[i].y * image_hight)
        annotated_image = cv2.circle(annotated_image, (get_x, get_y), 1, (0,0,255), -1)
      # cv2_imshow(annotated_image)
    video.append(frame)
    # cv2_imshow(img) # Note cv2_imshow, not cv2.imshow
    # cv2.waitKey(1) & 0xff

print(video)

# with mp_pose.Pose(
#     static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
#   for name, image in images.items():
#     # Convert the BGR image to RGB and process it with MediaPipe Pose.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
#     # Print nose landmark.
#     image_hight, image_width, _ = image.shape
#     if not results.pose_landmarks:
#       continue

#     Pose = []
#     Pose_Land = []
#     # print(mp_pose.PoseLandmark)
#     for i in mp_pose.PoseLandmark:
#       print(i)
#       # print(
#       #   f'{i}: ('
#       #   f'{results.pose_landmarks.landmark[i].x * image_width}, '
#       #   f'{results.pose_landmarks.landmark[i].y * image_hight}',
#       #   f'{results.pose_landmarks.landmark[i].z} )'
#       # )
#       position = {}
#       position['x'] = results.pose_landmarks.landmark[i].x
#       position['y'] = results.pose_landmarks.landmark[i].y
#       keypoint = {}
#       keypoint['position'] = position
#       keypoint['part'] = str(mp_pose.PoseLandmark(i))
#       Pose_Land.append(str(mp_pose.PoseLandmark(i)))
#       # position = Position(results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y)
#       # keypoint = Keypoint(position, i)
#       # print(keypoint)
#       Pose.append(keypoint)
#       # print(results.pose_landmarks.landmark[i].x * image_width)
#     # print(mp_pose)

#     # for part in mp_pose.PoseLandmark:
#     #   print(part)
#     #   print(results.pose_landmarks.landmark[part])
#     print(Pose_Land)
#     # Draw pose landmarks.
#     results_man.append(Pose)

#     annotated_image = image.copy()
#     for i in mp_pose.PoseLandmark:
#       get_x = int(results.pose_landmarks.landmark[i].x * image_width)
#       get_y = int(results.pose_landmarks.landmark[i].y * image_hight)
#       annotated_image = cv2.circle(annotated_image, (get_x, get_y), 30, (0,0,255), -1)
#     resize_and_show(annotated_image)


cv2.destroyAllWindows()
cap.release()
