import json
import cv2
import math
import numpy as np

from modules import pose_filtering
from modules import draw_tool as draw
from modules import similarity_check as sc

with open('pose.json') as json_file:
    input_json = json.load(json_file)

with open('video.json') as json_file:
    json_data = json.load(json_file)

# 해당 부분 수정을 통해 원하는 사진을 비교 가능, 현재 가능 index 0 ~ 3
input_num_1 = 3

# 포즈 선정후, 
pose1 = input_json[input_num_1]
cv_img1 = cv2.imread('./images/{}.jpeg'.format(input_num_1+1))
print(cv_img1.shape)

# 포즈에 맞게 원을 그린다.
annotated_image1 = draw.draw_circle_landmark(cv_img1, input_json[input_num_1], 30)

# 영상과 사진을 이어붙일 수 있게 resize
h, w, _ = annotated_image1.shape

DESIRED_HEIGHT = 720
DESIRED_WIDTH = 1280

if h < w:
    annotated_image1 = cv2.resize(annotated_image1, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
else:
    annotated_image1 = cv2.resize(annotated_image1, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

capture = cv2.VideoCapture("videos/Workout.mp4")

while cv2.waitKey(33) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_fps = int(capture.get(1))
    total_fps = capture.get(0)
    ret, frame = capture.read()
    if not ret:
        break
    if current_fps != 136:
        Similarity = sc.get_pose_similarity(json_data[current_fps]['Pose'], frame.shape, pose1, cv_img1.shape)
        annotated_image = draw.draw_circle_landmark(frame, pose_filtering.landmark_filtering(json_data[current_fps]['Pose']), 5)
    addh = cv2.hconcat([annotated_image, annotated_image1])
    text = 'Percentage : [ x : {}% ], [ y : {}% ]'.format(Similarity[0], Similarity[1])
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(addh, text, (0, 100), font, 1, (0,0,155), 1, cv2.LINE_AA)
    cv2.imshow("VideoFrame", addh)

capture.release()
cv2.destroyAllWindows()