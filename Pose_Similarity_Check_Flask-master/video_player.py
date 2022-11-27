import cv2
import math

capture = cv2.VideoCapture("videos/Workout.mp4")

DESIRED_HEIGHT = 720
DESIRED_WIDTH = 1280




# 해당 부분 수정을 통해 원하는 사진을 비교 가능, 현재 가능 index 0 ~ 3
input_num_1 = 3
input_num_2 = 4

cv_img1 = cv2.imread('./images/{}.jpeg'.format(input_num_1+1))
cv_img2 = cv2.imread('./images/{}.jpeg'.format(input_num_2+1))




h, w, _ = cv_img1.shape
# cv_img1 = cv2.resize(cv_img1, (1280, 720))
if h < w:
    cv_img1 = cv2.resize(cv_img1, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
else:
    cv_img1 = cv2.resize(cv_img1, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))


while cv2.waitKey(33) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    if not ret:
        break
    print(frame.shape)
    print(cv_img1.shape)
    addh = cv2.hconcat([frame, cv_img1])
    # print(frame)
    cv2.imshow("VideoFrame", addh)

capture.release()
cv2.destroyAllWindows()