import cv2
import numpy as np
from scipy.spatial.distance import cosine


# 관절 위치 추출 함수
def extract_joints(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)

    # 관절 추출 모델 로드
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

    # 추출된 관절 저장할 리스트
    joints = []

    # 추출된 관절 위치 찾기
    input_blob = cv2.dnn.blobFromImage(image, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(input_blob)
    output = net.forward()
    points = []
    for i in range(22):
        prob_map = output[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        x = int((image.shape[1] * point[0]) / output.shape[3])
        y = int((image.shape[0] * point[1]) / output.shape[2])
        if prob > 0.01:
            points.append((x, y))
        else:
            points.append(None)
    joints.append(points)

    # 관절 위치 반환
    return joints


# 이미지 경로
image_path1 = "./image1.jpg"
image_path2 = "./image2.jpg"
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 두 이미지에서 관절 위치 추출
joints1 = extract_joints(image_path1)
joints2 = extract_joints(image_path2)

# 관절 간 거리 계산하여 스켈레톤 만들기
skeleton1 = np.zeros((image1.shape[0], image1.shape[1], 3), np.uint8)
skeleton2 = np.zeros((image2.shape[0], image2.shape[1], 3), np.uint8)

for i in range(21):
    if joints1[0][i] and joints1[0][i+1]:
        cv2.line(skeleton1, joints1[0][i], joints1[0][i+1], (255, 255, 255), 2)
    if joints2[0][i] and joints2[0][i+1]:
        cv2.line(skeleton2, joints2[0][i], joints2[0][i+1], (255, 255, 255), 2)


# 코사인 유사도 계산
cosine_similarity = cosine(joints1, joints2)

# 스켈레톤 색상 지정
if cosine_similarity >= 0.9:
    color = (0, 255, 0)  # 녹색
elif cosine_similarity >= 0.7:
    color = (0, 0, 255)  # 빨간색
else:
    color = (255, 0, 0)  # 파란색

# 색상 지정한 스켈레톤 출력
skeleton = cv2.addWeighted(skeleton1, 0.5, skeleton2, 0.5, 0)
skeleton[skeleton == [0, 0, 0]] = color
cv2.imshow("Skeleton", skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()






