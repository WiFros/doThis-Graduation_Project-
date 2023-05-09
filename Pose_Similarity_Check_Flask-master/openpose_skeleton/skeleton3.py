import cv2
import numpy as np

# 모델 경로
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"

# 네트워크 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 이미지 경로
image_path1 = "image1.jpg"
image_path2 = "image2.jpg"

# 이미지 로드
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 이미지 크기 조정
image1 = cv2.resize(image1, (368, 368))
image2 = cv2.resize(image2, (368, 368))


# 이미지 전처리
blob1 = cv2.dnn.blobFromImage(image1, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
blob2 = cv2.dnn.blobFromImage(image2, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)

# 네트워크 입력 설정
net.setInput(blob1)
output1 = net.forward()
net.setInput(blob2)
output2 = net.forward()

# 각 부위의 중심 좌표 계산
points1 = []
points2 = []
for i in range(15):
    probMap1 = output1[0, i, :, :]
    probMap2 = output2[0, i, :, :]

    # 확률 맵 라벨링
    _, probMap1 = cv2.threshold(probMap1, 0.1, 255, cv2.THRESH_BINARY)
    _, probMap2 = cv2.threshold(probMap2, 0.1, 255, cv2.THRESH_BINARY)

    # CV_8UC1 형식으로 변환
    probMap1 = cv2.convertScaleAbs(probMap1)
    probMap2 = cv2.convertScaleAbs(probMap2)

    contour1, _ = cv2.findContours(probMap1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour2, _ = cv2.findContours(probMap2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 contour 찾기
    if len(contour1) > 0 and len(contour2) > 0:
        max_contour1 = max(contour1, key=cv2.contourArea)
        max_contour2 = max(contour2, key=cv2.contourArea)

        # 바운딩 박스 계산
        rect1 = cv2.boundingRect(max_contour1)
        rect2 = cv2.boundingRect(max_contour2)

        # 바운딩 박스 중심 좌표 계산
        center1 = (rect1[0] + rect1[2] / 2, rect1[1] + rect1[3] / 2)
        center2 = (rect2[0] + rect2[2] / 2, rect2[1] + rect2[3] / 2)

        # 좌표 리스트에 추가
        points1.append(center1)
        points2.append(center2)


# 각 부위의 유사도 계산
similarity = []
for i in range(len(points1)):
    x1, y1 = points1[i]
    x2, y2 = points2[i]
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    sim = max(0, 1 - dist / 50)
    print("73", sim)
    similarity.append(sim)

# 이미지 색칠하기
for i, sim in enumerate(similarity):
    x1, y1 = points1[i]
    x2, y2 = points2[i]
    if sim >= 0.5:
        color = (int(sim * 255), 0, int((1 - sim) * 255))
    else:
        color = (0, int(sim * 255), int((1 - sim) * 255))
    cv2.line(image1, (int(x1), int(y1)), (int(x2), int(y2)),  color, 3)
    print("85", (int(x1), int(y1)), (int(x2), int(y2)),  (255, 0, 0), 8)

# 결과 이미지 보여주기
cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.waitKey(0)

