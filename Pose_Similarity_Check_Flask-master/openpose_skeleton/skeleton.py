import cv2
import os
import math

def draw(frame, index):  # 1~5
    #-- 파츠명 선언
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                  "Background": 15}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                  ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                  ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

    #-- 모델 파일 불러오기
    protoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "pose_deploy_linevec_faster_4_stages.prototxt")
    weightsFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pose_iter_160000.caffemodel")
    #-- network 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    #-- 이미지 불러오기
    image = frame

    # -- 불러온 이미지에서 height, width, color를 가져옴
    imageHeight, imageWidth, _ = image.shape

    # -- network에 적용하기 위한 전처리
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    #-- 검출된 키포인트 적용
    points = []
    selectedPoints = []
    for i in range(0, 15):
        probMap = output[0, i, :, :]

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        #-- 원본 이미지에 맞도록 포인트 위치 적용
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

        #-- 검출된 포인트가 BODY_PARRTS와 대응되면 포인트 추가(검출 결과가 0.1보다 크면) / 검출했으나 해당 파츠가 없는 경우 None
        if prob > 0.1:
            # 점찍고 숫자 써줌 - 아래 2줄
            if i == 1 or i == 14 or i == 8 or i == 9 or i == 10:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)  #-- circle(이미지, 원의 중심, 반지름, 컬러)
            # cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA) # 각 랜드마크 표시
            points.append((int(x), int(y)))
        else:
            points.append(None)

    #-- 생성된 POSE_PAIRS에 맞게 라인 생성
    for pair in POSE_PAIRS:
        partA = pair[0]  #-- 머리
        partA = BODY_PARTS[partA]  # 0
        partB = pair[1]  #-- 목
        partB = BODY_PARTS[partB]  # 1

        if (partA == 1 and partB == 14) or (partA == 14 and partB == 8) or (partA == 8 and partB == 9) or (partA == 9 and partB == 10):
            print(partA, " 와 ", partB, " 연결\n")
            if points[partA] and points[partB]:
                selectedPoints.append((points[partA], points[partB]))

    return image, selectedPoints


# 점 쌍 간의 코사인 유사도를 계산하는 함수
def calculate_cosine_similarity(line1, line2):
    # Calculate the cosine similarity between two lines
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the vectors for each line
    vector1 = (x2 - x1, y2 - y1)
    vector2 = (x4 - x3, y4 - y3)

    # Calculate the dot product of the vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate the magnitudes of the vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return cosine_similarity

def main(frame1, frame2):
    index_arr = []
    advice_str = ""

    image1 = frame1
    image2 = frame2

    image11, selectedPoints1 = draw(image1, 1)
    image22, selectedPoints2 = draw(image2, 2)

    # 코사인 유사도에 따라 색상을 다르게 설정하여 스켈레톤 그리기
    for index, (line1, line2) in enumerate(zip(selectedPoints1, selectedPoints2)):
        cosine_similarity = calculate_cosine_similarity(line1, line2)
        if cosine_similarity > 0.95:
            color = (255, 0, 0)  # 파란색
        elif cosine_similarity > 0.9:
            color = (0, 255, 0)  # 초록색
        elif cosine_similarity > 0.8:
            color = (0, 165, 255)  # 주황색
            index_arr.append(index)
        else:
            color = (0, 0, 255)  # 빨간색
            index_arr.append(index)
        cv2.line(image11, line1[0], line1[1], color, 2)
        cv2.line(image22, line2[0], line2[1], color, 2)

    if 0 in index_arr:
        advice_str += "고개를 들어주세요\n"
    if 1 in index_arr:
        advice_str += "허리와 엉덩이 각을 주의해주세요\n"
    if 2 in index_arr:
        advice_str += "앉는 정도를 신경써주세요\n"
    if 3 in index_arr:
        advice_str += "무릎 각도를 주의해주세요\n"

    return image11, image22, advice_str
