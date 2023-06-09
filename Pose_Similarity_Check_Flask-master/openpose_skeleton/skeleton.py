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
            # if i == 1 or i == 14 or i == 8 or i == 9 or i == 10:
            #     cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)  #-- circle(이미지, 원의 중심, 반지름, 컬러)
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


def calculate_angle(point_A, point_B, point_C):
    # 공통된 좌표 B를 기준으로 선분 AB와 선분 BC의 벡터를 계산합니다
    vector_AB = (point_A[0] - point_B[0], point_A[1] - point_B[1])
    vector_BC = (point_C[0] - point_B[0], point_C[1] - point_B[1])

    # 선분 AB와 선분 BC의 벡터 내적을 계산합니다
    dot_product = vector_AB[0] * vector_BC[0] + vector_AB[1] * vector_BC[1]

    # 선분 AB와 선분 BC의 길이를 계산합니다
    length_AB = math.sqrt(vector_AB[0] ** 2 + vector_AB[1] ** 2)
    length_BC = math.sqrt(vector_BC[0] ** 2 + vector_BC[1] ** 2)

    # 두 벡터의 길이가 0이면 각도를 구할 수 없으므로 None을 반환합니다
    if length_AB == 0 or length_BC == 0:
        return None

    # 두 벡터의 내적을 이용하여 라디안으로 각도를 계산합니다
    cos_theta = dot_product / (length_AB * length_BC)
    angle_rad = math.acos(cos_theta)

    # 라디안 각도를 도로 변환하여 반환합니다
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def main(frame1, frame2):
    index_arr = []
    advice_str = ""

    image1 = frame1
    image2 = frame2

    image11, selectedPoints1 = draw(image1, 1)
    image22, selectedPoints2 = draw(image2, 2)

    # 코사인 유사도에 따라 색상을 다르게 설정하여 스켈레톤 그리기 -> 안하고 그냥 line 그리기 용도
    for index, (line1, line2) in enumerate(zip(selectedPoints1, selectedPoints2)):
        cv2.line(image11, line1[0], line1[1], (255,255,255), 2)
        cv2.line(image22, line2[0], line2[1], (255,255,255), 2)

    ######################################## 조인트 부분
    selectedPoints = [selectedPoints1, selectedPoints2]  # 선택된 점들의 리스트
    angles = []  # 각도를 저장할 이중배열

    for i, points in enumerate(selectedPoints):
        angle_list = []  # 한 점과 그 다음 점 사이의 각도를 저장할 리스트
        for j in range(len(points) - 1):
            # print("point{}{}".format(i, j), (points[j][0], points[j][1], points[j+1][1]))
            angle = calculate_angle(points[j][0], points[j][1], points[j + 1][1])
            # print("angle{}{}".format(i, j), angle)
            angle_list.append(angle)  # 각도를 리스트에 추가
        angles.append(angle_list)  # 점들의 각도 리스트를 이중배열에 추가

    angle_diff = []
    for j in range(len(angles[1])):
        angle_diff.append(abs(angles[0][j] - angles[1][j]))

    # 각 selectedpoint 지점에 angle_diff 값에 따라 색상 다르게 칠하기
    for i in range(len(angle_diff)):
        if angle_diff[i] < 5:
            radius = 2
            color = (255, 0, 0)
            cv2.putText(image22, "Good", (selectedPoints2[i][1][0] + 15, selectedPoints2[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        elif angle_diff[i] < 10:
            radius = 4
            color = (0, 255, 0)
            cv2.putText(image22, "Good", (selectedPoints2[i][1][0] + 15, selectedPoints2[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        elif angle_diff[i] < 30:
            radius = 6
            color = (0, 255, 255)
            cv2.putText(image22, "Not Bad", (selectedPoints2[i][1][0] + 15, selectedPoints2[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            index_arr.append(i)
        elif angle_diff[i] < 50:
            radius = 8
            color = (0, 165, 255)
            cv2.putText(image22, "Bad", (selectedPoints2[i][1][0] + 15, selectedPoints2[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            index_arr.append(i)
        elif angle_diff[i] < 180:
            radius = 10
            color = (0, 0, 255)
            cv2.putText(image22, "Bad", (selectedPoints2[i][1][0] + 15, selectedPoints2[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            index_arr.append(i)
        cv2.circle(image22, selectedPoints2[i][1], radius, color, thickness=-1, lineType=cv2.FILLED)  # -- circle(이미지, 원의 중심, 반지름, 컬러)

    if 0 in index_arr:
        advice_str += "상체를 펴주세요\n"
    else:
        advice_str += "상체 자세 좋네요!\n"
    if 1 in index_arr:
        advice_str += "앉는 정도를 신경써주세요\n"
    else:
        advice_str += "하체 자세 좋네요!\n"
    if 2 in index_arr:
        advice_str += "무릎 각도를 신경써주세요\n"
    else:
        advice_str += "무릎 접히는 정도 좋습니다!\n"


    return image11, image22, advice_str


image1 = cv2.imread("./image1.jpg")
image2 = cv2.imread("./image3.jpg")

i1, i2, advice = main(image1, image2)
print(advice)
cv2.imshow("Image 1", i1)
cv2.imshow("Image 2", i2)
cv2.waitKey(0)