import cv2
import numpy as np
import os
# def get_angle(p1 : list, p2 : list ,p3 : list, angle_vec : bool) -> float:
#     """ 
#         세점 사이의 끼인 각도 구하기
#     """
#     rad = np.arctan2(p3[1] - p1[1], p3[0] - p1[0]) - np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
#     deg = rad * (180 / np.pi)
#     if angle_vec:
#         deg = 360-abs(deg)
#     return abs(deg)

def draw(frame, index): #1~5
    #-- 파츠명 선언
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                    "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

    #-- 모델 파일 불러오기
    protoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "pose_deploy_linevec_faster_4_stages.prototxt")
    weightsFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pose_iter_160000.caffemodel")
    print('protoFile:',protoFile)
    print('weightsFile:',protoFile)
    #-- network 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    #-- 이미지 불러오기
    image = frame

    imageHeight, imageWidth, _ = image.shape #-- 불러온 이미지에서 height, width, color를 가져옴
    
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False) #-- network에 적용하기 위한 전처리

    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    #-- 검출된 키포인트 적용 
    points = []
    for i in range(0,15):
        probMap = output[0, i, :, :]
    
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        #-- 원본 이미지에 맞도록 포인트 위치 적용
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

        #-- 검출된 포인트가 BODY_PARRTS와 대응되면 포인트 추가(검출 결과가 0.1보다 크면) / 검출했으나 해당 파츠가 없는 경우 None    
        if prob > 0.1 :    
            # 점찍고 숫자 써줌 - 아래 2줄
            if i==1 or i==14 or i==8 or i==9 or i==10:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED) #-- circle(이미지, 원의 중심, 반지름, 컬러)
            # cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)

    #-- 생성된 POSE_PAIRS에 맞게 라인 생성
    for pair in POSE_PAIRS:
        partA = pair[0]             #-- 머리
        partA = BODY_PARTS[partA]   # 0
        partB = pair[1]             #-- 목
        partB = BODY_PARTS[partB]   # 1
        
        if((partA==1 and partB==14) or (partA==14 and partB==8) or (partA==8 and partB==9) or (partA==9 and partB==10)):
            print(partA," 와 ", partB, " 연결\n")
            if points[partA] and points[partB]:
                cv2.line(image, points[partA], points[partB], (0, 255, 0), 2)

    return image