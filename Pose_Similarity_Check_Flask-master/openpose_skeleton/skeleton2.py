import cv2
import pyopenpose as op
import numpy as np


# OpenPose 설정
params = dict()
params["model_folder"] = "openpose/models/"  # 모델 폴더 경로
params["model_pose"] = "COCO"  # 모델 타입
params["net_resolution"] = "-1x368"  # 이미지 해상도
params["disable_blending"] = True  # 이미지 합성 안 함
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 이미지 로드
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# OpenPose를 사용하여 관절 인식
datum1 = op.Datum()
datum1.cvInputData = img1
opWrapper.emplaceAndPop([datum1])
joints1 = datum1.poseKeypoints
connections = op.getPoseBodyPartMapping(op.PoseModel.COCO)

datum2 = op.Datum()
datum2.cvInputData = img2
opWrapper.emplaceAndPop([datum2])
joints2 = datum2.poseKeypoints

# 관절 좌표와 연결 정보를 저장
joint_coords = []
joint_connections = []
for joint_id in range(len(connections)):
    joint_pair = connections[joint_id]
    joint_coords.append((joints1[0][joint_pair[0]], joints1[0][joint_pair[1]], joints2[0][joint_pair[1]], joints2[0][joint_pair[0]]))
    joint_connections.append(joint_pair)

