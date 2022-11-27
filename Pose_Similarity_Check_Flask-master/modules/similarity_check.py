import math
import numpy as np
from . import pose_filtering

# 벡터화 및 scale 상수 전달
def vectorize(pose, img_shape):
    pose_vector = []
    image_hight, image_width, _ = img_shape

    translateX = math.inf
    translateY = math.inf
    scaler = -math.inf

    for i in pose:
        temp = [i['position']['x'] * image_width, i['position']['y'] * image_hight]
        pose_vector.append(temp)

        translateX = min(translateX, temp[0])
        translateY = min(translateY, temp[1])
        scaler = max(scaler, max(temp[0], temp[1]))
    return scale_translate(pose_vector, [translateX / scaler, translateY / scaler, scaler])
    # return pose_vector, [translateX / scaler, translateY / scaler, scaler]

# Scale하여 ROI 사이즈 조정
def scale_translate(pose_vector, vector_transformValues):
    transX, transY, scaler = vector_transformValues
    new_vector = pose_vector
    for i,part in enumerate(pose_vector):
        new_vector[i][0] = pose_vector[i][0] / scaler - transX
        new_vector[i][1] = pose_vector[i][1] / scaler - transY
    return new_vector

# L2 Normalize
def L2_normalize(pose_vector):
    absVectorPoseXY = [0,0]
    for position in pose_vector:
        absVectorPoseXY += np.power(position, 2)
    absVectorPoseXY = np.sqrt(absVectorPoseXY)
    pose_vector /= absVectorPoseXY
    return pose_vector

# 두 벡터간 코사인 유사도 측정
def get_cosine_similarity(pose1_vector, pose2_vector):
    v1DotV2 = 0
    absV1 = 0
    absV2 = 0
    for i, v1 in enumerate(pose1_vector):
        v2 = pose2_vector[i]
        v1DotV2 += v1 * v2
        absV1 += v1 * v1
        absV2 += v2 * v2
    absV1 = np.sqrt(absV1)
    absV2 = np.sqrt(absV2)
    return v1DotV2 / (absV1 * absV2)

# 두 포즈 데이터를 비교
def get_pose_similarity(pose1, pose1_shape, pose2, pose2_shape):

    # 순회하면서 포즈에서 불필요한 정보를 삭제한다
    pose1 = pose_filtering.landmark_filtering(pose1)
    pose2 = pose_filtering.landmark_filtering(pose2)

    # 포즈 벡터화
    pose1_vector = vectorize(pose1, pose1_shape)
    pose2_vector = vectorize(pose2, pose2_shape)

    # L2 정규화
    pose1_vector = L2_normalize(pose1_vector)
    pose2_vector = L2_normalize(pose2_vector)

    # 코사인 점수화
    return get_cosine_similarity(pose1_vector, pose2_vector)