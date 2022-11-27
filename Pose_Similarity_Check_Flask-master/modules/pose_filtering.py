# 포즈에서 어떠한 랜드마크를 사용할지와 해당 랜드마크만 제외하고 keypoints 값들을 조절할 수 있는 모듈

# 전체 포즈 리스트 (참고용)
# ['PoseLandmark.NOSE', 'PoseLandmark.LEFT_EYE_INNER', 'PoseLandmark.LEFT_EYE', 'PoseLandmark.LEFT_EYE_OUTER', 'PoseLandmark.RIGHT_EYE_INNER',
# 'PoseLandmark.RIGHT_EYE', 'PoseLandmark.RIGHT_EYE_OUTER', 'PoseLandmark.LEFT_EAR', 'PoseLandmark.RIGHT_EAR', 'PoseLandmark.MOUTH_LEFT',
# 'PoseLandmark.MOUTH_RIGHT', 'PoseLandmark.LEFT_SHOULDER', 'PoseLandmark.RIGHT_SHOULDER', 'PoseLandmark.LEFT_ELBOW', 'PoseLandmark.RIGHT_ELBOW',
# 'PoseLandmark.LEFT_WRIST', 'PoseLandmark.RIGHT_WRIST', 'PoseLandmark.LEFT_PINKY', 'PoseLandmark.RIGHT_PINKY', 'PoseLandmark.LEFT_INDEX',
# 'PoseLandmark.RIGHT_INDEX', 'PoseLandmark.LEFT_THUMB', 'PoseLandmark.RIGHT_THUMB', 'PoseLandmark.LEFT_HIP', 'PoseLandmark.RIGHT_HIP',
# 'PoseLandmark.LEFT_KNEE', 'PoseLandmark.RIGHT_KNEE', 'PoseLandmark.LEFT_ANKLE', 'PoseLandmark.RIGHT_ANKLE', 'PoseLandmark.LEFT_HEEL',
# 'PoseLandmark.RIGHT_HEEL', 'PoseLandmark.LEFT_FOOT_INDEX', 'PoseLandmark.RIGHT_FOOT_INDEX']

# 남겨둘 포즈, 일단 17개의 landmark만 적용한다
Poses = ['PoseLandmark.NOSE', 'PoseLandmark.LEFT_EYE', 'PoseLandmark.RIGHT_EYE', 'PoseLandmark.LEFT_EAR', 'PoseLandmark.RIGHT_EAR', 'PoseLandmark.LEFT_SHOULDER',
        'PoseLandmark.RIGHT_SHOULDER', 'PoseLandmark.LEFT_ELBOW', 'PoseLandmark.RIGHT_ELBOW', 'PoseLandmark.LEFT_WRIST', 'PoseLandmark.RIGHT_WRIST',
        'PoseLandmark.LEFT_HIP', 'PoseLandmark.RIGHT_HIP', 'PoseLandmark.LEFT_KNEE', 'PoseLandmark.RIGHT_KNEE', 'PoseLandmark.LEFT_ANKLE', 'PoseLandmark.RIGHT_ANKLE']

# 남겨둘 포즈, 8개의 landmark만 적용
# Poses = ['PoseLandmark.LEFT_ELBOW', 'PoseLandmark.RIGHT_ELBOW', 'PoseLandmark.LEFT_WRIST', 'PoseLandmark.RIGHT_WRIST',
#          'PoseLandmark.LEFT_KNEE', 'PoseLandmark.RIGHT_KNEE', 'PoseLandmark.LEFT_ANKLE', 'PoseLandmark.RIGHT_ANKLE']

# Poses에 있는 landmark만 남긴 리스트를 반환
def landmark_filtering(pose):
    for i in pose[:]:
        if not i['part'] in Poses:
            pose.remove(i)
    return pose