import cv2
import numpy as np
import mediapipe as mp
import glob
import os
import csv
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

'''
# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
'''

def fields_name():
    # CSV format
    fields = []
    #fields.append('file_name')
    for i in range(42):
        fields.append(str(i)+'_x')
        fields.append(str(i)+'_y')
        fields.append(str(i)+'_z')
    return fields

# save directory
save_csv_dir = './result/csv'
os.makedirs(save_csv_dir, exist_ok=True)
save_csv_name = 'landmark.csv'
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, \
    open(os.path.join(save_csv_dir, save_csv_name),
            'w', encoding='utf-8', newline="") as f:
  writer = csv.DictWriter(f, fieldnames=fields_name())
  writer.writeheader()

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue

    #LeftHandRecord
    record = {}
    #RightHandRecord
    record1={}

    for hand_world_landmarks, handedness in zip(results.multi_hand_world_landmarks, results.multi_handedness):
      handedness_index = 0


      if handedness.classification[0].label == 'Left':
          handedness_index = 0


          landmarks = hand_world_landmarks
          #Left hand landmark
          for i, landmark in enumerate(landmarks.landmark):
            record[str(i) + '_x'] = landmark.x
            record[str(i) + '_y'] = landmark.y
            record[str(i) + '_z'] = landmark.z



      elif handedness.classification[0].label == 'Right':

          handedness_index = 1
          landmarks = hand_world_landmarks
          #Right hand landmark
          for i, landmark in enumerate(landmarks.landmark):
            record1[str(i+21) + '_x'] = landmark.x
            record1[str(i+21) + '_y'] = landmark.y
            record1[str(i+21) + '_z'] = landmark.z

      # If Multi hand ditected, save record landmarks
      if np.sum(len(handedness.classification)) == 2:

          if handedness.classification[1].label == 'Left':
              #Left hand landmark
              handedness_index = 0
              landmarks = results.multi_hand_world_landmarks[1]
              landmarks = hand_world_landmarks

              for i, landmark in enumerate(landmarks.landmark):
                  record[str(i) + '_x'] = landmark.x
                  record[str(i) + '_y'] = landmark.y
                  record[str(i) + '_z'] = landmark.z

          elif handedness.classification[1].label == 'Right':
              #Right hand landmark
              handedness_index = 1
              landmarks = results.multi_hand_world_landmarks[1]
              landmarks = hand_world_landmarks

              for i, landmark in enumerate(landmarks.landmark):
                  record1[str(i+21) + '_x'] = landmark.x
                  record1[str(i+21) + '_y'] = landmark.y
                  record1[str(i+21) + '_z'] = landmark.z

    d={**record, **record1}
    writer.writerow(d)
    #for hand_world_landmarks in results.multi_hand_world_landmarks:
    #    mp_drawing.plot_landmarks(
    #    hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
        #Flip the image horizontally for a selfie-view display.

cap.release()