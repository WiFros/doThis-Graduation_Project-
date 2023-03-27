import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
import glob
import os
import csv
from utils import DLT, get_projection_matrix, write_keypoints_to_disk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [720, 1280]

# add here if you need more keypoints
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

pose_divided = {
    'torso': [11, 12, 23, 24],
    'rightarm': [12, 14, 16],
    'leftarm': [11, 13, 15],
    'rightleg': [24, 26, 28],
    'leftleg': [23, 25, 27]
}


def fields_name():
    # CSV format
    fields = []
    #fields.append('file_name')

    for i, parts in enumerate(pose_divided):
        for j, vertex in enumerate(pose_divided[parts]):
            print(parts, ':', vertex)
    return fields


def run_mp(input_stream1, input_stream2, P0, P1,mode):
    #fields_name()
    # For static images:
    if mode == 'image':
        kpts_cam0 = []
        kpts_cam1 = []
        kpts_3d = []


        IMAGE_FILES = ['media/Sample/ex_3.png']
        BG_COLOR = (192, 192, 192)  # gray
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.1) as pose:
            for idx, file in enumerate(IMAGE_FILES):
                image = cv.imread(file)
                image_height, image_width, _ = image.shape
                # Convert the BGR image to RGB before processing.
                results = pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

                if not results.pose_landmarks:
                    continue
                print(
                    f'Nose coordinates: ('
                    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
                )

                annotated_image = image.copy()
                # Draw segmentation on the image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                annotated_image = np.where(condition, annotated_image, bg_image)
                # Draw pose landmarks on the image.
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv.imwrite('media/Sample/frame1_annotated_image' + str(idx) + '.png', annotated_image)
                # Plot pose world landmarks.
                mp_drawing.plot_landmarks(
                    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


    # write Excel file each landmark position into x,y,z
    if mode == 'video':
        IMAGE_FILES_0 = 'media/Sample/frame1.jpeg'
        IMAGE_FILES_1 = 'media/Sample/ex_3.png'
        image0 = cv.imread(IMAGE_FILES_0)
        image1 = cv.imread(IMAGE_FILES_1)
        # input video stream
        cap0 = cv.imread(IMAGE_FILES_0)
        cap1 = cv.imread(IMAGE_FILES_1)
        caps = [cap0, cap1]
        image_height0, image_width0, _ = cap0.shape
        image_height1, image_height1, _ = cap1.shape


        # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
        #for cap in caps:
        #    cap.set(3, frame_shape[1])
        #    cap.set(4, frame_shape[0])

        # create body keypoints detector objects.
        pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # containers for detected keypoints for each camera. These are filled at each frame.
        # This will run you into memory issue if you run the program without stop
        kpts_cam0 = []
        kpts_cam1 = []
        kpts_3d = []
        i = 0;
        while i<3:
            i+=1
            # read frames from stream
            frame0 = cap0
            frame1 = cap1

            #if not ret0 or not ret1: break

            # crop to 720x720.
            # Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
            #if frame0.shape[1] != 720:
            #    frame0 = frame0[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]
            #    frame1 = frame1[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]

            # the BGR image to RGB.
            frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
            frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame0.flags.writeable = False
            frame1.flags.writeable = False
            results0 = pose0.process(frame0)
            results1 = pose1.process(frame1)
            #results0 = pose.process(cv.cvtColor(cap0, cv.COLOR_BGR2RGB))
            #results1 = pose.process(cv.cvtColor(cap1, cv.COLOR_BGR2RGB))
            # reverse changes
            frame0.flags.writeable = True
            frame1.flags.writeable = True
            frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
            frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

            # check for keypoints detection
            frame0_keypoints = []
            if results0.pose_landmarks:
                for i, landmark in enumerate(results0.pose_landmarks.landmark):
                    if i not in pose_keypoints: continue  # only save keypoints that are indicated in pose_keypoints
                    pxl_x = landmark.x * frame0.shape[1]
                    pxl_y = landmark.y * frame0.shape[0]
                    pxl_z = results0.pose_world_landmarks.landmark[i].z * (frame0.shape[0]+frame0.shape[1]) / 2
                    print(pxl_x)
                    print(pxl_y)
                    print(pxl_z)
                    pxl_x = int(round(pxl_x))
                    pxl_y = int(round(pxl_y))
                    pxl_z = int(round(pxl_y))
                    cv.circle(frame0, (pxl_x, pxl_y), 3, (0, 0, 255), -1)  # add keypoint detection points into figure

                    kpts = [pxl_x, pxl_y]
                    frame0_keypoints.append(kpts)
            else:
                # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
                frame0_keypoints = [[-1, -1]] * len(pose_keypoints)

            # this will keep keypoints of this frame in memory
            kpts_cam0.append(frame0_keypoints)

            frame1_keypoints = []
            if results1.pose_landmarks:
                for i, landmark in enumerate(results1.pose_landmarks.landmark):
                    if i not in pose_keypoints: continue
                    pxl_x = landmark.x * frame1.shape[1]
                    pxl_y = landmark.y * frame1.shape[0]
                    pxl_z = results1.pose_world_landmarks.landmark[i].z * (frame1.shape[0]+frame1.shape[1]) / 2

                    pxl_x = int(round(pxl_x))
                    pxl_y = int(round(pxl_y))
                    cv.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                    kpts = [pxl_x, pxl_y]
                    frame1_keypoints.append(kpts)

            else:
                # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
                frame1_keypoints = [[-1, -1]] * len(pose_keypoints)

            # update keypoints container
            kpts_cam1.append(frame1_keypoints)

            # calculate 3d position
            frame_p3ds = []
            for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
                if uv1[0] == -1 or uv2[0] == -1:
                    _p3d = [-1, -1, -1]
                else:
                    _p3d = DLT(P0, P1, uv1, uv2)  # calculate 3d position of keypoint
                frame_p3ds.append(_p3d)

            '''
            This contains the 3d position of each keypoint in current frame.
            For real time application, this is what you want.
            '''
            frame_p3ds = np.array(frame_p3ds).reshape((12, 3))
            kpts_3d.append(frame_p3ds)
            #print(frame0)
            # uncomment these if you want to see the full keypoints detections
            # mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # draw into plot
            # mp_drawing.plot_landmarks(
            #    results1.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.imshow('cam1', frame1)
            cv.imshow('cam0', frame0)
            cv.imwrite('media/Sample/frame0_annotated_image' + str(i) + '.png', frame0)
            cv.imwrite('media/Sample/frame1_annotated_image' + str(i) + '.png', frame1)

            k = cv.waitKey(1)
            if k & 0xFF == 27: break  # 27 is ESC key.

        cv.destroyAllWindows()
        #for cap in caps:
        #    cap.release()

        return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


if __name__ == '__main__':

    # this will load the sample videos if no camera ID is given
    input_stream1 = 'media/test2.mp4'
    input_stream2 = 'media/test2.mp4'

    # put camera id as command line arguements
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    # get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1,'video')
    #kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1,'image')

    # this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)