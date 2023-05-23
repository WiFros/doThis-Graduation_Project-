import cv2

def draw_circle_landmark(img, pose, r):
    annotated_image = img.copy()
    image_hight, image_width, _ = img.shape
    for i in pose:
        get_x = int(i['position']['x'] * image_width)
        get_y = int(i['position']['y'] * image_hight)
        annotated_image = cv2.circle(annotated_image, (get_x, get_y), r, (0,0,255), -1)
    return annotated_image