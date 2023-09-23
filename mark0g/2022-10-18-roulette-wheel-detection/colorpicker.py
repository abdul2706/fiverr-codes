import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def nothing(x):
    pass

# image_names = os.listdir('images')
# image = cv2.imread(os.path.join('images', image_names[0]))

cv2.namedWindow('trackbars')
# create trackbars for color change
cv2.createTrackbar('H-min', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('S-min', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('V-min', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('H-max', 'trackbars', 255, 255, nothing)
cv2.createTrackbar('S-max', 'trackbars', 255, 255, nothing)
cv2.createTrackbar('V-max', 'trackbars', 255, 255, nothing)


video_path = '2019-07-08 - Roulette Wheel Spins - Session 1 [30 Minutes].mp4'
cap = cv2.VideoCapture(video_path)
print('[CAP_PROP_FRAME_WIDTH]', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('[CAP_PROP_FRAME_HEIGHT]', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('[CAP_PROP_FRAME_COUNT]', cap.get(cv2.CAP_PROP_FRAME_COUNT))

# skip frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # image, image_gray, image_blur, image_binary = process_frame(image)
    # diff_image = image_blur - image2_blur
    # diff_image = cv2.GaussianBlur(diff_image, ksize=(3, 3), sigmaX=1.5)
    # _, diff_image_binary = cv2.threshold(diff_image, 250, 255, cv2.THRESH_BINARY)

    # image = cv2.imread(os.path.join('images', image_names[0]))
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # print('[image]', image.shape)
    image = cv2.resize(image, None, fx=0.2, fy=0.2)
    # print('[image][resize]', image.shape)
    placeholder_image = np.zeros_like(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    H_min = cv2.getTrackbarPos('H-min','trackbars')
    S_min = cv2.getTrackbarPos('S-min','trackbars')
    V_min = cv2.getTrackbarPos('V-min','trackbars')
    H_max = cv2.getTrackbarPos('H-max','trackbars')
    S_max = cv2.getTrackbarPos('S-max','trackbars')
    V_max = cv2.getTrackbarPos('V-max','trackbars')
    lower = np.array([H_min, S_min, V_min])
    upper = np.array([H_max, S_max, V_max])
    print('[lower, upper]', lower, upper)
    mask = cv2.inRange(image_hsv, lower, upper)
    image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    image = cv2.bitwise_and(image, image, mask=mask)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_blur = cv2.GaussianBlur(image_gray, (11, 11), 0)
    # image_gray_blur = cv2.medianBlur(image_gray, 11)
    # image_thresh = cv2.adaptiveThreshold(image_gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, image_thresh = cv2.threshold(image_gray_blur, 90, 255, cv2.THRESH_BINARY)
    # image_thresh, im_flood_fill_inv = fillhole(image_thresh)
    image_canny = cv2.Canny(image_thresh, 150, 250)

    contours, _ = cv2.findContours(image_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        contours_list.append([area, contour])
    contours_list = sorted(contours_list, key=lambda x: x[0], reverse=True)
    # print(len(contours_list))
    # print([cnt[0] for cnt in contours_list])
    # contours = [contours_list[i][1] for i in range(20)]
    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

    titles = ['image', 'image_hsv', 'image_gray', 'image_gray_blur', 'image_thresh', 'image_canny']
    plot_images = [image, image_hsv, image_gray, image_gray_blur, image_thresh, image_canny]
    grid_image = grid(plot_images, nrows=2, ncols=3)
    grid_image = resize_within_box(grid_image, d=1500)
    cv2.imshow('grid_image', grid_image)

    if cv2.waitKey(1) == ord('q'):
        break
