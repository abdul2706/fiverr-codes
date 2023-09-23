import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def nothing(x):
    pass

cv2.namedWindow('trackbars')
# create trackbars for: dp, minDist, param1, param2, minRadius, maxRadius
cv2.createTrackbar('dp', 'trackbars', 75, 100, nothing)
cv2.createTrackbar('minDist', 'trackbars', 10, 100, nothing)
cv2.createTrackbar('param1', 'trackbars', 50, 255, nothing)
cv2.createTrackbar('param2', 'trackbars', 80, 200, nothing)
cv2.createTrackbar('minRadius', 'trackbars', 0, 200, nothing)
cv2.createTrackbar('maxRadius', 'trackbars', 0, 200, nothing)

while True:
    image_name = os.listdir('frames')[0]
    image = cv2.imread(os.path.join('frames', image_name))
    image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    # print('[image][resize]', image.shape)
    placeholder_image = np.zeros_like(image)

    dp = cv2.getTrackbarPos('dp','trackbars')
    if dp <= 0: dp = 1
    minDist = cv2.getTrackbarPos('minDist','trackbars')
    if minDist <= 0: minDist = 1
    param1 = cv2.getTrackbarPos('param1','trackbars')
    if param1 <= 50: param1 = 50
    param2 = cv2.getTrackbarPos('param2','trackbars')
    if param2 <= 0: param2 = 1
    minRadius = cv2.getTrackbarPos('minRadius','trackbars')
    maxRadius = cv2.getTrackbarPos('maxRadius','trackbars')

    # 75, 25, 175, 42, 200, 0

    image_gray = bgr2gray(image)
    image_blur = cv2.GaussianBlur(image_gray, ksize=(7, 7), sigmaX=1.5)
    # _, image_binary = cv2.threshold(image_blur, param1, 255, cv2.THRESH_BINARY)

    # contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image_canny = cv2.Canny(image_binary, 150, 250)
    # contours, _ = cv2.findContours(image_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])
    image_laplacian = cv2.filter2D(image_blur, ddepth=-1, kernel=kernel)

    # hough circles
    circles = cv2.HoughCircles(image_laplacian, cv2.HOUGH_GRADIENT, dp=dp / 50, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = circles[0]
        # print('[circles]', circles.shape)
        circles = np.uint16(np.around(circles))
        # center = np.array([image.shape[1] / 2, image.shape[0] / 2])
        # average_circle = np.uint16(np.around(np.mean(circles, axis=0)))
        # print('[average_circle]', average_circle)
        # draw the outer circle
        # cv2.circle(image, (average_circle[0], average_circle[1]), average_circle[2], (255, 255, 0), 2)
        # draw the center of the circle
        # cv2.circle(image, (average_circle[0], average_circle[1]), 2, (0, 255, 255), 3)
        for i in circles:
            # if np.linalg.norm(i[:2] - center) < 30:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # show images as a single grid
    image_grid = grid([image, image_gray, image_blur, image_laplacian], 2, 2)
    image_grid = bgr2rgb(image_grid)
    # image_grid = cv2.resize(image_grid, dsize=None, fx=0.7, fy=0.7)
    cv2.imshow('image_grid', image_grid)

    if cv2.waitKey(1) == ord('q'):
        break
