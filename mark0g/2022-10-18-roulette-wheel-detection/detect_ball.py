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
cv2.createTrackbar('R-min', 'trackbars', 140, 255, nothing)
cv2.createTrackbar('G-min', 'trackbars', 140, 255, nothing)
cv2.createTrackbar('B-min', 'trackbars', 140, 255, nothing)
cv2.createTrackbar('R-max', 'trackbars', 160, 255, nothing)
cv2.createTrackbar('G-max', 'trackbars', 160, 255, nothing)
cv2.createTrackbar('B-max', 'trackbars', 160, 255, nothing)

video_path = '2019-07-08 - Roulette Wheel Spins - Session 1 [30 Minutes].mp4'
cap = cv2.VideoCapture(video_path)
print('[CAP_PROP_FRAME_WIDTH]', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('[CAP_PROP_FRAME_HEIGHT]', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('[CAP_PROP_FRAME_COUNT]', cap.get(cv2.CAP_PROP_FRAME_COUNT))

# skip frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 10000)
x_offset = 50

def process_frame(frame):
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = bgr2gray(frame)
    frame_blur = cv2.GaussianBlur(frame_gray, ksize=(7, 7), sigmaX=1.5)
    _, frame_binary = cv2.threshold(frame_blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return frame, frame_hsv, frame_gray, frame_blur, frame_binary

prev_avg_circles = []
kernel = np.ones((3, 3), np.uint8)
success, image_prev = cap.read()
image_prev, image_prev_hsv, image_prev_gray, image_prev_blur, image_prev_binary = process_frame(image_prev)
success, image = cap.read()
image, image_hsv, image_gray, image_blur, image_binary = process_frame(image)

while cap.isOpened():
    success, image_next = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    image_next, image_next_hsv, image_next_gray, image_next_blur, image_next_binary = process_frame(image_next)

    # use HoughCircles for finding the largest circle covering the circular board to clean the entire view
    if len(prev_avg_circles) < 5:
        dp, minDist, param1, param2, minRadius, maxRadius = 1.5, 25, 175, 30, 200, 210
        image_blur_circles = cv2.GaussianBlur(image_gray, ksize=(7, 7), sigmaX=1.5)
        circles = cv2.HoughCircles(image_blur_circles, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            circles = circles[0]
            # print('[circles]', circles.shape)
            circles = np.uint16(np.around(circles))
            # center = np.array([image.shape[1] / 2, image.shape[0] / 2])
            average_circle = np.uint16(np.around(np.mean(circles, axis=0)))
            if len(prev_avg_circles) < 5:
                prev_avg_circles.append(average_circle)
        average_circle = np.mean(prev_avg_circles, axis=0).astype(np.uint32)

    # draw the outer circle
    mask_image = np.zeros_like(image_binary)
    print(average_circle)
    cv2.circle(mask_image, (average_circle[0], average_circle[1]), average_circle[2], (255, 255, 255), -1)
    image = cv2.bitwise_and(image, image, mask=mask_image)
    image_gray = cv2.bitwise_and(image_gray, image_gray, mask=mask_image)
    image_blur = cv2.bitwise_and(image_blur, image_blur, mask=mask_image)
    image_prev_blur = cv2.bitwise_and(image_prev_blur, image_prev_blur, mask=mask_image)
    image_next_blur = cv2.bitwise_and(image_next_blur, image_next_blur, mask=mask_image)
    
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])
    image_gray_sharp = cv2.filter2D(image_blur, ddepth=-1, kernel=kernel)

    # take difference between two consecutive frames
    diff_image = image_blur - image_prev_blur
    # diff_image = cv2.GaussianBlur(diff_image, ksize=(11, 11), sigmaX=1.5)
    diff_image = cv2.medianBlur(diff_image, ksize=25)
    # diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_CLOSE, kernel, iterations=5)
    # diff_image = cv2.dilate(diff_image, None, iterations=5)
    # diff_image = cv2.erode(diff_image, None, iterations=5)
    _, diff_image = cv2.threshold(diff_image, 200, 255, cv2.THRESH_BINARY)
    # mask = cv2.inRange(cv2.merge([diff_image, diff_image, diff_image]), np.array([100, 100, 100]), np.array([120, 120, 120]))
    # diff_iamge = cv2.bitwise_and(diff_image, diff_image, mask=mask)
    # _, diff_image = cv2.threshold(diff_image, 250, 255, cv2.THRESH_OTSU)

    # image_gray_sharp = image_gray - image_blur

    # B G R
    R_min = cv2.getTrackbarPos('R-min','trackbars')
    G_min = cv2.getTrackbarPos('G-min','trackbars')
    B_min = cv2.getTrackbarPos('B-min','trackbars')
    R_max = cv2.getTrackbarPos('R-max','trackbars')
    G_max = cv2.getTrackbarPos('G-max','trackbars')
    B_max = cv2.getTrackbarPos('B-max','trackbars')
    lower = np.array([R_min, G_min, B_min])
    upper = np.array([R_max, G_max, B_max])
    print('[lower, upper]', lower, upper)
    inrange_input = cv2.merge([image_gray, image_gray, image_gray])
    ball_mask = cv2.inRange(inrange_input, lower, upper)

    # show images as a single grid
    image_grid = grid([image, image_gray, image_blur, ball_mask, diff_image, image_gray_sharp], 2, 3)
    image_grid = bgr2rgb(image_grid)
    image_grid = cv2.resize(image_grid, dsize=None, fx=0.7, fy=0.7)
    cv2.imshow('image_grid', image_grid)

    if cv2.waitKey(1) == ord('q'):
        break

    image_prev, image_prev_hsv, image_prev_gray, image_prev_blur, image_prev_binary = image, image_hsv, image_gray, image_blur, image_binary
    image, image_hsv, image_gray, image_blur, image_binary = image_next, image_next_hsv, image_next_gray, image_next_blur, image_next_binary

cap.release()
