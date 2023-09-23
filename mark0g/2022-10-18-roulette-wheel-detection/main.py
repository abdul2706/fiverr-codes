import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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
    height, width, c = frame.shape
    # frame = frame[:, int((width-height-x_offset)*0.4):int((width+height-x_offset)*0.6), :]
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = bgr2gray(frame)
    # frame_gray = cv2.equalizeHist(frame_gray)
    frame_blur = cv2.GaussianBlur(frame_gray, ksize=(7, 7), sigmaX=1.5)
    _, frame_binary = cv2.threshold(frame_blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # frame_binary = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -30)
    # kernel = np.ones((3, 3), np.uint8)
    # frame_binary = cv2.morphologyEx(frame_binary, cv2.MORPH_CLOSE, kernel, iterations=10)
    # frame_binary = cv2.dilate(frame_binary, kernel, iterations=1)
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

    # contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)

    # hough circles
    dp, minDist, param1, param2, minRadius, maxRadius = 1.5, 25, 175, 30, 200, 210
    circles = cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
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
    cv2.circle(mask_image, (average_circle[0], average_circle[1]), average_circle[2], (255, 255, 255), -1)
    image = cv2.bitwise_and(image, image, mask=mask_image)
    image_blur = cv2.bitwise_and(image_blur, image_blur, mask=mask_image)
    image_prev_blur = cv2.bitwise_and(image_prev_blur, image_prev_blur, mask=mask_image)
    image_next_blur = cv2.bitwise_and(image_next_blur, image_next_blur, mask=mask_image)
    # take difference between two consecutive frames
    diff_image = 2 * image_blur - image_prev_blur - image_next_blur
    diff_image = cv2.GaussianBlur(diff_image, ksize=(11, 11), sigmaX=1.5)
    diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_CLOSE, kernel, iterations=5)
    # frame_binary = cv2.dilate(frame_binary, kernel, iterations=1)

    # diff_image = cv2.dilate(diff_image, None, iterations=5)
    # diff_image = cv2.erode(diff_image, None, iterations=5)
    # _, diff_image_binary = cv2.threshold(diff_image, 200, 255, cv2.THRESH_BINARY)
    # _, diff_image_otsu = cv2.threshold(diff_image, 250, 255, cv2.THRESH_OTSU)

    # B G R
    white_lower = (120, 120, 120)
    white_upper = (160, 160, 160)
    diff_image_c3 = cv2.merge([diff_image, diff_image, diff_image])
    diff_image_mask = cv2.inRange(diff_image_c3, white_lower, white_upper)

    # image = gaussian_blur_c3(image, (21, 21), sigma=2)
    # image_hsv = gaussian_blur_c3(image_hsv, (21, 21), sigma=2)
    # H, S, V
    green_lower = (40, 0, 0)
    green_upper = (80, 255, 255)
    mask = cv2.inRange(image_hsv, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2, cv2.LINE_AA)

    # show images as a single grid
    image_grid = grid([image, mask, image_blur, diff_image, diff_image_mask], 2, 4)
    image_grid = bgr2rgb(image_grid)
    image_grid = cv2.resize(image_grid, dsize=None, fx=0.8, fy=0.8)
    cv2.imshow('image_grid', image_grid)

    if cv2.waitKey(1) == ord('q'):
        break

    image_prev, image_prev_hsv, image_prev_gray, image_prev_blur, image_prev_binary = image, image_hsv, image_gray, image_blur, image_binary
    image, image_hsv, image_gray, image_blur, image_binary = image_next, image_next_hsv, image_next_gray, image_next_blur, image_next_binary

cap.release()

# fast fourier transform
# f = np.fft.fft2(image)
# print('[f]', f.shape, np.min(f), np.max(f), np.mean(f), np.std(f))
# fshift = np.fft.fftshift(f)
# print('[fshift]', fshift.shape, np.min(fshift), np.max(fshift), np.mean(fshift), np.std(fshift))

# rows, cols, _ = image.shape
# crow, ccol = rows//2 , cols//2
# fshift2 = np.zeros_like(fshift)
# fshift2[crow-50:crow+50, ccol-50:ccol+50] = fshift[crow-50:crow+50, ccol-50:ccol+50]
# fshift = fshift2
# print('[fshift]', fshift.shape, np.min(fshift), np.max(fshift), np.mean(fshift), np.std(fshift))

# f_ishift = np.fft.ifftshift(fshift)
# print('[f_ishift]', f_ishift.shape, np.min(f_ishift), np.max(f_ishift), np.mean(f_ishift), np.std(f_ishift))
# img_back = np.fft.ifft2(f_ishift)
# print('[img_back]', img_back.shape, np.min(img_back), np.max(img_back), np.mean(img_back), np.std(img_back))
# img_back = np.real(img_back)
# print('[img_back]', img_back.shape, np.min(img_back), np.max(img_back), np.mean(img_back), np.std(img_back))

# magnitude_spectrum = 1 * np.log(np.abs(fshift) + 1) - 1
# print('[magnitude_spectrum]', magnitude_spectrum.shape, np.min(magnitude_spectrum), np.max(magnitude_spectrum), np.mean(magnitude_spectrum), np.std(magnitude_spectrum))

# plt.subplot(221)
# plt.imshow(bgr2rgb(image))
# plt.title('Input Image')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(222)
# plt.imshow(rgb255(magnitude_spectrum))
# plt.title('Magnitude Spectrum')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(223)
# plt.imshow(bgr2rgb(rgb255(img_back)))
# plt.title('Result in JET')
# plt.xticks([])
# plt.yticks([])
# plt.show()
