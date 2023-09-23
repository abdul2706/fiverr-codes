import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def bgr2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hsv2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def rgb255(image):
    image -= image.min()
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    return image

def resize_within_box(img, d=1000):
    h, w, c = img.shape
    rh, rw = d / h, d / w
    r = min(rh, rw)
    return cv2.resize(img, dsize=None, fx=r, fy=r)

def plt_img(img):
    if len(img.shape) == 2:
        img = cv2.merge([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def grid(images_list, nrows, ncols):
    placeholder_image = np.zeros_like(images_list[0])
    image_grid = None
    for i in range(nrows):
        image_row = None
        for j in range(ncols):
            index = i * ncols + j
            img = images_list[index] if index < len(images_list) else placeholder_image
            img = plt_img(img)
            # print('[img]', img.shape)
            image_row = np.hstack([image_row, img]) if image_row is not None else img
            # print('[grid][image_row]', image_row.shape)
        image_grid = np.vstack([image_grid, image_row]) if image_grid is not None else image_row
    # print('[image_grid]', image_grid.shape)
    return image_grid

def fillhole(input_image):
    h, w = input_image.shape[:2]
    im_flood_fill = input_image.copy().astype("uint8")
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out, im_flood_fill_inv

def gaussian_blur_c3(image_3c, ksize, sigma):
    """ blur 3-channel image """
    c1, c2, c3 = cv2.split(image_3c)
    c1 = cv2.GaussianBlur(c1, ksize=ksize, sigmaX=sigma)
    c2 = cv2.GaussianBlur(c2, ksize=ksize, sigmaX=sigma)
    c3 = cv2.GaussianBlur(c3, ksize=ksize, sigmaX=sigma)
    return cv2.merge([c1, c2, c3])
