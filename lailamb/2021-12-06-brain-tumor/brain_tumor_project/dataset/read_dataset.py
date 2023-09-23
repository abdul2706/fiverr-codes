import os
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing import image


def read_labels(labels_path):
    """
    This function is to read labels from the specified path.

    Args:
        * labels_path (string): path of the labels file needed to be read.

    Returns:
        * df_labels (data_frame): pandas dataframe of the labels.
    """

    # read labels
    df_labels = pd.read_csv(labels_path)

    # encoding: tumor = 1, non-tumor = 0
    df_labels['tumor'] = 1
    df_labels.loc[df_labels['label'] == 'no_tumor', 'tumor'] = 0

    # encoding: no_tumor = 0, meningioma_tumor = 1, glioma_tumor = 2, pituitary_tumor = 3
    df_labels['tumor_type'] = df_labels['tumor']
    df_labels['tumor_type'] = df_labels['label'].map({'no_tumor': 0, 'meningioma_tumor': 1, 'glioma_tumor': 2, 'pituitary_tumor': 3})

    # save csv with encoded labels
    basepath, _ = os.path.split(labels_path)
    df_labels.to_csv(os.path.join(basepath, 'p_labels.csv'))

    return df_labels.iloc[:100]


def read_images(images_path, mode='rgb', image_size=100):
    """
    This function is to read images from the specified path.

    Args:
        * images_path (string): path of the image folder needed to be read.
        * mode (str): 'rgb' for deeplearning models 'grayscale' for baseline models.
        * image_size (int): the size of the loaded image

    Returns:
        * images (list): list of loaded images.
    """

    images_name = os.listdir(images_path)
    images = []
    for image_name in tqdm(images_name[:100]):
        img_path = os.path.join(images_path, image_name)
        img = image.load_img(img_path, target_size=(image_size, image_size), color_mode=mode)
        img = np.array(img) / 255
        images.append(img)
    images = np.array(images)

    if mode == 'grayscale':
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

    return images


def crop_brain_contour(img, plot=False):
    """##### Crop images"""
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()

    return new_image


def generate_brain_contour_crops(images_path, crop_images_path):
    if not os.path.exists(crop_images_path):
        os.mkdir(crop_images_path)
    else:
        if len(os.listdir(crop_images_path)) == len(os.listdir(images_path)):
            return
    images_name = os.listdir(images_path)
    for image_name in tqdm(images_name):
        images = cv2.imread(os.path.join(images_path, image_name))
        crop_image = crop_brain_contour(images, False)
        filename = os.path.join(crop_images_path, image_name)
        if not os.path.exists(filename):
            cv2.imwrite(filename, crop_image)

# generate_brain_contour_crops(IMAGES_PATH)
