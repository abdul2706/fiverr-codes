import os
import re
import cv2
import time
import pytesseract
import numpy as np
import pandas as pd
from datetime import datetime

# helper function for resizing high resolution images before displaying
def resize(image, bound=512):
    h1, w1 = image.shape[:2]
    ratio = bound / max(h1, w1)
    image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return image

# tesseract: https://sourceforge.net/projects/tesseract-ocr-alt/files/
# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

PATH_EXCEL = 'excel_files'
now = datetime.now()
excel_filename = f'measurements-{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}.xlsx'
if not os.path.exists(PATH_EXCEL):
    os.mkdir(PATH_EXCEL)

PATH_IMAGES = ''
while not os.path.exists(PATH_IMAGES):
    print("Kindly enter the full path to image (jpg) folder: ")
    PATH_IMAGES = input('>> ')
    if not os.path.exists(PATH_IMAGES):
        print('Given path does not exists, try again...')

# scale-ratio of rectangle which is used to crop region around measurement values
SCALE_RATIO = 2

# get names of all images from folder
all_image_names = os.listdir(PATH_IMAGES)
print('images in specified folder:', all_image_names)

excel_data = pd.DataFrame(columns=['filename', 'measurement'])

# regex pattern for finding measurement digits in the output of tesseract
digit_pattern = re.compile(r'\d+', re.IGNORECASE)

# loop over all images
for i, image_name in enumerate(all_image_names):
    print('processing: ', image_name)

    # note the start time for current image
    t1 = time.time()

    # load image
    image_path = os.path.join(PATH_IMAGES, image_name)
    original_image = cv2.imread(image_path)

    # crop region of image where measurements are written
    h, w = original_image.shape[:2]
    original_image = original_image[int(0.25 * h):int(0.75 * h)]
    
    # create copy of original image before further processing
    image = original_image.copy()
    
    # rotate image to make measurements horizontal
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # convert to HSV for filtering red color measurements
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # set lower and upper red color limits
    lower_val = np.array([0, 100, 200])
    upper_val = np.array([1, 255, 255])

    # threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_val, upper_val)

    # apply mask to original image - this shows the red with black blackground
    image_measurement = cv2.bitwise_and(image, image, mask=mask)

    # convert to binary before better visibility of measurements text
    image_measurement = cv2.cvtColor(image_measurement, cv2.COLOR_BGR2GRAY)
    image_measurement = cv2.threshold(image_measurement, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make copy of image_measurement to extract blobs around measurement text
    morphed_image = image_measurement.copy()
    
    # morphological operations (https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html)
    # to remove thin lines near text
    morphed_image = cv2.morphologyEx(morphed_image, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
    # to convert text to white blobs
    morphed_image = cv2.morphologyEx(morphed_image, cv2.MORPH_DILATE, np.ones((50, 50), np.uint8))
    
    # find contours
    contours, hierarchy = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # convert image from grayscale to RGB
    image_measurement = cv2.cvtColor(image_measurement, cv2.COLOR_GRAY2BGR)
    # this variable will store combined regions of measurement texts in the image
    combined_image = None

    # iterate over all contours
    for contour in contours[::-1]:
        # reshape contour to required shape
        contour = np.array(contour).reshape(-1, 2)
        # get bounding box of rectangle surrounding the contour
        x1 = np.min(contour[:, 0])
        y1 = np.min(contour[:, 1])
        x2 = np.max(contour[:, 0])
        y2 = np.max(contour[:, 1])
        # get center of bounding box
        cx = np.average([x1, x2])
        cy = np.average([y1, y2])
        # shift origin of bounding box to (0, 0) before scaling
        x1 -= cx
        y1 -= cy
        x2 -= cx
        y2 -= cy
        # scale the bounding box and shift back to old location
        x1 = int(x1 * SCALE_RATIO + cx)
        y1 = int(y1 * SCALE_RATIO + cy)
        x2 = int(x2 * SCALE_RATIO + cx)
        y2 = int(y2 * SCALE_RATIO + cy)
        # confirm that x1, y1, x2, y2 are within bounds
        h, w = image_measurement.shape[:2]
        x1 = x1 if x1 > 0 else 0
        y1 = y1 if y1 > 0 else 0
        x2 = x2 if x2 < w else w
        y2 = y2 if y2 < h else h

        # crop the bounding box region
        cropped_region = image_measurement[y1:y2, x1:x2]
        # increase the size of cropped image, to enlarge the size of measurement text
        cropped_region = resize(cropped_region, 256)
        # apply mophological operations to imporove text visibility
        cropped_region = cv2.morphologyEx(cropped_region, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # remove noisy patched near measurement text
        sub_contours, hierarchy = cv2.findContours(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for sub_contour in sub_contours:
            sub_contour = np.array(sub_contour).reshape(-1, 2)
            # get bounding box of rectangle surrounding the contour
            a1 = np.min(sub_contour[:, 0])
            b1 = np.min(sub_contour[:, 1])
            a2 = np.max(sub_contour[:, 0])
            b2 = np.max(sub_contour[:, 1])
            w, h = (a2 - a1), (b2 - b1)
            if w > 0 and h > 0 and min(w, h) / max(w, h) <= 0.20:
                cv2.rectangle(cropped_region, (a1, b1), (a2, b2), (0, 0, 0), -1)

        # extrating the measurement text from the image as string
        string = pytesseract.image_to_string(cropped_region, config='--oem 1 --psm 6')
        # print('string:', string)

        # filter out measurements text
        for token in string.split('\n'):
            matched = digit_pattern.match(token)
            if matched:
                measurement = int(matched.group())
                if measurement > 0 and measurement != 100:
                    print('detected measurement:', measurement)
                    # append data for current image into excel_data
                    excel_data = excel_data.append({'filename': image_name, 'measurement': measurement}, ignore_index=True)

        # lines for debugging
        # uncomment these 3 lines to show the image_measurement
        # cv2.imshow('original_image', cv2.rotate(resize(original_image, 600), cv2.ROTATE_90_CLOCKWISE))
        # cv2.imshow('original_image', resize(original_image, 1200))
        # cv2.imshow('image_measurement', cv2.rotate(resize(image_measurement, 1600), cv2.ROTATE_90_COUNTERCLOCKWISE))
        # # cv2.imshow('image_measurement', resize(image_measurement, 600))
        # # cv2.imshow('cropped_region', cv2.rotate(resize(cropped_region, 600), cv2.ROTATE_90_COUNTERCLOCKWISE))
        # cv2.imshow('cropped_region', cropped_region)
        # if cv2.waitKey() == ord('q'):
        #     break
    
    # note the end time for current image
    t2 = time.time()
    # display time taken for current image
    print(f'time taken: {t2 - t1:.4} seconds')

    # write data to excel sheet after every 10 images
    if (i + 1) % 10 == 0:
        # print('excel_data:\n', excel_data)
        excel_data.to_excel(os.path.join(PATH_EXCEL, excel_filename), index=False)
        print(f'Data for 10 more images saved...')

# in the end, write all data to another excel sheet
excel_filename = f'measurements-final.xlsx'
excel_data.to_excel(os.path.join(PATH_EXCEL, excel_filename), index=False)
print(f'All data written to excel file: {excel_filename}')
print(f'All {len(all_image_names)} images processed')
input('Press enter key to exit...')
