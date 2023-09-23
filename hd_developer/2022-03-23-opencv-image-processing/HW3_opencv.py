from lib2to3.pgen2.token import OP
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
print(cv2.version.opencv_version)

class OpenCV_Image:


    def __init__(self, path_to_image=None, image_read_mode=None):
        """
        Creates opencv-img handling object
        :param path_to_image: path to image file
        :param image_read_mode: 0-Grayscale, 1-Color, -1-Unchanged
        """
        self.img = None
        if path_to_image != None:
            try:
                self.img = cv2.imread(path_to_image, image_read_mode)
            except Exception as e:
                print("Cannot read image:\n")
                print(e)

    def display_cv(self, window_name="Image"):
        """
        Displays image until key is pressed, only if img exists
        :param window_name: Named window as str option
        :return: None
        """
        if self.img is not None:
            cv2.imshow(window_name, self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Image not set")

    def save_img(self, out_file_path_name):
        """
        Saves image file to path with filename
        :param out_file_path_name: full path with file name
        :return: True if success, False else
        """
        if self.img is not None:
            try:
                cv2.imwrite(out_file_path_name, self.img)
            except Exception as e:
                print("Exception:\n")
                print(e)
                print("Cannot write image")
                return False
            return True
        else:
            print("Image does not exist")
            return False

    def display_plt(self):
        """
        Displays image using matplotlib, color conversion from bgr to rgb
        :return: None
        """
        rgb_im = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_im)
        plt.show()

    def select_ROI(self):
        """
        Select square Region of interest
        :return: region of interest [top_left_xpix, top_left_ypix, width, height]
        """
        r = cv2.selectROI("Select Square ROI then Spacebar", self.img, fromCenter=False, showCrosshair=False)
        cv2.destroyAllWindows()
        return r

    def return_border(self, border_pix_width):
        """
        Creates a border for current self.img; border type:  cv2.BORDER_CONSTANT
        :param border_pix_width: integer # of pixels to add to image on all sides for border
        :return: new opencv_image object
        """
        new_opencv_image = OpenCV_Image()
        new_opencv_image.img = cv2.copyMakeBorder(self.img, border_pix_width, border_pix_width, border_pix_width, border_pix_width, cv2.BORDER_CONSTANT, value=[0, 0, 255])
        return new_opencv_image

    def return_copypaste(self):
        """
        Uses image ROI to copy/paste a section of current self.img
        add 2 pixel border to pasted region
        :return: new opencv_image object
        """
        r = self.select_ROI()
        x, y, w, h = r
        selected_region = self.img[y:y+h, x:x+w]
        new_opencv_image = OpenCV_Image()
        new_opencv_image.img = selected_region
        return new_opencv_image.return_border(20)

    def split_image(self):
        """
        Splits current self.img in half, by xpix
        :return: tuple of 2 OpenCV_Image objects
        """
        width = self.img.shape[1]
        half_idx = width // 2
        left_half = OpenCV_Image()
        left_half.img = self.img[:, :half_idx]
        right_half = OpenCV_Image()
        right_half.img = self.img[:, half_idx:]
        return (left_half, right_half)

class OpenCV_Video:


    def __init__(self, path_to_video=None):
        """
        Creates opencv-video handling object
        :param path_to_video: path to video file
        """
        self.frame_stack = None
        self.xpix = None
        self.ypix = None
        self.num_frames = None
        self.fps = None
        if path_to_video != None:
            cap = cv2.VideoCapture(path_to_video) #create import object
            self.xpix = int(cap.get(3))  # frame width
            self.ypix = int(cap.get(4))  # frame height
            self.fps = int(cap.get(5))  # vid fps
            self.num_frames = int(cap.get(7))  # numframes
            self.frame_stack = np.zeros(
                (int(self.num_frames), self.ypix, self.xpix, 3), #numpy array to store frames[frame_number, rows, cols, bgr]
                dtype=np.uint8) # data type set same as opencv-image files
            for i in range(self.num_frames):
                ret, frame = cap.read() #read frame from video
                self.frame_stack[i, :, :, :] = frame #save frame to stack
            cap.release() #close import object
            #print(self.frame_stack.shape)

    def return_frame(self, frame_idx):
        """
        Returns frame from video frame stack (0 indexed)
        :param frame_idx: frame # to return; 0 <= integer < self.num_frames
        :return: frame: single image of video stack as OpenCV_Image
        """
        if type(frame_idx) is int: #check that is integer
            if (0 <= frame_idx) and (self.num_frames > frame_idx): # check that int is within frames
                frame_data = self.frame_stack[frame_idx, :, :, :] #remove frame from stack
                frame = OpenCV_Image() #create empty opencv-image object
                frame.img = frame_data #set image data
                return frame #return opencv-object
        return None # only if frame_idx does not meet reqs

def opencv_classes_test():
    """
    test function, imports 1 image and 1 video from local directory.
    Displays image in opencv and plt
    Displays Frame from video
    :return: None
    """
    print("Select Image")
    files = os.listdir()
    for i, file in enumerate(files):
        print(i, file)
    selection = input("Select Image Index\n")
    file_val = files[int(selection)]
    img = OpenCV_Image(os.path.join(os.getcwd(), file_val), 1)
    roi = img.select_ROI()
    print(roi)
    img.display_plt()
    img.display_cv("testwindow")
    print("Select Video")
    for i, file in enumerate(files):
        print(i, file)
    selection = input("Select Video Index\n")
    vid_val = files[int(selection)]
    vid = OpenCV_Video(os.path.join(os.getcwd(), vid_val))
    vid_frame = vid.return_frame(5)
    vid_frame.display_cv("Frame # 5")

def blend_images(opcv_img_1, weight_1, opcv_img_2, weight_2):
    """
    Blends two OpenCV_Image objects according to parameter weights
    :param opcv_img_1: OpenCV_Image object
    :param weight_1: weight to blend opcv_img_1
    :param opcv_img_2: OpenCV_Image object
    :param weight_2: weight to blend opcv_img_2
    :return: new OpenCV_Image object
    """
    blended_image = OpenCV_Image()
    blended_image.img = cv2.addWeighted(opcv_img_1.img, weight_1, opcv_img_2.img, weight_2, 0)
    return blended_image

def hw3_main():
    """
    main() function for hw3
    Goals:
        0: try to create an output directory "YourName_Images", save all images here
        Your Choice images:
        1: import 2 local image files
        2: Add border to one image, show then save that image
        3: use select_ROI() and return_copypaste() to edit the second image
        Canvas Video:
        4: import local video file
        5: remove 3 frames from video (equally spaced)
        for each frame extracted:
            6: split video frame in half
            7: blend the two halves into a single image
            8: save image
    :return: None
    """
    # pass
    # 0: try to create an output directory "YourName_Images", save all images here
    output_directory = 'ARK_Images'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # 1: import 2 local image files
    image1 = OpenCV_Image(os.path.join(os.getcwd(), 'image1.jpeg'), 1)
    image2 = OpenCV_Image(os.path.join(os.getcwd(), 'image2.jpeg'), 1)

    # 2: Add border to one image, show then save that image
    image1_with_border = image1.return_border(20)
    image1_with_border.display_cv('image1_with_border')
    image1_with_border.save_img(os.path.join(output_directory, 'image1_with_border.jpg'))

    # 3: use select_ROI() and return_copypaste() to edit the second image
    image2_copypaste = image2.return_copypaste()
    image2_copypaste.display_cv('image2_copypaste')
    image2_copypaste.save_img(os.path.join(output_directory, 'image2_copypaste.jpg'))
    
    # 4: import local video file
    local_video = OpenCV_Video(os.path.join(os.getcwd(), 'Videos of cells flowing in a microfluidic cell culture device for studying immune cell behavior_1080p.mp4'))

    # 5: remove 3 frames from video (equally spaced)
    frame_idx1 = 0
    frame_idx2 = (local_video.num_frames - 1) // 2
    frame_idx3 = local_video.num_frames - 1

    print(frame_idx2 - frame_idx1, frame_idx3 - frame_idx2, frame_idx3 - frame_idx1)

    frame_image1 = local_video.return_frame(frame_idx1)
    frame_image2 = local_video.return_frame(frame_idx2)
    frame_image3 = local_video.return_frame(frame_idx3)

    # for each frame extracted:
    #     6: split video frame in half
    #     7: blend the two halves into a single image
    #     8: save image
    frame1_left_half, frame1_right_half = frame_image1.split_image()
    frame1_blend = blend_images(frame1_left_half, 0.5, frame1_right_half, 0.5)
    frame1_blend.save_img(os.path.join(output_directory, 'frame1_blend.jpg'))
    
    frame2_left_half, frame2_right_half = frame_image2.split_image()
    frame2_blend = blend_images(frame2_left_half, 0.5, frame2_right_half, 0.5)
    frame2_blend.save_img(os.path.join(output_directory, 'frame2_blend.jpg'))

    frame3_left_half, frame3_right_half = frame_image3.split_image()
    frame3_blend = blend_images(frame3_left_half, 0.5, frame3_right_half, 0.5)
    frame3_blend.save_img(os.path.join(output_directory, 'frame3_blend.jpg'))

if __name__ == '__main__':
    # opencv_classes_test()
    hw3_main()
