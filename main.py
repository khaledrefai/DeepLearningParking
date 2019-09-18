import yaml
import numpy as np
import cv2
# This file contains all the main external libs we'll use
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
from samples.coco import coco

import sys

from PIL import Image, ImageOps


from pathlib import Path
import json
from PIL import ImageDraw, ImageFont

import pdb



# path references
fn = "camer1.webm" #3
fn_yaml = "main_park_points - Copy.yml"
fn_out =  "outputvideo_01.avi"
global_str = "Last change at: "
change_pos = 0.00
dict =  {
        'text_overlay': True,
        'parking_overlay': True,
        'parking_id_overlay': True,
        'parking_detection': True,
        'motion_detection': False,
        'min_area_motion_contour': 500, # area given to detect motion
        'park_laplacian_th': 2.8,
        'park_sec_to_wait': 1, # 4   wait time for changing the status of a region
        'start_frame': 1, # begin frame from specific frame number
        'show_ids': True, # shows id on each region
        'classifier_used': True,
        'save_video': True
        }

# Set from video
cap = cv2.VideoCapture(fn)
video_info = {  'fps':    cap.get(cv2.CAP_PROP_FPS),
                'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.6),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.6),
                'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

cap.set(cv2.CAP_PROP_POS_FRAMES, dict['start_frame']) # jump to frame number specified


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 32
    DETECTION_MIN_CONFIDENCE = 0.2
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    #BACKBONE = "resnet50"
    MAX_GT_INSTANCES=1
    DETECTION_MAX_INSTANCES=1
config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)



'''arch=resnet101
PATH = "./data/"
sz=224
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=False)
trn_tfms, val_tfms = tfms_from_model(arch,sz) # get transformations
'''
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def run_classifier(imgs):
    #im = val_tfms(img)
    #learn.precompute=False # We'll pass in a raw image, not activations
    #log_preds = learn.predict_array(im[None])
    #pred = np.argmax(log_preds, axis=1)
    #image = skimage.io.imread(img)
    print(len(imgs))
    results = model.detect(imgs, verbose=1)
    park_sts=[]
    for res in results:
        print(res['class_ids'])
        if 1 in  res['class_ids'] or res['class_ids'].size ==0:
            park_sts.append(False)
        else:
            park_sts.append(True)
    return park_sts


# Define the codec and create VideoWriter object
if dict['save_video']:
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D') # options: ('P','I','M','1'), ('D','I','V','X'), ('M','J','P','G'), ('X','V','I','D')
    out = cv2.VideoWriter(fn_out, fourcc, 25.0,(video_info['width'], video_info['height']))

# # initialize the HOG descriptor/person detector. Take a lot of processing power.
# if dict['pedestrian_detection']:
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Use Background subtraction
if dict['motion_detection']:
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)


# Read YAML data (parking space polygons)
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.load(stream)

parking_contours = []
parking_bounding_rects = []
parking_mask = []
parking_data_motion = []
park_imgs =[]
park_img_ids=[]

if parking_data != None:
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        #points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
        #points_shifted[:,1] = points[:,1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                    color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask==255
        parking_mask.append(mask)

kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # morphological kernel
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5,19))
if parking_data != None:
    parking_status = [False]*len(parking_data)
    parking_buffer = [None]*len(parking_data)
# bw = ()


def print_parkIDs(park, coor_points, frame_rev):
    moments = cv2.moments(coor_points)
    centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
    # putting numbers on marked regions
    cv2.putText(frame_rev, str(park['id']), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)





while (cap.isOpened()):
    video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
    video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Index of the frame to be decoded/captured next
    ret, frame_initial = cap.read()
    if ret == True:
        frame = cv2.resize(frame_initial, None, fx=0.6, fy=0.6)
    if ret == False:
        print("Video ended")
        break

    # Background Subtraction
    frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
    # frame_blur = frame_blur[150:1000, 100:1800]
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    # Drawing the Overlay. Text overlay at the left corner of screen
    if dict['text_overlay']:
        str_on_frame = "%d/%d" % (video_cur_frame, video_info['num_of_frames'])
        cv2.putText(frame_out, str_on_frame, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_out, global_str + str(round(change_pos, 2)) + 'sec', (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # motion detection for all objects
    if dict['motion_detection']:
        fgmask = fgbg.apply(frame_blur)
        bw = np.uint8(fgmask == 255) * 255
        bw = cv2.erode(bw, kernel_erode, iterations=1)
        bw = cv2.dilate(bw, kernel_dilate, iterations=1)
        (_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
            if cv2.contourArea(c) < dict['min_area_motion_contour']:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # detecting cars and vacant spaces with LAPLACIAN
    '''if dict['parking_detection']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]),
                       rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation

            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
            # cv2.imshow('oir', laplacian)
            points[:, 0] = points[:, 0] - rect[0]  # shift contour to roi
            points[:, 1] = points[:, 1] - rect[1]
            delta = np.mean(np.abs(laplacian * parking_mask[ind]))
            # if(delta<2.5):
            # print("ind, del", ind, delta)
            status = delta < dict['park_laplacian_th']
            # If detected a change in parking status, save the current time
            if status != parking_status[ind] and parking_buffer[ind] == None:
                parking_buffer[ind] = video_cur_pos
                change_pos = video_cur_pos
            # If status is still different than the one saved and counter is open
            elif status != parking_status[ind] and parking_buffer[ind] != None:
                if video_cur_pos - parking_buffer[ind] > dict['park_sec_to_wait']:
                    parking_status[ind] = status
                    parking_buffer[ind] = None
            # If status is still same and counter is open
            elif status == parking_status[ind] and parking_buffer[ind] != None:
                parking_buffer[ind] = None
         '''
    # changing the color on the basis on status change occured in the above section and putting numbers on areas

    #desired_size = 255

    if dict['parking_overlay']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if True: #parking_status[ind]
                color = (0, 255, 0)
                rect = parking_bounding_rects[ind]
                roi_gray_ov = frame[rect[1]:(rect[1] + rect[3]),
                              rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
                #old_size = roi_gray_ov.shape[:2]
                #ratio = float(desired_size) / max(old_size)
                #new_size = tuple([int(x * ratio) for x in old_size])
                #roi_gray_ov = cv2.resize(roi_gray_ov, (new_size[1], new_size[0]))
                #delta_w = desired_size - new_size[1]
                #delta_h = desired_size - new_size[0]
                #top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                #left, right = delta_w // 2, delta_w - (delta_w // 2)

                #color = [0, 0, 0]
                #roi_gray_ov = cv2.copyMakeBorder(roi_gray_ov, top, bottom, left, right, cv2.BORDER_CONSTANT,
                #                            value=color)
                park_imgs.append(roi_gray_ov)
                park_img_ids.append(ind)
                cv2.imwrite(
                    "test\car_318__id{}_   .jpg".format( ind),
                    roi_gray_ov)
                '''res = run_classifier(roi_gray_ov)
                cv2.imwrite(
                    "test\car_318_{}_id{}_   .jpg".format(res, ind ),
                    roi_gray_ov)
                if res:
                    parking_data_motion.append(parking_data[ind])
                    color = (0, 255, 0)
            else:
                color = (0, 0, 255)
                '''

        park_sts = run_classifier(park_imgs)
        index = 0
        while index < len(park_sts):
            print(index)
            id = park_img_ids[index]
            park = parking_data[id]
            points = np.array(park['points'])
            color = (0, 255, 0)
            print("id {} , status {} ".format(id,park_sts[index]))
            if park_sts[index]:
                color = (0, 0, 255)
            cv2.drawContours(frame_out, [points], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)
            if dict['show_ids']:
                print_parkIDs(park, points, frame_out)
            index += 1
    park_imgs.clear()
    park_img_ids.clear()
    park_sts.clear()
    cv2.imshow('frame', frame_out)
    # write the output frames
    if dict['save_video']:
        #         if video_cur_frame % 35 == 0: # take every 30 frames
        out.write(frame_out)

    # Display video


    #     if video_cur_frame < 256:
    cv2.imwrite('./videos/frameset/%d.jpg' % video_cur_frame, frame_out)
    #     else:
    #       break

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        cv2.imwrite('%d.jpg' % video_cur_frame, frame_out)
    elif k == ord('j'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 1000)  # jump 1000 frames
    elif k == ord('u'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 500)  # jump 500 frames
    if cv2.waitKey(33) == 27:
        break

cv2.waitKey(0)
cap.release()
if dict['save_video']: out.release()
cv2.destroyAllWindows()





import glob
import os

ROOT_DIR = os.getcwd()
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "frameset")
images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
# Sort the images by integer index
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

outvid = os.path.join(VIDEO_DIR, "out.mp4")

# print(images)
# for image in images:
#   img = cv2.imread(image)
#   print(img)


def make_video(outvid, images=images, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

# Directory of images to run detection on
make_video(outvid, images, fps=30)

# !mv frameset/ ./videos/