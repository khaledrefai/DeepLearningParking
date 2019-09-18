"""
 
"""

from threading import Thread
# Root directory of the project
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
from samples.coco import coco
import numpy as np
import os
import cv2
import yaml
import time
import  tensorflow as tf

fn_yaml = "main_park_points.yml"
park_sec_to_wait = 1
show_ids = True
park_laplacian_th = 2.8

# Read YAML data (parking space polygons)
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.load(stream)

parking_contours = []
parking_bounding_rects = []
parking_mask = []
parking_data_motion = []

if parking_data != None:
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
        points_shifted[:,1] = points[:,1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                    color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask==255
        parking_mask.append(mask)


if parking_data != None:
    parking_status = [False]*len(parking_data)
    parking_buffer = [None]*len(parking_data)
# bw = ()



# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
config = InferenceConfig()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=COCO_MODEL_PATH, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

graph = tf.get_default_graph()

def run_classifier(img):
        # im = val_tfms(img)
        # learn.precompute=False # We'll pass in a raw image, not activations
        # log_preds = learn.predict_array(im[None])
        # pred = np.argmax(log_preds, axis=1)
        # image = skimage.io.imread(img)

        results = model.detect([img], verbose=1)
        r = results[0]
        print(r['class_ids'])
        if r['class_ids'].size == 0:
            return False
        if 3 in r['class_ids']:
            return True
        else:
            return False

def print_parkIDs(park, coor_points, frame_rev):
    moments = cv2.moments(coor_points)
    centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
    # putting numbers on marked regions
    cv2.putText(frame_rev, str(park['id']), (centroid[0] + 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] - 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] + 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] - 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

class SaveImage(Thread):
    """
    Class used to store image
    """

    def __init__(self,video_cur_pos,frame_out,frame_gray ,frame=None):
        Thread.__init__(self)
        self.frame = frame
        self.stopped = False
        self.video_cur_pos = video_cur_pos
        self.frame_out=frame_out
        self.frame_gray = frame_gray
        self.counter=0
    def run(self):
         while not self.stopped:
             global graph
             with graph.as_default():
                     for ind, park in enumerate(parking_data):
                         points = np.array(park['points'])
                         rect = parking_bounding_rects[ind]
                         roi_gray = self.frame_gray[rect[1]:(rect[1] + rect[3]),
                                    rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation

                         laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
                         # cv2.imshow('oir', laplacian)
                         points[:, 0] = points[:, 0] - rect[0]  # shift contour to roi
                         points[:, 1] = points[:, 1] - rect[1]
                         delta = np.mean(np.abs(laplacian * parking_mask[ind]))
                         # if(delta<2.5):
                         # print("ind, del", ind, delta)
                         status = delta < park_laplacian_th
                         # If detected a change in parking status, save the current time
                         if status != parking_status[ind] and parking_buffer[ind] == None:
                             parking_buffer[ind] = self.video_cur_pos
                             change_pos = self.video_cur_pos
                         # If status is still different than the one saved and counter is open
                         elif status != parking_status[ind] and parking_buffer[ind] != None:
                             if self.video_cur_pos - parking_buffer[ind] > park_sec_to_wait:
                                 parking_status[ind] = status
                                 parking_buffer[ind] = None
                         # If status is still same and counter is open
                         elif status == parking_status[ind] and parking_buffer[ind] != None:
                             parking_buffer[ind] = None

                     for ind, park in enumerate(parking_data):
                         points = np.array(park['points'])
                         if parking_status[ind]:
                             color = (0, 255, 0)
                             rect = parking_bounding_rects[ind]
                             roi_gray_ov = self.frame[rect[1]:(rect[1] + rect[3]),
                                           rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
                             res = run_classifier(roi_gray_ov)
                             cv2.imwrite(
                                 "test\car_318_{}_id{}_   .jpg".format(res, ind),
                                 roi_gray_ov)
                             if res:
                                 parking_data_motion.append(parking_data[ind])
                                 color = (0, 255, 0)
                         else:
                             color = (0, 0, 255)

                         cv2.drawContours(self.frame_out, [points], contourIdx=-1,
                                          color=color, thickness=2, lineType=cv2.LINE_8)
                         if show_ids:
                             print_parkIDs(park, points, self.frame_out)
                     cv2.imshow("Video", self.frame_out)
                     cv2.imwrite('./videos/frameset/%d.jpg' % self.counter, self.frame_out)
                     self.counter += 1
                     k = cv2.waitKey(1)
                     if k % 256 == 27:
                         # ESC pressed
                         self.stopped = True



    def stop(self):
        self.stopped = True
        
        
        '''    def start(self):
        Thread(target=self.save, args=()).start()
        return self
'''