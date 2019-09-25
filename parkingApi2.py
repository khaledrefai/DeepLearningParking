from flask import jsonify, render_template, send_from_directory
from flask import request
from flask import Flask,Response
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
import tempfile
from os import path
import logging
import tensorflow as tf




app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
min_threshold = 0.5
graph = tf.get_default_graph()
# Root directory of the project
ROOT_DIR = os.path.abspath("../")




@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def Motion():
    return render_template('index.html')


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/images/<path:path>')
def send_gfx(path):
    return send_from_directory('images', path)

@app.route('/js/<path:path>')
def send_t(path):
    return send_from_directory('js', path)

@app.route('/sass/<path:path>')
def send_sass(path):
    return send_from_directory('sass', path)

@app.route('/<path:path>')
def get_file(path):
    return send_from_directory('', path)



@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "not ok", "message": "this server could not understand your request"})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "not found", "message": "route not found"})


@app.errorhandler(500)
def not_found(e):
    return jsonify({"status": "internal error", "message": "internal error occurred in server"})



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
        'park_sec_to_wait': 4, # 4   wait time for changing the status of a region
        'start_frame': 1, # begin frame from specific frame number
        'show_ids': True, # shows id on each region
        'classifier_used': True,
        'save_video': True
        }


fn = "camer1.webm"  # 3
cap = cv2.VideoCapture(fn)
# Set from video

video_info = {  'fps':    cap.get(cv2.CAP_PROP_FPS),
                'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.6),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.6),
                'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

cap.set(cv2.CAP_PROP_POS_FRAMES, dict['start_frame']) # jump to frame number specified


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_cars.h5"
frame_out =None


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    #BACKBONE = "resnet50"
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
    imgscopy = imgs.copy()
    print(len(imgscopy))
    results = model.detect(imgscopy, verbose=1)
    park_sts=[]
    imgs.clear()
    park_imgs.clear()
    for res in results:
        print(res['class_ids'])
        if 1 in  res['class_ids'] or res['class_ids'].size ==0:
            park_sts.append(False)
        else:
            park_sts.append(True)
    return park_sts



# Read YAML data (parking space polygons)
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.load(stream)

parking_contours = []
parking_bounding_rects = []
parking_mask = []
parking_data_motion = []
park_imgs =[]
park_img_ids=[]
parking_yml_box=[]
if parking_data != None:
    for park in parking_data:

        points = np.array(park['points'])
        rect = cv2.boundingRect(points)


        points_shifted = points.copy()
        points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
        points_shifted[:,1] = points[:,1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        parking_yml_box.append([rect[1] -30,rect[0],rect[1] +rect[2]-20 ,rect[3]+rect[0] +20 ])
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


parking_yml_box = np.array(parking_yml_box)
# Filter to only cars
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        print(class_ids[i])
        if class_ids[i] in [1]:
            car_boxes.append(box)

    return np.array(car_boxes)

temp = np.array(4,)
# spotted parking spaces

parked_car_boxes1 = [None] * 11

def checkEqual2(iterator):
   print(iterator)
#     return len(set(iterator)) <= 1


def gen():
    free_space_frames = 0
    count = 0
    parked_car_boxes = None
    greenpoints=[]
    redpoints=[]
    last_pos=0
    global graph
    with graph.as_default():
        while (cap.isOpened()):
            video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ret, frame_initial = cap.read()
            if not ret:
                 break
            #frame_initial = frame_initial[:, :, ::-1]
            frame = cv2.resize(frame_initial, None, fx=0.6, fy=0.6)
            if video_cur_pos - last_pos> dict['park_sec_to_wait'] or last_pos==0:
                last_pos=video_cur_pos
                results = model.detect([frame], verbose=0)
                greenpoints.clear()
                redpoints.clear()
                # Mask R-CNN assumes we are running detection on multiple images.
                # We only passed in one image to detect, so only grab the first result.
                r = results[0]

                # The r variable will now have the results of detection:
                # - r['rois'] are the bounding box of each detected object
                # - r['class_ids'] are the class id (type) of each detected object
                # - r['scores'] are the confidence scores for each detection
                # - r['masks'] are the object masks for each detected object (which gives you the object outline)
                #if parked_car_boxes is None:
                    # This is the first frame of video - assume all the cars detected are in parking spaces.
                    # Save the location of each car as a parking space box and go to the next frame of video.
                #    parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])

                #else:
                    # Get where cars are currently located in the frame
                car_boxes = get_car_boxes(r['rois'], r['class_ids'])


                # See how much those cars overlap with the known parking spaces
                overlaps = utils.compute_overlaps(parking_yml_box, car_boxes)

                # Assume no spaces are free until we find one that is free
                free_space = False

                # Loop through each known parking space box
                for parking_area, overlap_areas,park in zip(parking_yml_box, overlaps,parking_data):

                    # For this parking space, find the max amount it was covered by any
                    # car that was detected in our image (doesn't really matter which car)
                    max_IoU_overlap = np.max(overlap_areas)

                    # Get the top-left and bottom-right coordinates of the parking area
                    y1, x1, y2, x2 = parking_area
                    points = np.array(park['points'])
                    ind = park['id']
                    # Check if the parking space is occupied by seeing if any car overlaps
                    # it by more than 0.15 using IoU
                    if max_IoU_overlap < 0.20 : #and  ind not in (11,18,19,26)
                        # Parking space not occupied! Draw a green box around it
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.drawContours(frame, [points], contourIdx=-1,
                                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)
                        greenpoints.append(points)
                        # Flag that we have seen at least one open space
                        free_space = True
                        count+=1
                    else:
                        # Parking space is still occupied - draw a red box around it
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.drawContours(frame, [points], contourIdx=-1,
                                         color=(0, 0, 255), thickness=1, lineType=cv2.LINE_8)
                        redpoints.append(points)
                    #print_parkIDs(park, points, frame)
                    # Write the IoU measurement inside the box
                    font = cv2.FONT_HERSHEY_DUPLEX
                    #cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))
                    #cv2.putText(frame, 'Total vacant ' + str(count-4) ,(5, 30),
                    #           cv2.FONT_HERSHEY_DUPLEX,
                    #            0.8, (255, 0, 0), 1, cv2.LINE_8)

                    # If at least one space was free, start counting frames
                    # This is so we don't alert based on one frame of a spot being open.
                    # This helps prevent the script triggered on one bad detection.
                    if free_space:
                        free_space_frames += 1
                    else:
                        # If no spots are free, reset the count
                        free_space_frames = 0

                        # If a space has been free for several frames, we are pretty sure it is really free!
                    if free_space_frames > 100:
                        # Write SPACE AVAILABLE!! at the top of the screen
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

            else:
                cv2.drawContours(frame, greenpoints, contourIdx=-1,
                                 color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)
                cv2.drawContours(frame, redpoints, contourIdx=-1,
                                 color=(0, 0, 255), thickness=1, lineType=cv2.LINE_8)

            # Show the frame of video on the screen
            #         cv2.imshow('Video', frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                print("frame is none")







if __name__ == "__main__":
    with app.app_context():
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        print("Starting server on http://localhost:8080")
        print("Serving ...", app.run(host='0.0.0.0', port=8080))
        print("Finished !")
        print("Done !")


