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




app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
min_threshold = 0.5
model_file = 'temp/checkpoints/initial-model.h5'
image_dir = 'tests/images'


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
frame_out =None


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



@app.route('/detect', methods=['POST'])
def postimage():
    global graph
    with graph.as_default():
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            file = request.files.get('upload')
            filename, ext = os.path.splitext(file.filename)
            # if ext not in ('.png', '.jpg', '.jpeg'):
            #    return 'File extension not allowed.'
            tmp = tempfile.TemporaryDirectory()
            temp_storage = path.join(tmp.name, file.filename)
            file.save(temp_storage)
            image = cv2.imread(temp_storage)
            frame_out = frame.copy()
            color = (0, 255, 0)
            rect = parking_bounding_rects[ind]
            roi_gray_ov = image[rect[1]:(rect[1] + rect[3]),
                          rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation

            park_imgs.append(roi_gray_ov)
            park_img_ids.append(ind)

        park_sts = run_classifier(park_imgs)
        index = 0
        while index < len(park_sts):
            print(index)
            id = park_img_ids[index]
            park = parking_data[id]
            points = np.array(park['points'])
            color = (0, 255, 0)
            print("id {} , status {} ".format(id, park_sts[index]))
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
    return "ok"




def gen():
        ret, jpeg = cv2.imencode('.jpg', frame_out)
        if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                print("frame is none")



@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    with app.app_context():
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        print("Starting server on http://localhost:2000")
        print("Serving ...", app.run(host='0.0.0.0', port=2000))
        print("Finished !")
        print("Done !")


