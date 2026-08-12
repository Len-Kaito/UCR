
from PIL import Image
import math

from PIL import Image
import math

import os
import time
import argparse
import numpy as np
import onnxruntime
import sys
import time
import cv2
# import pycuda.autoinit  
from utils.yolo_classes import get_cls_dict, COCO_CLASSES_LIST

from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
# from utils.control import road_lines
from imutils.video import VideoStream
import time
sess_options = onnxruntime.SessionOptions()

### Client ###
# from controller import Controller, road_lines, remove_small_contours

session_sign = onnxruntime.InferenceSession(
    'sign_new.onnx', None, providers=['CUDAExecutionProvider'])
input_name_sign = session_sign.get_inputs()[0].name

img = cv2.imread("classify.png")
img = cv2.resize(img, (30, 30))
img = img.astype('float32')/255
img = img.reshape(1, 30, 30, 3)

pred = session_sign.run(None,{input_name_sign:img})
print(pred)