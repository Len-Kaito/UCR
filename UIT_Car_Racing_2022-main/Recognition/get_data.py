
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
import pycuda.autoinit  
from utils.yolo_classes import get_cls_dict, COCO_CLASSES_LIST

from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
# from utils.control import road_lines
from imutils.video import VideoStream
import time
from UITCar import UITCar

Car = UITCar()

def gstreamer_pipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor_mode=0 !"
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

i = 9700
cam = VideoStream(src=gstreamer_pipeline()).start()   

while True:
    if Car.button == 1:
        img = cam.read()                    #đọc ảnh từ camera
        if img is None:
            print("None")
            break
        if i % 20 == 0:
            cv2.imwrite(f"data/a_{i}.png", img)
        cv2.imshow("img", img)
        i+=1
        cv2.waitKey(1)
    else:
        print("Done Save")
        pass