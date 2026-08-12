
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
# sess_options = onnxruntime.SessionOptions()

### Client ###
from controller import Controller, road_lines, remove_small_contours, Car

Control = Controller(1, 0.05)

SSH = False
DETEC = True
SHOW_IMG = True
PRINT = False
session_lane = onnxruntime.InferenceSession( 'weights/lane_mod.onnx', None, providers=['CPUExecutionProvider'])
input_name_lane = session_lane.get_inputs()[0].name
session_sign = onnxruntime.InferenceSession('weights/my_sign.onnx', None, providers=['CUDAExecutionProvider'])
input_name_sign = session_sign.get_inputs()[0].name

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

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    #model yolo mặc định 6 class
    parser.add_argument(
        '-c', '--category_num', type=int, default=5,
        help='number of object categories [6]')
    #chọn model
    parser.add_argument(
        '-m', '--model', type=str, default='yolo_final-416',
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):

    # be tap lam quen :D
    imgs = os.listdir('warnup')
    for img in imgs:
        img = cv2.imread(os.path.join('warnup', img))
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        _ = vis.draw_bboxes(
            img, boxes, confs, clss, session_sign, input_name_sign)
    print("done warnup")
            
    # fps = 0.02
    # start_time_detect=time.time()
    frame = 0

    lst_area = np.zeros(5)
    
    #### Visualize ####
    # result = cv2.VideoWriter('segment.avi', 
    #                      cv2.VideoWriter_fourcc(*'MJPG'),
    #                      15, (640, 360))

    while True:
        tic = time.time()
        img = cam.read()                    #đọc ảnh từ camera
        if img is None:
            print("None")
            break
        # try:        
        copyImage = np.copy(img) 

        #### Segmentation ####           
        DAsegmentation = road_lines(copyImage, session=session_lane, inputname=input_name_lane)    #hàm detect làn đường trả về ảnh đã detect, góc lái, với điểm center dự đoán      
        DAsegmentation = remove_small_contours(DAsegmentation)

        #### Object Detection ####
        # if frame % 2 == 0:
        boxes, confs, clss = trt_yolo.detect(img, conf_th)              #trả về boxes: chứa tọa độ bounding box, phần trăm dự đoán, và phân lớp
        if(len(confs)>0):                                               #nếu nhận diện được biển báo
            #chỉ lấy ra đối tượng có phần trăm dự đoán cao nhất
            index=np.argmax(confs)
            confs = [confs[index]]
            
            #### Class Name ####
            clss = [clss[index]]

        # hàm vẽ bounding box lên ảnh, trả về ảnh đã vẽ, center của các đối tượng, phân lớp, và diện tích của boundingbox
        img, class_TS, area = vis.draw_bboxes(
            img, boxes, confs, clss, session=session_sign, inputname=input_name_sign, frame_id = frame)

        lst_area[1:] = lst_area[0:-1]
        lst_area[0] = area
        ############# Control Car ############# 
        pred = np.copy(DAsegmentation)

        sendBack_angle, sendBack_speed = Control(pred, sendBack_speed=28, height=9, signal=class_TS, area=area)
        print("*******UTE-AI-Sandbox TOP1*******")
        #### Visualize ####
        # result.write(img)
        # cv2.waitKey(1)
        # if Car.button == 2:  # ESC key: quit program
        #     break

        #####------Show hình-----------------------
        # if SHOW_IMG:
            # img[:DAsegmentation.shape[0],img.shape[1]-DAsegmentation.shape[1]:,:]=DAsegmentation
            # cv2.imshow('merge', pred)

        frame += 1
        # toc = time.time()
        # fps = 1.0 / (toc - tic)
        # print(fps)
        # key = cv2.waitKey(1)
        # if key == 27:  # ESC key: quit program
        #     break
            
        # except Exception as inst:
        #     print(type(inst))    # the exception instance
        #     print(inst)
        #     pass
    Car.setAngle(0)
    Car.setSpeed_cm(0)
    
    # del Car
    # result.release()
    # print("The video was successfully saved")

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    print("---------------------------------------------")

    cam = VideoStream(src=gstreamer_pipeline()).start()            #đọc camera
    print("********************************************")
    cls_dict = get_cls_dict(args.category_num)                                  #lấy danh sách tên các lớp   
    print(cls_dict, args.category_num)
    #---------------------------Kiểm tra model có bị lỗi không để lấy kích thước ảnh                            
    yolo_dim = args.model.split('-')[-1]                                        
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
    #---------------------------------------------------------------------------------------
    print("===============================================")
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)                   #load model yolo
    print("++++++++++++++++++Read model done+++++++++++++++++")
    vis = BBoxVisualization(cls_dict)                                           #gọi instance BBoxVisualization với thông số mặc định là danh sách các lớp
    # m=None
    loop_and_detect(cam, trt_yolo, conf_th=0.7, vis=vis)                        #vào vòng lặp để dự đoán liên tục
    
    cam.stream.release()                                                        #release camera
    cam.stop()
    cv2.destroyAllWindows()   
                                                      #tắt hết các windows của images

if __name__ == '__main__':
    main()