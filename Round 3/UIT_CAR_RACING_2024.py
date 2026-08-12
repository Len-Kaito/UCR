from lib.model.trt_seg import TRT, Run, gstreamer_pipeline
import cv2
import time
import numpy as np
from lib.control.UITCar import UITCar
import torch
import yaml
import time
import math
from lib.model.trt import trt_model
from lib.utils.utils import letterbox, non_max_suppression, scale_boxes, detect, gstreamer_pipeline
from lib.utils.plots import box_label, Colors


def detect(model, im, device):

    # Preprocessing
    im = np.stack([letterbox(im, (320, 320), stride=32, auto=False)[0]])  # resize
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.half()
    im /= 255  # 0 - 255 to 0.0 - 1.0

    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Prediction
    det = model(im)
    det = non_max_suppression(det, conf_thres=0.5, iou_thres=0.45, classes=None, agnostic=True, max_det=1000)

    return det, im

with open('lib/cfg/cfg.yaml', encoding='ascii', errors='ignore') as f:
    cfg = yaml.safe_load(f)

names = ["DownHill", "GreenLight", "TurnLeft", "NoTurnRight", "NoStraight", "Parking", "RedLight", "Straight", "UpHill", "YellowLight"]


def run_yolo(model, im, device):
    # im = cv2.imread(im)
    imc = np.copy(im)
    # st = time.time()
    pred, im = detect(model, im, device)
    # end = time.time()
    # print(f"Inference time: {end-st} with {round(1/(end-st))} fps")

    colors = Colors()
    classes = None
    for _, det in enumerate(pred):  # per image                       
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], imc.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                classes = names[c]
                imc = box_label(imc, xyxy, label, color=colors(c, True))

    return imc, classes

last_time = time.time()
st = time.time()
angle_2 =0
CHECKPOINT = 70
LANEWIGHT = 55            # Độ rộng đường (pixel)
IMAGESHAPE = [320, 184]
tmp =False
temp =False
temp_1 =False
temp_2 =False
temp_3 =False
in_turn = False
ready_turn = None
turn_start_time = 0
start_time_2 = 0
start_time_3 = 0
start_time_4 = 0
wait = 0
cnt = 0
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def calculate(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def check_nga_tu(checkpoint, checkpoint2, image):
    h, w = image.shape

    line_row = image[checkpoint, :]
    center = image[checkpoint2, :]
    
    flag = True
    center_min_x = 0
    center_max_x = 0
    
    for x, y in enumerate(center):
if y == 255 and flag:
            flag = False
            center_min_x = x
        elif y == 255:
            center_max_x = x
    
    flag = True
    min_x = 0
    max_x = 0
    
    for x, y in enumerate(line_row):
        if y == 255 and flag:
            flag = False
            min_x = x
        elif y == 255:
            max_x = x
    if center_min_x == 0 and min_x == 0 and center_max_x>=200 and center_max_x<250 and max_x >= 200 and max_x < 250:
        return True
    else:
        return False
        

def Mid_of_road(CHECKPOINT, image):
    h, w = image.shape
    _lineRow = image[CHECKPOINT, :] 
    count = 0
    sumCenter = 0
    centerArg = int(IMAGESHAPE[0]/2)
    minx=0
    maxx=0
    first_flag=True
    for x, y in enumerate(_lineRow):
        if y == 255 and first_flag:
            first_flag=False
            minx=x
        elif y == 255:
            maxx=x
        
    # centerArg = int(sumCenter/count)
    centerArg=int((minx+maxx+50)//2)
    count=maxx-minx

    # print(minx,maxx,centerArg)
    # print(centerArg, count)

    if (count < LANEWIGHT):
        if (centerArg < int(IMAGESHAPE[0]/2)):
            centerArg -= LANEWIGHT - count
        else:
            centerArg += LANEWIGHT - count
    print(f"minx {minx}  center {centerArg} maxx {maxx}")
    # image=cv2.line(image,(centerArg,CHECKPOINT),(int(IMAGESHAPE[0]/2),IMAGESHAPE[1]),(255,0,0),1)
    return centerArg
    

def AngCal(image, dt):
    lt=time.time()
    global tmp
    global st
    global angle_2
    global turn_start_time
    global in_turn
    global temp_1
    global temp_2
    global temp_3
    if abs(lt - st) < 4.5:
        return angle_2
    tmp = False
    pid = PIDController(kp=0.8, ki=0.05, kd=0.00188)
    h, w = image.shape
    mid = Mid_of_road(130, image)

    angle = pid.calculate(mid - int(w/2), dt)
    print(f"angle {angle}")
    if angle > 60:
        angle = 60
        if angle > 70:    
            tmp = True
            angle_2= 80
            st = time.time()
    elif angle < -60:
        angle = -40
        if angle < -70:
            tmp = True
            st = time.time()
            angle_2 =-80
    elif angle in range(-5,5):
        angle = 0
    return angle

def Angle(image, dt, classes):
    print(classes)
    global in_turn
    global ready_turn
    global temp_1
    global temp_2
    global temp_3
    global turn_start_time
    global start_time_2
    global start_time_3
    global start_time_4
    global wait
    global cnt
    speed = 15
    angle = 0
    
    if classes =='UpHill':
        angle = AngCal(image, dt)
        return (angle, 45)
    elif classes =='DownHill':
        angle = AngCal(image, dt)
        return (-7, 20)
    elif classes =='Parking':
        start_time_2 =time.time()
        angle = AngCal(image, dt)
        temp_1 == True
    elif classes == None and temp_1 == True:
        lt = time.time() - start_time_2
        if lt > 2:
            return (0, 0)
        return (0 ,15)
elif classes in ["NoStraight", "NoTurnRight", "TurnLeft"]:
        print(classes)
        temp_2 = True
        wait = time.time()
        turn_start_time = time.time()
        angle = AngCal(image, dt)
    elif temp_2 == True and classes == None:
        print(classes)
        if time.time() - wait >= 2:
            angle = -80
            elapsed_time = time.time() - turn_start_time
            if elapsed_time >= 5.4:
                temp_2 = False
                angle = 0
        else:
            angle = AngCal(image, dt)
            return (angle, 15)
    else:
        angle = AngCal(image, dt)
    print(angle)
    return (angle, speed)
if __name__ == "__main__":
    car = UITCar()
    car.setMotorMode(0)


    #Load model seg
    model = TRT("weights/model_AU_6.engine")

    device = 'cuda:0'
    yolo_model = trt_model('weights/obj.engine')
    yolo_model = yolo_model.half()
    print("FINISHED LOADING MODEL")
    yolo_model.warmup(imgsz=(1, 3, 320, 320))  # warmup
    print("FINISHED WARMING UP")

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    assert cap.isOpened(), "Camera failed"

    car.OLED_Print("Seg-model: done", 2)

    st = time.time()
    els = time.time()
    bien_hieu = None

    while True:
        _, frame = cap.read()
        if time.time() - els >= 2:
            frame = cv2.resize(frame, (320, 320))
            img, bien_hieu = run_yolo(yolo_model, frame, device)
            cv2.imwrite('./obj.jpg', img)
            els = time.time()
        if bien_hieu == 'GreenLight':
            print("BURRRRRR")
            break


    start = time.time()
    while True:
        current = time.time()
        dt = current - last_time
        last_time = current
        _, frame = cap.read()
        # print(frame)
        yolo_image, segment_image = Run(model, frame)
        #print(1/(time.time()-start))
        cv2.imwrite('./test_cam.jpg', segment_image)

        if time.time() - els >= 2:
            yolo_image = cv2.resize(yolo_image, (320, 320))
            img, bien_hieu = run_yolo(yolo_model, yolo_image, device)
            cv2.imwrite('./obj.jpg', img)
            els = time.time()

        print(bien_hieu)

        last = time.time() - start
        if last < 11:
            angle = -7.4
            speed = 15
        else:
            angle,speed = Angle(segment_image, dt, bien_hieu)
        car.OLED_Print(f"Angle PID: {angle}", 3)
        print(angle)
        car.setAngle((-1)*(angle))      
        car.setSpeed_cm(speed)
        
        
        """ angel khi bắt được biển báo nên là 0"""