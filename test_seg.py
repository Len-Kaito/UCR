from lib.model.trt_seg import TRT, Run, gstreamer_pipeline
import cv2
import time
import numpy as np
from lib.control.UITCar import UITCar
import math

last_time = time.time()
st = time.time()
angle_2 =0
CHECKPOINT = 70
LANEWIGHT = 55            # Độ rộng đường (pixel)
IMAGESHAPE = [320, 184]
tmp =False
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

def Chech_nga_tu(checkpoint, checkpoint2, image):
    h, w = image.shape

    line_row = image[CHECKPOINT, :]
    center = image[h-5, :]
    
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
    if center_min_x == min_x and center_max_x == max_x:
        print()
        

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
    if tmp == True and lt - st <2.0:
        return angle_2
    tmp = False
    pid = PIDController(kp=0.8, ki=0.05, kd=0.00188)
    h, w = image.shape
    mid = Mid_of_road(125, image)

    angle = pid.calculate(mid - int(w/2), dt)
    print(f"angle {angle}")
    if angle > 40:
        angle = 60
        if angle > 90:    
            tmp = True
            angle_2= 80
            st = time.time()
    elif angle < -40:
        angle = -40
        if angle < -90:
            tmp = True
            st = time.time()
            angle_2 =-80
    elif angle in range(-5,5):
        angle = 0
    return angle

if __name__ == "__main__":
    car = UITCar()
    car.setMotorMode(0)


    #Load model seg
    model = TRT("weights/model_AU_6.engine")
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    assert cap.isOpened(), "Camera failed"

    car.OLED_Print("Seg-model: done", 2)

    st = time.time()

    while True:
        print("-------------",time.time()-st)
        st = time.time()
        current = time.time()
        dt = current - last_time
        last_time = current

        #Khoi dong cam va bat dau lay segment
        start = time.time()
        _, frame = cap.read()
        print("Cam: ",time.time() - start)
        # print(frame)
        start = time.time()
        _, segment_image = Run(model, frame)
        print("Seg: ",time.time() - start)
        #print(1/(time.time()-start))
        # cv2.imwrite('./test_cam.jpg', segment_image)
        start = time.time()
        angle = AngCal(segment_image, dt)
        car.OLED_Print(f"Angle PID: {angle}", 3)
        print("AngCal: ",time.time() - start)
        
        start = time.time()
        car.setAngle((-1)*(angle))
        car.setSpeed_cm(15)
        print("car set: ",time.time() - start)