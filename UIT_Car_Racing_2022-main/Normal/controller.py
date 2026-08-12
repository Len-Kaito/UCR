import time
import numpy as np
import cv2
from UITCar import UITCar

Car = UITCar()

pre_t = time.time()
error_arr = np.zeros(5)

def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove
    
#   Hàm dự đoán làn đường dựa vào ảnh từ camera
def road_lines(image, session, inputname):
	# Crop ảnh lại, lấy phần ảnh có làn đườngs
	image = image[200:, :, :]
	small_img = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
	# cv2.imshow('image',small_img)
	small_img = small_img/255
	small_img = np.array(small_img, dtype=np.float32)
	small_img = small_img[None, :, :, :]
	prediction = session.run(None, {inputname: small_img})
	prediction = np.squeeze(prediction)
	prediction = np.where(prediction < 0.5, 0, 255)
	# prediction = prediction.reshape(small_img.shape[0], small_img.shape[1])
	prediction = prediction.astype(np.uint8)

	return prediction

class Controller:
    def __init__(self, kp, kd):
        self.__kp = kp
        self.__kd = kd
        self.__LANEWIGHT = 70            # Độ rộng đường (pixel)

        self.mode = 0

        Car.setMotorMode(0)

    def findingLane(self, frame, height):
        arr_normal = []
        lineRow = frame[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [frame.shape[1] * 1 // 3, frame.shape[1] * 2 // 3]

        self.minLane = min(arr_normal)
        self.maxLane = max(arr_normal)
        
        center = int((self.minLane + self.maxLane) / 2)
        width=self.maxLane-self.minLane

        if (width < self.__LANEWIGHT):
            if (center < int(frame.shape[1]/2)):
                center -= self.__LANEWIGHT - width
            else :
                center += self.__LANEWIGHT - width

        error = frame.shape[1]//2 - center

        return error

    def __PID(self, error):
        global pre_t
        # global error_arr
        error_arr[1:] = error_arr[0:-1]
        error_arr[0] = error
        P = error*self.__kp
        delta_t = time.time() - pre_t
        pre_t = time.time()
        D = (error-error_arr[1])/delta_t*self.__kd
        angle = P + D
        
        if abs(angle)>=45:
            angle = np.sign(angle)*45

        return int(angle)

    def __show(self, signal, area):
        Car.OLED_Print(f"{signal}     {area}", line = 0, left = 0)
        Car.OLED_Print(f"{self.minLane}", line = 1, left = 0)
        Car.OLED_Print(f"{self.maxLane}", line = 2, left = 0)
        Car.OLED_Print(f"{self.error}", line = 3, left = 0)
    
    def __start_stop(self, sendBack_angle, sendBack_speed):
        if Car.button == 1:
            Car.setAngle(sendBack_angle+5)
            Car.setSpeed_rad(sendBack_speed)
        else:
            Car.setAngle(0)
            Car.setSpeed_rad(0)

    def __call__(self, frame, sendBack_speed, height, signal, area):
        ####### Angle Processing #######
        self.error = self.findingLane(frame, height)
        sendBack_angle = self.__PID(self.error)

        ####### Speed Processing #######
        if self.error >= 20:
            sendBack_speed = 15

        ####### Show Info #######
        self.__show(signal, area)
        print(signal, area)
        ####### Start-Stop #######
        self.__start_stop(sendBack_angle, sendBack_speed)

        return sendBack_angle, sendBack_speed