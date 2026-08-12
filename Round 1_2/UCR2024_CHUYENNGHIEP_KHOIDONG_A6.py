from CEEC_Library import GetStatus, GetRaw, GetSeg, AVControl ,CloseSocket
import cv2
import numpy as np
import math
import time

last_time = time.time()
MA_MAU = dict()
MA_MAU["Road"] = [255,21,180]
MA_MAU["Thang"] = [255,1,255]
MA_MAU["Trai"] = [255,255,1]
MA_MAU["Phai"] = [179,178,255]
MA_MAU["Cam_Phai"] = [128,128,128]
MA_MAU["Cam_Trai"] = [179, 255, 179]
MA_MAU["Dung"] = [89,89,89]

in_turn =False
temp = False
turn_start_time = 0
turn_start_time_2 = 0
error =0
check_line = False
ready_turn = None
x = 0
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

# def detect_intersection(image):
#     """Phát hiện khi xe đã đến gần giao lộ dựa vào độ rộng của làn đường."""
#     gray = Change_to_gray(image)
#     mid_top = Mid_of_road(130, gray)
#     mid_bottom = Mid_of_road(170, gray)

#     # Nếu sự khác biệt giữa độ rộng ở hai vị trí này lớn, xe đã gần đến giao lộ
#     return abs(mid_top - mid_bottom) > 130

def Change_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray*(255/np.max(gray))).astype(np.uint8)
    return gray.copy()

def detect_segment_action(segment_image, MA_MAU):
    h, w, _ = segment_image.shape

    # Lặp qua một số vùng ảnh đại diện để tăng hiệu suất (ví dụ: 50 pixel ngẫu nhiên)
    for y in np.random.randint(0, h, 50):
        for x in np.random.randint(0, w, 50):
            pixel = segment_image[y, x]

            # So sánh với các mã màu trong MA_MAU
            for action, color in MA_MAU.items():
                if (pixel == color).all():
                    return action  # Trả về hành động nếu phát hiện mã màu

    return 'none'  # Không tìm thấy hành động nào

def Mid_of_road(checkpoint, image):
    global x
    line_row = image[checkpoint, :]
    line = np.where(line_row == 255)[0]
    if len(line) == 0:
        return 0

    min_x = line[0]
    max_x = line[-1]

    if max_x - min_x == 319 and x == 0:
        return int((max_x+min_x + 10)/2)
    return int((max_x+min_x + 50 + x)/2)
 
def Chech_nga_tu(checkpoint, checkpoint2, gray):
    line_row = gray[checkpoint, :]
    line1 = np.where(line_row == MA_MAU["Road"])[0]
    line_row_2 = gray[checkpoint2, :]
    line2 = np.where(line_row_2 == MA_MAU["Road"])[0]
    if len(line1) == 0 or len(line2) == 0:
        return False
    if line1[0] == line2[0] and line1[-1] == line2[-1]:
        return True
    else:
        return False

def AngCal(image, dt):

    pid = PIDController(kp=0.08, ki=0.01, kd=0.0188)

    gray = Change_to_gray(image)
    h, w = gray.shape

    mid = Mid_of_road(140, image)

    angle = pid.calculate(mid - int(w/2), dt)

    if angle > 25:
        angle = 25
    elif angle < -25:
        angle = -25

    return angle

def Angle(image, dt):
    global in_turn
    global turn_start_time
    global turn_start_time_2
    global temp
    global error
    global check_line
    global ready_turn
    global x
    global cnt
    speed = 35
    angle = 0
    action = detect_segment_action(segment_image, MA_MAU)

    if action != 'Road' and action != 'Thang' and action != "Dung":
        ready_turn = action
        temp = True
    elif action == "Dung" or ready_turn == "Dung":
        ready_turn = "Dung"
        speed = 0
    if temp:
        if action == "Trai" or action == "Cam_Phai":
            x = -112
            turn_start_time_2 = time.time()
        elif action =="Phai" or action == "Cam_Trai":
            x = 10
            turn_start_time_2 = time.time()
        print(f"action {ready_turn}")
        check_line = Chech_nga_tu(133,170, segment_image )
        if check_line and not in_turn:
            print("Phát hiện giao lộ! Bắt đầu rẽ...")
            in_turn = True  # Chuyển sang trạng thái rẽ
            turn_start_time = time.time()  # Lưu thời điểm bắt đầu rẽ

            # Quyết định rẽ trái hoặc phải
            if ready_turn == "Trai":
                print("Rẽ trái...")
                error = -25  # Rẽ trái
            elif ready_turn == "Phai":
                print("Rẽ phải...")
                error = 25  # Rẽ phải
            elif ready_turn == "Cam_Phai":
                print("Rẽ trái...")
                error = -25
            elif ready_turn =="Cam_Trai":
                print("Rẽ phải...")
                error = 25
            # Tính góc lái bằng PID cho giao lộ
            angle =AngCal(image, dt)
            speed = 35
            if ready_turn == "Dung":
                speed = 0
        elif in_turn:
            # Nếu đang trong quá trình rẽ, tiếp tục theo dõi
            elapsed_time = time.time() - turn_start_time
            angle = AngCal(image, dt)
            # Nếu đã rẽ xong (sau 2 giây), quay lại bám đường
            if elapsed_time > 1.1: # thoi gian queo
                print("Thoát khỏi giao lộ, quay lại bám đường")
                in_turn = False  # Thoát trạng thái rẽ
                temp = False
                check_line = False
                error = 0
                ready_turn = None
                x = 0
            elif elapsed_time > 0.472: # thoi gian di thang
                angle = error 
            # Tiếp tục rẽ nếu chưa xong
        else:
            # Nếu không gặp giao lộ, dùng PID làn đường
            angle = AngCal(image, dt)
            if ready_turn == "Cam_Phai" or ready_turn == "Trai":
                elapsed_time_2 = time.time() - turn_start_time_2
                if elapsed_time_2 > 7: # thoi gian queo
                    # print("a")
                    ready_turn = None
                    x = 0

            # Điều chỉnh tốc độ tùy thuộc vào góc lái
            if ready_turn != "Dung":
                if abs(angle) < 0.2:
                    cnt+=1
                    speed = 50
                    if cnt > 10:
                        speed = 35
                elif abs(angle) < 0.7:
                    cnt+=1
                    speed = 45 
                    if cnt > 10:
                        speed = 20
                elif abs(angle) < 1:
                    cnt+=1
                    speed = 40
                    if cnt > 10:
                        speed = 20
                elif abs(angle) < 3:
                    speed = 35
                elif abs(angle) < 5:
                    speed = 33 # Đường cong nhẹ
                elif abs(angle) < 15:
                    speed = 30
                else:
                    cnt = 0
                    speed = 25
    else:
        angle = AngCal(image, dt)
        if ready_turn == "Cam_Phai" or ready_turn == "Trai":
            elapsed_time_2 = time.time() - turn_start_time_2
            if elapsed_time_2 > 7: # thoi gian queo
                # print("a")
                ready_turn = None
                x = 0

        # Điều chỉnh tốc độ tùy thuộc vào góc lái
        if ready_turn != "Dung":
            if abs(angle) < 0.2:
                cnt+=1
                speed = 50
                if cnt > 10:
                    speed = 20
            elif abs(angle) < 0.7:
                cnt+=1
                speed = 45  # Đường thẳng, tăng tốc
                if cnt > 10:
                    speed = 20
            elif abs(angle) < 1:
                cnt+=1
                speed = 40
                if cnt > 10:
                    speed = 20
            elif abs(angle) < 3:
                speed = 35
            elif abs(angle) < 5:
                speed = 33 # Đường cong nhẹ
            elif abs(angle) < 15:
                speed = 30
            else:
                cnt = 0
                speed = 25
    return (angle, speed)


if __name__ == "__main__":
    try:
        angle = 0
        speed = 35
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            print(state)
            # print("Let's go!!!")
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            cv2.imshow("Raw", raw_image)
            cv2.imshow("Segment", segment_image)

            angle, speed = Angle(segment_image, dt)
            # print(f"speed = {speed}")
            if ready_turn == "Trai" or ready_turn == "Phai" or ready_turn == "Cam_Trai" or ready_turn == "Cam_Phai":
                if abs(angle) < 2:
                    speed = 35
                elif abs(angle) < 15:
                    speed = 30
                else:
                    speed = 25
            if x == -112 and ready_turn != "Dung":
                speed = 25
            AVControl(speed, angle) # maxspeed = 90, max steering angle = 25

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        CloseSocket()
