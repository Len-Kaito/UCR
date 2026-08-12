import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm điều khiển xe (mô phỏng)
def turn_left():
    print("The car is turning left!")

def detect_objects(frame, net, layer_names, classes):
    # Chuyển ảnh thành blob để đưa vào mạng
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Lấy đầu ra của mạng
    outs = net.forward(layer_names)

    # Lọc kết quả: Chỉ lấy các bounding box có độ tin cậy cao
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Giảm ngưỡng confidence xuống 0.3
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    # Áp dụng non-maxima suppression để loại bỏ các bounding box trùng lặp
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)

    # Biến để kiểm tra nếu phát hiện "rẽ trái"
    turn_left_detected = False

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = str(classes[class_id])
            confidence = confidences[i]
            print(label)
            # Vẽ bounding box và hiển thị class
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Kiểm tra nếu phát hiện lớp "rẽ trái"
            if label == "turn_left":  # Giả sử "turn_left" là lớp nhận diện biển báo rẽ trái
                turn_left_detected = True

    # Nếu phát hiện biển báo rẽ trái, thực hiện hành động rẽ trái
    if turn_left_detected:
        print("Detected 'turn_left' class. The car will turn left!")
        turn_left()  # Gọi hàm điều khiển xe

    # Hiển thị ảnh với Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Đọc class labels từ tệp
def load_classes(file):
    with open(file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Tải mô hình yolov4
def load_yolo_model(cfg_file, weights_file):
    net = cv2.dnn.readNet(weights_file, cfg_file)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(net.getUnconnectedOutLayers())
    return net, output_layers

# Main function
if __name__ == "__main__":
    # Đọc class labels (giả sử bạn có file coco.names hoặc một file class tương tự)
    classes = load_classes("obj.names")

    # Tải mô hình yolov4
    net, output_layers = load_yolo_model("yolov4.cfg", "yolov4.weights")

    # Đọc ảnh từ file
    img_path = "1.jpg"  # Đặt đường dẫn đến tệp ảnh của bạn
    image = cv2.imread(img_path)

    if image is not None:
        detect_objects(image, net, output_layers, classes)
    else:
        print("Không thể đọc ảnh từ file. Kiểm tra đường dẫn ảnh!")
