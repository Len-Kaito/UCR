from ultralytics import YOLO

if __name__ == "__main__":
    # 1️⃣ Khởi tạo model
    model = YOLO("yolov8n.pt")

    # 2️⃣ Đường dẫn file data.yaml
    data_yaml = "./data.yaml"

    # 3️⃣ Train model
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=320,
        batch=8,
        workers=0,          # ❗️Quan trọng: Windows nên đặt =0 để tránh multiprocessing
        name="checkpoint"
    )
