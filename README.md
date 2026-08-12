# UIT Car Racing 2024 — Bảng Chuyên Nghiệp

Source code thi đấu cuộc thi UIT Car Racing 2024, bảng Chuyên nghiệp.

Xe tự hành nhận diện làn đường bằng segmentation, nhận diện biển báo bằng YOLO, điều khiển góc lái và tốc độ bằng PID.

## Cấu trúc thư mục

```
├── Round 1_2/          # Code thi vòng 1 & 2 (chạy trên simulator, dùng segment map có sẵn)
├── Round 3/            # Code thi vòng 3 (chạy trên xe thật với Jetson, dùng TensorRT)
├── Train_Seg/          # Train model segmentation làn đường (UNet nhỏ, export ONNX)
├── Train_yolo/         # Train YOLOv8n nhận diện biển báo giao thông
├── yolo/darknet/       # Darknet framework (tham khảo)
├── socket.py           # Nhận ảnh từ xe qua socket, lưu dataset
└── test_seg.py         # Test segmentation + PID trên xe thật
```

## Cách hoạt động

### Vòng 1 & 2 — Simulator

Xe nhận ảnh segment có sẵn từ simulator (qua thư viện `CEEC_Library`/`client_lib`). Mỗi pixel mang màu tương ứng loại biển báo:

| Màu | Ý nghĩa |
|-----|---------|
| `[255, 21, 180]` | Đường (Road) |
| `[255, 255, 1]` | Rẽ trái |
| `[179, 178, 255]` | Rẽ phải |
| `[128, 128, 128]` | Cấm rẽ phải |
| `[179, 255, 179]` | Cấm rẽ trái |
| `[255, 1, 255]` | Đi thẳng |
| `[89, 89, 89]` | Dừng |

Xe dùng PID bám giữa làn (quét hàng pixel ngang, tìm biên trái/phải đường trắng, lấy trung điểm). Khi gặp ngã tư thì rẽ theo biển báo đã nhận diện trước đó, tốc độ điều chỉnh theo độ cong.

### Vòng 3 — Xe thật (Jetson)

- **Segmentation**: UNet nhỏ (base=16 channels) train trên dataset tự gán nhãn, export ONNX → TensorRT. Input 320×180, output binary mask làn đường.
- **Object Detection**: YOLOv8n train trên 5 class biển báo (`Trai`, `Cam_Trai`, `Cam_Phai`, `Phai`, `Thang`), export TensorRT.
- **Điều khiển**: PID tương tự vòng 1&2. Thêm xử lý đèn xanh/đỏ, lên/xuống dốc, đỗ xe.

Camera lấy ảnh qua GStreamer, model inference trên GPU, điều khiển xe qua thư viện `UITCar`.

## Training

### Segmentation

```bash
cd Train_Seg

# Tạo augmented data (tuỳ chọn)
python generate_dataset.py --src <đường_dẫn_source> --dst <đường_dẫn_aug> --variants 2

# Train
python train.py --epochs 60 --lr 1e-3

# Export ONNX
python convert_onnx.py
```

Loss: BCE + Dice. Early stopping theo IoU trên validation set.

### YOLO

```bash
cd Train_yolo
python train.py
```

Train YOLOv8n, 50 epoch, input 320×320. Cần sửa đường dẫn trong `data.yaml` cho đúng máy.

## Ghi chú

- Dataset và checkpoint không có trong repo này.
- Thư mục `Map/` chứa file giả lập nặng, cũng không được push lên.
- Code vòng 1&2 cần thư viện `CEEC_Library` / `client_lib.so` từ BTC.
- Code vòng 3 cần board Jetson với thư viện `UITCar` và TensorRT.
