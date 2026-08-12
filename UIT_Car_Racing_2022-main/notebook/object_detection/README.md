### Convert YOLO to TensorRT
```python
    cd ../../Recognition/yolo
    python3 yolo_to_onnx.py -m "model name"
    python3 onnx_to_tensorrt.py -m "model name"
```

## Traffic Sign Object Detection

### YoloV4 tiny

[Darknet](https://github.com/AlexeyAB/darknet)

## Data
[Dataset One Class](https://github.com/dotrannhattuong/UIT_Car_Racing_2022/tree/main/dataset/object_detection/data_1class).

## Note: Chỉ training với 1 class Sign