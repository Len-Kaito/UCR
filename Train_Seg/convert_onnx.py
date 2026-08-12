import torch
from prepare_dataloaders import IMG_W, IMG_H
from train import UNetSmall  # <-- import trực tiếp lớp UNetSmall

# ========= Cấu hình =========
PTH_PATH   = r"C:\1. UIT car racing bang chuyen nghiep\Train_Seg\_checkpoints\lane_segment_best.pth"
ONNX_PATH  = r"C:\1. UIT car racing bang chuyen nghiep\Train_Seg\_checkpoints\model.onnx"
DEVICE     = "cuda"  # hoặc "cpu"
OPSET      = 11
# ============================

def convert(pth_path, onnx_path, device=DEVICE, opset=OPSET):
    # 1) Khởi tạo ĐÚNG kiến trúc đã train
    model = UNetSmall(in_ch=3, out_ch=1, base=16)  # <-- base=16 như lúc train
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 2) Dummy input đúng kích thước train (HxW từ prepare_dataloaders)
    dummy = torch.randn(1, 3, IMG_H, IMG_W).to(device)

    # 3) Export ONNX
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=opset, export_params=True, do_constant_folding=True
    )
    print(f"[OK] Exported {onnx_path} (input 1x3x{IMG_H}x{IMG_W}, opset={opset})")

if __name__ == "__main__":
    convert(PTH_PATH, ONNX_PATH)
