import os, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image

# ====== lấy loader & cấu hình ảnh từ file bước 1 ======
from prepare_dataloaders import build_loaders, IMG_W, IMG_H

# ====== (khuyến nghị) dùng segmentation_models_pytorch ======
# Fallback: nếu không có SMP, sẽ dùng UNet tự code rất nhỏ gọn
# def make_model(backbone="efficientnet_b0", in_ch=3, out_ch=1):
#     try:
#         import segmentation_models_pytorch as smp
#         model = smp.Unet(
#             encoder_name=backbone,      # "resnet18", "resnet34", "mobilenet_v2", ...
#             encoder_weights="imagenet", # giúp hội tụ nhanh với data nhỏ
#             in_channels=in_ch,
#             classes=out_ch
#         )
#         return model, True
#     except Exception as e:
#         print(f"[WARN] Không import được segmentation_models_pytorch ({e}). Dùng UNet đơn giản fallback.")
#         return UNetSmall(in_ch, out_ch), False

def make_model(backbone=None, in_ch=3, out_ch=1, base=16):
    print(f"[INFO] Using UNetSmall backbone (base={base})")
    return UNetSmall(in_ch=in_ch, out_ch=out_ch, base=base), False


# ====== UNet nhỏ (fallback) ======
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad nếu lệch kích thước
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.up1 = Up(base*8, base*4)
        self.up2 = Up(base*4, base*2)
        self.up3 = Up(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)
        return self.outc(x)

# ====== Loss & Metrics ======
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        # logits: (B,1,H,W), targets: (B,1,H,W) in {0,1}
        probs = torch.sigmoid(logits)
        num = 2.0 * torch.sum(probs * targets)
        den = torch.sum(probs + targets) + self.eps
        return 1.0 - (num / den)

def binary_iou(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets) - inter + eps
    return (inter + eps) / union

def binary_dice(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = torch.sum(preds * targets)
    return (2*inter + eps) / (torch.sum(preds) + torch.sum(targets) + eps)

# ====== Train / Val loops ======
def train_one_epoch(model, loader, optimizer, bce, dice, device, scaler=None):
    from torch import amp as _amp
    use_amp = scaler is not None
    model.train()
    tot_bce, tot_dice = 0.0, 0.0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with _amp.autocast('cuda'):
                logits = model(x)
                loss = bce(logits, y) + dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = bce(logits, y) + dice(logits, y)
            loss.backward()
            optimizer.step()

        # log train stats (tính lại trên CPU để tránh lẫn với graph)
        with torch.no_grad():
            tot_bce += bce(logits, y).item()
            tot_dice += (1.0 - dice(logits, y).item())

    n = len(loader)
    return tot_bce / n, tot_dice / n


@torch.no_grad()
def validate(model, loader, bce, dice, device):
    model.eval()
    tot_loss, tot_iou, tot_dice = 0.0, 0.0, 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        loss = bce(logits, y) + dice(logits, y)
        tot_loss += loss.item()
        tot_iou  += binary_iou(logits, y).item()
        tot_dice += binary_dice(logits, y).item()
    n = len(loader)
    return tot_loss/n, tot_iou/n, tot_dice/n

# ====== Lưu dự đoán mẫu ======
@torch.no_grad()
def save_val_previews(model, loader, out_dir, device, max_samples=8):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        for i in range(x.size(0)):
            if saved >= max_samples: return
            img = x[i].cpu().numpy().transpose(1,2,0)
            # de-normalize theo ImageNet
            mean = np.array([0.485,0.456,0.406], dtype=np.float32)
            std  = np.array([0.229,0.224,0.225], dtype=np.float32)
            img = (img * std + mean).clip(0,1)

            gt  = y[i,0].cpu().numpy()
            pr  = preds[i,0].cpu().numpy()

            # overlay prediction (trắng)
            overlay = img.copy()
            overlay[pr > 0.5] = 1.0

            def to_u8(a): return (a*255).astype(np.uint8)
            Image.fromarray(to_u8(img)).save(f"{out_dir}/img_{saved:02d}.png")
            Image.fromarray((gt*255).astype(np.uint8)).save(f"{out_dir}/gt_{saved:02d}.png")
            Image.fromarray((pr*255).astype(np.uint8)).save(f"{out_dir}/pred_{saved:02d}.png")
            Image.fromarray(to_u8(overlay)).save(f"{out_dir}/overlay_{saved:02d}.png")
            saved += 1

# ====== Export ONNX ======
def export_onnx(model, onnx_path, h=IMG_H, w=IMG_W, device="cuda", opset=9):
    model.eval()
    dummy = torch.randn(1, 3, h, w).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=opset, export_params=True, do_constant_folding=True
    )
    print(f"[OK] Exported ONNX -> {onnx_path} (opset {opset}, input 1x3x{h}x{w})")

# ====== Main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", type=str, default="efficientnet_b0",
                    help="resnet18,resnet34,mobilenet_v2,... (nếu SMP khả dụng)")
    ap.add_argument("--batch-accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--no-amp", action="store_true", help="tắt mixed precision")
    ap.add_argument("--save-dir", type=str, default="./_checkpoints")
    ap.add_argument("--save-name", type=str, default="lane_segment_best.pth")
    ap.add_argument("--export-onnx", action="store_true", help="xuất ONNX sau khi train")
    ap.add_argument("--onnx-path", type=str, default="./lane_segment.onnx")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # loaders từ bước 1
    dl_train, dl_val, _, _ = build_loaders()

    # model
    model, used_smp = make_model(args.backbone, in_ch=3, out_ch=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    bce  = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    use_amp = (device.type == "cuda" and not args.no_amp)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_iou = -1.0
    best_path = os.path.join(args.save_dir, args.save_name)
    patience, best_epoch, no_improve = 5, 0, 0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_bce, tr_dice = train_one_epoch(model, dl_train, optimizer, bce, dice, device, scaler)
        val_loss, val_iou, val_dice = validate(model, dl_val, bce, dice, device)
        scheduler.step(val_iou)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train BCE: {tr_bce:.4f} | Train Dice: {tr_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | {dt:.1f}s")

        # save best theo IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] Saved to {best_path} (IoU={best_iou:.4f})")
            # lưu vài dự đoán mẫu mỗi khi có best
            save_val_previews(model, dl_val, "./_val_preds", device, max_samples=8)
            best_epoch, no_improve = epoch, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}, best at {best_epoch}")
                break
    print(f"Training done. Best IoU={best_iou:.4f}. Best weights: {best_path}")

    # export onnx nếu bật cờ
    # if args.export_onnx:
    #     # load best rồi export để chắc chắn
    #     model.load_state_dict(torch.load(best_path, map_location=device))
    #     model.to(device).eval()
    #     export_onnx(model, args.onnx_path, h=IMG_H, w=IMG_W, device=device, opset=9)

if __name__ == "__main__":
    main()
