import os, random, math, json
from glob import glob
from typing import List, Tuple
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ===================== #
# Config
# ===================== #
TRAIN_SOURCES = [
    r"C:\1. UIT car racing bang chuyen nghiep\dataset\dataset_seg\source",        # luôn có
    r"C:\1. UIT car racing bang chuyen nghiep\dataset\dataset_seg\aug",    # nếu chưa tạo augment thì cứ để vẫn ổn (tool sẽ bỏ qua nếu không tồn tại)
]
VAL_SOURCE = r"C:\1. UIT car racing bang chuyen nghiep\dataset\dataset_seg\source"   # chỉ validate bằng dữ liệu gốc

AUG_WEIGHT = 0.3 
DATASET_DIR = r"C:\1. UIT car racing bang chuyen nghiep\dataset\dataset_seg\source"                 # chứa images/, masks/, label_classes.json
IMAGES_DIR  = os.path.join(DATASET_DIR, "images")
MASKS_DIR   = os.path.join(DATASET_DIR, "masks")
LABEL_JSON  = os.path.join(DATASET_DIR, "label_classes.json")

IMG_W, IMG_H = 320, 180                 # giữ 16:9 (đổi nếu cần)
TRAIN_RATIO  = 0.8
RANDOM_SEED  = 24
BATCH_SIZE   = 8
NUM_WORKERS  = 2 
SHUFFLE_TRAIN = True
PREVIEW_DIR  = r"C:\1. UIT car racing bang chuyen nghiep\dataset\dataset_seg\preview"             # lưu ảnh kiểm tra

# Chuẩn hoá ImageNet (dùng nếu encoder pretrain)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

os.makedirs(PREVIEW_DIR, exist_ok=True)

def list_pairs_from_root(root_dir: str):
    images_dir = os.path.join(root_dir, "images")
    masks_dir  = os.path.join(root_dir, "masks")
    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        return []
    return list_pairs(images_dir, masks_dir)


def load_label_palette():
    """Đọc label_classes.json (nếu có) để biết chỉ số màu PNG cho lớp 'Road'.
       Không bắt buộc; ta vẫn binarize mask > 0 sau khi đọc.
    """
    if not os.path.isfile(LABEL_JSON):
        return None
    try:
        with open(LABEL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def list_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    """Ghép cặp file ảnh/mask theo cùng tên (đuôi .png/.jpg đều hỗ trợ)."""
    exts = ["*.png", "*.jpg", "*.jpeg"]
    img_files = []
    for e in exts:
        img_files.extend(glob(os.path.join(images_dir, e)))
    img_files = sorted(img_files)

    pairs = []
    for img_path in img_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        # Thử các phần mở rộng cho mask theo cùng basename
        mpath = None
        for e in exts:
            cand = os.path.join(masks_dir, f"{name}{e[1:]}")
            if os.path.isfile(cand):
                mpath = cand
                break
        if mpath is None:
            # fallback: tìm trực tiếp trong masks theo mọi ext
            cands = []
            for e in exts:
                cands.extend(glob(os.path.join(masks_dir, f"{name}{e[1:]}")))
            if cands:
                mpath = cands[0]
        if mpath is not None:
            pairs.append((img_path, mpath))

    return pairs


class LaneSegDataset(Dataset):
    def __init__(self, pairs, img_size=(IMG_W, IMG_H), augment=False, normalize=True):
        self.pairs = pairs
        self.w, self.h = img_size
        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)

    def _to_tensor(self, img_np):
        # img_np: HxWxC in [0,1]
        img_np = img_np.astype(np.float32)
        # Normalize if requested
        if self.normalize:
            img_np = (img_np - np.array(NORM_MEAN, dtype=np.float32)) / np.array(NORM_STD, dtype=np.float32)
        # HWC -> CHW
        img_np = np.transpose(img_np, (2, 0, 1))
        return torch.from_numpy(img_np)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image (RGB) & mask (L/gray)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize (bilinear cho ảnh, nearest cho mask)
        img = img.resize((self.w, self.h), Image.BILINEAR)
        mask = mask.resize((self.w, self.h), Image.NEAREST)

        # Augmentation đơn giản (train only)
        if self.augment:
            # Horizontal flip 50%
            if random.random() < 0.5:
                img  = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # Brightness jitter ảnh (mask không đổi)
            if random.random() < 0.3:
                img = ImageEnhance.Brightness(img).enhance(0.7 + 0.6 * random.random())
            if random.random() < 0.3:
                img = ImageEnhance.Contrast(img).enhance(0.7 + 0.6 * random.random())

        # To numpy
        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.uint8)

        # Binarize mask: >0 -> 1
        mask_bin = (mask_np > 0).astype(np.float32)  # HxW in {0,1}

        # To tensor
        img_t = self._to_tensor(img_np)                      # 3xHxW
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0)     # 1xHxW

        sample = {
            "image": img_t,
            "mask": mask_t,
            "image_path": img_path,
            "mask_path": mask_path
        }
        return sample


def split_train_val(pairs, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(math.floor(train_ratio * n))
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]
    return train_pairs, val_pairs


def save_preview(sample, idx, prefix="train"):
    """Lưu ảnh gốc, mask và overlay để kiểm tra nhanh."""
    img = sample["image"]
    mask = sample["mask"]
    # Tensor -> numpy (de-normalize để xem)
    img_np = img.numpy().transpose(1,2,0)
    # de-normalize
    img_np = (img_np * np.array(NORM_STD) + np.array(NORM_MEAN))
    img_np = np.clip(img_np, 0, 1)
    mask_np = mask.numpy()[0]  # HxW

    # overlay trắng trên vùng mask
    overlay = img_np.copy()
    overlay[mask_np > 0.5] = 1.0

    def to_uint8(x): return (x * 255).astype(np.uint8)

    Image.fromarray(to_uint8(img_np)).save(os.path.join(PREVIEW_DIR, f"{prefix}_{idx:03d}_img.png"))
    Image.fromarray((mask_np*255).astype(np.uint8)).save(os.path.join(PREVIEW_DIR, f"{prefix}_{idx:03d}_mask.png"))
    Image.fromarray(to_uint8(overlay)).save(os.path.join(PREVIEW_DIR, f"{prefix}_{idx:03d}_overlay.png"))


def build_loaders():
    # ===== 1) VAL: chỉ từ VAL_SOURCE (split train/val từ gốc) =====
    base_pairs = list_pairs_from_root(VAL_SOURCE)
    assert len(base_pairs) > 0, f"Không tìm thấy cặp ảnh/mask trong {VAL_SOURCE}/images & {VAL_SOURCE}/masks"
    print(f"[BASE] Found {len(base_pairs)} pairs in '{VAL_SOURCE}'")

    train_base, val_pairs = split_train_val(base_pairs)
    print(f"[SPLIT on BASE] Train_base: {len(train_base)} | Val: {len(val_pairs)}")

    # ===== 2) TRAIN: gộp train_base + toàn bộ (hoặc một phần) từ các TRAIN_SOURCES còn lại =====
    extra_pairs = []
    for src in TRAIN_SOURCES:
        if src == VAL_SOURCE:
            # đã tính ở bước 1; chỉ thêm phần train_base
            continue
        ps = list_pairs_from_root(src)
        if len(ps) == 0:
            print(f"[INFO] Skip source '{src}' (không có hoặc trống).")
            continue
        # optionally downsample augment
        if AUG_WEIGHT < 1.0:
            k = int(len(ps) * AUG_WEIGHT)
            random.Random(RANDOM_SEED).shuffle(ps)
            ps = ps[:max(k, 1)]
        extra_pairs.extend(ps)
        print(f"[EXTRA] Found {len(ps)} pairs from '{src}' (after weight={AUG_WEIGHT})")

    train_pairs = train_base + extra_pairs
    print(f"[TRAIN TOTAL] {len(train_pairs)} (base {len(train_base)} + extra {len(extra_pairs)})")

    # ===== 3) Dataset & Dataloader =====
    ds_train = LaneSegDataset(train_pairs, img_size=(IMG_W, IMG_H), augment=True,  normalize=True)
    ds_val   = LaneSegDataset(val_pairs,   img_size=(IMG_W, IMG_H), augment=False, normalize=True)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    # ===== 4) Sanity check nhanh trên TRAIN =====
    batch = next(iter(dl_train))
    x, y = batch["image"], batch["mask"]
    print(f"[Sanity] train batch image: {x.shape} | mask: {y.shape} | pos_ratio: {(y.numpy()>0.5).mean():.4f}")

    # Lưu preview vài mẫu từ train/val
    for i in range(min(3, len(ds_train))):
        save_preview(ds_train[i], i, prefix="train")
    for i in range(min(2, len(ds_val))):
        save_preview(ds_val[i], i, prefix="val")

    print(f"Saved preview PNGs to: {os.path.abspath(PREVIEW_DIR)}")
    return dl_train, dl_val, ds_train, ds_val



if __name__ == "__main__":
    _ = load_label_palette()  # chỉ để tham khảo; pipeline vẫn nhị phân hóa mask >0
    build_loaders()
