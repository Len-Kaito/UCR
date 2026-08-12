import os, random, argparse
from glob import glob
from pathlib import Path
import numpy as np
import cv2

# pip install albumentations==1.4.6 opencv-python
import albumentations as A

def build_aug_pipeline(img_h=180, img_w=320):
    import albumentations as A
    import cv2

    return A.Compose(
        [
            # --- Photometric only (không biến hình học) ---
            # Độ sáng/tương phản, gamma, màu sắc nhẹ
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.75),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=8, p=0.35),

            # --- Bóng cây (chỉ shadow, KHÔNG vật thể che cam) ---
            # num_shadows_lower/upper phải có đủ để tránh lỗi randint(None,…)
            # shadow_roi chọn toàn ảnh; nếu muốn bóng chủ yếu từ nửa trên, dùng (0,0.0,1,0.6)
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                p=0.60
            ),

            # --- Sun flare nhẹ để mô phỏng hướng ánh sáng, KHÔNG che khuất mạnh ---
            A.RandomSunFlare(
                src_radius=60,
                flare_roi=(0.0, 0.0, 1.0, 0.5),  # chỉ ở nửa trên bầu trời
                angle_lower=0.0,
                p=0.20
            ),

            # (TÙY CHỌN) Blur rất nhẹ để mô phỏng rung camera nhỏ, KHÔNG bắt buộc
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.20),

            # --- Resize cuối về kích thước input của model ---
            A.Resize(height=img_h, width=img_w, interpolation=cv2.INTER_LINEAR),
        ],
        is_check_shapes=False
    )


def binarize_mask(mask):
    # Nhận mask uint8 sau augment; nhị phân hoá chắc chắn về {0,1}
    return (mask > 127).astype(np.uint8)

def process_one(img_path, mask_path, out_img_dir, out_mask_dir, aug, n_variants, keep_original=False):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)   # BGR
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"[WARN] Skip (cannot read): {img_path}")
        return 0

    base = os.path.splitext(os.path.basename(img_path))[0]

    saved = 0

    # Optionally output a standardized (resize-only) original
    if keep_original:
        transformed = aug(image=img, mask=mask)  # vẫn đi qua Resize cuối
        img_o = transformed["image"]
        mask_o = transformed["mask"]
        mask_o = binarize_mask(mask_o)*255
        cv2.imwrite(os.path.join(out_img_dir, f"{base}_orig.png"), img_o)
        cv2.imwrite(os.path.join(out_mask_dir, f"{base}_orig.png"), mask_o)
        saved += 1

    for k in range(n_variants):
        transformed = aug(image=img, mask=mask)
        img_a = transformed["image"]
        mask_a = transformed["mask"]

        # đảm bảo mask nhị phân
        mask_a = binarize_mask(mask_a) * 255

        out_img = os.path.join(out_img_dir, f"{base}_a{k:02d}.png")
        out_msk = os.path.join(out_mask_dir, f"{base}_a{k:02d}.png")
        cv2.imwrite(out_img, img_a)
        cv2.imwrite(out_msk, mask_a)
        saved += 1

    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=r"C:\1. UIT car racing bang chuyen nghiep\dataset\dataset_seg\source", help="thư mục gốc chứa images/, masks/")
    ap.add_argument("--dst", type=str, default=r"C:\1. UIT car racing bang chuyen nghiep\dataset\dataset_seg\aug", help="thư mục xuất images/, masks/")
    ap.add_argument("--variants", type=int, default=2, help="số biến thể/ảnh")
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--img_h", type=int, default=384)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--keep-original", action="store_true", help="xuất thêm bản chuẩn hoá (resize-only)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_img_dir = os.path.join(args.src, "images")
    src_msk_dir = os.path.join(args.src, "masks")
    dst_img_dir = os.path.join(args.dst, "images")
    dst_msk_dir = os.path.join(args.dst, "masks")
    Path(dst_img_dir).mkdir(parents=True, exist_ok=True)
    Path(dst_msk_dir).mkdir(parents=True, exist_ok=True)

    # liệt kê cặp ảnh
    img_paths = sorted(glob(os.path.join(src_img_dir, "*.png")) + glob(os.path.join(src_img_dir, "*.jpg")))
    total_saved = 0

    aug = build_aug_pipeline(img_h=args.img_h, img_w=args.img_w)

    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        # tìm mask cùng tên theo nhiều đuôi
        m = None
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = os.path.join(src_msk_dir, name + ext)
            if os.path.isfile(cand):
                m = cand
                break
        if m is None:
            print(f"[MISS] mask không thấy cho: {img_path}")
            continue

        total_saved += process_one(img_path, m, dst_img_dir, dst_msk_dir, aug, args.variants, keep_original=args.keep_original)

    print(f"[DONE] Tạo augment: {total_saved} files (ảnh + mask) vào: {args.dst}")

if __name__ == "__main__":
    main()
