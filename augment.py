import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import random
from natsort import natsorted
import albumentations as A
import re
import shutil  # Import shutil for file copying

# ==============================================================================
# --- CONFIGURATION VARIABLES ---
# ==============================================================================

# --- Seeding ---
RANDOM_SEED = 0

# --- Paths ---
HIGH_RES_INPUT_DIR = "data/raw/flir_frames7/"
LOW_RES_INPUT_DIR  = "data/raw/webcam_frames7/"

AUGMENTED_BASE_SAVE_PATH = "data/augmented/"

# --- Dataset Splitting ---
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1

# --- Augmentation Settings ---
AUGMENTATIONS_PER_IMAGE      = 3
CHECK_ALBUMENTATIONS_SHAPES  = False

SSR_SHIFT_LIMIT    = 0.08
SSR_SCALE_LIMIT    = 0.0
SSR_ROTATE_LIMIT   = 10
ET_ALPHA           = 0.3
ET_SIGMA           = 5
RBC_BRIGHTNESS_LIMIT = 0.15
RBC_CONTRAST_LIMIT   = 0.15
GN_VAR_LIMIT       = (10.0, 50.0)

# ==============================================================================
# --- SCRIPT START ---
# ==============================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Setup output directories
train_save_path_lr = os.path.join(AUGMENTED_BASE_SAVE_PATH, "train", "low_res")
train_save_path_hr = os.path.join(AUGMENTED_BASE_SAVE_PATH, "train", "high_res")
val_save_path_lr   = os.path.join(AUGMENTED_BASE_SAVE_PATH, "val",   "low_res")
val_save_path_hr   = os.path.join(AUGMENTED_BASE_SAVE_PATH, "val",   "high_res")
test_save_path_lr  = os.path.join(AUGMENTED_BASE_SAVE_PATH, "test",  "low_res")
test_save_path_hr  = os.path.join(AUGMENTED_BASE_SAVE_PATH, "test",  "high_res")

for p in [train_save_path_lr, train_save_path_hr,
          val_save_path_lr,   val_save_path_hr,
          test_save_path_lr,  test_save_path_hr]:
    os.makedirs(p, exist_ok=True)


def get_image_pairs(lr_dir, hr_dir):
    lr_files = natsorted(glob.glob(os.path.join(lr_dir, "frame_*.png")))
    hr_files = natsorted(glob.glob(os.path.join(hr_dir, "frame_*.png")))
    pairs = []

    for lr_path in lr_files:
        base = re.match(r"(frame_\d+)", os.path.basename(lr_path))
        if not base:
            continue
        base = base.group(1)
        for hr_path in hr_files:
            name = os.path.basename(hr_path)
            if name.startswith(f"{base}_") and name.endswith(".png"):
                pairs.append((lr_path, hr_path))
                break
    return pairs


def augment_and_save(low_res_img, high_res_img,
                     save_lr_dir, save_hr_dir,
                     file_idx, num_aug):
    h_lr, w_lr = low_res_img.shape[:2]
    h_hr, w_hr = high_res_img.shape[:2]

    geom = A.Compose([
        # no 90° rotations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=SSR_SHIFT_LIMIT,
                           scale_limit=SSR_SCALE_LIMIT,
                           rotate_limit=SSR_ROTATE_LIMIT,
                           border_mode=cv2.BORDER_CONSTANT,
                           p=0.75),
        A.GridDistortion(distort_limit=0.2, p=0.3),
        A.ElasticTransform(alpha=ET_ALPHA, sigma=ET_SIGMA, p=0.3)
    ], additional_targets={'mask': 'image'},
       is_check_shapes=CHECK_ALBUMENTATIONS_SHAPES)

    lr_intensity = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=RBC_BRIGHTNESS_LIMIT,
                                   contrast_limit=RBC_CONTRAST_LIMIT, p=0.7),
        A.RandomGamma(gamma_limit=(85, 115), p=0.7),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=15,
                             val_shift_limit=10, p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.GaussNoise(var_limit=GN_VAR_LIMIT, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05),
                       intensity=(0.1, 0.3), p=0.3),
        ], p=0.6),
        A.ColorJitter(brightness=0.1, contrast=0.1,
                      saturation=0.1, hue=0.05, p=0.5),
        A.Resize(height=120, width=160, interpolation=cv2.INTER_LINEAR)
    ], is_check_shapes=CHECK_ALBUMENTATIONS_SHAPES)

    for _ in range(num_aug):
        # upscale LR to HR size
        lr_up = cv2.resize(low_res_img, (w_hr, h_hr),
                           interpolation=cv2.INTER_LINEAR)

        out = geom(image=high_res_img, mask=lr_up)
        aug_hr = out['image']
        aug_lr_up = out['mask']

        # downscale LR back to original LR size + intensity aug
        aug_lr = cv2.resize(aug_lr_up, (w_lr, h_lr),
                            interpolation=cv2.INTER_AREA)
        aug_lr = lr_intensity(image=aug_lr)['image']

        cv2.imwrite(os.path.join(save_lr_dir, f"aug_{file_idx}.png"), aug_lr)
        cv2.imwrite(os.path.join(save_hr_dir, f"aug_{file_idx}.png"), aug_hr)
        file_idx += 1

    return file_idx


# Main
all_pairs = get_image_pairs(LOW_RES_INPUT_DIR, HIGH_RES_INPUT_DIR)
if not all_pairs:
    print("No image pairs found.")
    exit()

random.shuffle(all_pairs)
n = len(all_pairs)
t = int(TRAIN_RATIO * n)
v = t + int(VAL_RATIO * n)

splits = {
    "Train":      (all_pairs[:t],   train_save_path_lr, train_save_path_hr, True),
    "Validation": (all_pairs[t:v],  val_save_path_lr,   val_save_path_hr,   False),
    "Test":       (all_pairs[v:],   test_save_path_lr,  test_save_path_hr,  False),
}

for name, (pairs, lr_dir, hr_dir, augment) in splits.items():
    idx = 0
    print(f"\nProcessing {name}: {len(pairs)} pairs")
    for lr_path, hr_path in tqdm(pairs, desc=name):
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        if lr_img is None or hr_img is None:
            print(f"Warning: skipping {lr_path}, {hr_path}")
            continue

        # ensure HR is 480×640 (rotate portrait first)
        h_hr, w_hr = hr_img.shape[:2]
        if h_hr > w_hr:
            hr_img = cv2.rotate(hr_img, cv2.ROTATE_90_CLOCKWISE)
            h_hr, w_hr = hr_img.shape[:2]
        if (h_hr, w_hr) != (480, 640):
            hr_img = cv2.resize(hr_img, (640, 480),
                                interpolation=cv2.INTER_AREA)

        if augment:
            idx = augment_and_save(
                lr_img, hr_img, lr_dir, hr_dir, idx, AUGMENTATIONS_PER_IMAGE
            )
        else:
            # resize LR to 120×160
            h_lr, w_lr = lr_img.shape[:2]
            if (h_lr, w_lr) != (120, 160):
                lr_img = cv2.resize(lr_img, (160, 120),
                                    interpolation=cv2.INTER_LINEAR)

            out_lr = os.path.join(lr_dir, f"orig_{idx}.png")
            out_hr = os.path.join(hr_dir, f"orig_{idx}.png")
            cv2.imwrite(out_lr, lr_img)
            cv2.imwrite(out_hr, hr_img)
            idx += 1

    action = "augmented" if augment else "processed"
    print(f"{name} {action} {idx} image pairs")

print("\nDone.")
