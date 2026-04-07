# import os
# import shutil
# import random
# from pathlib import Path

# # =========================
# # CONFIG
# # =========================
# SOURCE_DIR = "/home/dulmin/FYPML/data/YOLO_data/stationary/images"
# OUTPUT_DIR = "/home/dulmin/FYPML/data/YOLO_data/split/stationary"

# TRAIN_RATIO = 0.7
# SEED = 42

# IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# def make_dirs():
#     for split in ["train", "val"]:
#         os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)


# def get_image_label_pairs():
#     source = Path(SOURCE_DIR)
#     pairs = []

#     for file in source.iterdir():
#         if file.suffix.lower() in IMAGE_EXTENSIONS:
#             label_file = file.with_suffix(".txt")
#             if label_file.exists():
#                 pairs.append((file, label_file))
#             else:
#                 print(f"Warning: label not found for image {file.name}")

#     return pairs


# def split_dataset(pairs):
#     random.seed(SEED)
#     random.shuffle(pairs)

#     train_size = int(len(pairs) * TRAIN_RATIO)
#     train_pairs = pairs[:train_size]
#     val_pairs = pairs[train_size:]

#     return train_pairs, val_pairs


# def copy_pairs(pairs, split_name):
#     split_dir = os.path.join(OUTPUT_DIR, split_name)

#     for image_path, label_path in pairs:
#         dst_img = os.path.join(split_dir, image_path.name)
#         dst_lbl = os.path.join(split_dir, label_path.name)

#         shutil.copy2(image_path, dst_img)
#         shutil.copy2(label_path, dst_lbl)


# def main():
#     make_dirs()

#     pairs = get_image_label_pairs()
#     print(f"Total valid image-label pairs found: {len(pairs)}")

#     if len(pairs) == 0:
#         print("No valid image-label pairs found.")
#         return

#     train_pairs, val_pairs = split_dataset(pairs)

#     print(f"Train pairs: {len(train_pairs)}")
#     print(f"Val pairs: {len(val_pairs)}")

#     copy_pairs(train_pairs, "train")
#     copy_pairs(val_pairs, "val")

#     print("\nDataset split completed successfully.")
#     print(f"Train folder: {OUTPUT_DIR}/train")
#     print(f"Val folder:   {OUTPUT_DIR}/val")


# if __name__ == "__main__":
#     main()

import os
import shutil
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
IMAGE_DIR = "/home/dulmin/FYPML/data/YOLO_data/stationary/images"
LABEL_DIR = "/home/dulmin/FYPML/data/YOLO_data/stationary/labels"
OUTPUT_DIR = "/home/dulmin/FYPML/data/YOLO_data/split/stationary"

TRAIN_RATIO = 0.7
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)


def get_image_label_pairs():
    image_dir = Path(IMAGE_DIR)
    label_dir = Path(LABEL_DIR)
    pairs = []

    for image_file in image_dir.iterdir():
        if image_file.suffix.lower() in IMAGE_EXTENSIONS:
            label_file = label_dir / f"{image_file.stem}.txt"

            if label_file.exists():
                pairs.append((image_file, label_file))
            else:
                print(f"Warning: label not found for image {image_file.name}")

    return pairs


def split_dataset(pairs):
    random.seed(SEED)
    random.shuffle(pairs)

    train_size = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    return train_pairs, val_pairs


def copy_pairs(pairs, split_name):
    image_out_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    label_out_dir = os.path.join(OUTPUT_DIR, split_name, "labels")

    for image_path, label_path in pairs:
        dst_img = os.path.join(image_out_dir, image_path.name)
        dst_lbl = os.path.join(label_out_dir, label_path.name)

        shutil.copy2(image_path, dst_img)
        shutil.copy2(label_path, dst_lbl)


def main():
    make_dirs()

    pairs = get_image_label_pairs()
    print(f"Total valid image-label pairs found: {len(pairs)}")

    if len(pairs) == 0:
        print("No valid image-label pairs found.")
        return

    train_pairs, val_pairs = split_dataset(pairs)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    print("\nDataset split completed successfully.")
    print(f"Train images: {OUTPUT_DIR}/train/images")
    print(f"Train labels: {OUTPUT_DIR}/train/labels")
    print(f"Val images:   {OUTPUT_DIR}/val/images")
    print(f"Val labels:   {OUTPUT_DIR}/val/labels")


if __name__ == "__main__":
    main()