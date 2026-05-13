from pathlib import Path
import shutil
import random
import yaml

# ============================================================
# PATHS
# ============================================================
source_root = Path(r"C:\Users\omars\OneDrive\Desktop\AIProj2\dataset_balanced_fm")

output_root = Path(r"C:\Users\omars\OneDrive\Desktop\AIProj2\final_yolo_dataset_fm")

# ============================================================
# SETTINGS
# ============================================================
SEED = 42
random.seed(SEED)

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FINAL_CLASSES = [
    "plastic_bottle",
    "paper_waste",
    "can",
    "general_bin",
    "recycle_bin",
]

# Final YOLO class IDs
CLASS_IDS = {
    "plastic_bottle": 0,
    "paper_waste": 1,
    "can": 2,
    "general_bin": 3,
    "recycle_bin": 4,
}


def clear_output_folder(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def make_dirs():
    for split in ["train", "valid", "test"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def find_label(image_path: Path, labels_dir: Path):
    label_path = labels_dir / f"{image_path.stem}.txt"
    return label_path if label_path.exists() else None


def copy_pair(image_path: Path, label_path: Path, split: str, new_stem: str):
    new_image_path = output_root / split / "images" / f"{new_stem}{image_path.suffix.lower()}"
    new_label_path = output_root / split / "labels" / f"{new_stem}.txt"

    shutil.copy2(image_path, new_image_path)
    shutil.copy2(label_path, new_label_path)


print("=" * 80)
print("CREATING FINAL YOLO TRAIN/VALID/TEST DATASET")
print("=" * 80)

clear_output_folder(output_root)
make_dirs()

summary = {
    "train": {},
    "valid": {},
    "test": {},
}

for class_name in FINAL_CLASSES:
    class_dir = source_root / class_name
    images_dir = class_dir / "images"
    labels_dir = class_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"\n[SKIPPED] Missing images or labels folder for: {class_name}")
        continue

    image_files = [
        img for img in images_dir.iterdir()
        if img.suffix.lower() in IMAGE_EXTS
    ]

    valid_pairs = []

    for img in image_files:
        label = find_label(img, labels_dir)
        if label is not None:
            valid_pairs.append((img, label))

    random.shuffle(valid_pairs)

    total = len(valid_pairs)

    train_count = int(total * TRAIN_RATIO)
    valid_count = int(total * VALID_RATIO)
    test_count = total - train_count - valid_count

    train_pairs = valid_pairs[:train_count]
    valid_pairs_split = valid_pairs[train_count:train_count + valid_count]
    test_pairs = valid_pairs[train_count + valid_count:]

    split_data = {
        "train": train_pairs,
        "valid": valid_pairs_split,
        "test": test_pairs,
    }

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_pairs)}")
    print(f"Valid: {len(valid_pairs_split)}")
    print(f"Test: {len(test_pairs)}")

    for split, pairs in split_data.items():
        summary[split][class_name] = len(pairs)

        for i, (img, label) in enumerate(pairs):
            new_stem = f"{class_name}_{split}_{i:04d}"
            copy_pair(img, label, split, new_stem)

# ============================================================
# CREATE data.yaml
# ============================================================
data_yaml = {
    "path": str(output_root),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(FINAL_CLASSES),
    "names": FINAL_CLASSES,
}

with open(output_root / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("\n" + "=" * 80)
print("FINAL SPLIT SUMMARY")
print("=" * 80)

for split in ["train", "valid", "test"]:
    print(f"\n{split.upper()}:")
    split_total = 0

    for class_name in FINAL_CLASSES:
        count = summary[split].get(class_name, 0)
        split_total += count
        print(f"  {class_name}: {count}")

    print(f"  Total {split}: {split_total}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Final YOLO dataset saved to: {output_root}")
print(f"data.yaml saved to: {output_root / 'data.yaml'}")