from pathlib import Path

dataset_root = Path(r"C:\Users\omars\OneDrive\Desktop\AIProj2\final_yolo_dataset_part2")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for split in ["train", "valid", "test"]:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"

    images = [img for img in images_dir.iterdir() if img.suffix.lower() in IMAGE_EXTS]
    labels = list(labels_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    missing_labels = []

    for img in images:
        label = labels_dir / f"{img.stem}.txt"
        if not label.exists():
            missing_labels.append(img.name)

    if missing_labels:
        print(f"Missing labels: {len(missing_labels)}")
        print(missing_labels[:10])
    else:
        print("Status: OK - every image has a label")