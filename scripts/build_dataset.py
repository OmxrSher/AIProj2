import os
import shutil

# Use the balanced source folder
SOURCE_ROOT = r"dataset_sources\roboflow"

DATASETS = {
    "plastic_bottles": {
        "folder": "plastic_bottles",
        "class_map": {0: 0},
    },
    "cans": {
        "folder": "cans",
        "class_map": {0: 1},
    },
    "paper": {
        "folder": "paper",
        "class_map": {0: 2},
    },
    "bins_a": {
        "folder": "bins_a",
        "class_map": {0: 3, 1: 4},
    },
    "bins_b": {
        "folder": "bins_b",
        "class_map": {0: 3, 1: 4},
    },
}

OUT_IMAGES = r"custom_dataset\all\images"
OUT_LABELS = r"custom_dataset\all\labels"

VALID_EXTS = (".jpg", ".jpeg", ".png")


def ensure_clean_output():
    if os.path.exists(OUT_IMAGES):
        shutil.rmtree(OUT_IMAGES)
    if os.path.exists(OUT_LABELS):
        shutil.rmtree(OUT_LABELS)

    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_LABELS, exist_ok=True)


def process_dataset(dataset_name, folder_name, class_map, start_index):
    dataset_path = os.path.join(SOURCE_ROOT, folder_name, "train")
    img_dir = os.path.join(dataset_path, "images")
    lbl_dir = os.path.join(dataset_path, "labels")

    if not os.path.exists(img_dir):
        print(f"Skipping {dataset_name}: missing {img_dir}")
        return start_index, 0

    count = 0
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(VALID_EXTS)])

    for file in files:
        src_img = os.path.join(img_dir, file)
        base, ext = os.path.splitext(file)
        src_lbl = os.path.join(lbl_dir, base + ".txt")

        new_base = f"{dataset_name}_{start_index:06d}"
        start_index += 1

        dst_img = os.path.join(OUT_IMAGES, new_base + ext.lower())
        dst_lbl = os.path.join(OUT_LABELS, new_base + ".txt")

        shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            with open(src_lbl, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                old_cls = int(parts[0])
                new_cls = class_map[old_cls]
                new_line = " ".join([str(new_cls)] + parts[1:])
                new_lines.append(new_line)

            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

        count += 1

    return start_index, count


def main():
    ensure_clean_output()

    global_index = 0
    total = 0

    print("=== MERGING + REMAPPING DATASETS ===")

    for dataset_name, cfg in DATASETS.items():
        global_index, copied = process_dataset(
            dataset_name=dataset_name,
            folder_name=cfg["folder"],
            class_map=cfg["class_map"],
            start_index=global_index,
        )
        total += copied
        print(f"{dataset_name}: copied {copied} images")

    print(f"\nDone. Total merged images: {total}")
    print(f"Images folder: {OUT_IMAGES}")
    print(f"Labels folder: {OUT_LABELS}")


if __name__ == "__main__":
    main()