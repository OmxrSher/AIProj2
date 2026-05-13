from pathlib import Path
import shutil
import yaml
import re
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
source_root = Path(r"C:\Users\omars\OneDrive\Desktop\AIProj2\dataset_sources\roboflow_u")

output_root = Path(r"C:\Users\omars\OneDrive\Desktop\AIProj2\dataset_grouped_by_class2")

# Set True first if you want to preview without copying
DRY_RUN = False

# Set True to delete old grouped folder before creating new one
# Recommended if you are re-running after adding new datasets
CLEAR_OUTPUT = True

# ============================================================
# FINAL CLASSES
# ============================================================
FINAL_CLASSES = {
    "plastic_bottle": 0,
    "paper_waste": 1,
    "can": 2,
    "general_bin": 3,
    "recycle_bin": 4,
}

# ============================================================
# FOLDERS TO PROCESS
# These are the folders shown in your screenshot
# ============================================================
FOLDERS_TO_PROCESS = [
    "bins_a",
    "bins_b",
    "can.yolov8",
    "Cans.yolov8 (1)",
    "Cans2",
    "Paper- Plastic - Wet Waste.yolov8",
    "paper.yolov8 (1)",
    "Plastic bottles",
    "Plastic Bottles2",
    "recycle2",
    "recycling",

    # Keep this here in case the folder still exists from before
    "paper trash.yolov8",
]

# ============================================================
# MANUAL CLASS REMAPPING RULES
# Key = source folder name
# Value = original Roboflow class names remapped to final classes
# Use None to ignore a class
# ============================================================
CLASS_REMAP = {
    # Plastic bottles
    "Plastic bottles": {
        "plastic-bottle": "plastic_bottle",
        "plastic bottle": "plastic_bottle",
        "Plastic Bottle": "plastic_bottle",
        "plastic_bottle": "plastic_bottle",
    },

    "Plastic Bottles2": {
        "plastic-bottle": "plastic_bottle",
        "plastic bottle": "plastic_bottle",
        "Plastic Bottle": "plastic_bottle",
        "plastic_bottle": "plastic_bottle",
    },

    # Paper waste
    "paper.yolov8 (1)": {
        "paper": "paper_waste",
        "Paper": "paper_waste",
        "paper_waste": "paper_waste",
    },

    "paper trash.yolov8": {
        "paper": "paper_waste",
        "Paper": "paper_waste",
        "paper_waste": "paper_waste",
    },

    # IMPORTANT:
    # For this mixed dataset, ignore Plastic and only keep Paper
    "Paper- Plastic - Wet Waste.yolov8": {
        "Paper": "paper_waste",
        "paper": "paper_waste",
        "Plastic": None,
        "plastic": None,
        "Wet Waste": None,
        "wet waste": None,
        "Wet_Waste": None,
    },

    # Cans
    "can.yolov8": {
        "metal-can": "can",
        "metal can": "can",
        "can": "can",
        "Can": "can",
    },

    "Cans.yolov8 (1)": {
        "cans": "can",
        "Cans": "can",
        "can": "can",
        "Can": "can",
    },

    "Cans2": {
        "Metal": "can",
        "metal": "can",
        "can": "can",
        "Can": "can",
    },

    # Bins
    "bins_a": {
        "General Bin": "general_bin",
        "general bin": "general_bin",
        "general_bin": "general_bin",
        "Recycle Bin": "recycle_bin",
        "recycle bin": "recycle_bin",
        "recycle_bin": "recycle_bin",
    },

    "bins_b": {
        "General Bin": "general_bin",
        "general bin": "general_bin",
        "general_bin": "general_bin",
        "Recycle Bin": "recycle_bin",
        "recycle bin": "recycle_bin",
        "recycle_bin": "recycle_bin",
    },

    # New recycling bin folders
    "recycle2": {
        "Recycle Bin": "recycle_bin",
        "recycle bin": "recycle_bin",
        "recycling bin": "recycle_bin",
        "Recycling Bin": "recycle_bin",
        "recycle_bin": "recycle_bin",
        "recycling_bin": "recycle_bin",
        "recycle": "recycle_bin",
        "recycling": "recycle_bin",
    },

    "recycling": {
        "Recycle Bin": "recycle_bin",
        "recycle bin": "recycle_bin",
        "recycling bin": "recycle_bin",
        "Recycling Bin": "recycle_bin",
        "recycle_bin": "recycle_bin",
        "recycling_bin": "recycle_bin",
        "recycle": "recycle_bin",
        "recycling": "recycle_bin",
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")


def normalize_class_name(name: str) -> str:
    """
    Makes class names easier to compare.
    Example:
    'Recycle Bin' -> 'recycle bin'
    'recycle_bin' -> 'recycle bin'
    """
    name = name.strip().lower()
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    name = re.sub(r"\s+", " ", name)
    return name


def load_class_names(data_yaml_path: Path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")

    if isinstance(names, list):
        return names

    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]

    raise ValueError(f"Could not read names from {data_yaml_path}")


def find_label_for_image(image_path: Path):
    """
    Finds matching YOLO label file:
    train/images/img.jpg -> train/labels/img.txt
    valid/images/img.jpg -> valid/labels/img.txt
    test/images/img.jpg -> test/labels/img.txt
    """
    parts = list(image_path.parts)

    if "images" in parts:
        parts[parts.index("images")] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        if label_path.exists():
            return label_path

    # Fallback search
    possible_labels = list(image_path.parents[1].rglob(f"{image_path.stem}.txt"))
    return possible_labels[0] if possible_labels else None


def get_target_class(dataset_name: str, original_class_name: str):
    """
    First tries exact manual mapping.
    Then tries normalized mapping.
    Then uses safe keyword fallback for obvious cases.
    """

    remap_rules = CLASS_REMAP.get(dataset_name, {})

    # 1. Exact match
    if original_class_name in remap_rules:
        return remap_rules[original_class_name]

    # 2. Normalized match
    original_norm = normalize_class_name(original_class_name)

    for source_name, target_name in remap_rules.items():
        if normalize_class_name(source_name) == original_norm:
            return target_name

    # 3. Safe fallback by folder name and class name
    folder_norm = normalize_class_name(dataset_name)

    # Plastic folders
    if "plastic" in folder_norm and "bottle" in folder_norm:
        if "plastic" in original_norm or "bottle" in original_norm:
            return "plastic_bottle"

    # Paper folders
    if "paper" in folder_norm:
        if "paper" in original_norm:
            return "paper_waste"
        if "plastic" in original_norm:
            return None
        if "wet" in original_norm:
            return None

    # Can folders
    if "can" in folder_norm:
        if "can" in original_norm or "metal" in original_norm:
            return "can"

    # Bin folders
    if "bins" in folder_norm:
        if "general" in original_norm:
            return "general_bin"
        if "recycle" in original_norm or "recycling" in original_norm:
            return "recycle_bin"

    # Recycling folders
    if "recycle" in folder_norm or "recycling" in folder_norm:
        if "recycle" in original_norm or "recycling" in original_norm or "bin" in original_norm:
            return "recycle_bin"

    return "UNMAPPED"


def copy_remapped_dataset():
    if CLEAR_OUTPUT and output_root.exists() and not DRY_RUN:
        print(f"Clearing old output folder: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skipped_stats = defaultdict(int)
    total_images_copied = 0
    total_labels_copied = 0

    print("=" * 80)
    print("REMAPPING AND GROUPING ALL DATASETS")
    print("=" * 80)

    for dataset_name in FOLDERS_TO_PROCESS:
        dataset_folder = source_root / dataset_name

        if not dataset_folder.exists():
            print(f"\n[SKIPPED] Folder not found: {dataset_name}")
            continue

        data_yaml = dataset_folder / "data.yaml"

        if not data_yaml.exists():
            print(f"\n[SKIPPED] No data.yaml found in: {dataset_name}")
            continue

        class_names = load_class_names(data_yaml)

        print(f"\nProcessing folder: {dataset_name}")
        print(f"Original classes: {class_names}")

        images = [
            p for p in dataset_folder.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ]

        for image_path in images:
            label_path = find_label_for_image(image_path)

            if label_path is None or not label_path.exists():
                skipped_stats[f"{dataset_name} - missing_label"] += 1
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                label_lines = [line.strip() for line in f.readlines() if line.strip()]

            kept_lines_by_target = defaultdict(list)

            for line in label_lines:
                parts = line.split()

                if len(parts) < 5:
                    skipped_stats[f"{dataset_name} - invalid_label_line"] += 1
                    continue

                try:
                    original_class_id = int(parts[0])
                except ValueError:
                    skipped_stats[f"{dataset_name} - invalid_class_id"] += 1
                    continue

                if original_class_id >= len(class_names):
                    skipped_stats[f"{dataset_name} - class_id_out_of_range"] += 1
                    continue

                original_class_name = class_names[original_class_id]
                target_class = get_target_class(dataset_name, original_class_name)

                if target_class == "UNMAPPED":
                    skipped_stats[f"{dataset_name} - unmapped_class_{original_class_name}"] += 1
                    continue

                if target_class is None:
                    skipped_stats[f"{dataset_name} - ignored_{original_class_name}"] += 1
                    continue

                target_class_id = FINAL_CLASSES[target_class]

                # Replace original class ID with final class ID
                new_line = " ".join([str(target_class_id)] + parts[1:])
                kept_lines_by_target[target_class].append(new_line)

            # Copy image and remapped label into final class folder
            for target_class, new_label_lines in kept_lines_by_target.items():
                class_output_dir = output_root / target_class
                images_output_dir = class_output_dir / "images"
                labels_output_dir = class_output_dir / "labels"

                images_output_dir.mkdir(parents=True, exist_ok=True)
                labels_output_dir.mkdir(parents=True, exist_ok=True)

                unique_prefix = safe_name(dataset_name)
                new_image_name = f"{unique_prefix}_{image_path.name}"
                new_label_name = f"{Path(new_image_name).stem}.txt"

                new_image_path = images_output_dir / new_image_name
                new_label_path = labels_output_dir / new_label_name

                if not DRY_RUN:
                    shutil.copy2(image_path, new_image_path)

                    with open(new_label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(new_label_lines) + "\n")

                stats[target_class] += 1
                total_images_copied += 1
                total_labels_copied += 1

        print(f"Finished: {dataset_name}")

    print("\n" + "=" * 80)
    print("FINAL GROUPED DATASET SUMMARY")
    print("=" * 80)

    for class_name in FINAL_CLASSES:
        print(f"{class_name}: {stats[class_name]} images")

    print("-" * 80)
    print(f"Total copied images: {total_images_copied}")
    print(f"Total copied label files: {total_labels_copied}")

    print("\n" + "=" * 80)
    print("SKIPPED / IGNORED SUMMARY")
    print("=" * 80)

    if skipped_stats:
        for reason, count in skipped_stats.items():
            print(f"{reason}: {count}")
    else:
        print("No skipped files.")

    print("\nDone.")
    print(f"Saved grouped dataset to: {output_root}")


if __name__ == "__main__":
    copy_remapped_dataset()