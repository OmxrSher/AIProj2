from pathlib import Path

grouped_root = Path(r"C:\Users\omars\OneDrive\Desktop\AIProj2\custom_dataset2")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print("=" * 60)
print("GROUPED CLASS IMAGE COUNT")
print("=" * 60)

total = 0

for class_folder in sorted(grouped_root.iterdir()):
    if class_folder.is_dir():
        images_dir = class_folder / "images"
        count = sum(
            1 for img in images_dir.iterdir()
            if img.suffix.lower() in IMAGE_EXTS
        )

        print(f"{class_folder.name}: {count} images")
        total += count

print("-" * 60)
print(f"TOTAL IMAGES: {total}")