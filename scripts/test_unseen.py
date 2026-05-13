from ultralytics import YOLO
from pathlib import Path
import sys


def main() -> None:
    # =========================
    # Paths for AIProj
    # =========================
    base_dir = Path(r"C:\Users\omars\OneDrive\Desktop\AIProj2")

    model_path = base_dir / "results" / "yolov8l_custom_dataset2_final" / "weights" / "best.pt"
    source_path = base_dir / "demo_images"
    output_project = base_dir / "results"
    output_name = "Infer_unseen"

    # =========================
    # Basic validation
    # =========================
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    if not source_path.exists():
        print(f"[ERROR] Source folder not found: {source_path}")
        sys.exit(1)

    if source_path.is_dir():
        image_files = list(source_path.glob("*.*"))
        if not image_files:
            print(f"[ERROR] No files found in source folder: {source_path}")
            sys.exit(1)

    print("[INFO] Loading model...")
    model = YOLO(str(model_path))

    print("[INFO] Running inference...")
    results = model.predict(
        source=str(source_path),
        conf=0.15,
        iou=0.5,
        save=True,
        show=True,
        project=str(output_project),
        name=output_name,
        exist_ok=True,
        imgsz=640,
        line_width=2,
        save_txt=False,
        save_conf=True,
        verbose=True
    )

    output_folder = output_project / output_name

    print("\n[INFO] Inference completed successfully.")
    print(f"[INFO] Results saved to: {output_folder}")
    print(f"[INFO] Number of result items: {len(results)}")


if __name__ == "__main__":
    main()