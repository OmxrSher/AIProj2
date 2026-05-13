from ultralytics import YOLO

model = YOLO(
    "/content/drive/MyDrive/AIProj2/runs_custom_dataset2/yolov8l_custom_dataset2_final/weights/best.pt"
)

model.val(
    data="/content/custom_dataset2/data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    plots=True,
    project="/content/drive/MyDrive/AIProj2/test_results_custom_dataset2",
    name="final_test_eval"
)
