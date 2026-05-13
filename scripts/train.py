from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data="/content/custom_dataset2/data.yaml",
    epochs=70,
    imgsz=640,
    batch=16,

    project="/content/drive/MyDrive/AIProj2/runs_custom_dataset2",
    name="yolov8l_custom_dataset2_final",

    patience=20,
    save=True,
    plots=True,
    cache=True,
    workers=4,

    dropout=0.05,
    weight_decay=0.0005,

    mosaic=0.2,
    close_mosaic=10,
    mixup=0.0,
    copy_paste=0.0,

    fliplr=0.5,
    scale=0.15,
    translate=0.08,
    degrees=5.0,

    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,

    seed=42
)