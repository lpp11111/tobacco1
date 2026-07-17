from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def train():
    print("=" * 50)
    print("开始训练 YOLOv8 中尺度烟草分割模型")
    print("=" * 50)

    model_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\yolov8m-seg.pt"
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    yaml_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\t\data.yaml"
    project_dir = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\runs\middle_seg"

    print(f"开始训练 (epochs=100, imgsz=640, batch=2)...")
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=2,
        workers=0,
        project=project_dir,
        name="tobacco_middle",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        patience=100,
        cos_lr=False,
        mosaic=0,
        perspective=0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        amp=False
    )

    print("\n中尺度烟草模型训练完成！")
    print(f"权重保存路径：{project_dir}/tobacco_middle/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    train()
