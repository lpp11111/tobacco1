from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def train():
    print("=" * 50)
    print("开始训练 YOLOv8 中尺度medium分割模型")
    print("=" * 50)

    # 加载官方预训练权重 yolov8n‑seg.pt
    model_path = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/yolov8n-seg.pt"
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 你的数据集yaml
    yaml_path = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/t/data.yaml"
    # 训练结果保存文件夹
    project_dir = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/runs/medium_seg"

    print(f"开始训练 (epochs=10, imgsz=640, batch=2)...")
    results = model.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        batch=2,                # 4050笔记本显存有限改为batch=2，之前batch=8显存不够
        workers=0,
        project=project_dir,
        name="tobacco_medium_final",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        patience=7,             # 早停防止过拟合
        cos_lr=True,
        mosaic=0.0,             # 关闭mosaic避开ultralytics8.4.9版本报索引越界bug
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3
    )

    print("\n中尺度medium训练完成！")
    print(f"模型权重保存路径: {project_dir}/tobacco_medium_final/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    train()
