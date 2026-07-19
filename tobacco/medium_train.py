from ultralytics import YOLO
import torch
import torch.nn as nn
from multiprocessing import freeze_support

def train():
    print("=" * 50)
    print("开始训练 YOLOv8n 中尺度病害检测模型（RGB+NDVI四通道）")
    print("=" * 50)

    # 更换为纯检测权重，不再使用seg分割模型
    model_path = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/yolov8n.pt"
    model = YOLO(model_path)

    # 适配4通道RGB+NDVI输入
    old_conv = model.model.model[0].conv
    new_conv = nn.Conv2d(4, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=old_conv.bias is not None)
    new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
    new_conv.weight.data[:, 3:4, :, :] = old_conv.weight.data.mean(dim=1, keepdim=True)
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data
    model.model.model[0].conv = new_conv

    yaml_path = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/medium/data.yaml"
    project_dir = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/runs/medium_seg"

    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=2,
        workers=0,
        project=project_dir,
        name="tobacco_medium_detect_opt",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        patience=15,
        cos_lr=True,
        mosaic=0.8,
        fliplr=0.5,
        flipud=0.3,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        box=1.0,
        cls=1.2,
        iou=0.5,
        conf=0.15
    )

    print(f"训练完成，最优权重路径：{project_dir}/tobacco_medium_detect_opt/weights/best.pt")

if __name__ == "__main__":
    freeze_support()
    train()