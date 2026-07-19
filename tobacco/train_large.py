from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def train():
    print("=" * 50)
    print("开始训练 YOLOv8m-seg 大尺度烟草（圈地+株数统计）模型")
    print("=" * 50)

    model_path = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/yolov8m-seg.pt"
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    yaml_path = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/large/data.yaml"
    project_dir = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/runs/large_seg_count"

    results = model.train(
        data=yaml_path,
        epochs=100,
        patience=20,
        imgsz=640,
        batch=8,
        workers=0,
        project=project_dir,
        name="tobacco_large_m_count",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',

        # 删掉了mask=0.75，这个参数新版不识别
        box=0.85,
        iou=0.6,
        conf=0.2,
        cls=0.6,

        mosaic=1.0,
        hsv_h=0.03,
        hsv_s=0.5,
        hsv_v=0.4,
        flipud=0.3,
        fliplr=0.5
    )

    print("\n训练完成！")
    print(f"最优权重：{project_dir}/tobacco_large_m_count/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    train()