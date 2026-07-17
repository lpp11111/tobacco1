from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def train_small():
    print("="*60)
    print("        启动小尺度 small 烟草近景分割训练")
    print("数据集路径：./small 近景叶片特写 640×640")
    print("="*60)

    # 预训练权重
    model_weight = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/yolov8n-seg.pt"
    print(f"加载初始化权重：{model_weight}")
    model = YOLO(model_weight)

    # small数据集yaml文件（和截图文件夹完全对应）
    data_yaml = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/small/data.yaml"
    # 训练结果单独存放，和large、medium区分
    save_dir = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/runs/small_seg"

    print("训练参数：epochs=10 imgsz=640 batch=8")
    train_results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=8,
        workers=0,  # 规避Windows共享内存报错
        project=save_dir,
        name="tobacco_small",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
        # 数据增强，弥补数据集无内置增强
        mosaic=1.0,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4
    )

    print("\n✅ small小尺度模型训练完成")
    print(f"最优模型：{save_dir}/tobacco_small/weights/best.pt")

if __name__ == "__main__":
    freeze_support()
    train_small()