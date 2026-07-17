from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def train():
    print("=" * 50)
    print("开始训练 YOLOv8 大尺度烟草分割模型")
    print("=" * 50)

    # 1. 加载本地预训练分割权重（已匹配你根目录的文件）
    model_path = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/yolov8n-seg.pt"
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 2. 数据集配置文件路径（已匹配你本地large数据集）
    yaml_path = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/large/data.yaml"
    # 训练结果保存路径
    project_dir = "E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/runs/large_seg"

    # 3. 启动训练
    print(f"开始训练 (epochs=10, imgsz=640, batch=8)...")
    results = model.train(
        data=yaml_path,
        epochs=10,  # 训练轮次，可自行调整
        imgsz=640,  # 图片尺寸，和数据集预处理一致
        batch=8,  # 显存不足改成4/2
        workers=0,  # 禁用多进程，彻底解决Windows共享内存报错
        project=project_dir,
        name="tobacco_large",
        exist_ok=True,  # 同名目录自动覆盖
        device=0 if torch.cuda.is_available() else 'cpu',  # 有GPU自动用GPU，无GPU自动切CPU
    )

    print("\n训练完成！")
    print(f"最优模型权重保存在: {project_dir}/tobacco_large/weights/best.pt")


if __name__ == "__main__":
    freeze_support()  # Windows多进程强制保护，解决启动报错
    train()