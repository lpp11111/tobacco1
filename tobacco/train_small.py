from ultralytics import YOLO
import torch
import torch.nn as nn
from multiprocessing import freeze_support

def train_small():
    print("="*60)
    print("        启动小尺度 small 烟草近景分割训练")
    print("数据集：5-30m低空近景单株/叶片特写，640×640 4通道RGB+NDVI")
    print("分类体系：4级病害精细分级（健康/轻度/中度/重度）")
    print("="*60)

    # 预训练权重
    model_path = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/yolov8n-seg.pt"
    print(f"加载初始化权重：{model_path}")
    model = YOLO(model_path)

    # ========== 兼容新版ultralytics 4通道输入替换代码 ==========
    net = model.model.model
    first_conv = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            first_conv = m
            break
    if first_conv is not None:
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        new_conv.weight.data[:, :3, :, :] = first_conv.weight.data
        new_conv.weight.data[:, 3:4, :, :] = torch.mean(first_conv.weight.data, dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data
        for name, module in net.named_modules():
            if module is first_conv:
                parent_name = name.rsplit(".", 1)[0]
                layer_name = name.rsplit(".", 1)[1]
                parent_module = net
                for part in parent_name.split("."):
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, layer_name, new_conv)
                break
    # ==================================================================

    data_yaml = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/small/data.yaml"
    save_dir = r"E:/Yolov8-Remose-Image-Dataset-Process-Tool-Set-main/runs/small_seg"

    print("训练基础参数：epochs=100 imgsz=640 batch=2")
    train_results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=2,
        workers=0,
        project=save_dir,
        name="tobacco_small_4ch",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
        cos_lr=True,
        patience=15,
        cls=1.2,
        iou=0.5,
        conf=0.15,
        mosaic=1.0,
        fliplr=0.5,
        flipud=0.3,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4
    )

    print("\n✅ small小尺度4通道分割模型训练完成")
    print(f"最优模型权重路径：{save_dir}/tobacco_small_4ch/weights/best.pt")

if __name__ == "__main__":
    freeze_support()
    train_small()