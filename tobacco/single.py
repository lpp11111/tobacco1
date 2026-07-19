import rasterio
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 替换成你chunk里真实存在的切片路径
tile_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\chunk\tile_22_x11440_y0.tif"
weight_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\runs\middle_seg\tobacco_middle\weights\best.pt"
conf_thresh = 0.01

# 读取切片 (C,H,W)
with rasterio.open(tile_path) as src:
    img_chw = src.read()
print(f"原始维度 (C,H,W): {img_chw.shape}")

# 1. 选取标准RGB三通道 R=Band1(0), G=Band2(1), B=Band3(2)
rgb_chw = img_chw[[0, 1, 2], :, :]
# 2. 转置：C,H,W → H,W,C
rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
print(f"转置后维度 (H,W,C): {rgb_hwc.shape}")

# 3. 浮点0~1拉伸至0~255 uint8
rgb_hwc = (rgb_hwc * 255).clip(0, 255).astype(np.uint8)
# 新增：转为连续内存数组，解决绘图报错
rgb_hwc = np.ascontiguousarray(rgb_hwc)

# 输出可视化原图
Image.fromarray(rgb_hwc).save("debug_origin_tile.jpg")
print("已生成可视化原图：debug_origin_tile.jpg")

# 单张推理测试
model = YOLO(weight_path)
results = model(rgb_hwc, imgsz=640, conf=conf_thresh)
res = results[0]

det_num = len(res.boxes) if res.boxes is not None else 0
print(f"单张识别烟苗数量：{det_num}")
if det_num > 0:
    print("预测置信度：", res.boxes.conf.cpu().numpy())
    # 现在可以正常保存标记图
    res.save("debug_pred_mark.jpg")
    print("已生成带分割标记图：debug_pred_mark.jpg")
else:
    print("该切片无检测目标，请更换切片或调整波段")
