import rasterio
import numpy as np

# 原始多光谱tif路径（你切片用的那张原始影像）
origin_tif = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\rs_data\result.tif"
# 输出NDVI栅格路径
ndvi_out = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\ndvi_result.tif"

with rasterio.open(origin_tif) as src:
    meta = src.meta.copy()
    data = src.read()
    # R波段 index=0，NIR波段 index=3
    red = data[0].astype(np.float32)
    nir = data[3].astype(np.float32)

# 计算NDVI，分母避免除0
ndvi = (nir - red) / (nir + red + 1e-8)
# 限制值域 -1 ~ 1
ndvi = np.clip(ndvi, -1, 1)

# 修改栅格元数据：单波段、浮点
meta.update(count=1, dtype="float32")
with rasterio.open(ndvi_out, "w", **meta) as dst:
    dst.write(ndvi, 1)

print("NDVI栅格生成完成：", ndvi_out)
