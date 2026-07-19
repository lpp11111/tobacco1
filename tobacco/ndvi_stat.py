import os
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask

# 路径配置
shp_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\tobacco_plant.shp"
ndvi_tif_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\ndvi_result.tif"
output_shp = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\tobacco_plant_ndvi.shp"
ndvi_thresh = 0.2

# 读取矢量，不再强制转4326，和NDVI同平面投影
gdf = gpd.read_file(shp_path)

ndvi_mean_list = []
disease_flag_list = []

with rasterio.open(ndvi_tif_path) as src_ndvi:
    for idx, geom in enumerate(gdf["geometry"]):
        try:
            crop_arr, _ = mask(src_ndvi, [geom], crop=True, nodata=-9999)
            crop_arr = crop_arr[0]
            valid_data = crop_arr[crop_arr > -1]
            if len(valid_data) == 0:
                mean_val = -9999
                print(f"第{idx}株无有效NDVI像素")
            else:
                mean_val = np.mean(valid_data)
        except Exception as e:
            mean_val = -9999
            print(f"第{idx}株裁剪异常：{str(e)}")

        ndvi_mean_list.append(mean_val)
        if mean_val != -9999 and mean_val < ndvi_thresh:
            disease_flag_list.append("疑似病害")
        else:
            disease_flag_list.append("健康")

# 新增字段
gdf["ndvi_mean"] = ndvi_mean_list
gdf["disease"] = disease_flag_list

# 输出矢量
gdf.to_file(output_shp, encoding="utf-8")

# 统计
valid_ndvi = [v for v in ndvi_mean_list if v != -9999]
if len(valid_ndvi) > 0:
    print("==== 所有烟株NDVI统计 ====")
    print("最小值：", np.min(valid_ndvi))
    print("平均值：", np.mean(valid_ndvi))
    print("最大值：", np.max(valid_ndvi))
    healthy_count = len(gdf[gdf["disease"] == "健康"])
    disease_count = len(gdf[gdf["disease"] == "疑似病害"])
    print(f"健康植株：{healthy_count} 株")
    print(f"低NDVI疑似病害植株：{disease_count} 株")
else:
    print("警告：所有烟株轮廓均未匹配到有效NDVI像素")
print(f"矢量输出完成：{output_shp}")
