import rasterio

# 原始识别影像
src_tif = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\rs_data\result.tif"
# 刚重新生成的NDVI
ndvi_tif = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\ndvi_result.tif"

with rasterio.open(src_tif) as s:
    print("原始tif边界：", s.bounds)
    print("原始tif坐标系：", s.crs)

with rasterio.open(ndvi_tif) as n:
    print("NDVI tif边界：", n.bounds)
    print("NDVI tif坐标系：", n.crs)
