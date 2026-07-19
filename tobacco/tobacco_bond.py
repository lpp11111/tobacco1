import rasterio
import numpy as np

def compute_ndvi():
    multi_tif = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\rs_data\result.tif"
    out_ndvi = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\ndvi_result.tif"
    with rasterio.open(multi_tif) as src:
        red = np.asarray(src.read(2), dtype=np.float32)
        nir = np.asarray(src.read(4), dtype=np.float32)
        profile = src.profile
    denominator = nir + red
    denominator[denominator == 0] = np.nan
    ndvi = (nir - red) / denominator
    ndvi = np.clip(ndvi, -1, 1)
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(out_ndvi, "w", **profile) as dst:
        dst.write(ndvi, 1)
    print("NDVI栅格生成完毕")

if __name__ == "__main__":
    compute_ndvi()

