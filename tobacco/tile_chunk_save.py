import os
import rasterio
from rasterio.windows import Window

def main():
    src_tif = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\rs_data\result.tif"
    out_dir = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\chunk"
    os.makedirs(out_dir, exist_ok=True)

    crop_size = 640
    overlap = 120
    step = crop_size - overlap

    with rasterio.open(src_tif) as src:
        img_w = src.width
        img_h = src.height
        tile_idx = 0
        print(f"原图宽{img_w}  原图高{img_h}")

        # 预生成坐标，杜绝死循环
        y_list = []
        y = 0
        while y <= img_h - crop_size:
            y_list.append(y)
            y += step
        if y_list[-1] + crop_size < img_h:
            y_list.append(img_h - crop_size)

        for y in y_list:
            x_list = []
            x = 0
            while x <= img_w - crop_size:
                x_list.append(x)
                x += step
            if x_list[-1] + crop_size < img_w:
                x_list.append(img_w - crop_size)

            for x in x_list:
                win = Window(x, y, crop_size, crop_size)
                data = src.read(window=win)
                mean_pixel = data.mean()
                print(f"切片{tile_idx} x{x},y{y} 像素均值:{mean_pixel:.3f}")

                # 过滤空白切片
                if mean_pixel < 0.02:
                    tile_idx += 1
                    continue

                profile = src.profile.copy()
                win_transform = src.window_transform(win)
                profile.update(width=crop_size, height=crop_size, transform=win_transform)
                save_name = f"tile_{tile_idx}_x{x}_y{y}.tif"
                save_full_path = os.path.join(out_dir, save_name)
                with rasterio.open(save_full_path, "w", **profile) as dst:
                    dst.write(data)
                tile_idx += 1

    print("===== 切片全部完成 =====")
    print(f"有效切片总数：{tile_idx}")

if __name__ == "__main__":
    main()
