import os
import rasterio
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon
import geopandas as gpd

def compute_polygon_iou(poly1: Polygon, poly2: Polygon):
    if not (poly1.is_valid and poly2.is_valid):
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0

def nms_global(poly_list, conf_list, iou_thresh=0.5):
    keep_idx = []
    sorted_idx = sorted(range(len(conf_list)), key=lambda x: conf_list[x], reverse=True)
    while len(sorted_idx) > 0:
        idx = sorted_idx.pop(0)
        keep_idx.append(idx)
        remove_index = []
        for j in range(len(sorted_idx)):
            j_idx = sorted_idx[j]
            iou = compute_polygon_iou(poly_list[idx], poly_list[j_idx])
            if iou > iou_thresh:
                remove_index.append(j)
        for j in reversed(remove_index):
            del sorted_idx[j]
    out_poly = [poly_list[i] for i in keep_idx]
    out_conf = [conf_list[i] for i in keep_idx]
    return out_poly, out_conf

def main():
    tif_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\rs_data\result.tif"
    weight_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\runs\middle_seg\tobacco_middle\weights\best.pt"
    save_shp = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\tobacco_plant.shp"
    crop_size = 640
    overlap = 120
    conf_thresh = 0.1
    step = crop_size - overlap
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(weight_path).to(device)

    all_polygon_geo = []
    all_conf = []
    all_cls = []

    with rasterio.open(tif_path) as src:
        img_w = src.width
        img_h = src.height
        gt = src.transform
        crs = src.crs
        print(f"图像宽度：{img_w}，图像高度：{img_h}")
        total_y = (img_h + step - 1) // step
        total_x = (img_w + step - 1) // step
        total_tiles = total_y * total_x
        tile_cnt = 0

        y_off = 0
        while y_off < img_h:
            x_off = 0
            while x_off < img_w:
                if x_off + crop_size > img_w:
                    x_off = img_w - crop_size
                    window = rasterio.windows.Window(x_off, y_off, crop_size, crop_size)
                    img_array = src.read(window=window)
                    # img_array (4,640,640)
                    # Band‑0:Green Band‑1:Red 构造 [Red,Green,Green]适配YOLO RGB输入顺序
                    rgb_img = img_array[[1, 0, 0], :, :]
                    results = model.predict(rgb_img, imgsz=crop_size, conf=conf_thresh, verbose=False)
                    tile_cnt += 1
                    print(f"已处理切片：{tile_cnt}/{total_tiles}, x_off:{x_off}, y_off:{y_off}")
                    for res in results:
                        if res.masks is None or res.boxes is None:
                            continue
                        masks_xy = res.masks.xyn
                        confs = res.boxes.conf.cpu().numpy()
                        cls_ids = res.boxes.cls.cpu().numpy()
                        for seg_idx, seg_points in enumerate(masks_xy):
                            geo_points = []
                            for (nx, ny) in seg_points:
                                px = x_off + nx * crop_size
                                py = y_off + ny * crop_size
                                gx = gt[0] + px * gt[1] + py * gt[2]
                                gy = gt[3] + px * gt[4] + py * gt[5]
                                geo_points.append((gx, gy))
                            poly = Polygon(geo_points)
                            if poly.is_valid and poly.area > 1e-8:
                                all_polygon_geo.append(poly)
                                all_conf.append(float(confs[seg_idx]))
                                all_cls.append(int(cls_ids[seg_idx]))
                    # 加一个缓存切片的机制，避免每次都切一遍
                    break
                window = rasterio.windows.Window(x_off, y_off, crop_size, crop_size)
                img_array = src.read(window=window)
                rgb_img = img_array[[1, 0, 0], :, :]
                results = model.predict(rgb_img, imgsz=crop_size, conf=conf_thresh, verbose=False)
                tile_cnt += 1
                print(f"已处理切片：{tile_cnt}/{total_tiles}, x_off:{x_off}, y_off:{y_off}")
                for res in results:
                    if res.masks is None or res.boxes is None:
                        continue
                    masks_xy = res.masks.xyn
                    confs = res.boxes.conf.cpu().numpy()
                    cls_ids = res.boxes.cls.cpu().numpy()
                    for seg_idx, seg_points in enumerate(masks_xy):
                        geo_points = []
                        for (nx, ny) in seg_points:
                            px = x_off + nx * crop_size
                            py = y_off + ny * crop_size
                            gx = gt[0] + px * gt[1] + py * gt[2]
                            gy = gt[3] + px * gt[4] + py * gt[5]
                            geo_points.append((gx, gy))
                        poly = Polygon(geo_points)
                        if poly.is_valid and poly.area > 1e-8:
                            all_polygon_geo.append(poly)
                            all_conf.append(float(confs[seg_idx]))
                            all_cls.append(int(cls_ids[seg_idx]))
                x_off += step
            y_off += step
    print("切片推理完毕，开始全局NMS去重")


    final_poly, final_conf = nms_global(all_polygon_geo, all_conf, iou_thresh=0.1)
    final_cls = [all_cls[i] for i in range(len(final_poly)) if all_polygon_geo[i] in final_poly]
    gdf = gpd.GeoDataFrame({
        "confidence": final_conf,
        "class_id": final_cls,
        "geometry": final_poly
    })
    gdf.crs = crs
    gdf.to_file(save_shp, encoding="utf-8")
    print(f"矢量shp输出完成：{save_shp}")
    print(f"识别到烟苗总数：{len(final_poly)}")
    print("现阶段说明：模型仅完成植株定位，现在模型精度不足，暂不区分病虫害，后面依靠NDVI判别")

if __name__ == "__main__":
    main()
