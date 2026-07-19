import os
import torch
import rasterio
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import geopandas as gpd
from pathlib import Path

def compute_polygon_iou(poly1: Polygon, poly2: Polygon):
    try:
        if not (poly1.is_valid and poly2.is_valid):
            return 0.0
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return inter_area / union_area if union_area > 1e-12 else 0.0
    except Exception:
        return 0.0

def global_nms(poly_list, conf_list, iou_thresh=0.1):
    if len(poly_list) == 0:
        return [], []
    # 绑定多边形与原始下标，解决index匹配失败
    poly_with_idx = [(poly, idx) for idx, poly in enumerate(poly_list)]
    tree = STRtree([p for p, idx in poly_with_idx])

    idx_sort = sorted(range(len(conf_list)), key=lambda x: conf_list[x], reverse=True)
    keep = []
    while idx_sort:
        cur = idx_sort.pop(0)
        cur_poly = poly_list[cur]
        keep.append(cur)
        drop_idx = []
        # 查询相交多边形，直接取出原始下标
        hit_polys = tree.query(cur_poly)
        hit_ids = []
        for p_hit in hit_polys:
            for p_ori, ori_idx in poly_with_idx:
                if p_hit is p_ori:
                    hit_ids.append(ori_idx)
                    break
        # 遍历剩余待处理索引
        for i in range(len(idx_sort)):
            test_idx = idx_sort[i]
            if test_idx not in hit_ids:
                continue
            iou = compute_polygon_iou(cur_poly, poly_list[test_idx])
            if iou > iou_thresh:
                drop_idx.append(i)
        # 倒序删除
        for i in reversed(drop_idx):
            idx_sort.pop(i)
    out_polys = [poly_list[i] for i in keep]
    out_confs = [conf_list[i] for i in keep]
    return out_polys, out_confs

def main():
    chunk_dir = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\chunk"
    origin_tif = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\rs_data\result.tif"
    weight_path = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\runs\middle_seg\tobacco_middle\weights\best.pt"
    out_shp = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\tobacco_project\output\tobacco_plant.shp"

    crop_size = 640
    conf_thresh = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(weight_path).to(device)

    all_polys = []
    all_confs = []
    all_cls = []

    with rasterio.open(origin_tif) as src:
        gt = src.transform
        crs = src.crs

    tile_paths = list(Path(chunk_dir).glob("*.tif"))
    total_tile = len(tile_paths)
    tile_cnt = 0

    for tile_path in tile_paths:
        tile_cnt += 1
        tile_name = tile_path.stem
        parts = tile_name.split("_")
        x_str = parts[2]
        y_str = parts[3]
        x_off = int(x_str.replace("x", ""))
        y_off = int(y_str.replace("y", ""))

        with rasterio.open(str(tile_path)) as tile_src:
            tile_data = tile_src.read()

        rgb = tile_data[[0, 1, 2], :, :]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb / rgb.max() * 255).clip(0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        results = model.predict(rgb, imgsz=crop_size, conf=conf_thresh, verbose=False)
        res = results[0]
        if res.masks is None:
            print(f"切片{tile_cnt}无识别目标")
            continue

        xyn_raw = res.masks.xyn
        if torch.is_tensor(xyn_raw):
            masks = xyn_raw.cpu().numpy()
        else:
            masks = [np.array(points) for points in xyn_raw]

        conf_raw = res.boxes.conf
        if torch.is_tensor(conf_raw):
            confs = conf_raw.cpu().numpy()
        else:
            confs = np.array(conf_raw)

        cls_raw = res.boxes.cls
        if torch.is_tensor(cls_raw):
            clses = cls_raw.cpu().numpy()
        else:
            clses = np.array(cls_raw)

        for idx, mask_xy in enumerate(masks):
            geo_coords = []
            for (nx, ny) in mask_xy:
                col = x_off + nx * crop_size
                row = y_off + ny * crop_size
                gx = gt.c + col * gt.a + row * gt.b
                gy = gt.f + col * gt.d + row * gt.e
                geo_coords.append((gx, gy))
            if len(geo_coords) < 4:
                continue
            poly = Polygon(geo_coords)
            if poly.is_valid and poly.area > 1e-10:
                all_polys.append(poly)
                all_confs.append(float(confs[idx]))
                all_cls.append(int(clses[idx]))

        print(f"推理切片：{tile_cnt}/{total_tile} x_off:{x_off}, y_off:{y_off}")

    print("全部切片推理完毕，执行全局NMS去重")
    print(f"NMS去重前检测到轮廓总数：{len(all_polys)}")
    final_polys, final_confs = global_nms(all_polys, all_confs, iou_thresh=0.1)
    print(f"NMS去重后剩余有效烟草轮廓：{len(final_polys)}")
    final_cls = [all_cls[i] for i, p in enumerate(all_polys) if p in final_polys]

    gdf = gpd.GeoDataFrame({
        "confidence": final_confs,
        "class_id": final_cls,
        "geometry": final_polys
    }, crs=crs)
    out_dir = Path(out_shp).parent
    out_dir.mkdir(exist_ok=True)
    gdf.to_file(out_shp, encoding="utf-8")
    print(f"矢量shp输出完成：{out_shp}")
    print(f"==== 本次识别烟草总数量：{len(final_polys)} ====")

if __name__ == "__main__":
    main()
