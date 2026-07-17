import os

def convert_bbox_to_seg(label_dir):
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_dir, filename)
            new_lines = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = parts[0]
                    # 取出归一化包围盒 xmin,ymin,xmax,ymax
                    x1 = parts[1]
                    y1 = parts[2]
                    x2 = parts[3]
                    y2 = parts[4]
                    # 矩形四个顶点作为分割轮廓，满足YOLO‑Seg格式要求
                    seg_line = f"{cls_id} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}\n"
                    new_lines.append(seg_line)
            with open(file_path, "w", encoding="utf-8") as out_f:
                out_f.writelines(new_lines)

train_labels = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\t\train\labels"
val_labels = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\t\valid\labels"
convert_bbox_to_seg(train_labels)
convert_bbox_to_seg(val_labels)
print("标签格式转换完成")
