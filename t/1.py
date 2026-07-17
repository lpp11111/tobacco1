import os

def check_wrong_seg_labels():
    root = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\t"
    label_folders = [os.path.join(root, "train/labels"), os.path.join(root, "valid/labels")]
    nc = 3
    empty_txt = []
    wrong_cls = []
    bad_polygon = []
    for folder in label_folders:
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                empty_txt.append(file)
                continue
            lines = content.splitlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                if cls < 0 or cls >= nc:
                    wrong_cls.append((file, cls))
                coords = parts[1:]
                # 判断轮廓坐标全部为0
                if all(x == 0 for x in coords):
                    bad_polygon.append(file)
    print("======检测结果======")
    if len(empty_txt):
        print(f"空标签文件共{len(empty_txt)}个，示例：{empty_txt[:5]}")
    if len(wrong_cls):
        print(f"类别ID越界文件：{wrong_cls[:5]}")
    if len(bad_polygon):
        print(f"轮廓坐标全为0的坏标签（根源报错文件）：{bad_polygon[:5]}")
    if len(empty_txt) == 0 and len(wrong_cls) == 0 and len(bad_polygon) == 0:
        print("标签文件本身没问题，问题出在Mosaic增强，执行步骤2")

if __name__ == "__main__":
    check_wrong_seg_labels()
