import os
import random
import shutil

# ----------------配置你的路径----------------
root = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\t"
train_img_dir = os.path.join(root, "train", "images")
train_label_dir = os.path.join(root, "train", "labels")
val_img_dir = os.path.join(root, "valid", "images")
val_label_dir = os.path.join(root, "valid", "labels")

val_ratio = 0.2   # 拿出20%样本做验证集，中尺度数据集黄金比例

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

img_list = [f for f in os.listdir(train_img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
random.seed(42)   # 设置随机种子，保证每次划分结果固定，可复现
random.shuffle(img_list)
val_num = int(len(img_list) * val_ratio)
val_img_names = img_list[:val_num]

for img_name in val_img_names:
    # 移动图片
    shutil.move(os.path.join(train_img_dir, img_name), os.path.join(val_img_dir, img_name))
    # 匹配标签，后缀换成txt
    label_name = os.path.splitext(img_name)[0] + ".txt"
    shutil.move(os.path.join(train_label_dir, label_name), os.path.join(val_label_dir, label_name))

print(f"划分完成：训练集{len(img_list)-val_num}张，验证集{val_num}张")
