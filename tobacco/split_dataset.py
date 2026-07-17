import os
import shutil
import random

# 已匹配你的本地路径，无需修改
base_dir = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\t"
val_ratio = 0.2  # 验证集占比20%，600张图拆分出120张验证集
random.seed(42)

# 路径定义
train_img_dir = os.path.join(base_dir, "train", "images")
train_label_dir = os.path.join(base_dir, "train", "labels")
val_img_dir = os.path.join(base_dir, "valid", "images")
val_label_dir = os.path.join(base_dir, "valid", "labels")

# 自动创建验证集文件夹
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有图片并随机打乱
img_list = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(img_list)
val_num = int(len(img_list) * val_ratio)
val_imgs = img_list[:val_num]

# 同步复制图片和对应分割标签
for img_name in val_imgs:
    label_name = os.path.splitext(img_name)[0] + ".txt"
    shutil.copy(os.path.join(train_img_dir, img_name), os.path.join(val_img_dir, img_name))
    if os.path.exists(os.path.join(train_label_dir, label_name)):
        shutil.copy(os.path.join(train_label_dir, label_name), os.path.join(val_label_dir, label_name))

print(f"拆分完成，验证集共{val_num}张图片")