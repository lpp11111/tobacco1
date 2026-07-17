import os

def clean_empty_labels(root_dir):
    folders = ["train", "valid"]
    for folder in folders:
        label_dir = os.path.join(root_dir, folder, "labels")
        image_dir = os.path.join(root_dir, folder, "images")
        for filename in os.listdir(label_dir):
            if filename.endswith(".txt"):
                txt_path = os.path.join(label_dir, filename)
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if len(content) == 0:
                    os.remove(txt_path)
                    stem = os.path.splitext(filename)[0]
                    for suffix in [".png", ".jpg"]:
                        img_path = os.path.join(image_dir, stem + suffix)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            print(f"删除空标签对应的图片：{stem}{suffix}")

root = r"E:\Yolov8-Remose-Image-Dataset-Process-Tool-Set-main\t"
clean_empty_labels(root)
