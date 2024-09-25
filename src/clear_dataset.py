import os
from PIL import Image, UnidentifiedImageError
from config import train_dict, test_dict


def delete_corrupted_images(image_folder):
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        try:
            with Image.open(img_path) as img:
                img.verify()  # 尝试验证图片
            print(f"{img_file} is valid.")
        except (IOError, SyntaxError, UnidentifiedImageError) as e:
            print(f"Error: {img_file} is corrupted or unreadable. Deleting...")
            os.remove(img_path)  # 删除损坏的图片


# 训练集和数据集中有损坏的图片，将他们删除
delete_corrupted_images(train_dict['image_dir'])
delete_corrupted_images(test_dict['image_dir'])