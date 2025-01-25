import os
import shutil
from os import getcwd

# 数据集划分类型
sets = ['train', 'val', 'test']

# 请在此填写您的类别列表（如果后续还需使用到类别信息的话）
classes = ['0']

abs_path = os.getcwd()
print(abs_path)

# 确保data/labels目录存在
if not os.path.exists('data/detection/train/labels'):
    os.makedirs('data/detection/train/labels')

# 根目录（含有标签txt文件的目录）
src_label_dir = r"data\detection\train\labels"

for image_set in sets:
    image_ids = open('data/detection/train/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('data/detection/train/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        # 源标签文件路径
        src_label_path = os.path.join(src_label_dir, image_id + ".txt")
        # 目标标签文件路径
        dst_label_path = os.path.join('data/detection/train/labels', image_id + ".txt")
        
        # 将标签文件复制到指定目录
        if os.path.exists(src_label_path):
            shutil.copyfile(src_label_path, dst_label_path)
        else:
            # 若对应标签不存在，可根据需要决定是否报错或跳过
            print(f"Warning: Label file not found for {image_id}")

        # 写入对应的数据集txt文件记录图片的绝对路径
        list_file.write(abs_path + '/data/detection/train/images/%s.jpg\n' % (image_id))
    list_file.close()
