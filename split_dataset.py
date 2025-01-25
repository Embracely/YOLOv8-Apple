import os
import random

# 设置随机种子（可选），以便结果可重复
random.seed(0)

# 原始图片目录
imgfilepath = 'data/detection/train/images'
# 输出txt文件路径
txtsavepath = 'data/detection/train/ImageSets'
os.makedirs(txtsavepath, exist_ok=True)

# 获取所有图片文件名
total_imgs = [f for f in os.listdir(imgfilepath) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

# 总数
num = len(total_imgs)

# 数据集比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 随机打乱数据
random.shuffle(total_imgs)

# 根据比例计算各子集数量
train_count = int(num * train_ratio)
val_count = int(num * val_ratio)
test_count = num - train_count - val_count  # 剩余的分配给test

# 分配数据集
train_set = total_imgs[:train_count]
val_set = total_imgs[train_count:train_count + val_count]
test_set = total_imgs[train_count + val_count:]

# trainval包含train和val
trainval_set = train_set + val_set

# 写入文件
ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

# 写入trainval.txt
for img in trainval_set:
    name = os.path.splitext(img)[0] + '\n'  # 去掉扩展名
    ftrainval.write(name)

# 写入test.txt
for img in test_set:
    name = os.path.splitext(img)[0] + '\n'
    ftest.write(name)

# 写入train.txt
for img in train_set:
    name = os.path.splitext(img)[0] + '\n'
    ftrain.write(name)

# 写入val.txt
for img in val_set:
    name = os.path.splitext(img)[0] + '\n'
    fval.write(name)

ftrainval.close()
ftest.close()
ftrain.close()
fval.close()

print("数据集划分完成!")
