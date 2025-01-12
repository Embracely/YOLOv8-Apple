import os
from ultralytics import YOLO
import math

# 配置路径（请根据实际情况修改）
model_path = r'E:\UOB\MV\project\ultralytics-main\runs\detect\train-cbam-seam\weights\best.pt' 
data_dir = r"E:\UOB\MV\project\ultralytics-main\my_data\counting\train"
images_dir = os.path.join(data_dir, "images")
ground_truth_file = os.path.join(data_dir, "train_ground_truth.txt")

img_ext = '.jpg'  # 如果图片后缀为.png请根据情况修改

# 加载模型
model = YOLO(model_path)

# 从ground_truth文件中读取图片名称和真实苹果数
# 文件格式：img_001.jpg,10
image_names = []
ground_truth_counts = []
with open(ground_truth_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 2:
            print(f"警告：行格式不正确: {line}")
            continue
        img_name = parts[0].strip()
        gt_str = parts[1].strip()
        # 转换为整数
        try:
            gt_count = int(gt_str)
        except ValueError:
            print(f"警告：无法解析真实苹果数: {gt_str}")
            continue
        image_names.append(img_name)
        ground_truth_counts.append(gt_count)

# 确保有数据
N = len(image_names)
if N == 0:
    raise ValueError("没有从train_ground_truth.txt中读取到任何有效图像数据。")

# 初始化用于计算误差的列表
absolute_errors = []
squared_errors = []
percentage_errors = []

for img_name, gt_count in zip(image_names, ground_truth_counts):
    img_path = os.path.join(images_dir, img_name)

    if not os.path.exists(img_path):
        # 尝试加上扩展名 img_ext 看是否存在
        if not img_name.endswith(img_ext):
            img_path_alt = os.path.join(images_dir, img_name + img_ext)
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
            else:
                print(f"警告：图片文件 {img_name} 不存在。")
                continue
        else:
            print(f"警告：图片文件 {img_name} 不存在。")
            continue

    # 使用模型预测
    results = model.predict(img_path, verbose=False)
    predicted_count = len(results[0].boxes)

    diff = predicted_count - gt_count
    absolute_errors.append(abs(diff))
    squared_errors.append(diff**2)

    # 若使用MAPE需要gt_count > 0
    if gt_count > 0:
        percentage_error = abs(diff) / gt_count
        percentage_errors.append(percentage_error)

# 计算指标
if len(absolute_errors) == 0:
    print("没有有效的数据进行统计。")
else:
    # MAE
    MAE = sum(absolute_errors) / len(absolute_errors)

    # MSE
    MSE = sum(squared_errors) / len(squared_errors)

    # RMSE
    RMSE = math.sqrt(MSE)

    print(f"MAE (平均绝对误差): {MAE:.2f}")
    print(f"MSE (均方误差): {MSE:.2f}")
    print(f"RMSE (均方根误差): {RMSE:.2f}")

    if len(percentage_errors) > 0:
        MAPE = (sum(percentage_errors) / len(percentage_errors)) * 100
        print(f"MAPE (平均绝对百分比误差): {MAPE:.2f}%")
    else:
        print("由于真实值为0的情况，无法计算MAPE或没有合适的样本计算MAPE。")
