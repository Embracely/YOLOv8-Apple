import os
import cv2
import numpy as np

# 类别设置，如果只有苹果一类，则 classes = ['apple'] 并 class_id = 0
classes = ['apple']
class_id = 0

images_dir = r'detection/train/images'  # 放置训练和测试图片的目录
masks_dir = r'detection/train/masks'    # 原始的mask所在位置
labels_dir = r'detection/train/labels'
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

example_saved = False  # 用于确保只保存一次示例输出

for img_file in image_files:
    mask_file = img_file.replace('.jpg','.png') # 根据你的mask格式修改
    mask_path = os.path.join(masks_dir, mask_file)

    if not os.path.exists(mask_path):
        continue

    # 读取图片和mask
    img_path = os.path.join(images_dir, img_file)
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    h, w = image.shape[:2]

    # Gamma矫正 (可根据需要微调gamma值)
    gamma = 0.1
    mask_float = mask.astype(np.float32) / 255.0
    mask_gamma = np.power(mask_float, gamma) * 255
    mask_gamma = mask_gamma.astype(np.uint8)

    # Otsu阈值化
    _, binary = cv2.threshold(mask_gamma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 分水岭算法流程开始
    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 对距离图进行阈值化获取前景
    # 根据实际情况微调阈值0.3 -> 0.5等
    ret, sure_fg = cv2.threshold(dist, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    # 形态学处理（可根据情况调整/尝试不同参数）
    kernel = np.ones((3,3), np.uint8)
    sure_fg = cv2.dilate(sure_fg, kernel, iterations=3) #可选的腐蚀处理前景

    # 找到背景区域
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    # 未确定区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记连通组件
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 分水岭
    markers = markers.astype(np.int32)
    markers = cv2.watershed(image, markers)

    # 为每个独立标记计算轮廓和边界框
    unique_labels = np.unique(markers)
    unique_labels = unique_labels[(unique_labels > 1)]  # 去除背景和分水岭线

    # 创建同名txt文件，用于存储YOLO标签
    txt_path = os.path.join(labels_dir, img_file.replace('.jpg','.txt'))
    with open(txt_path, 'w') as f:
        for label_id in unique_labels:
            obj_mask = np.uint8(markers == label_id) * 255
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x,y,box_w,box_h = cv2.boundingRect(cnt)
                x_center = (x + box_w/2) / w
                y_center = (y + box_h/2) / h
                w_norm = box_w / w
                h_norm = box_h / h
                f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

    # 示例输出（只做一次）
    if not example_saved:
        # 保存分水岭分割后的binary对比
        cv2.imwrite('binary/binary_watershed.jpg', binary)

        # 在原图上绘制Bounding Box示例
        draw_img = image.copy()
        for label_id in unique_labels:
            obj_mask = np.uint8(markers == label_id) * 255
            cts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cts:
                x,y,box_w,box_h = cv2.boundingRect(c)
                cv2.rectangle(draw_img, (x, y), (x+box_w, y+box_h), (0, 255, 0), 2)

        cv2.imwrite('watershed_bounding_box_example.png', draw_img)
        example_saved = True

print("处理完成。已在当前目录下输出 watershed_binary.png 和 watershed_bounding_box_example.png 两个示例文件（仅一次）。")
