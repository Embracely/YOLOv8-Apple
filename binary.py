import cv2
import numpy as np

# 读取二值化后的图像
binary = cv2.imread('binary/binary_gamma_otsu.png', cv2.IMREAD_GRAYSCALE)

# 确保前景为白(255)，背景为黑(0)
# 如果颜色反转（背景是白，前景是黑），可以翻转颜色
# binary = cv2.bitwise_not(binary)

# 距离变换，计算前景区域到背景的距离
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

# 阈值化距离图，提取核心区域
ret, sure_fg = cv2.threshold(dist, 0.7, 1.0, cv2.THRESH_BINARY)

# 转换为uint8类型
sure_fg = (sure_fg * 255).astype(np.uint8)

# 扩大前景区域，让分水岭的标记更加清晰
kernel = np.ones((3, 3), np.uint8)
sure_fg = cv2.dilate(sure_fg, kernel, iterations=3)

# 计算背景区域
sure_bg = cv2.dilate(binary, kernel, iterations=3)

# 计算未知区域
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通区域
ret, markers = cv2.connectedComponents(sure_fg)

# 分水岭要求背景标记不能为0，故将标记值加1
markers = markers + 1

# 将未知区域标记为0
markers[unknown == 255] = 0

# 对二值化图像执行分水岭算法
markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

# 可视化分割结果
# 将分水岭线标记为红色
output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # 转换为彩色图以便绘制
output[markers == -1] = [0, 0, 255]  # 分水岭线用红色标记

# 为每个分割区域随机分配颜色
for label in np.unique(markers):
    if label > 1:  # 忽略背景和分水岭线
        obj_mask = np.uint8(markers == label)
        output[obj_mask == 255] = np.random.randint(0, 255, size=3).tolist()

cv2.imwrite("watershed_binary_only.jpg", output)
