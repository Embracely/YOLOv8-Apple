import os
import cv2
import numpy as np

# Class settings. If there is only one class (e.g., apple), then classes = ['apple'] and class_id = 0
classes = ['apple']
class_id = 0

images_dir = r'detection/train/images'  # Directory containing training and testing images
masks_dir = r'detection/train/masks'    # Directory where original masks are located
labels_dir = r'detection/train/labels'  # Directory to save preprocessed labels
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

for img_file in image_files:
    mask_file = img_file.replace('.jpg', '.png')  # Modify based on your mask format
    mask_path = os.path.join(masks_dir, mask_file)

    if not os.path.exists(mask_path):
        continue

    # Read the image and mask
    img_path = os.path.join(images_dir, img_file)
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    height, width = image.shape[:2]

    # Gamma correction (adjust gamma value as needed)
    gamma = 0.1
    mask_float = mask.astype(np.float32) / 255.0
    mask_gamma = np.power(mask_float, gamma) * 255
    mask_gamma = mask_gamma.astype(np.uint8)

    # Otsu's thresholding
    _, binary = cv2.threshold(mask_gamma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Start of watershed algorithm process
    # Distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Threshold the distance image to obtain the foreground
    # Adjust the threshold value from 0.3 to 0.5 as needed
    _, sure_fg = cv2.threshold(dist, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    # Morphological operations (adjust parameters as needed)
    kernel = np.ones((3, 3), np.uint8)
    sure_fg = cv2.dilate(sure_fg, kernel, iterations=3)  # Optional dilation on foreground

    # Find the background area
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = markers.astype(np.int32)
    markers = cv2.watershed(image, markers)

    # Calculate contours and bounding boxes for each unique marker
    unique_labels = np.unique(markers)
    unique_labels = unique_labels[unique_labels > 1]  # Remove background and watershed lines

    # Create a text file with the same name to store YOLO labels
    txt_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))
    with open(txt_path, 'w') as f:
        for label_id in unique_labels:
            obj_mask = np.uint8(markers == label_id) * 255
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, box_w, box_h = cv2.boundingRect(cnt)
                x_center = (x + box_w / 2) / width
                y_center = (y + box_h / 2) / height
                w_norm = box_w / width
                h_norm = box_h / height
                f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")
