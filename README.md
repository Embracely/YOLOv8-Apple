# YOLOv8-Apple

## Introduction
YOLOv8-Apple is an improved YOLOv8 model designed to effectively detect small-scale apples and handle severe occlusions. A backbone network was constructed by replacing traditional stride convolutions and pooling layers with SPD-Conv to minimize the loss of fine-grained information, using SEAM to enhance feature representations of unoccluded regions and compensate for occluded ones at the neck, and integrating a lightweight CBAM attention mechanism to strengthen the representation capability of small targets. The results of comprehensive experiments on the MinneApple dataset demonstrate that the mAP of YOLOv8-Apple increases by 3.8%, and MAE and RMSE of counting tasks decrease by 11.8% and 28.8%, respectively.

## Frame Structure
<img src="structure.png" alt="Apple Detection Example" title="YOLOv8-Apple Structure" width="800">


## Evaluation

### Apple Detection and Counting

Below are the detection and counting results under different improvement configurations.

| SPD-Conv | SEAM | CBAM | mAP0.5 | F1 Score | Precision | Recall | MAE  | RMSE  |
|:--------:|:----:|:----:|:------:|:--------:|:---------:|:------:|:----:|:-----:|
|          |      |      | 0.725  | 0.709    | 0.753     | 0.670  | 2.97 | 13.28 |
| ✔        |      |      | 0.759  | 0.724    | 0.769     | 0.686  | 2.65 | 11.08 |
|          | ✔    |      | 0.744  | 0.704    | 0.755     | 0.659  | 2.54 | 9.49  |
|          |      | ✔    | 0.738  | 0.722    | 0.758     | 0.688  | 2.89 | 12.24 |
| ✔        | ✔    |      | 0.763  | 0.730    | 0.761     | 0.704  | 2.77 | 9.29  |
|          | ✔    | ✔    | 0.747  | 0.710    | 0.754     | 0.670  | 2.98 | 12.80 |
| ✔        |      | ✔    | 0.764  | 0.717    | 0.766     | 0.673  | 2.95 | 9.29  |
| ✔        | ✔    | ✔    | 0.763  | 0.726    | 0.768     | 0.688  | 2.62 | 9.45  |


## Environmental Requirements

1. **Create a Python Virtual Environment**  
   ```bash
   conda create -n {name} python=3.8

2. **Activate the Virtual Environment
   ```bash
   conda activate {name}

3. **Install Pytorch**  
   ```bash
   pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
---

## Step-Through Example

### Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/Embracely/YOLOv8-Apple.git

### Dataset

1. **Download the [MinneApple Dataset](https://conservancy.umn.edu/items/e1bb4015-e92a-4295-822c-d21d277ecfbd) into the data folder**  
    
2. **Convert the dataset’s mask tags YOLO annotation files**
   ```bash
   python mask2yolo.py
---

## Pretrained Model

You can download the pretrained weights at [yolov8-apple.pt](https://drive.google.com/file/d/10qP2b4g4UT-748k4UHBdf1XS_CPb2A6Y/view?usp=drive_link).

---

## Training

### Data Pre-processing

1. Split the data set

```bash
python split_dataset.py
```
2. Converting the format of data set

```bash
python voc_label.py
```

### Train your model on MinneApple

```bash
python train.py
```

## Test
```bash
python test_counting.py
```
