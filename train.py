from ultralytics import YOLO
 
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-seam.yaml')  # 从YAML建立一个新模型
    model.train(data=r'E:\UOB\MV\project\ultralytics-main\wheat.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                single_cls=False,  # 是否是单类别检测
                batch=8,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                )