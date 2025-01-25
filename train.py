from ultralytics import YOLO
 
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('yolov8-apple.pt')  # 从YAML建立一个新模型，可在同一路径下更换带训练模型的yaml文件
    model.train(data=r'E:\UOB\MV\project\ultralytics-main\wheat.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                single_cls=False, 
                batch=8,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                )
