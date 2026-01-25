# train.py
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("yolov8n.pt") 

# 2. 학습 설정
model.train(
    data="data.yaml",      
    epochs=30,            
    imgsz=224,             
    
    augment=True,          
    degrees=10.0,         
    fliplr=0.5,            
    hsv_v=0.2,             
    mosaic=0.0,            
    mixup=0.0
)