from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("yolov8n.pt") 

# 2. 학습 및 실시간 데이터 증강 설정
model.train(
    data="data.yaml",      # 데이터 설정 파일
    epochs=10,             # 학습 횟수
    imgsz=224,             # 전처리된 크기에 맞춤
    augment=True,          # 데이터 증강 활성화 
    degrees=15.0,          # 15도 범위 내 무작위 회전
    fliplr=0.5,            # 50% 확률로 좌우 반전
    hsv_v=0.2              # 밝기 변화
)