import os
import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# 폴더 구조 생성
paths = [
    'datasets/train/images', 'datasets/train/labels',
    'datasets/valid/images', 'datasets/valid/labels'
]
for path in paths:
    os.makedirs(path, exist_ok=True)

# Hugging Face 데이터 로드
print("데이터셋 로드 중...")
ds = load_dataset("ethz/food101", split='train', streaming=True)
samples = list(ds.take(100)) # 총 100장 추출

print("데이터셋 생성 중...")
for i, item in enumerate(tqdm(samples)):
    # 이미지 변환 (PIL -> OpenCV)
    image = np.array(item['image'])
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 크기 조정: 224x224
    image = cv2.resize(image, (224, 224)) 
    
    # 데이터 분할 및 저장
    if i < 80:
        base_path = 'datasets/train'
    else:
        base_path = 'datasets/valid'
    
    img_save_path = f'{base_path}/images/food_{i}.jpg'
    lbl_save_path = f'{base_path}/labels/food_{i}.txt'
    
    # 이미지 저장
    cv2.imwrite(img_save_path, image)
    
    # 4) 라벨 파일 생성: YOLO 형식
    # 박스 크기를 0.7로 설정
    with open(lbl_save_path, 'w') as f:
        f.write("0 0.5 0.5 0.7 0.7")

print("\n데이터셋 준비 완료.")
