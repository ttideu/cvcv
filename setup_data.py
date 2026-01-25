import os
import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# 1. 폴더 구조 생성 (이미지 및 라벨 폴더 포함)
paths = [
    'datasets/train/images', 'datasets/train/labels',
    'datasets/valid/images', 'datasets/valid/labels',
    'preprocessed_samples'
]
for path in paths:
    os.makedirs(path, exist_ok=True)

# 2. Hugging Face 데이터 로드 (Food-101 데이터셋 활용) [cite: 230]
print("데이터셋 로드 중...")
ds = load_dataset("ethz/food101", split='train', streaming=True)
samples = list(ds.take(100)) # 총 100장 추출

print("이미지 전처리 및 데이터셋(이미지+라벨) 생성 중...")
for i, item in enumerate(tqdm(samples)):
    # 이미지 변환 (PIL -> OpenCV)
    image = np.array(item['image'])
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # [업무 요청서 전처리 필수 사항]
    # 1) 크기 조정: 224x224 [cite: 230]
    image = cv2.resize(image, (224, 224)) 
    # 2) 노이즈 제거: Blur 필터 적용 [cite: 232]
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 3) 데이터 분할 및 저장 (8:2 비율)
    if i < 80:
        base_path = 'datasets/train'
    else:
        base_path = 'datasets/valid'
    
    img_save_path = f'{base_path}/images/food_{i}.jpg'
    lbl_save_path = f'{base_path}/labels/food_{i}.txt'
    
    # 이미지 저장
    cv2.imwrite(img_save_path, image)
    
    # 4) 라벨 파일(.txt) 생성: YOLO 형식 [cite: 403]
    # 음식 전체를 하나의 객체로 지정 (클래스 0, 중심 0.5 0.5, 크기 1.0 1.0)
    with open(lbl_save_path, 'w') as f:
        f.write("0 0.5 0.5 1.0 1.0")
    
    # [제출 항목] 전처리된 샘플 이미지 5장 별도 저장 [cite: 240]
    if i < 5:
        cv2.imwrite(f'preprocessed_samples/sample_{i}.jpg', image)

print("\n=== 모든 준비가 완료되었습니다 ===")
print("1. 'datasets' 폴더: 학습 및 검증 데이터 (이미지+라벨)")
print("2. 'preprocessed_samples' 폴더: 제출용 이미지 5장")
print("3. 이제 'train.py'를 실행하여 모델 학습을 시작하세요.")