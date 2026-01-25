# AI 기반 데이터 모델링 및 OpenCV를 활용한 결과 시각화

## 개요
**YOLOv8** 모델을 활용하여 음식 이미지를 탐지하는 AI 모델을 구축합니다.
**Food-101** 데이터셋을 활용하였으며, 위치 라벨이 없는 데이터셋의 한계를 극복하기 위해 중앙 고정 라벨링 을 적용했습니다.

## 실행
setup_data.py - train.py - result.py

## 구조
```text
📦 Project Root
┣ 📂 datasets           # 데이터셋 저장소 (자동 생성)
┣ 📂 runs               # 학습 결과 및 모델 가중치 저장소 (자동 생성)
┣ 📜 setup_data.py      # 데이터 다운로드, 전처리 및 라벨링 생성
┣ 📜 train.py           # YOLOv8 모델 학습 설정 및 실행
┣ 📜 result.py          # 학습 결과 평가 및 그래프 시각화
┣ 📜 data.yaml          # 데이터셋 경로 및 클래스 설정
┗ 📜 README.md          # 프로젝트 설명 문서
```

## 환경
* Python 3.8+
* Ultralytics YOLOv8
* OpenCV, NumPy, Pandas, Matplotlib, Hugging Face Datasets

### 필수 라이브러리

pip install ultralytics opencv-python numpy pandas matplotlib datasets tqdm


