# AI 기반 데이터 모델링 및 OpenCV를 활용한 결과 시각화

## 개요
**YOLOv8** 모델을 활용하여 음식 이미지를 탐지하는 AI 모델을 구축합니다.<br>
**Food-101** 데이터셋을 활용하였으며,<br>
모든 이미지에 대해 중앙 좌표(0.5, 0.5)를 기준으로 너비와 높이가 이미지의 70%를 차지하는 박스를 자동으로 생성하여 라벨을 부여했습니다.
```text
setup_data.py

with open(lbl_save_path, 'w') as f:
        f.write("0 0.5 0.5 0.7 0.7")
```

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





