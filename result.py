import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

def get_latest_train_dir():
    # runs/detect/train* 폴더들 중 최근에 생성된 폴더를 찾습니다.
    train_dirs = glob.glob('runs/detect/train*')
    if not train_dirs:
        raise FileNotFoundError("학습 결과 폴더가 없습니다.")
    return max(train_dirs, key=os.path.getmtime)

# 학습 폴더 감지
latest_dir = get_latest_train_dir()
best_model_path = os.path.join(latest_dir, "weights", "best.pt")
results_csv_path = os.path.join(latest_dir, "results.csv")

print(f"최신 학습 경로 감지됨: {latest_dir}")

# 모델 성능 평가
print("\n모델 성능 평가 중...")
model = YOLO(best_model_path)
metrics = model.val(data="data.yaml", verbose=False)

print("\n" + "="*30)
print(f"최종 성능 지표 (Path: {latest_dir})")
print(f" - Precision (정밀도): {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f" - Recall (재현율):    {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f" - mAP50 (평균 정밀도): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print("="*30 + "\n")

# 학습 결과 그래프 출력
if os.path.exists(results_csv_path):
    print("성능 그래프 생성 중...")
    
    # 결과 CSV 읽기 및 공백 제거
    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/precision(B)'], label="Precision", marker='o')
    plt.plot(df['epoch'], df['metrics/recall(B)'], label="Recall", marker='s')

    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Model Performance")
    plt.legend()
    plt.grid(True)

    # 결과 저장 및 출력
    save_path = "performance_graph.png"
    plt.savefig(save_path)
    print(f"그래프 저장 완료: {save_path}")
    plt.show()
else:
    print("results.csv 파일을 찾을 수 없어 그래프를 그릴 수 없습니다.")