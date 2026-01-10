import cv2
import numpy as np
import os

# 0. 폴더 설정 및 필터링 기준
output_dir = 'preprocessed_samples'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

BRIGHTNESS_THRESHOLD = 50  # 이 값보다 낮으면 제거 (너무 어두움)
MIN_AREA_THRESHOLD = 500   # 가장 큰 객체의 픽셀 면적이 이보다 작으면 제거

# 1. 이미지 로드
image = cv2.imread('sample.jpg')

if image is None:
    print("오류: 이미지를 로드할 수 없습니다.")
else:
    # --- [필터링 1] 평균 밝기 검사 ---
    temp_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(temp_gray)
    
    # --- [필터링 2] 객체 크기 검사 (이진화 후 윤곽선 면적 계산) ---
    _, thresh = cv2.threshold(temp_gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max([cv2.contourArea(c) for c in contours]) if contours else 0

    # 이상치 체크
    if avg_brightness < BRIGHTNESS_THRESHOLD:
        print(f"이미지 제거: 너무 어두움 (밝기: {avg_brightness:.2f})")
    elif max_area < MIN_AREA_THRESHOLD:
        print(f"이미지 제거: 객체가 너무 작음 (면적: {max_area})")
    else:
        print(f"검증 통과 (밝기: {avg_brightness:.2f}, 면적: {max_area}) - 처리를 시작합니다.")

        # --- [기능 1] 크기 조정 (224x224) ---
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{output_dir}/1_resized.jpg', resized)

        # --- [기능 3] 노이즈 제거 (Blur 필터) ---
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        cv2.imwrite(f'{output_dir}/2_blurred.jpg', blurred)

        # --- [기능 2] 색상 변환 및 정규화 (Grayscale & Normalize) ---
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # 파일 저장용 (0~255)
        norm_for_save = (gray.astype(np.float32) / 255.0 * 255).astype(np.uint8)
        cv2.imwrite(f'{output_dir}/3_gray_normalized.jpg', norm_for_save)

        # --- [기능 4] 데이터 증강 (Data Augmentation) ---
        # 4-1. 좌우 반전
        flipped = cv2.flip(resized, 1)
        cv2.imwrite(f'{output_dir}/4_aug_flipped.jpg', flipped)

        # 4-2. 90도 회전 (시계 방향)
        rotated_90 = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f'{output_dir}/5_aug_rotated_90.jpg', rotated_90)

        # 4-3. 색상 변화 (밝기 증강)
        brightened = cv2.convertScaleAbs(resized, alpha=1.2, beta=10)
        cv2.imwrite(f'{output_dir}/6_aug_brightened.jpg', brightened)

        print(f"전처리 및 증강 완료. '{output_dir}' 폴더를 확인하세요.")