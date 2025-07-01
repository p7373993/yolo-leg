# ✅ 1. import 및 모델 로드
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import urllib.request

# 모델 파일명과 URL
checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
checkpoint_path = "sam_vit_l_0b3195.pth"

# 파일이 없으면 다운로드
if not os.path.exists(checkpoint_path):
    print(f"📥 {checkpoint_path} 파일이 없어 다운로드를 시작합니다...")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
    print("✅ 다운로드 완료!")
else:
    print("✅ checkpoint 파일이 이미 존재합니다.")


# SAM 모델 로드
sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path).to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

# ✅ 2. 이미지 불러오기 (경로 수정!)
image_path = "a.jpg"  # ← 여기에 실제 이미지 경로 입력
image_bgr = cv2.imread(image_path)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ✅ 3. 마스크 자동 생성
masks = mask_generator.generate(image)
print(f"🔍 생성된 마스크 수: {len(masks)}")

# ✅ 4. 마스크 시각화 함수
import random


def show_masks_on_image(image, masks):
    image = image.copy()
    for mask in masks:
        seg = mask["segmentation"]
        color = [random.randint(0, 255) for _ in range(3)]
        image[seg] = [int(c * 0.6 + ic * 0.4) for c, ic in zip(color, image[seg])]
    return image


# ✅ 5. 전체 마스크 시각화
vis_image = show_masks_on_image(image, masks)
plt.figure(figsize=(10, 10))
plt.imshow(vis_image)
plt.axis("off")
plt.title("Segmented Masks (SAM)")
plt.show()

# ✅ 6. 가장 큰 마스크 선택 + 시각화
biggest = max(masks, key=lambda x: x["area"])
mask = biggest["segmentation"]

image_with_one_mask = image.copy()
image_with_one_mask[mask] = [255, 0, 0]  # 빨간색 표시

plt.figure(figsize=(10, 10))
plt.imshow(image_with_one_mask)
plt.axis("off")
plt.title("Largest Mask Only")
plt.show()

# ✅ 7. 가장 큰 마스크 저장 (흑백 이미지로)
binary_mask = mask.astype(np.uint8) * 255
cv2.imwrite("food_mask.png", binary_mask)
print("✅ 마스크 저장 완료: food_mask.png")
