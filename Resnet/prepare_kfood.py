import os
import shutil
import random
from glob import glob
from tqdm import tqdm

# 원본 데이터 경로
SRC_ROOT = "한국 음식 이미지/kfood"
# 임시 합치기 폴더
DST_MERGED = "dataset"
# 최종 분할 폴더
DST_SPLIT = "dataset_split"

os.makedirs(DST_MERGED, exist_ok=True)
os.makedirs(DST_SPLIT, exist_ok=True)

# 1. 음식명 기준으로 이미지 합치기
for bigcat in os.listdir(SRC_ROOT):
    bigcat_path = os.path.join(SRC_ROOT, bigcat)
    if not os.path.isdir(bigcat_path):
        continue
    for food in os.listdir(bigcat_path):
        food_path = os.path.join(bigcat_path, food)
        if not os.path.isdir(food_path):
            continue
        dst_food_dir = os.path.join(DST_MERGED, food)
        os.makedirs(dst_food_dir, exist_ok=True)
        # 이미지 확장자별로 모두 복사
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img in glob(os.path.join(food_path, ext)):
                fname = os.path.basename(img)
                new_fname = f"{bigcat}_{food}_{fname}"
                shutil.copy(img, os.path.join(dst_food_dir, new_fname))

# 2. train/val 분할 (test까지 원하면 아래 주석 해제)
split_ratio = [0.8, 0.2]  # train, val
# split_ratio = [0.8, 0.1, 0.1]  # train, val, test

for food in tqdm(os.listdir(DST_MERGED)):
    food_dir = os.path.join(DST_MERGED, food)
    images = [
        f for f in os.listdir(food_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(images)
    n = len(images)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    # n_test = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        # "test": images[n_train + n_val:],
    }

    for split, split_imgs in splits.items():
        split_dir = os.path.join(DST_SPLIT, split, food)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_imgs:
            shutil.copy(os.path.join(food_dir, img), os.path.join(split_dir, img))

    # # test set까지 원할 때
    # split_dir = os.path.join(DST_SPLIT, "test", food)
    # os.makedirs(split_dir, exist_ok=True)
    # for img in splits["test"]:
    #     shutil.copy(os.path.join(food_dir, img), os.path.join(split_dir, img))

print("✅ 폴더 구조 변환 완료!")
