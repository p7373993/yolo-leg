import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import kagglehub


def convert_mask_to_yolo_seg(mask_path, image_shape):
    """
    하나의 마스크 이미지를 YOLO 세그멘테이션 라벨(.txt) 형식으로 변환합니다.
    각 객체의 외곽선(polygon) 좌표를 찾아 정규화합니다.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    img_h, img_w = image_shape
    yolo_labels = []

    unique_class_ids = np.unique(mask)

    for class_id in unique_class_ids:
        if class_id == 0:  # 배경(0)은 건너뜁니다.
            continue

        # YOLO 클래스 ID는 0부터 시작하므로, 원본 ID에서 1을 빼줍니다.
        yolo_class_id = int(class_id) - 1

        # 현재 클래스에 해당하는 이진 마스크를 생성합니다.
        binary_mask = np.uint8(mask == class_id)

        # 이진 마스크에서 외곽선을 찾습니다.
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # 너무 작은 객체(노이즈)는 무시합니다.
            if cv2.contourArea(contour) < 20:
                continue

            # 외곽선 좌표를 0~1 사이의 값으로 정규화합니다.
            contour = contour.astype(np.float32)
            contour[:, 0, 0] /= img_w  # x 좌표 정규화
            contour[:, 0, 1] /= img_h  # y 좌표 정규화

            # YOLO 형식에 맞게 좌표를 1차원 리스트로 펼칩니다.
            polygon_points = contour.flatten().tolist()

            # 최종 라벨 문자열을 만듭니다: <class_id> <x1> <y1> <x2> <y2> ...
            label_line = f"{yolo_class_id} " + " ".join(
                [f"{p:.6f}" for p in polygon_points]
            )
            yolo_labels.append(label_line)

    return yolo_labels


def process_and_organize_dataset(base_path, yolo_base_path):
    """
    FoodSeg103 데이터셋 전체를 다운로드하고 YOLO 형식으로 변환하여 저장합니다.
    """
    # ★★★ 오류 수정: 경로에 'FoodSeg103' 상위 폴더를 추가합니다. ★★★
    dataset_root = os.path.join(base_path, "FoodSeg103")
    image_dir = os.path.join(dataset_root, "Images", "img_dir")
    mask_dir = os.path.join(dataset_root, "Images", "ann_dir")

    # YOLO 데이터셋 폴더 구조 생성
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_base_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_base_path, "labels", split), exist_ok=True)
    print(f"YOLO 데이터셋 폴더가 '{yolo_base_path}'에 생성되었습니다.")

    # 데이터셋 분할 처리 (train/test -> train/val)
    for split in ["train", "test"]:
        yolo_split = "val" if split == "test" else "train"
        print(
            f"\n'{split}' 데이터셋을 처리하여 YOLO '{yolo_split}' 세트로 변환합니다..."
        )

        img_split_dir = os.path.join(image_dir, split)
        mask_split_dir = os.path.join(mask_dir, split)

        image_files = [f for f in os.listdir(img_split_dir) if f.endswith(".jpg")]

        for img_name in tqdm(image_files, desc=f"Converting {split} set"):
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(img_split_dir, img_name)
            mask_path = os.path.join(mask_split_dir, base_name + ".png")

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape

            yolo_labels = convert_mask_to_yolo_seg(mask_path, (h, w))

            if yolo_labels:
                dest_img_path = os.path.join(
                    yolo_base_path, "images", yolo_split, img_name
                )
                cv2.imwrite(dest_img_path, img)

                label_path = os.path.join(
                    yolo_base_path, "labels", yolo_split, base_name + ".txt"
                )
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_labels))

    print("\n모든 데이터셋 처리가 성공적으로 완료되었습니다!")


def create_yaml_file(base_path, yolo_base_path):
    """
    YOLO 학습에 필요한 data.yaml 설정 파일을 생성합니다.
    """
    # ★★★ 오류 수정: 경로에 'FoodSeg103' 상위 폴더를 추가합니다. ★★★
    dataset_root = os.path.join(base_path, "FoodSeg103")
    class_names = []
    with open(
        os.path.join(dataset_root, "category_id.txt"), "r", encoding="utf-8"
    ) as f:
        lines = f.readlines()
        sorted_lines = sorted(lines, key=lambda x: int(x.strip().split("\t")[0]))
        for line in sorted_lines:
            parts = line.strip().split("\t")
            if parts[0] != "0":
                class_names.append(parts[1])

    yaml_data = {
        "path": os.path.abspath(yolo_base_path),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_file_path = os.path.join(yolo_base_path, "data.yaml")
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data, f, sort_keys=False, allow_unicode=True, default_flow_style=False
        )

    print(f"\n'data.yaml' 파일이 다음 경로에 생성되었습니다: {yaml_file_path}")
    print("\n--- YAML 파일 내용 ---")
    print(yaml.dump(yaml_data, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    print("FoodSeg103 데이터셋 다운로드를 시작합니다...")
    dataset_path = kagglehub.dataset_download("ggrill/foodseg103")
    print(f"데이터셋이 '{dataset_path}'에 다운로드 되었습니다.")

    yolo_dataset_path = "./FoodSeg103_YOLO_seg"

    process_and_organize_dataset(dataset_path, yolo_dataset_path)

    create_yaml_file(dataset_path, yolo_dataset_path)
