import os
import shutil
import random
from tqdm import tqdm
import logging
from pathlib import Path

# 로깅 설정: 진행 상황을 터미널에 명확하게 표시
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 경로 설정 ---
# 이 스크립트 파일이 있는 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).resolve().parent

# 원본 데이터(kfood)가 있는 경로
SRC_ROOT = PROJECT_ROOT / "한국 음식 이미지" / "kfood"
# 1단계: 모든 이미지를 클래스별로 모아둘 임시 폴더
DST_MERGED = PROJECT_ROOT / "dataset"
# 2단계: 최종적으로 train/val 분할될 폴더
DST_SPLIT = PROJECT_ROOT / "dataset_split"


def setup_directories():
    """스크립트 실행 전, 기존 폴더를 정리하고 새 폴더를 생성합니다."""
    logging.info(f"기존 폴더 삭제 중: {DST_MERGED}, {DST_SPLIT}")
    if DST_MERGED.exists():
        shutil.rmtree(DST_MERGED)
    if DST_SPLIT.exists():
        shutil.rmtree(DST_SPLIT)

    logging.info("새로운 폴더 생성 중...")
    DST_MERGED.mkdir(exist_ok=True)
    DST_SPLIT.mkdir(exist_ok=True)
    logging.info("폴더 설정 완료.")


def merge_images_by_class():
    """
    복잡한 kfood 폴더 구조에서 이미지를 스캔하여
    'dataset/음식명/' 구조로 통합합니다.
    이중으로 된 폴더(예: 국/국/미역국)도 처리합니다.
    """
    logging.info("Step 1: 클래스별 이미지 통합 시작...")
    if not SRC_ROOT.exists():
        logging.error(f"원본 데이터 폴더를 찾을 수 없습니다: {SRC_ROOT}")
        raise FileNotFoundError(f"원본 데이터 폴더를 찾을 수 없습니다: {SRC_ROOT}")

    # 이미지 파일이 들어있는 가장 하위 폴더들을 모두 찾음
    image_parent_folders = set()
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img_path in SRC_ROOT.rglob(ext):
            image_parent_folders.add(img_path.parent)

    if not image_parent_folders:
        logging.error("통합할 이미지를 원본 폴더에서 찾지 못했습니다.")
        raise FileNotFoundError("통합할 이미지를 원본 폴더에서 찾지 못했습니다.")

    for food_path in tqdm(sorted(list(image_parent_folders)), desc="이미지 통합 중"):
        class_name = food_path.name

        try:
            relative_parts = food_path.relative_to(SRC_ROOT).parts
            big_category = relative_parts[0] if relative_parts else "기타"
        except (ValueError, IndexError):
            big_category = "기타"

        dst_food_dir = DST_MERGED / class_name
        dst_food_dir.mkdir(exist_ok=True)

        image_files = (
            list(food_path.glob("*.jpg"))
            + list(food_path.glob("*.jpeg"))
            + list(food_path.glob("*.png"))
        )

        for img_src_path in image_files:
            original_fname = img_src_path.name
            new_fname = f"{big_category}_{class_name}_{original_fname}"
            dst_img_path = dst_food_dir / new_fname
            shutil.copy(str(img_src_path), str(dst_img_path))

    logging.info("Step 1: 이미지 통합 완료.")


def split_train_val(split_ratio=(0.8, 0.2)):
    """
    통합된 이미지들을 train/val 세트로 분할하여
    'dataset_split/' 폴더로 복사합니다.
    """
    logging.info("Step 2: Train/Val 데이터 분할 시작...")

    class_dirs = [d for d in DST_MERGED.iterdir() if d.is_dir()]

    if not class_dirs:
        logging.warning("분할할 클래스 폴더가 없습니다. Step 1을 먼저 확인하세요.")
        return

    (DST_SPLIT / "train").mkdir(exist_ok=True)
    (DST_SPLIT / "val").mkdir(exist_ok=True)

    for class_dir in tqdm(class_dirs, desc="Train/Val 분할 중"):
        class_name = class_dir.name

        images = [
            f
            for f in class_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        random.shuffle(images)

        n = len(images)
        if n == 0:
            logging.warning(f"'{class_name}' 클래스에 이미지가 없어 건너뜁니다.")
            continue

        n_train = int(n * split_ratio[0])

        train_imgs = images[:n_train]
        val_imgs = images[n_train:]

        # Train 폴더로 복사
        train_split_dir = DST_SPLIT / "train" / class_name
        train_split_dir.mkdir(exist_ok=True)
        for img_path in train_imgs:
            shutil.copy(str(img_path), str(train_split_dir / img_path.name))

        # Val 폴더로 복사
        val_split_dir = DST_SPLIT / "val" / class_name
        val_split_dir.mkdir(exist_ok=True)
        for img_path in val_imgs:
            shutil.copy(str(img_path), str(val_split_dir / img_path.name))

    logging.info("Step 2: 데이터 분할 완료.")


if __name__ == "__main__":
    try:
        setup_directories()
        merge_images_by_class()
        split_train_val()
        logging.info(
            "✅ 모든 작업 완료! 'dataset_split' 폴더가 올바르게 생성되었습니다."
        )
        logging.info(f"   Train/Val 데이터 경로: {DST_SPLIT}")
    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생: {e}")
