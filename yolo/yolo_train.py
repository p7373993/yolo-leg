from ultralytics import YOLO

if __name__ == "__main__":
    # --- 스크립트 시작 확인용 print문 ---
    print("train.py 스크립트 실행 시작...")
    print("YOLO 모델을 로드합니다. 처음 실행 시 모델 파일을 다운로드할 수 있습니다...")

    # 미리 학습된 yolov8n-seg.pt 모델을 로드하여 시작합니다.
    # 'n'은 nano 모델을 의미하며, 가장 가볍고 빠릅니다. (s, m, l, x 순으로 커짐)
    model = YOLO("yolov8n-seg.pt")

    # --- 학습 시작 확인용 print문 ---
    print("\n모델 로드 완료. 이제 학습을 시작합니다.")
    print(
        "데이터셋 스캔 및 캐싱 작업으로 인해 로그 출력이 잠시 지연될 수 있습니다. 기다려주세요..."
    )

    # 모델 학습을 시작합니다.
    # data: data.yaml 파일의 경로
    # epochs: 전체 데이터셋을 몇 번 반복 학습할지 결정 (테스트로 30으로 시작, 나중에 늘리세요)
    # imgsz: 학습에 사용할 이미지 크기
    # batch: 한 번에 몇 개의 이미지를 처리할지 결정 (GPU 메모리에 따라 조절)
    results = model.train(
        data="FoodSeg103_YOLO_seg/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        name="foodseg103_yolov8n_seg",
    )

    print(
        "학습이 완료되었습니다. 결과는 'runs/segment/foodseg103_yolov8n_seg' 폴더에 저장됩니다."
    )
