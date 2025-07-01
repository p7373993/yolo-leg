from ultralytics import YOLO
from PIL import Image

if __name__ == "__main__":
    # 학습으로 생성된 최고의 모델 가중치 파일을 로드합니다.
    model_path = "runs/segment/foodseg103_yolov8n_seg/weights/best.pt"
    model = YOLO(model_path)

    # 예측하고 싶은 이미지 경로를 지정합니다.
    # 예시: 'path/to/your/test_image.jpg'
    # FoodSeg103 데이터셋의 검증 이미지 중 하나를 사용해 보겠습니다.
    source_image_path = "FoodSeg103_YOLO_seg/images/val/img_00000000.jpg"

    # 모델 예측 실행
    results = model.predict(source=source_image_path)

    # 결과 확인
    # results[0]에 첫 번째 이미지에 대한 결과가 담겨 있습니다.
    # 결과 이미지를 화면에 표시합니다.
    result_image = Image.fromarray(results[0].plot()[..., ::-1])  # BGR to RGB
    result_image.show()

    print(f"예측이 완료되었습니다. 결과는 'runs/segment/predict' 폴더에 저장됩니다.")

    # 각 객체의 세부 정보 출력 (옵션)
    for r in results:
        for i, mask in enumerate(r.masks.xy):
            class_id = int(r.boxes.cls[i])
            class_name = model.names[class_id]
            confidence = float(r.boxes.conf[i])
            print(f"객체 {i+1}: {class_name} (신뢰도: {confidence:.2f})")
            # print(f"  마스크 좌표: {mask}") # 좌표가 너무 길어서 주석 처리
