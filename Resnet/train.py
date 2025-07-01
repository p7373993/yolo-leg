import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from pathlib import Path

# ✅ 경로 설정 (스크립트 실행 위치에 무관하게)
# 이 스크립트 파일(train.py)의 절대 경로
SCRIPT_PATH = Path(__file__).resolve()
# 프로젝트 루트 (train.py -> resnet -> YOLO_FoodSeg)
PROJECT_ROOT = SCRIPT_PATH.parent.parent
# 데이터셋 경로
DATA_PATH = PROJECT_ROOT / "dataset_split"
# 모델 저장 경로
SAVE_PATH = SCRIPT_PATH.parent / "resnet_kfood.pth"


# ✅ 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = len(os.listdir(DATA_PATH / "train"))  # 음식 클래스 수 자동 감지
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 전처리 (ImageNet 기반 ResNet은 이 정규화 필수)
# 리사이즈와 정규화를 저렇게 한 이유는 사전학습을 ImageNet으로 했기 때문에 이미지 크기와 정규화 값을 맞춰줘야 함
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ✅ 데이터셋
train_dataset = datasets.ImageFolder(DATA_PATH / "train", transform=transform)
val_dataset = datasets.ImageFolder(DATA_PATH / "val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ ResNet18 모델 불러오기 (사전학습 + 마지막 출력층 수정)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ✅ 손실함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

# ✅ 검증 및 평가
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu()

        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

# ✅ 모델 저장
torch.save(model.state_dict(), SAVE_PATH)
