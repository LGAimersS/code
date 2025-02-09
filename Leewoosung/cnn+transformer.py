import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from google.colab import drive

# 구글 드라이브 마운트
drive.mount('/content/drive')

# 전처리된 데이터 로드
train_df = pd.read_csv("/content/drive/MyDrive/open/processed_train.csv")
test_df = pd.read_csv("/content/drive/MyDrive/open/processed_test.csv")
sample_submission = pd.read_csv("/content/drive/MyDrive/open/sample_submission.csv")

# 데이터 분할
y = train_df['임신 성공 여부']
X = train_df.drop(columns=['임신 성공 여부'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch Dataset 생성
class FertilityDataset(data.Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

train_dataset = FertilityDataset(X_train, y_train)
val_dataset = FertilityDataset(X_val, y_val)
test_dataset = FertilityDataset(test_df, None)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN + Transformer 모델 정의
class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNTransformerModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=4), num_layers=2
        )
        self.fc = nn.Linear(32 * (input_dim // 2), num_classes)
        self.softmax = nn.Softmax(dim=1)  # 확률값 변환

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.cnn(x).permute(2, 0, 1)  # (seq_len, batch, features)
        x = self.transformer(x).permute(1, 0, 2).flatten(start_dim=1)  # (batch, features)
        x = self.fc(x)
        return self.softmax(x)

# 모델 초기화
input_dim = X_train.shape[1]
num_classes = len(y_train.unique())
model = CNNTransformerModel(input_dim, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        # Initialize correct and total as separate variables
        correct = 0  # Initialize correct to 0
        total = 0    # Initialize total to 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Accuracy: {correct / total:.4f}")

# 모델 학습 실행
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# 모델 저장
model_path = "/content/drive/MyDrive/open/cnn_transform_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 테스트 데이터 예측
model.eval()
predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probabilities = outputs[:, 1]  # Positive class probability
        predictions.extend(probabilities.cpu().numpy())

# 샘플 제출 파일 양식에 맞춰 저장
submission = sample_submission.copy()
submission['probability'] = predictions
submission.to_csv("/content/drive/MyDrive/open/final_submission.csv", index=False)

# 결과 확인
print("Final Submission Sample:")
print(submission.head())
