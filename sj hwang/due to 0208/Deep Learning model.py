# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Using device: {device}")

# 평가 지표 계산 함수
# 평가 지표 계산 함수 (메모리 최적화)
def evaluate_model(y_true, y_pred, model_name):
    try:
        # 🔹 1D 배열로 변환 (2D 또는 다른 차원 방지)
        y_true = np.array(y_true).reshape(-1).astype(np.float32)
        y_pred = np.array(y_pred).reshape(-1).astype(np.float32)

        # 🔹 크기 맞추기 (예측값이 더 많을 경우 자르기)
        min_length = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_length], y_pred[:min_length]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # 0으로 나누는 문제 방지

        print(f"📊 {model_name} 평가 결과:")
        print(f"✅ MSE  (Mean Squared Error): {mse:.4f}")
        print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"✅ MAE  (Mean Absolute Error): {mae:.4f}")
        print(f"✅ MAPE (Mean Absolute Percentage Error): {mape:.2f}%\n")

        return mse, rmse, mae, mape

    except Exception as e:
        print(f"🚨 {model_name}: 평가 오류 발생 - {e}")
        return None


# 데이터셋 클래스
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, seq_length=10):
        self.data = data
        self.target = target
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.target[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 모델 정의
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()  # 🔹 비연속적인 텐서를 연속적인 텐서로 변환
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)



class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=2):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(hidden_size, num_heads, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.to(device)
    loss_history = []
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return loss_history

# 시각화 함수
def plot_loss(loss_history, model_name, method):
    plt.figure(figsize=(10,5))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name}_{method} Training Loss")
    plt.legend()
    plt.show()

# 메인 실행 함수
def train_and_evaluate_dl_models(train_path, test_path, method):
    print(f"\n📂 데이터 로드 중... ({method})")
    # 데이터 로드
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 숫자형 컬럼만 선택 + NaN 제거
    num_features = train_df.select_dtypes(include=[np.number]).dropna()

    if num_features.shape[1] == 0:
        print(f"🚨 {method}: 변환할 숫자형 컬럼이 없음! 데이터 확인 필요.")
        print("현재 데이터 타입:\n", train_df.dtypes)
        print("현재 데이터 샘플:\n", train_df.head())
        return

    # MinMaxScaler 적용
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(num_features.drop(columns=["임신 성공 여부"], errors="ignore"))

    # NaN 값이 있는지 확인
    if np.isnan(X_train).sum() > 0:
        print(f"🚨 {method}: MinMaxScaler 변환 후에도 NaN 값 존재! → 제거 진행")
        X_train = np.nan_to_num(X_train)

    # 변환된 데이터 확인
    print(f"✅ {method} 변환된 X_train 값 확인: Min={np.min(X_train)}, Max={np.max(X_train)}")

    y_train = num_features["임신 성공 여부"].values

    # 데이터셋 및 데이터로더 생성
    dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 모델들 정의
    models = {
        "RNN": RNNModel(input_size=X_train.shape[1], hidden_size=64, output_size=1),
        "LSTM": LSTMModel(input_size=X_train.shape[1], hidden_size=64, output_size=1),
        "GRU": GRUModel(input_size=X_train.shape[1], hidden_size=64, output_size=1),
        "Transformer": TransformerModel(input_size=X_train.shape[1], hidden_size=64, output_size=1)
    }

    # 손실 함수 및 최적화 기법
    criterion = nn.MSELoss()
    optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}

    # 모델 학습 및 평가
    for model_name, model in models.items():
        print(f"🚀 {model_name} 모델 학습 시작... ({method})")
        loss_history = train_model(model, train_loader, criterion, optimizers[model_name])
        plot_loss(loss_history, model_name, method)

        # ✅ 예측 수행 (연속적인 텐서 변환 추가)
        X_test_tensor = torch.tensor(X_train[-len(test_df):], dtype=torch.float32).to(device).contiguous()
        X_test_tensor = X_test_tensor.view(X_test_tensor.shape[0], 1, -1)  # 🔹 RNN 입력 형태 변환

        y_pred = model(X_test_tensor).cpu().detach().numpy()

        # 평가 및 결과 저장
        evaluate_model(y_train[-len(y_pred):], y_pred, f"{model_name}_{method}")
        output_dir = "./DeepLearning_result"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{output_dir}/{model_name}_{method}_sample.csv"
        pd.DataFrame({"ID": test_df["ID"], "probability": y_pred.flatten()}).to_csv(output_filename, index=False)
        print(f"✅ {model_name} 결과 저장 완료! ({output_filename})")

# 실행 코드
if __name__ == "__main__":
    methods = ["linear", "poly", "mean", "knn"]
    for method in methods:
        train_and_evaluate_dl_models(f"./preprocessing/train_{method}.csv", f"./preprocessing/test_{method}.csv", method)
