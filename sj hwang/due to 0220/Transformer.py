import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


#############################################
# 1. 전처리된 데이터 파일 불러오기
#############################################
def load_preprocessed_data():
    if os.path.exists('./datasets/transformed_train_df.csv') and os.path.exists('./datasets/transformed_test_df.csv'):
        print("전처리된 파일(transformed_train_df.csv, transformed_test_df.csv)을 불러옵니다.")
        knn_train = pd.read_csv('./datasets/transformed_train_df.csv')
        knn_test = pd.read_csv('./datasets/transformed_test_df.csv')
        return knn_train, knn_test
    else:
        raise FileNotFoundError("전처리된 파일이 없습니다. transformed_train_df.csv, transformed_test_df.csv 파일을 준비해주세요.")


#############################################
# 2. PyTorch Dataset 및 Transformer 모델 정의
#############################################
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X: numpy array of features
        y: numpy array of labels (또는 None)
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class TabTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        """
        input_dim: 피처 수 (각 피처를 하나의 토큰으로 취급)
        d_model: 임베딩 차원
        nhead: 멀티헤드 어텐션 헤드 수
        num_layers: Transformer 레이어 반복 횟수
        dropout: dropout 비율
        """
        super(TabTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 각 피처(스칼라)를 d_model 차원으로 임베딩
        self.input_layer = nn.Linear(1, d_model)
        # learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 최종 출력층 (이진 분류)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        x = self.input_layer(x)  # [batch_size, input_dim, d_model]
        x = x + self.pos_embedding  # 위치 임베딩 추가
        x = self.transformer_encoder(x)  # Transformer Encoder 통과
        x = x.mean(dim=1)  # 모든 토큰에 대해 평균 풀링
        out = self.output_layer(x)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]


#############################################
# 3. Transformer 모델 학습 및 평가 함수
#############################################
def train_and_evaluate(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)

        # 검증 (클린 데이터 기준)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds.extend(torch.sigmoid(outputs).cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
        preds_binary = [1 if p >= 0.5 else 0 for p in preds]
        val_acc = accuracy_score(targets, preds_binary)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    return best_val_acc, best_model_state


#############################################
# 4. 메인 함수
#############################################
def main():
    print("==== 전처리된 데이터 불러오기 ====")
    knn_train, knn_test = load_preprocessed_data()

    print("\n==== 데이터 준비 중 ====")
    X = knn_train.drop(columns=['ID', '임신 성공 여부']).values
    y = knn_train['임신 성공 여부'].values
    test_ids = knn_test['ID'].values
    X_test = knn_test.drop(columns=['ID']).values

    # 데이터 스케일링
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Dataset 및 DataLoader 구성
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 하이퍼파라미터 후보 (필요 시 조정)
    hyperparams_list = [
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'epochs': 10},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'epochs': 20},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 20},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 20},
    ]

    best_acc = 0.0
    best_params = None
    best_state = None

    print("\n==== Transformer 모델 하이퍼파라미터 튜닝 시작 ====")
    for idx, params in enumerate(hyperparams_list):
        print(f"\n===== Experiment {idx + 1} =====")
        for key, value in params.items():
            print(f"  {key}: {value}")
        model = TabTransformer(input_dim=X_train.shape[1],
                               d_model=params['d_model'],
                               nhead=params['nhead'],
                               num_layers=params['num_layers'],
                               dropout=params['dropout']).to(device)
        val_acc, model_state = train_and_evaluate(model, train_loader, val_loader, device,
                                                  epochs=params['epochs'], lr=params['lr'])
        print(f"Experiment {idx + 1} 최종 검증 정확도: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_state = model_state.copy()

    print("\n==== 최적 하이퍼파라미터 선정 완료 ====")
    print("최적 파라미터:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"최종 검증 정확도: {best_acc:.4f}")

    # 최종 모델 재학습 (전체 데이터 사용)
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    full_dataset = TabularDataset(X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    final_model = TabTransformer(input_dim=X_full.shape[1],
                                 d_model=best_params['d_model'],
                                 nhead=best_params['nhead'],
                                 num_layers=best_params['num_layers'],
                                 dropout=best_params['dropout']).to(device)
    final_model.load_state_dict(best_state)
    final_val_acc, final_state = train_and_evaluate(final_model, full_loader, full_loader, device,
                                                    epochs=best_params['epochs'], lr=best_params['lr'])
    final_model.load_state_dict(final_state)

    final_model.eval()
    test_dataset_pt = TabularDataset(X_test)
    test_loader_pt = DataLoader(test_dataset_pt, batch_size=64, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for X_batch in test_loader_pt:
            X_batch = X_batch.to(device)
            outputs = final_model(X_batch)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
    result_df = pd.DataFrame({'ID': test_ids, 'probability': all_preds})
    result_df.to_csv('result transformed and transformer.csv', index=False)
    print("최종 결과가 'result transformed and transformer.csv' 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()

