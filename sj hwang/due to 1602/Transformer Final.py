import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# ========= KNN 전처리 함수 (범주형 처리 포함) =========
def knn_preprocessing():
    # knn_train.csv와 knn_test.csv 파일이 있으면 로드
    if os.path.exists('knn_train.csv') and os.path.exists('knn_test.csv'):
        print("KNN 전처리 파일이 존재합니다. 파일을 불러옵니다.")
        knn_train = pd.read_csv('knn_train.csv')
        knn_test = pd.read_csv('knn_test.csv')
    else:
        print("KNN 전처리 파일이 없습니다. 원본 데이터로부터 전처리합니다.")
        # 원본 데이터 경로
        train = pd.read_csv('./datasets/train.csv')
        test = pd.read_csv('./datasets/test.csv')

        # train: 타깃 컬럼 '임신 성공 여부'와 ID 'ID'를 제외한 피처들,
        # test: ID 'ID'를 제외한 피처들
        train_features = train.drop(columns=['ID', '임신 성공 여부'])
        test_features = test.drop(columns=['ID'])

        # 범주형 데이터(문자열)를 숫자로 변환 (train과 test를 합쳐서 처리)
        for col in train_features.columns:
            if train_features[col].dtype == 'object':
                combined = pd.concat([train_features[col], test_features[col]], axis=0)
                # na_sentinel 인자 제거
                codes, uniques = pd.factorize(combined)
                # factorize 결과에서 NA는 기본적으로 -1로 처리되므로, 이를 np.nan으로 변경
                codes = pd.Series(codes).replace(-1, np.nan).values
                train_features[col] = codes[:len(train_features)]
                test_features[col] = codes[len(train_features):]

        # KNNImputer (n_neighbors=5) 적용
        imputer = KNNImputer(n_neighbors=5)
        train_imputed = pd.DataFrame(imputer.fit_transform(train_features),
                                     columns=train_features.columns)
        test_imputed = pd.DataFrame(imputer.transform(test_features),
                                    columns=test_features.columns)

        # ID와 타깃 재결합
        knn_train = pd.concat([train[['ID']].reset_index(drop=True),
                               train_imputed,
                               train[['임신 성공 여부']].reset_index(drop=True)], axis=1)
        knn_test = pd.concat([test[['ID']].reset_index(drop=True), test_imputed], axis=1)

        # 파일 저장
        knn_train.to_csv('knn_train.csv', index=False)
        knn_test.to_csv('knn_test.csv', index=False)
        print("KNN 전처리 완료 및 knn_train.csv, knn_test.csv 저장됨.")

    return knn_train, knn_test


# ========= PyTorch Dataset =========
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


# ========= Transformer 기반 Tabular 모델 =========
class TabTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        """
        input_dim: 피처의 개수 (각 피처를 하나의 토큰으로 취급)
        d_model: 임베딩 차원
        nhead: 멀티헤드 어텐션 헤드 수
        num_layers: TransformerEncoderLayer 반복 횟수
        dropout: dropout 비율
        """
        super(TabTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 각 피처(스칼라)를 d_model 차원으로 임베딩 (동일한 선형층을 모든 피처에 적용)
        self.input_layer = nn.Linear(1, d_model)
        # learnable positional embedding (각 피처에 대한 고유 위치 임베딩)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 분류를 위한 최종 출력층 (이후 sigmoid를 적용하여 확률 출력)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        # 각 피처를 하나의 토큰으로 보기 위해 차원 확장: [batch_size, input_dim, 1]
        x = x.unsqueeze(-1)
        # 모든 피처에 대해 선형 임베딩 (공유된 가중치)
        x = self.input_layer(x)  # [batch_size, input_dim, d_model]
        # 위치 임베딩 추가
        x = x + self.pos_embedding  # [batch_size, input_dim, d_model]
        # Transformer Encoder 통과
        x = self.transformer_encoder(x)  # [batch_size, input_dim, d_model]
        # 모든 토큰에 대해 평균(pooling)
        x = x.mean(dim=1)  # [batch_size, d_model]
        # 최종 출력
        out = self.output_layer(x)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]


# ========= 학습 및 평가 함수 =========
def train_and_evaluate(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_state = None

    # 각 epoch별 손실과 검증 정확도를 기록할 리스트
    epoch_train_losses = []
    epoch_val_accs = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_loss = np.mean(train_losses)
        epoch_train_losses.append(avg_loss)

        # validation
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                preds.extend(torch.sigmoid(outputs).cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
        preds_binary = [1 if p >= 0.5 else 0 for p in preds]
        val_acc = accuracy_score(targets, preds_binary)
        epoch_val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} => Avg Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    return best_val_acc, best_model_state, epoch_train_losses, epoch_val_accs


# ========= 메인 함수 =========
def main():
    print("==== 데이터 전처리 시작 ====")
    knn_train, knn_test = knn_preprocessing()

    print("\n==== 데이터 준비 중 ====")
    X = knn_train.drop(columns=['ID', '임신 성공 여부']).values
    y = knn_train['임신 성공 여부'].values
    test_ids = knn_test['ID'].values
    X_test = knn_test.drop(columns=['ID']).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 실험 리스트에 epoch 수도 포함 (예: 10, 20, 30)
    hyperparams_list = [
        # d_model=32, nhead=4, num_layers=2, dropout=0.1
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'epochs': 10},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'epochs': 20},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'epochs': 30},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 10},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 20},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 30},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-4, 'epochs': 10},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-4, 'epochs': 20},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-4, 'epochs': 30},

        # d_model=64, nhead=4, num_layers=2, dropout=0.2
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 10},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 20},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 30},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 5e-4, 'epochs': 10},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 5e-4, 'epochs': 20},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 5e-4, 'epochs': 30},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-4, 'epochs': 10},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-4, 'epochs': 20},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-4, 'epochs': 30},

        # d_model=32, nhead=8, num_layers=3, dropout=0.1
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 10},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 20},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 30},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 1e-4, 'epochs': 10},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 1e-4, 'epochs': 20},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 1e-4, 'epochs': 30},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 5e-5, 'epochs': 10},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 5e-5, 'epochs': 20},
        {'d_model': 32, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1, 'lr': 5e-5, 'epochs': 30},
    ]


    best_overall_acc = 0.0
    best_hyperparams = None
    best_state = None
    best_history = None  # (train_loss_history, val_acc_history)

    input_dim = X_train.shape[1]

    print("\n==== 하이퍼파라미터 실험 시작 ====")
    for idx, params in enumerate(hyperparams_list):
        print(f"\n===== Experiment {idx + 1} =====")
        print("실험 파라미터:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        model = TabTransformer(input_dim=input_dim,
                               d_model=params['d_model'],
                               nhead=params['nhead'],
                               num_layers=params['num_layers'],
                               dropout=params['dropout']).to(device)

        val_acc, model_state, train_loss_history, val_acc_history = train_and_evaluate(
            model, train_loader, val_loader, device, epochs=params['epochs'], lr=params['lr']
        )
        print(f"Experiment {idx + 1} 최종 검증 정확도: {val_acc:.4f}")

        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_hyperparams = params
            best_state = model_state.copy()
            best_history = (train_loss_history, val_acc_history)

    print("\n==== 최적 하이퍼파라미터 선정 완료 ====")
    print("최적 파라미터:")
    for key, value in best_hyperparams.items():
        print(f"  {key}: {value}")
    print(f"최종 검증 정확도: {best_overall_acc:.4f}")

    # 최적 모델 재구축 및 상태 로드
    best_model = TabTransformer(input_dim=input_dim,
                                d_model=best_hyperparams['d_model'],
                                nhead=best_hyperparams['nhead'],
                                num_layers=best_hyperparams['num_layers'],
                                dropout=best_hyperparams['dropout']).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    print("\n==== 테스트 데이터 예측 중 ====")
    all_preds = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = best_model(X_batch)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())

    result_df = pd.DataFrame({'ID': test_ids, 'probability': all_preds})
    result_df.to_csv('result.csv', index=False)
    print("최종 결과가 'result.csv' 파일로 저장되었습니다.")

    # ===== 학습 과정 그래프 출력 =====
    if best_history is not None:
        train_loss_history, val_acc_history = best_history
        epochs_range = range(1, len(train_loss_history) + 1)

        plt.figure(figsize=(12, 5))

        # 서브플롯 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_loss_history, marker='o', color='blue')
        plt.title('Epoch별 Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # 서브플롯 2: Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, val_acc_history, marker='o', color='green')
        plt.title('Epoch별 Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)

        plt.suptitle("최적 모델 학습 과정", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("training_history.png")
        plt.show()
        print("\n학습 과정 그래프가 'training_history.png'로 저장되었으며, 화면에 출력되었습니다.")


if __name__ == "__main__":
    main()
