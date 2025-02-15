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

# XGBoost 관련 라이브러리
from xgboost import XGBClassifier

#########################################
# 1. 데이터 전처리 (KNN + 범주형 처리 포함)
#########################################
def knn_preprocessing():
    if os.path.exists('knn_train.csv') and os.path.exists('knn_test.csv'):
        print("KNN 전처리 파일이 존재합니다. 파일을 불러옵니다.")
        knn_train = pd.read_csv('knn_train.csv')
        knn_test = pd.read_csv('knn_test.csv')
    else:
        print("KNN 전처리 파일이 없습니다. 원본 데이터로부터 전처리합니다.")
        train = pd.read_csv('./datasets/train.csv')
        test = pd.read_csv('./datasets/test.csv')
        train_features = train.drop(columns=['ID', '임신 성공 여부'])
        test_features = test.drop(columns=['ID'])
        for col in train_features.columns:
            if train_features[col].dtype == 'object':
                combined = pd.concat([train_features[col], test_features[col]], axis=0)
                codes, uniques = pd.factorize(combined)
                codes = pd.Series(codes).replace(-1, np.nan).values
                train_features[col] = codes[:len(train_features)]
                test_features[col] = codes[len(train_features):]
        imputer = KNNImputer(n_neighbors=5)
        train_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns)
        test_imputed = pd.DataFrame(imputer.transform(test_features), columns=test_features.columns)
        knn_train = pd.concat([train[['ID']].reset_index(drop=True),
                               train_imputed,
                               train[['임신 성공 여부']].reset_index(drop=True)], axis=1)
        knn_test = pd.concat([test[['ID']].reset_index(drop=True), test_imputed], axis=1)
        knn_train.to_csv('knn_train.csv', index=False)
        knn_test.to_csv('knn_test.csv', index=False)
        print("KNN 전처리 완료 및 knn_train.csv, knn_test.csv 저장됨.")
    return knn_train, knn_test

#########################################
# 2. PyTorch Dataset 및 모델 정의
#########################################
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# Transformer 기반 모델
class TabTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(TabTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_layer = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        x = self.input_layer(x)  # [batch_size, input_dim, d_model]
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.output_layer(x)
        return out.squeeze(1)

# 간단한 MLP 모델
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x).squeeze(1)

#########################################
# 3. 모델별 학습 및 평가 함수
#########################################
# PyTorch 모델 학습 (TabTransformer, MLP 공용)
def train_and_evaluate_pytorch(model, train_loader, val_loader, device, epochs, lr):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_loss = np.mean(train_losses)
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
        print(f"[PyTorch] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    return best_val_acc, best_model_state

# XGBoost 모델 학습
def train_and_evaluate_xgb(X_train_np, y_train_np, X_val_np, y_val_np, params):
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_np, y_train_np)
    preds = model.predict(X_val_np)
    val_acc = accuracy_score(y_val_np, preds)
    print(f"[XGBoost] Validation Accuracy: {val_acc:.4f} with params: {params}")
    return val_acc, model

#########################################
# 4. 메인 함수: 후보 모델별 튜닝 및 최종 선택
#########################################
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

    # 학습/검증 분리
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # PyTorch용 Dataset
    train_dataset = TabularDataset(X_train_np, y_train_np)
    val_dataset = TabularDataset(X_val_np, y_val_np)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 후보 모델별 하이퍼파라미터 그리드
    results = {}

    ###########################
    # 4-1. TabTransformer 후보
    ###########################
    print("\n==== TabTransformer 후보 실험 시작 ====")
    tab_transformer_params = [
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'epochs': 20},
        {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 5e-4, 'epochs': 20},
        {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'epochs': 20},
    ]
    best_acc_tt = 0.0
    best_params_tt = None
    best_state_tt = None
    for params in tab_transformer_params:
        print("\n[TabTransformer] 실험 파라미터:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        model = TabTransformer(input_dim=X_train_np.shape[1],
                               d_model=params['d_model'],
                               nhead=params['nhead'],
                               num_layers=params['num_layers'],
                               dropout=params['dropout']).to(device)
        val_acc, model_state = train_and_evaluate_pytorch(model, train_loader, val_loader, device, epochs=params['epochs'], lr=params['lr'])
        if val_acc > best_acc_tt:
            best_acc_tt = val_acc
            best_params_tt = params
            best_state_tt = model_state.copy()
    results['TabTransformer'] = {'val_acc': best_acc_tt, 'params': best_params_tt, 'state': best_state_tt}

    ###########################
    # 4-2. MLP 후보
    ###########################
    print("\n==== MLP 후보 실험 시작 ====")
    mlp_params_list = [
        {'hidden_dims': [64, 32], 'dropout': 0.1, 'lr': 1e-3, 'epochs': 20},
        {'hidden_dims': [128, 64], 'dropout': 0.2, 'lr': 5e-4, 'epochs': 20},
        {'hidden_dims': [64, 32], 'dropout': 0.1, 'lr': 5e-4, 'epochs': 20},
    ]
    best_acc_mlp = 0.0
    best_params_mlp = None
    best_state_mlp = None
    for params in mlp_params_list:
        print("\n[MLP] 실험 파라미터:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        model = MLP(input_dim=X_train_np.shape[1], hidden_dims=params['hidden_dims'], dropout=params['dropout']).to(device)
        val_acc, model_state = train_and_evaluate_pytorch(model, train_loader, val_loader, device, epochs=params['epochs'], lr=params['lr'])
        if val_acc > best_acc_mlp:
            best_acc_mlp = val_acc
            best_params_mlp = params
            best_state_mlp = model_state.copy()
    results['MLP'] = {'val_acc': best_acc_mlp, 'params': best_params_mlp, 'state': best_state_mlp}

    ###########################
    # 4-3. XGBoost 후보
    ###########################
    print("\n==== XGBoost 후보 실험 시작 ====")
    xgb_params_list = [
        {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100},
        {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 150},
        {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 200},
    ]
    best_acc_xgb = 0.0
    best_params_xgb = None
    best_model_xgb = None
    for params in xgb_params_list:
        print("\n[XGBoost] 실험 파라미터:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        val_acc, model_xgb = train_and_evaluate_xgb(X_train_np, y_train_np, X_val_np, y_val_np, params)
        if val_acc > best_acc_xgb:
            best_acc_xgb = val_acc
            best_params_xgb = params
            best_model_xgb = model_xgb
    results['XGBoost'] = {'val_acc': best_acc_xgb, 'params': best_params_xgb, 'model': best_model_xgb}

    # 결과 비교
    print("\n==== 후보 모델별 검증 결과 ====")
    for model_name, res in results.items():
        print(f"{model_name}: Best Val Acc = {res['val_acc']:.4f}, Params = {res['params']}")

    # 최종 최고의 후보 선택
    best_model_name = max(results, key=lambda k: results[k]['val_acc'])
    print(f"\n최종 최고의 모델: {best_model_name} (Val Acc: {results[best_model_name]['val_acc']:.4f})")

    #########################################
    # 5. 최종 모델 재학습 및 테스트 예측, 파일 저장
    #########################################
    # 전체 학습 데이터를 합쳐서 최종 모델 재학습
    X_full = np.concatenate([X_train_np, X_val_np], axis=0)
    y_full = np.concatenate([y_train_np, y_val_np], axis=0)

    if best_model_name in ['TabTransformer', 'MLP']:
        # PyTorch 모델 재학습
        full_dataset = TabularDataset(X_full, y_full)
        full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
        if best_model_name == 'TabTransformer':
            best_params = results['TabTransformer']['params']
            final_model = TabTransformer(input_dim=X_full.shape[1],
                                         d_model=best_params['d_model'],
                                         nhead=best_params['nhead'],
                                         num_layers=best_params['num_layers'],
                                         dropout=best_params['dropout']).to(device)
            # 초기 상태로부터 최적 상태로 재학습
        elif best_model_name == 'MLP':
            best_params = results['MLP']['params']
            final_model = MLP(input_dim=X_full.shape[1], hidden_dims=best_params['hidden_dims'], dropout=best_params['dropout']).to(device)
        # 재학습 (여기서는 epochs=best_params['epochs']로 다시 학습)
        print(f"\n==== {best_model_name} 최종 모델 재학습 시작 ====")
        final_val_acc, final_state = train_and_evaluate_pytorch(final_model, full_loader, full_loader, device, epochs=best_params['epochs'], lr=best_params['lr'])
        final_model.load_state_dict(final_state)
        # 테스트 예측
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
        result_filename = f"result_{best_model_name}.csv"
        result_df.to_csv(result_filename, index=False)
        torch.save(final_model.state_dict(), f"best_{best_model_name}.pth")
        print(f"최종 결과가 '{result_filename}' 파일로 저장되었습니다.")
        print(f"모델 상태도 'best_{best_model_name}.pth'로 저장되었습니다.")
    elif best_model_name == 'XGBoost':
        # XGBoost 모델 재학습
        final_xgb = XGBClassifier(**results['XGBoost']['params'], use_label_encoder=False, eval_metric='logloss')
        final_xgb.fit(X_full, y_full)
        preds = final_xgb.predict(X_test)
        result_df = pd.DataFrame({'ID': test_ids, 'probability': preds})
        result_filename = f"result_{best_model_name}.csv"
        result_df.to_csv(result_filename, index=False)
        final_xgb.save_model(f"best_{best_model_name}.model")
        print(f"최종 결과가 '{result_filename}' 파일로 저장되었습니다.")
        print(f"모델 파일은 'best_{best_model_name}.model'로 저장되었습니다.")

if __name__ == "__main__":
    main()
