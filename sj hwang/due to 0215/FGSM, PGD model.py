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
from itertools import product

warnings.filterwarnings("ignore")


#############################################
# 1. 데이터 전처리 (KNN + 범주형 데이터 처리)
#############################################
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


#############################################
# 2. PyTorch Dataset 및 Transformer 모델 정의
#############################################
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
        self.input_layer = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # [batch, input_dim, 1]
        x = self.input_layer(x)  # [batch, input_dim, d_model]
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.output_layer(x)
        return out.squeeze(1)


#############################################
# 3. Adversarial Attack 함수 (FGSM, PGD)
#############################################
def fgsm_attack(model, X, y, epsilon, criterion):
    X_adv = X.clone().detach().requires_grad_(True)
    outputs = model(X_adv)
    loss = criterion(outputs, y)
    model.zero_grad()
    loss.backward()
    grad = X_adv.grad.data
    X_adv = X_adv + epsilon * grad.sign()
    return X_adv.detach()


def pgd_attack(model, X, y, epsilon, alpha, iters, criterion):
    X_adv = X.clone().detach()
    for i in range(iters):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        loss = criterion(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = X_adv.grad.data
        X_adv = X_adv + alpha * grad.sign()
        perturbation = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        X_adv = X + perturbation
        X_adv = X_adv.detach()
    return X_adv


#############################################
# 4. Adversarial Training 및 평가 함수
#############################################
def train_and_evaluate_adv(model, train_loader, val_loader, device, epochs, lr, attack_params, attack_mode):
    """
    attack_params: 딕셔너리, 예) {'epsilon': 0.1, 'alpha': 0.01, 'pgd_iters': 3}
    attack_mode: 'fgsm', 'pgd', 'combined'
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_model_state = None
    train_loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if attack_mode == 'fgsm':
                X_adv = fgsm_attack(model, X_batch, y_batch, attack_params['epsilon'], criterion)
                loss = criterion(model(X_adv), y_batch)
            elif attack_mode == 'pgd':
                X_adv = pgd_attack(model, X_batch, y_batch, attack_params['epsilon'], attack_params['alpha'],
                                   attack_params['pgd_iters'], criterion)
                loss = criterion(model(X_adv), y_batch)
            elif attack_mode == 'combined':
                mid = X_batch.shape[0] // 2
                if mid == 0:
                    X_adv = fgsm_attack(model, X_batch, y_batch, attack_params['epsilon'], criterion)
                    loss = criterion(model(X_adv), y_batch)
                else:
                    X_batch1, y_batch1 = X_batch[:mid], y_batch[:mid]
                    X_batch2, y_batch2 = X_batch[mid:], y_batch[mid:]
                    X_adv1 = fgsm_attack(model, X_batch1, y_batch1, attack_params['epsilon'], criterion)
                    X_adv2 = pgd_attack(model, X_batch2, y_batch2, attack_params['epsilon'], attack_params['alpha'],
                                        attack_params['pgd_iters'], criterion)
                    X_adv = torch.cat([X_adv1, X_adv2], dim=0)
                    y_adv = torch.cat([y_batch1, y_batch2], dim=0)
                    loss = criterion(model(X_adv), y_adv)
            else:
                loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_loss = np.mean(batch_losses)
        train_loss_history.append(avg_loss)
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
        val_acc_history.append(val_acc)
        print(f"[{attack_mode.upper()}] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    return best_val_acc, best_model_state, train_loss_history, val_acc_history


#############################################
# 5. 메인 함수: adversarial training 하이퍼파라미터 튜닝 및 최종 선택
#############################################
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 조합을 더 많이 생성 (FGSM, PGD, Combined 각 모드별)
    adv_config_list = []
    for attack_mode in ['combined', 'fgsm', 'pgd']:
        if attack_mode == 'combined':
            epsilons = [0.05, 0.1, 0.2]
            alphas = [0.005, 0.01, 0.02]
            pgd_iters_list = [3, 5]
            lrs = [1e-3, 5e-4]
            epochs_list = [20, 30]
            d_models = [32, 64]
            nheads = [4, 8]
            num_layers_list = [2, 3]
            dropouts = [0.1, 0.2]
            for (epsilon, alpha, pgd_iters, lr, epochs, d_model, nhead, num_layers, dropout) in product(
                    epsilons, alphas, pgd_iters_list, lrs, epochs_list, d_models, nheads, num_layers_list, dropouts):
                adv_config_list.append({
                    'attack_mode': attack_mode,
                    'epsilon': epsilon,
                    'alpha': alpha,
                    'pgd_iters': pgd_iters,
                    'lr': lr,
                    'epochs': epochs,
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dropout': dropout
                })
        elif attack_mode == 'fgsm':
            epsilons = [0.05, 0.1, 0.2]
            lrs = [1e-3, 5e-4]
            epochs_list = [20, 30]
            d_models = [32, 64]
            nheads = [4, 8]
            num_layers_list = [2, 3]
            dropouts = [0.1, 0.2]
            for (epsilon, lr, epochs, d_model, nhead, num_layers, dropout) in product(
                    epsilons, lrs, epochs_list, d_models, nheads, num_layers_list, dropouts):
                adv_config_list.append({
                    'attack_mode': attack_mode,
                    'epsilon': epsilon,
                    'lr': lr,
                    'epochs': epochs,
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dropout': dropout
                })
        elif attack_mode == 'pgd':
            epsilons = [0.05, 0.1, 0.2]
            alphas = [0.005, 0.01, 0.02]
            pgd_iters_list = [3, 5]
            lrs = [1e-3, 5e-4]
            epochs_list = [20, 30]
            d_models = [32, 64]
            nheads = [4, 8]
            num_layers_list = [2, 3]
            dropouts = [0.1, 0.2]
            for (epsilon, alpha, pgd_iters, lr, epochs, d_model, nhead, num_layers, dropout) in product(
                    epsilons, alphas, pgd_iters_list, lrs, epochs_list, d_models, nheads, num_layers_list, dropouts):
                adv_config_list.append({
                    'attack_mode': attack_mode,
                    'epsilon': epsilon,
                    'alpha': alpha,
                    'pgd_iters': pgd_iters,
                    'lr': lr,
                    'epochs': epochs,
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dropout': dropout
                })

    print(f"\n총 adversarial 하이퍼파라미터 조합 수: {len(adv_config_list)}")

    best_overall_acc = 0.0
    best_adv_config = None
    best_adv_state = None
    best_history = None

    print("\n==== Adversarial Training 하이퍼파라미터 튜닝 시작 ====")
    for idx, config in enumerate(adv_config_list):
        print(f"\n[Experiment {idx + 1}] - Attack Mode: {config['attack_mode']}")
        for k, v in config.items():
            if k not in ['attack_mode']:
                print(f"  {k}: {v}")
        model = TabTransformer(input_dim=X_train.shape[1],
                               d_model=config['d_model'],
                               nhead=config['nhead'],
                               num_layers=config['num_layers'],
                               dropout=config['dropout']).to(device)
        attack_params = {'epsilon': config['epsilon']}
        if config['attack_mode'] in ['pgd', 'combined']:
            attack_params['alpha'] = config['alpha']
            attack_params['pgd_iters'] = config['pgd_iters']
        val_acc, model_state, train_loss_hist, val_acc_hist = train_and_evaluate_adv(
            model, train_loader, val_loader, device,
            epochs=config['epochs'], lr=config['lr'],
            attack_params=attack_params,
            attack_mode=config['attack_mode']
        )
        print(f"Experiment {idx + 1} 최종 검증 정확도: {val_acc:.4f}")
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_adv_config = config
            best_adv_state = model_state.copy()
            best_history = (train_loss_hist, val_acc_hist)

    print("\n==== 최적 adversarial training 구성 선정 완료 ====")
    print("최적 구성:")
    for k, v in best_adv_config.items():
        print(f"  {k}: {v}")
    print(f"최종 검증 정확도: {best_overall_acc:.4f}")

    # 학습 과정 그래프 시각화
    if best_history is not None:
        train_loss_hist, val_acc_hist = best_history
        epochs_range = range(1, len(train_loss_hist) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_loss_hist, marker='o', color='blue')
        plt.title('Epoch별 Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, val_acc_hist, marker='o', color='green')
        plt.title('Epoch별 Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.suptitle("최적 Adversarial Training 학습 과정", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("training_history_adv.png")
        plt.show()
        print("학습 과정 그래프가 'training_history_adv.png'로 저장되었으며, 화면에 출력되었습니다.")

    # 최종 모델 재학습 (전체 데이터 사용)
    print("\n==== 최종 모델 재학습 시작 (전체 데이터) ====")
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    full_dataset = TabularDataset(X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    final_model = TabTransformer(input_dim=X_full.shape[1],
                                 d_model=best_adv_config['d_model'],
                                 nhead=best_adv_config['nhead'],
                                 num_layers=best_adv_config['num_layers'],
                                 dropout=best_adv_config['dropout']).to(device)
    final_model.load_state_dict(best_adv_state)
    final_val_acc, final_state, _, _ = train_and_evaluate_adv(
        final_model, full_loader, full_loader, device,
        epochs=best_adv_config['epochs'], lr=best_adv_config['lr'],
        attack_params={'epsilon': best_adv_config['epsilon'],
                       'alpha': best_adv_config.get('alpha', 0.0),
                       'pgd_iters': best_adv_config.get('pgd_iters', 0)},
        attack_mode=best_adv_config['attack_mode']
    )
    final_model.load_state_dict(final_state)

    # 테스트 데이터 예측 및 결과 저장
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
    result_filename = "result_Transformer_adv.csv"
    result_df.to_csv(result_filename, index=False)
    torch.save(final_model.state_dict(), "best_Transformer_adv.pth")
    print(f"\n최종 결과가 '{result_filename}' 파일로 저장되었습니다.")
    print("모델 상태는 'best_Transformer_adv.pth'로 저장되었습니다.")


if __name__ == "__main__":
    main()
