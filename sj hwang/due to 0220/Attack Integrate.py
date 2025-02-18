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
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


#############################################
# 1. 전처리된 데이터 파일 불러오기
#############################################
def load_preprocessed_data():
    if os.path.exists('./datasets/transformed_train_df.csv') and os.path.exists('./datasets/transformed_test_df.csv'):
        print("전처리된 파일(transformed_train_df.csv, transformed_test_df.csv)을 불러옵니다.")
        train_df = pd.read_csv('./datasets/transformed_train_df.csv')
        test_df = pd.read_csv('./datasets/transformed_test_df.csv')
        return train_df, test_df
    else:
        raise FileNotFoundError("전처리된 파일이 없습니다. transformed_train_df.csv, transformed_test_df.csv 파일을 준비해주세요.")


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
# 3. Adversarial Attack 함수들: FGSM, PGD, CW
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


def cw_attack(model, X, y, c=1e-2, kappa=0, lr=1e-2, iters=10):
    # CW attack for binary classification.
    # 변환: y==1 -> target = -1, y==0 -> target = 1
    delta = torch.zeros_like(X, requires_grad=True)
    optimizer = optim.Adam([delta], lr=lr)
    y_sign = torch.where(y == 1, -torch.ones_like(y), torch.ones_like(y))
    for i in range(iters):
        adv_X = X + delta
        outputs = model(adv_X)
        loss_adv = torch.clamp(kappa - y_sign * outputs, min=0)
        loss = torch.norm(delta, p=2) ** 2 + c * torch.mean(loss_adv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (X + delta).detach()


#############################################
# 4. Adversarial Training 및 평가 함수
#############################################
def train_and_evaluate_adv(model, train_loader, val_loader, device, epochs=10, lr=1e-3, attack_params=None,
                           attack_mode='fgsm'):
    """
    attack_params: dictionary, e.g. {'epsilon':0.1, 'alpha':0.01, 'pgd_iters':3, 'c':1e-2, 'kappa':0, 'cw_lr':1e-2, 'cw_iters':10}
    attack_mode: 'fgsm', 'pgd', 'cw', 'combined'
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_state = None

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
            elif attack_mode == 'cw':
                X_adv = cw_attack(model, X_batch, y_batch,
                                  c=attack_params.get('c', 1e-2),
                                  kappa=attack_params.get('kappa', 0),
                                  lr=attack_params.get('cw_lr', 1e-2),
                                  iters=attack_params.get('cw_iters', 10))
                loss = criterion(model(X_adv), y_batch)
            elif attack_mode == 'combined':
                batch_size = X_batch.shape[0]
                if batch_size < 3:
                    X_adv = fgsm_attack(model, X_batch, y_batch, attack_params['epsilon'], criterion)
                    loss = criterion(model(X_adv), y_batch)
                else:
                    split = batch_size // 3
                    X_batch_fgsm, y_batch_fgsm = X_batch[:split], y_batch[:split]
                    X_batch_pgd, y_batch_pgd = X_batch[split:2 * split], y_batch[split:2 * split]
                    X_batch_cw, y_batch_cw = X_batch[2 * split:], y_batch[2 * split:]
                    X_adv_fgsm = fgsm_attack(model, X_batch_fgsm, y_batch_fgsm, attack_params['epsilon'], criterion)
                    X_adv_pgd = pgd_attack(model, X_batch_pgd, y_batch_pgd, attack_params['epsilon'],
                                           attack_params['alpha'], attack_params['pgd_iters'], criterion)
                    X_adv_cw = cw_attack(model, X_batch_cw, y_batch_cw,
                                         c=attack_params.get('c', 1e-2),
                                         kappa=attack_params.get('kappa', 0),
                                         lr=attack_params.get('cw_lr', 1e-2),
                                         iters=attack_params.get('cw_iters', 10))
                    X_adv = torch.cat([X_adv_fgsm, X_adv_pgd, X_adv_cw], dim=0)
                    y_adv = torch.cat([y_batch_fgsm, y_batch_pgd, y_batch_cw], dim=0)
                    loss = criterion(model(X_adv), y_adv)
            else:
                loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_loss = np.mean(batch_losses)

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
        print(f"[{attack_mode.upper()}] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    return best_val_acc, best_model_state


#############################################
# 5. 메인 함수: adversarial training (FGSM, PGD, CW, combined)
#############################################
def main():
    print("==== 전처리된 데이터 불러오기 ====")
    train_df, test_df = load_preprocessed_data()

    print("\n==== 데이터 준비 중 ====")
    X = train_df.drop(columns=['ID', '임신 성공 여부']).values
    y = train_df['임신 성공 여부'].values
    test_ids = test_df['ID'].values
    X_test = test_df.drop(columns=['ID']).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 고정된 adversarial 공격 파라미터 (충분하다고 판단되는 값)
    attack_params = {
        'epsilon': 0.1,
        'alpha': 0.01,
        'pgd_iters': 3,
        'c': 1e-2,
        'kappa': 0,
        'cw_lr': 1e-2,
        'cw_iters': 10
    }

    # 사용할 공격 모드: 'fgsm', 'pgd', 'cw', 'combined'
    attack_modes = ['fgsm', 'pgd', 'cw', 'combined']

    # 하이퍼파라미터 (고정)
    hyperparams = {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'epochs': 20}

    results = {}
    for mode in attack_modes:
        print(f"\n==== Adversarial Training: Attack Mode = {mode.upper()} ====")
        model = TabTransformer(input_dim=X_train.shape[1],
                               d_model=hyperparams['d_model'],
                               nhead=hyperparams['nhead'],
                               num_layers=hyperparams['num_layers'],
                               dropout=hyperparams['dropout']).to(device)
        val_acc, model_state = train_and_evaluate_adv(model, train_loader, val_loader, device,
                                                      epochs=hyperparams['epochs'],
                                                      lr=hyperparams['lr'],
                                                      attack_params=attack_params,
                                                      attack_mode=mode)
        results[mode] = {'val_acc': val_acc, 'state': model_state}
        print(f"Attack Mode {mode.upper()} - Best Val Acc: {val_acc:.4f}")

    best_mode = max(results, key=lambda m: results[m]['val_acc'])
    print(f"\n최종 최고의 공격 모드: {best_mode.upper()} (Val Acc: {results[best_mode]['val_acc']:.4f})")

    # 최종 모델 재학습 (전체 데이터 사용) with best_mode
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    full_dataset = TabularDataset(X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    final_model = TabTransformer(input_dim=X_full.shape[1],
                                 d_model=hyperparams['d_model'],
                                 nhead=hyperparams['nhead'],
                                 num_layers=hyperparams['num_layers'],
                                 dropout=hyperparams['dropout']).to(device)
    final_model.load_state_dict(results[best_mode]['state'])
    final_val_acc, final_state = train_and_evaluate_adv(final_model, full_loader, full_loader, device,
                                                        epochs=hyperparams['epochs'],
                                                        lr=hyperparams['lr'],
                                                        attack_params=attack_params,
                                                        attack_mode=best_mode)
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
    result_df.to_csv('result_Transformer_adv.csv', index=False)
    torch.save(final_model.state_dict(), "best_Transformer_adv.pth")
    print(f"\n최종 결과가 'result_Transformer_adv.csv' 파일로 저장되었습니다.")
    print("모델 상태는 'best_Transformer_adv.pth'로 저장되었습니다.")


if __name__ == "__main__":
    main()

