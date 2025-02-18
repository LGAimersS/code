import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import mode
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import WeightedRandomSampler

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Custom Dataset for PyTorch
class FertilityDataset(Dataset):
    def __init__(self, features, targets=None):
        if isinstance(features, (pd.DataFrame, pd.Series)):
            features = features.values
        self.features = torch.FloatTensor(features)

        if targets is not None:
            if isinstance(targets, pd.Series):
                targets = targets.values
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

# Training function for deep learning models
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, scheduler=None):
    best_val_loss = float('inf')
    best_model = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted.view(-1) == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets.view(-1, 1))

                val_loss += loss.item()
                predicted = (outputs.data > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted.view(-1) == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    model.load_state_dict(best_model)
    return model

# Neural Network Models
class EnhancedTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.embedding1 = nn.Linear(input_dim, 256)
        self.embedding2 = nn.Linear(input_dim, 128)

        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=num_heads,
            dropout=dropout,
            activation='gelu'
        )
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=num_heads//2,
            dropout=dropout,
            activation='gelu'
        )

        self.transformer1 = nn.TransformerEncoder(encoder_layer1, num_layers)
        self.transformer2 = nn.TransformerEncoder(encoder_layer2, num_layers)

        self.fc = nn.Sequential(
            nn.Linear(384, 192),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.embedding1(x).unsqueeze(1)
        x2 = self.embedding2(x).unsqueeze(1)

        x1 = self.transformer1(x1).squeeze(1)
        x2 = self.transformer2(x2).squeeze(1)

        x_combined = torch.cat([x1, x2], dim=1)
        return self.fc(x_combined)

class EnhancedCNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.SELU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.SELU(),
            )

        self.proj1 = nn.Conv1d(32, 64, kernel_size=1)
        self.proj2 = nn.Conv1d(64, 128, kernel_size=1)
        self.input_proj = nn.Conv1d(1, 32, kernel_size=1)
        self.conv1 = conv_block(32, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc = nn.Sequential(
            nn.Linear(256 * 16, 512),
            nn.SELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        
        identity = x
        x = self.conv1(x)
        identity = self.proj1(identity)
        x = x + identity

        identity = x
        x = self.conv2(x)
        identity = self.proj2(identity)
        x = x + identity

        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
class EnhancedFertilityEnsemble:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.transformer = None
        self.cnn = None
        self.lgbm = None
        self.xgb = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, X):
        if 'ID' in X.columns:
            X = X.drop('ID', axis=1)
        elif 'id' in X.columns:
            X = X.drop('id', axis=1)

        if isinstance(X, pd.DataFrame):
            poly = PolynomialFeatures(degree=2, include_bias=False)
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            X_poly = poly.fit_transform(X[numeric_cols])
            X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(numeric_cols))

            selector = VarianceThreshold(threshold=0.01)
            X_poly_selected = selector.fit_transform(X_poly_df)
            X_poly_cols = X_poly_df.columns[selector.get_support()]

            X = pd.concat([X, pd.DataFrame(X_poly_selected, columns=X_poly_cols)], axis=1)
            X = X.values

        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca, X_scaled

    def train(self, X, y, n_splits=5, batch_size=32):
        X_pca, X_scaled = self.prepare_data(X)

        # Initialize arrays for predictions
        transformer_preds = np.zeros(len(X))
        cnn_preds = np.zeros(len(X))
        lgbm_preds = np.zeros(len(X))
        xgb_preds = np.zeros(len(X))

        # StratifiedKFold 설정
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_pca, y)):
            print(f"\nFold {fold+1}/{n_splits}")
            print("-" * 60)

            # Split data
            X_train_pca, X_val_pca = X_pca[train_idx], X_pca[val_idx]
            X_train_scaled, X_val_scaled = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Apply SMOTE only to training data
            smote = SMOTE(random_state=42)
            X_train_pca_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)
            X_train_scaled_resampled, _ = smote.fit_resample(X_train_scaled, y_train)

            # Prepare data loaders with weights
            weights = compute_sample_weight(
                class_weight='balanced',
                y=y_train_resampled
            )
            train_dataset = FertilityDataset(X_train_pca_resampled, y_train_resampled)
            val_dataset = FertilityDataset(X_val_pca, y_val)
            train_sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Train models
            # Transformer
            self.transformer = EnhancedTransformerModel(self.n_components).to(self.device)
            optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            criterion = nn.BCELoss(weight=torch.tensor([3.0]).to(self.device))
            self.transformer = train_model(
                self.transformer, train_loader, val_loader, criterion, optimizer, self.device,
                epochs=20, scheduler=scheduler
            )

            # CNN
            self.cnn = EnhancedCNNModel(self.n_components).to(self.device)
            optimizer = torch.optim.AdamW(self.cnn.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            self.cnn = train_model(
                self.cnn, train_loader, val_loader, criterion, optimizer, self.device,
                epochs=20, scheduler=scheduler
            )

            # LightGBM
            self.lgbm = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                num_leaves=31,
                max_depth=7,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42
            )
            callbacks = [lgb.early_stopping(stopping_rounds=50)]
            self.lgbm.fit(
                X_train_scaled_resampled, y_train_resampled,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=callbacks
            )

            # XGBoost
            scale_pos_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
            self.xgb = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.xgb.fit(
                X_train_scaled_resampled, y_train_resampled,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50
            )

            # Make predictions
            self.transformer.eval()
            self.cnn.eval()
            with torch.no_grad():
                val_tensor = torch.FloatTensor(X_val_pca).to(self.device)
                transformer_preds[val_idx] = self.transformer(val_tensor).cpu().numpy().ravel()
                cnn_preds[val_idx] = self.cnn(val_tensor).cpu().numpy().ravel()

            lgbm_preds[val_idx] = self.lgbm.predict_proba(X_val_scaled)[:, 1]
            xgb_preds[val_idx] = self.xgb.predict_proba(X_val_scaled)[:, 1]

        # Weighted ensemble predictions
        ensemble_preds = (
            0.3 * transformer_preds +
            0.3 * cnn_preds +
            0.2 * lgbm_preds +
            0.2 * xgb_preds
        )
        final_preds = (ensemble_preds > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y, final_preds)
        auc = roc_auc_score(y, ensemble_preds)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': final_preds,
            'probabilities': ensemble_preds,
            'model_predictions': {
                'transformer': transformer_preds,
                'cnn': cnn_preds,
                'lgbm': lgbm_preds,
                'xgb': xgb_preds
            }
        }

    def predict(self, X):
        X_pca, X_scaled = self.prepare_data(X)
        X_tensor = torch.FloatTensor(X_pca).to(self.device)

        self.transformer.eval()
        self.cnn.eval()
        with torch.no_grad():
            transformer_preds = self.transformer(X_tensor).cpu().numpy().ravel()
            cnn_preds = self.cnn(X_tensor).cpu().numpy().ravel()

        lgbm_preds = self.lgbm.predict_proba(X_scaled)[:, 1]
        xgb_preds = self.xgb.predict_proba(X_scaled)[:, 1]

        ensemble_preds = (
            0.3 * transformer_preds +
            0.3 * cnn_preds +
            0.2 * lgbm_preds +
            0.2 * xgb_preds
        )
        return (ensemble_preds > 0.5).astype(int), ensemble_preds

# Main execution
if __name__ == "__main__":
    # Load data
    train_data = pd.read_csv('transformed_train_df.csv')
    test_data = pd.read_csv('transformed_test_df.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    
    X = train_data.drop('임신 성공 여부', axis=1)
    y = train_data['임신 성공 여부']

    # Train model
    ensemble = EnhancedFertilityEnsemble(n_components=50)
    results = ensemble.train(X, y, n_splits=5, batch_size=32)

    # Make predictions on test data
    predictions, probabilities = ensemble.predict(test_data)

    # Save predictions
    sample_submission['probability'] = probabilities
    sample_submission.to_csv('ensemble_submission.csv', index=False)

    # Print results
    print("\nTraining Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print("\nPrediction Distribution:")
    print(sample_submission['probability'].describe())