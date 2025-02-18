import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

from google.colab import drive
drive.mount('/content/drive')

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.drop(columns=['ID'], errors='ignore', inplace=True)
    test_df.drop(columns=['ID'], errors='ignore', inplace=True)

    return train_df, test_df

def handle_missing_values(train_df, test_df, target_col):
    num_imputer = SimpleImputer(strategy="mean")  # 수치형 변수: 평균 대체
    cat_imputer = SimpleImputer(strategy="most_frequent")  # 범주형 변수: 최빈값 대체

    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    train_df[numerical_cols] = num_imputer.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = num_imputer.transform(test_df[numerical_cols])

    train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
    test_df[categorical_cols] = cat_imputer.transform(test_df[categorical_cols])

    return train_df, test_df, numerical_cols, categorical_cols

def encode_categorical_features(train_df, test_df, categorical_cols):
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return train_df, test_df

def scale_numerical_features(train_df, test_df, numerical_cols):
    scaler = StandardScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    return train_df, test_df

def prepare_cnn_lstm_input(train_df, test_df, target_col):
    X = train_df.drop(columns=[target_col], errors='ignore')
    y = train_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = np.expand_dims(X_train.values, axis=1)  # (batch, sequence_length=1, feature_dim)
    X_val = np.expand_dims(X_val.values, axis=1)
    X_test = np.expand_dims(test_df.values, axis=1)

    return X_train, X_val, X_test, y_train, y_val

def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train[:, 0, :]).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_val[:, 0, :]).to_csv(f"{output_dir}/X_val.csv", index=False)
    pd.DataFrame(X_test[:, 0, :]).to_csv(f"{output_dir}/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
    pd.DataFrame(y_val).to_csv(f"{output_dir}/y_val.csv", index=False)

    print(f"전처리 완료! 데이터가 {output_dir}에 저장되었습니다.")

def preprocess_for_cnn_lstm(train_path, test_path, output_dir):
    target_col = '임신 성공 여부'
    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df, numerical_cols, categorical_cols = handle_missing_values(train_df, test_df, target_col)
    train_df, test_df = encode_categorical_features(train_df, test_df, categorical_cols)
    train_df, test_df = scale_numerical_features(train_df, test_df, numerical_cols)
    X_train, X_val, X_test, y_train, y_val = prepare_cnn_lstm_input(train_df, test_df, target_col)
    save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val)

# 실행 코드
train_path = "/content/drive/MyDrive/open/train.csv"
test_path = "/content/drive/MyDrive/open/test.csv"
output_dir = "/content/drive/MyDrive/open/preprocessed_data"
preprocess_for_cnn_lstm(train_path, test_path, output_dir)
