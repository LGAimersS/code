import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

from google.colab import drive
drive.mount('/content/drive')

# 데이터 로드
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.drop(columns=['ID'], errors='ignore', inplace=True)
    test_df.drop(columns=['ID'], errors='ignore', inplace=True)

    return train_df, test_df

# 결측값 처리
def handle_missing_values(train_df, test_df):
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    if '임신 성공 여부' in numerical_cols:
        numerical_cols.remove('임신 성공 여부')

    train_df[numerical_cols] = num_imputer.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = num_imputer.transform(test_df[numerical_cols])

    train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
    test_df[categorical_cols] = cat_imputer.transform(test_df[categorical_cols])

    return train_df, test_df, numerical_cols, categorical_cols

# 범주형 변수 인코딩
def encode_categorical_features(train_df, test_df, categorical_cols):
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        category_mapping = {category: idx for idx, category in enumerate(le.classes_)}
        test_df[col] = test_df[col].map(category_mapping).fillna(-1).astype(int)

    return train_df, test_df

# 데이터 정규화 (표준화)
def scale_numerical_features(train_df, test_df, numerical_cols):
    scaler = StandardScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    return train_df, test_df

# 데이터 분할 및 CNN 입력 변환
def prepare_for_cnn(train_df, test_df):
    X = train_df.drop(columns=['임신 성공 여부'])
    y = train_df['임신 성공 여부']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_cnn = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_cnn = test_df.values.reshape(test_df.shape[0], test_df.shape[1], 1)

    return X_train_cnn, X_val_cnn, X_test_cnn, y_train, y_val, X_train, X_val, test_df

# 전체 전처리 실행 함수
def preprocess_data(train_path, test_path, output_dir):
    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df, numerical_cols, categorical_cols = handle_missing_values(train_df, test_df)
    train_df, test_df = encode_categorical_features(train_df, test_df, categorical_cols)
    train_df, test_df = scale_numerical_features(train_df, test_df, numerical_cols)
    X_train_cnn, X_val_cnn, X_test_cnn, y_train, y_val, X_train, X_val, test_df = prepare_for_cnn(train_df, test_df)

    # 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # CSV 파일 저장
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    test_df.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)

    return X_train_cnn, X_val_cnn, X_test_cnn, y_train, y_val

# 실행 코드
if __name__ == "__main__":
    train_path = "/content/drive/MyDrive/open/train.csv"
    test_path = "/content/drive/MyDrive/open/test.csv"
    output_dir = "preprocessed_data"  # 저장할 폴더 지정

    preprocess_data(train_path, test_path, output_dir)
    print("전처리가 완료되었습니다. CSV 파일이 저장되었습니다.")
