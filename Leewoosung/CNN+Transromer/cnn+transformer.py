import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os

from google.colab import drive
drive.mount('/content/drive')

# 데이터 로드 함수
def load_preprocessed_data(data_dir):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv", delimiter=",", header=0).values
    X_val = pd.read_csv(f"{data_dir}/X_val.csv", delimiter=",", header=0).values
    X_test = pd.read_csv(f"{data_dir}/X_test.csv", delimiter=",", header=0).values
    y_train = pd.read_csv(f"{data_dir}/y_train.csv", delimiter=",", header=0).values.flatten()
    y_val = pd.read_csv(f"{data_dir}/y_val.csv", delimiter=",", header=0).values.flatten()

    # CNN 입력 형식 변환
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_val, X_test, y_train, y_val

# CNN + Transformer 모델 정의
def create_cnn_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # CNN Layer
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Transformer Encoder (Reshape to ensure proper input shape)
    x = layers.Reshape((1, x.shape[-1]))(x)  # Reshape to (batch, seq_len=1, feature_dim)
    num_heads = 4
    key_dim = 64
    transformer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    transformer = layers.LayerNormalization()(transformer + x)
    transformer = layers.Flatten()(transformer)  # Flatten back to (batch, feature_dim)

    # Fully Connected Layers
    x = layers.Dense(128, activation="relu")(transformer)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model

# 데이터 디렉토리 지정
data_dir = "/content/drive/MyDrive/open/preprocessed_data/"
X_train, X_val, X_test, y_train, y_val = load_preprocessed_data(data_dir)

# 데이터 차원 확인
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")

# 모델 생성
model = create_cnn_transformer_model((X_train.shape[1], 1))

# 모델 컴파일
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# 테스트 데이터 예측
predictions = model.predict(X_test)

# 예측 결과를 sample_submission 형식으로 저장
sample_submission_path = "/content/drive/MyDrive/open/sample_submission.csv"
sample_submission = pd.read_csv(sample_submission_path)
sample_submission["probability"] = predictions.flatten()
output_path = f"/content/drive/MyDrive/open/predictions.csv"
sample_submission.to_csv(output_path, index=False)

print(f"모델 학습 및 예측이 완료되었습니다. 결과가 {output_path}에 저장되었습니다.")
