import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os

from google.colab import drive
drive.mount('/content/drive')

def load_preprocessed_data(data_dir):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv").values
    X_val = pd.read_csv(f"{data_dir}/X_val.csv").values
    X_test = pd.read_csv(f"{data_dir}/X_test.csv").values
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.flatten()
    y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.flatten()

    X_train = np.expand_dims(X_train, axis=1)  # (batch, sequence_length=1, feature_dim)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    return X_train, X_val, X_test, y_train, y_val

def create_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Transformer Encoder Block
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = layers.LayerNormalization()(x + inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.LayerNormalization()(x)

    # Flatten before passing to fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model

def train_transformer_model(data_dir, output_model_path):
    X_train, X_val, X_test, y_train, y_val = load_preprocessed_data(data_dir)

    model = create_transformer_model((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)
    model.save(output_model_path)

    return model, X_test

def make_predictions(model, X_test, sample_submission_path, output_submission_path):
    predictions = model.predict(X_test).flatten()

    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission["probability"] = predictions
    sample_submission.to_csv(output_submission_path, index=False)

    print(f"예측 완료, 결과가 {output_submission_path}에 저장되었습니다.")

# 실행 코드
data_dir = "/content/drive/MyDrive/open/Transformer/preprocessed_data"
sample_submission_path = "/content/drive/MyDrive/open/sample_submission.csv"
output_submission_path = "/content/drive/MyDrive/open/Transformer/preprocessed_data/predictions.csv"

# 모델 학습
model, X_test = train_transformer_model(data_dir)

# 예측 및 제출 파일 저장
make_predictions(model, X_test, sample_submission_path, output_submission_path)
