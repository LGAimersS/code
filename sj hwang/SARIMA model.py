# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima  # 최적의 (p, d, q) 탐색
import warnings
import os

# FutureWarning 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# 평가 지표 계산 함수
def evaluate_model(y_true, y_pred, model_name):
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"🚨 {model_name}: 평가 지표를 계산할 충분한 데이터가 없음")
        return None

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"📊 {model_name} 평가 결과:")
    print(f"✅ MSE  (Mean Squared Error): {mse:.4f}")
    print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"✅ MAE  (Mean Absolute Error): {mae:.4f}")
    print(f"✅ MAPE (Mean Absolute Percentage Error): {mape:.2f}%\n")

    return mse, rmse, mae, mape

# 시각화 함수 (실제값 vs 예측값 비교)
def plot_predictions(y_true, y_pred, model_name):
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"🚨 {model_name}: 시각화를 위한 데이터가 부족함")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="red")  # 일반 선 그래프
    plt.legend()
    plt.title(f"{model_name} Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

# 잔차 분석 (Residual Plot)
def plot_residuals(y_true, y_pred, model_name):
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"🚨 {model_name}: 잔차 분석을 위한 데이터가 부족함")
        return

    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"{model_name} Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

# SARIMA 모델 학습 및 예측 함수
def train_and_evaluate_sarima(train_path, test_path, method):
    try:
        # 1️⃣ 데이터 로드
        print(f"\n📂 데이터 로드 중... ({method})")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # 2️⃣ 학습 데이터에서 '임신 성공 여부' 가져오기
        if "임신 성공 여부" not in train_df.columns:
            print(f"🚨 {method}: '임신 성공 여부' 컬럼이 없음")
            return

        y_train = train_df["임신 성공 여부"]

        # 3️⃣ NaN 값 제거 (필수)
        y_train = y_train.dropna()

        # 4️⃣ 데이터가 비어있는지 확인
        if len(y_train) == 0:
            print(f"🚨 {method}: 훈련 데이터가 비어 있음 → 모델 학습 건너뜀")
            return

        # 5️⃣ 최적의 SARIMA 파라미터 찾기 (auto_arima 사용)
        print(f"🔍 최적의 SARIMA 파라미터 찾는 중... ({method})")
        optimal_sarima = auto_arima(y_train, seasonal=True, m=12, trace=True)
        best_order = optimal_sarima.order
        best_seasonal_order = optimal_sarima.seasonal_order
        print(f"✅ 최적의 SARIMA 파라미터: {best_order}, 계절성: {best_seasonal_order}")

        # 6️⃣ SARIMA 모델 훈련
        print(f"🚀 SARIMA 모델 학습 시작... ({method})")
        model = SARIMAX(y_train, order=best_order, seasonal_order=best_seasonal_order)
        model_fit = model.fit()

        # 7️⃣ 테스트 데이터 예측
        print(f"🔮 테스트 데이터 예측 중... ({method})")
        y_pred = model_fit.forecast(steps=len(test_df))
        y_pred = np.nan_to_num(y_pred)  # NaN이 나오면 0으로 대체

        # 8️⃣ 평가 지표 출력
        evaluate_model(y_train[-len(y_pred):], y_pred, f"SARIMA_{method}")

        # 9️⃣ 시각화 (실제값 vs 예측값 비교)
        plot_predictions(y_train[-len(y_pred):], y_pred, f"SARIMA_{method}")
        plot_residuals(y_train[-len(y_pred):], y_pred, f"SARIMA_{method}")

        # 🔟 원본 테스트 데이터와 병합하여 모든 ID 포함
        original_test_df = pd.read_csv("./datasets/test.csv")[["ID"]]
        sarima_results = pd.DataFrame({"ID": test_df["ID"], "probability": y_pred})

        final_submission = original_test_df.merge(sarima_results, on="ID", how="left")
        final_submission["probability"] = final_submission["probability"].fillna(0)

        # 1️⃣1️⃣ 결과 저장
        output_dir = "./Sarima_result"
        os.makedirs(output_dir, exist_ok=True)  # 디렉터리 자동 생성
        output_filename = f"{output_dir}/SARIMA_{method}_sample.csv"
        final_submission.to_csv(output_filename, index=False)
        print(f"✅ 예측 결과 저장 완료! ({output_filename})")

    except Exception as e:
        print(f"🚨 오류 발생: {e}")

# 🔥 실행 코드 (5가지 전처리 방식별 학습)
if __name__ == "__main__":
    methods = ["remove", "linear", "poly", "mean", "knn"]
    train_files = {m: f"./preprocessing/train_{m}.csv" for m in methods}
    test_files = {m: f"./preprocessing/test_{m}.csv" for m in methods}

    for method in methods:
        train_and_evaluate_sarima(train_files[method], test_files[method], method)
