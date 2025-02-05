# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 📂 데이터 로드
print("📂 데이터 로드 중...")
train_df = pd.read_csv("./datasets/train.csv")
test_df = pd.read_csv("./datasets/test.csv")

# 숫자형 변수만 선택 (전처리 적용 대상)
train_numeric = train_df.select_dtypes(include=[np.number])
test_numeric = test_df.select_dtypes(include=[np.number])

# 문자형 데이터 (전처리 대상 아님)
train_categorical = train_df.select_dtypes(exclude=[np.number])
test_categorical = test_df.select_dtypes(exclude=[np.number])

# 1️⃣ 삭제 방법 (Remove)
print("🚀 전처리: 삭제(Remove)")
train_remove = train_numeric.dropna()
test_remove = test_numeric.dropna()

# 2️⃣ 선형 보간법 (Linear Interpolation 후 NaN 삭제)
print("🚀 전처리: 선형 보간(Linear Interpolation) 후 NaN 삭제")
train_linear = train_numeric.interpolate(method="linear").dropna()
test_linear = test_numeric.interpolate(method="linear").dropna()

# 3️⃣ 다항 보간법 (Polynomial Interpolation 후 NaN 삭제)
print("🚀 전처리: 다항 보간(Polynomial Interpolation) 후 NaN 삭제")
train_poly = train_numeric.interpolate(method="polynomial", order=2).dropna()
test_poly = test_numeric.interpolate(method="polynomial", order=2).dropna()

# 4️⃣ 평균 대체 (Mean Imputation)
print("🚀 전처리: 평균 대체(Mean Imputation)")
train_mean = train_numeric.fillna(train_numeric.mean())
test_mean = test_numeric.fillna(test_numeric.mean())

# 5️⃣ KNN 대체 (KNN Imputation)
print("🚀 전처리: KNN 대체(KNN Imputation)")
knn_imputer = KNNImputer(n_neighbors=5)
train_knn = pd.DataFrame(knn_imputer.fit_transform(train_numeric), columns=train_numeric.columns)
test_knn = pd.DataFrame(knn_imputer.fit_transform(test_numeric), columns=test_numeric.columns)

# 문자형 데이터 다시 결합 (삭제 방식 제외)
train_linear = pd.concat([train_linear, train_categorical], axis=1)
test_linear = pd.concat([test_linear, test_categorical], axis=1)

train_poly = pd.concat([train_poly, train_categorical], axis=1)
test_poly = pd.concat([test_poly, test_categorical], axis=1)

train_mean = pd.concat([train_mean, train_categorical], axis=1)
test_mean = pd.concat([test_mean, test_categorical], axis=1)

train_knn = pd.concat([train_knn, train_categorical], axis=1)
test_knn = pd.concat([test_knn, test_categorical], axis=1)

# 🚀 5가지 전처리 결과 저장
print("💾 전처리된 데이터 저장 중...")

train_remove.to_csv("./preprocessing/train_remove.csv", index=False)
test_remove.to_csv("./preprocessing/test_remove.csv", index=False)

train_linear.to_csv("./preprocessing/train_linear.csv", index=False)
test_linear.to_csv("./preprocessing/test_linear.csv", index=False)

train_poly.to_csv("./preprocessing/train_poly.csv", index=False)
test_poly.to_csv("./preprocessing/test_poly.csv", index=False)

train_mean.to_csv("./preprocessing/train_mean.csv", index=False)
test_mean.to_csv("./preprocessing/test_mean.csv", index=False)

train_knn.to_csv("./preprocessing/train_knn.csv", index=False)
test_knn.to_csv("./preprocessing/test_knn.csv", index=False)

print("✅ 모든 전처리 완료 및 파일 저장 완료!")
