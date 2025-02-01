## 데이터 분석에 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
test_data = pd.read_csv('./dataset/test.csv')
test_df = pd.DataFrame(data=test_data)
train_data = pd.read_csv('./dataset/train.csv')
train_df = pd.DataFrame(data=train_data)
print(test_df)
print(train_df)

#----------------------------
## 각 데이터는 일단, 시계열 데이터임을 알 수 있음. 시계열 데이터에 알맞게 데이터를 분석해야 함.
'''
데이터 전처리 방식은 매우 다양하게 존재함. 
그 중 나는 5가지 방식으로 전처리를 진행할 예정. 제일 성능 좋은 것으로 실험해볼 것임. 
1. 삭제 (Remove): 결측값이 있는 행을 삭제
2. 선형 보간법 (Linear Interpolation): 결측값을 선형적으로 보간
3. 다항 보간법 (Polynomial Interpolation): 결측값을 다항식으로 보간
4. 평균 대체 (Mean Imputation): 결측값을 해당 열의 평균값으로 대체
5. KNN 대체 (K-Nearest Neighbors Imputation): KNN을 이용하여 결측값을 대체
'''
from sklearn.impute import KNNImputer

# 데이터 로드
test_data = pd.read_csv('./dataset/test.csv')
train_data = pd.read_csv('./dataset/train.csv')

# 원본 데이터프레임 생성
test_df = pd.DataFrame(data=test_data)
train_df = pd.DataFrame(data=train_data)

# 숫자형 데이터만 선택
test_numeric = test_df.select_dtypes(include=[np.number])
train_numeric = train_df.select_dtypes(include=[np.number])

# 문자형 데이터 (전처리 대상 아님)
test_categorical = test_df.select_dtypes(exclude=[np.number])
train_categorical = train_df.select_dtypes(exclude=[np.number])

## 1. 삭제 방법 (숫자형 변수에서만 적용)
delete_test_numeric = test_numeric.dropna()
delete_train_numeric = train_numeric.dropna()

# 문자형 데이터를 원래대로 유지
delete_test_df = pd.concat([delete_test_numeric, test_categorical], axis=1)
delete_train_df = pd.concat([delete_train_numeric, train_categorical], axis=1)

## 2. 선형 보간법 (숫자형 변수에서만 적용)
data_linear_interpolate_test = test_numeric.interpolate(method='linear')
data_linear_interpolate_train = train_numeric.interpolate(method='linear')

# 문자형 데이터 유지
data_linear_interpolate_test = pd.concat([data_linear_interpolate_test, test_categorical], axis=1)
data_linear_interpolate_train = pd.concat([data_linear_interpolate_train, train_categorical], axis=1)

## 3. 다항 보간법 (2차 다항식, 숫자형 변수만)
data_poly_interpolate_test = test_numeric.interpolate(method='polynomial', order=2)
data_poly_interpolate_train = train_numeric.interpolate(method='polynomial', order=2)

# 문자형 데이터 유지
data_poly_interpolate_test = pd.concat([data_poly_interpolate_test, test_categorical], axis=1)
data_poly_interpolate_train = pd.concat([data_poly_interpolate_train, train_categorical], axis=1)

## 4. 평균 대체 (숫자형 변수에서만 적용)
data_mean_fill_test = test_numeric.fillna(test_numeric.mean())
data_mean_fill_train = train_numeric.fillna(train_numeric.mean())

# 문자형 데이터 유지
data_mean_fill_test = pd.concat([data_mean_fill_test, test_categorical], axis=1)
data_mean_fill_train = pd.concat([data_mean_fill_train, train_categorical], axis=1)

## 5. KNN 대체 (숫자형 변수에서만 적용)
knn_imputer = KNNImputer(n_neighbors=5)
test_data_knn_fill_numeric = pd.DataFrame(knn_imputer.fit_transform(test_numeric), columns=test_numeric.columns)
train_data_knn_fill_numeric = pd.DataFrame(knn_imputer.fit_transform(train_numeric), columns=train_numeric.columns)

# 문자형 데이터 유지
test_data_knn_fill = pd.concat([test_data_knn_fill_numeric, test_categorical], axis=1)
train_data_knn_fill = pd.concat([train_data_knn_fill_numeric, train_categorical], axis=1)

# 확인용 출력 (숫자형 변수만 NaN 값 처리됨)
print('test')
print("Original Data - NaN count:\n", test_df.isnull().sum())
print("Data after dropping rows:\n", delete_test_df.isnull().sum())
print("Data after linear interpolation:\n", data_linear_interpolate_test.isnull().sum())
print("Data after polynomial interpolation:\n", data_poly_interpolate_test.isnull().sum())
print("Data after mean imputation:\n", data_mean_fill_test.isnull().sum())
print("Data after KNN imputation:\n", test_data_knn_fill.isnull().sum())

print('train')
print("Original Data - NaN count:\n", train_df.isnull().sum())
print("Data after dropping rows:\n", delete_train_df.isnull().sum())
print("Data after linear interpolation:\n", data_linear_interpolate_train.isnull().sum())
print("Data after polynomial interpolation:\n", data_poly_interpolate_train.isnull().sum())
print("Data after mean imputation:\n", data_mean_fill_train.isnull().sum())
print("Data after KNN imputation:\n", train_data_knn_fill.isnull().sum())

delete_test_df.to_csv("./dataset/test_remove.csv", index=False)
delete_train_df.to_csv("./dataset/train_remove.csv", index=False)

data_linear_interpolate_test.to_csv("./dataset/test_linear.csv", index=False)
data_linear_interpolate_train.to_csv("./dataset/train_linear.csv", index=False)

data_poly_interpolate_test.to_csv("./dataset/test_poly.csv", index=False)
data_poly_interpolate_train.to_csv("./dataset/train_poly.csv", index=False)

data_mean_fill_test.to_csv("./dataset/test_mean.csv", index=False)
data_mean_fill_train.to_csv("./dataset/train_mean.csv", index=False)

test_data_knn_fill.to_csv("./dataset/test_knn.csv", index=False)
train_data_knn_fill.to_csv("./dataset/train_knn.csv", index=False)

print("모든 전처리 데이터가 CSV 파일로 저장되었습니다!")
