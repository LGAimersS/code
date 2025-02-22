# -*- coding: utf-8 -*-
"""combine_case.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hYZpZfL7kGmtls8AJ3DCKjGG9z5DLe0P
"""

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

from google.colab import drive
drive.mount('/content/drive')

"""###<h2>DataLoad"""

import pandas as pd
import numpy as np
train_df = pd.read_csv('/content/drive/MyDrive/lg aimers/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/lg aimers/test.csv')

# 데이터 합치기
total_data = pd.concat([train_df, test_df], axis=0)

# 각 컬럼의 고유값 확인
for column in total_data.columns:
    print(f"\n[{column}] 고유값:")
    print(total_data[column].unique())
    print(f"고유값 개수: {total_data[column].nunique()}")

"""<h2>Separate"""

def create_treatment_type_columns(data):
    # 기본 시술 유형 리스트
    treatment_types = ['ICSI', 'IVF', 'IUI', 'BLASTOCYST', 'AH',
                      'IVI', 'ICI', 'GIFT', 'FER', 'Generic DI']

    # 결과를 저장할 새로운 데이터프레임 생성
    new_columns = pd.DataFrame(index=data.index)

    # 각 시술 유형별로 확인
    for treatment in treatment_types:
        # 해당 시술이 포함되어 있으면 1, 아니면 0
        new_columns[treatment] = data['특정 시술 유형'].str.contains(treatment, na=False).astype(int)

    # Unknown과 NaN을 하나의 열로 처리
    new_columns['Unknown_or_NaN'] = (
        (data['특정 시술 유형'].isna()) |
        (data['특정 시술 유형'].str.contains('Unknown', na=False))
    ).astype(int)

    return new_columns

# 실행
result = create_treatment_type_columns(train_df)
train_df = pd.concat([train_df, result], axis=1)
result = create_treatment_type_columns(test_df)
test_df = pd.concat([test_df, result], axis=1)

train_df.drop(['특정 시술 유형'], axis=1, inplace=True)
test_df.drop(['특정 시술 유형'], axis=1, inplace=True)

def create_make_type_columns(data):
    # 기본 시술 유형 리스트
    make_types = ['현재 시술용', '난자 저장용', '배아 저장용', '기증용', '연구용']

    # 결과를 저장할 새로운 데이터프레임 생성
    new_columns = pd.DataFrame(index=data.index)

    # 각 시술 유형별로 확인
    for make in make_types:
        # 해당 시술이 포함되어 있으면 1, 아니면 0
        new_columns[make] = data['배아 생성 주요 이유'].str.contains(make, na=False).astype(int)

    # Unknown과 NaN을 하나의 열로 처리
    new_columns['생성_NaN'] = (
        (data['배아 생성 주요 이유'].isna()) |
        (data['배아 생성 주요 이유'].str.contains('Unknown', na=False))
    ).astype(int)

    return new_columns

# 실행
result = create_make_type_columns(train_df)
train_df = pd.concat([train_df, result], axis=1)
result = create_make_type_columns(test_df)
test_df = pd.concat([test_df, result], axis=1)

train_df.drop(['배아 생성 주요 이유'], axis=1, inplace=True)
test_df.drop(['배아 생성 주요 이유'], axis=1, inplace=True)

train_df.columns

# 새로운 '배란 유형' 열 생성
train_df['배란 유형'] = np.where(
    train_df['배란 자극 여부'] == 0, 'A',  # 배란 자극 여부가 0인 경우
    np.where(
        (train_df['배란 자극 여부'] == 1) & (train_df['배란 유도 유형'] == '세트로타이드 (억제제)'), 'B',  # 조건 2
        np.where(
            (train_df['배란 자극 여부'] == 1) & (train_df['배란 유도 유형'] == '생식선 자극 호르몬'), 'C',  # 조건 3
            np.where(
                (train_df['배란 자극 여부'] == 1) &
                ((train_df['배란 유도 유형'] == '기록되지 않은 시행') |
                 (train_df['배란 유도 유형'] == '알 수 없음')), np.nan,  # 조건 4
                np.nan  # 나머지 경우
            )
        )
    )
)

# 새로운 '배란 유형' 열 생성
test_df['배란 유형'] = np.where(
    test_df['배란 자극 여부'] == 0, 'A',  # 배란 자극 여부가 0인 경우
    np.where(
        (test_df['배란 자극 여부'] == 1) & (test_df['배란 유도 유형'] == '세트로타이드 (억제제)'), 'B',  # 조건 2
        np.where(
            (test_df['배란 자극 여부'] == 1) & (test_df['배란 유도 유형'] == '생식선 자극 호르몬'), 'C',  # 조건 3
            np.where(
                (test_df['배란 자극 여부'] == 1) &
                ((test_df['배란 유도 유형'] == '기록되지 않은 시행') |
                 (test_df['배란 유도 유형'] == '알 수 없음')), np.nan,  # 조건 4
                np.nan  # 나머지 경우
            )
        )
    )
)

train_df.drop(['배란 자극 여부'], axis=1, inplace=True)
test_df.drop(['배란 자극 여부'], axis=1, inplace=True)

train_df.drop(['배란 유도 유형'], axis=1, inplace=True)
test_df.drop(['배란 유도 유형'], axis=1, inplace=True)

"""<h3> fill - 유전 검사 원인, 결과"""

# 해당하는 모든 열을 리스트로 저장
columns = ['착상 전 유전 검사 사용 여부', '착상 전 유전 진단 사용 여부', '남성 주 불임 원인',
           '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인', '부부 주 불임 원인',
           '부부 부 불임 원인', '불명확 불임 원인', '불임 원인 - 난관 질환', '불임 원인 - 남성 요인',
           '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제',
           '불임 원인 - 자궁내막증', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인',
           '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태']

# 모든 열이 NaN인 행을 찾아서 마스크 생성
all_train_nan_mask = train_df[columns].isna().all(axis=1)

# 마스크를 사용하여 해당 행들의 열들만 0으로 채우기
train_df.loc[all_train_nan_mask, columns] = 0

all_test_nan_mask = train_df[columns].isna().all(axis=1)

# 마스크를 사용하여 해당 행들의 열들만 0으로 채우기
train_df.loc[all_test_nan_mask, columns] = 0

# 검사할 열 목록
columns = ['남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인',
           '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인', '불임 원인 - 난관 질환',
           '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인',
           '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', '불임 원인 - 정자 농도',
           '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태']

# 모든 열이 0인 행 찾기
all_zero_mask = (train_df[columns] == 0).all(axis=1)

# 하나라도 1인 행 찾기
any_one_mask = (train_df[columns] == 1).any(axis=1)

# 조건에 따라 '착상 전 유전 진단 사용 여부' 열 업데이트
# 모든 값이 0인 경우 0으로, 하나라도 1이 있는 경우 1로 채우기
train_df.loc[all_zero_mask & train_df['착상 전 유전 진단 사용 여부'].isna(), '착상 전 유전 진단 사용 여부'] = 0
train_df.loc[any_one_mask & train_df['착상 전 유전 진단 사용 여부'].isna(), '착상 전 유전 진단 사용 여부'] = 1

# 모든 열이 0인 행 찾기
all_zero_mask = (test_df[columns] == 0).all(axis=1)

# 하나라도 1인 행 찾기
any_one_mask = (test_df[columns] == 1).any(axis=1)

# 조건에 따라 '착상 전 유전 진단 사용 여부' 열 업데이트
# 모든 값이 0인 경우 0으로, 하나라도 1이 있는 경우 1로 채우기
test_df.loc[all_zero_mask & test_df['착상 전 유전 진단 사용 여부'].isna(), '착상 전 유전 진단 사용 여부'] = 0
test_df.loc[any_one_mask & test_df['착상 전 유전 진단 사용 여부'].isna(), '착상 전 유전 진단 사용 여부'] = 1

# 착상 전 유전 진단 사용 여부가 0이고 착상 전 유전 검사 사용 여부가 NaN인 경우를 찾아 0으로 채우기
mask = (train_df['착상 전 유전 진단 사용 여부'] == 0) & (train_df['착상 전 유전 검사 사용 여부'].isna())
train_df.loc[mask, '착상 전 유전 검사 사용 여부'] = 0
mask = (test_df['착상 전 유전 진단 사용 여부'] == 0) & (test_df['착상 전 유전 검사 사용 여부'].isna())
test_df.loc[mask, '착상 전 유전 검사 사용 여부'] = 0
mask = (train_df['착상 전 유전 진단 사용 여부'] == 1) & (train_df['착상 전 유전 검사 사용 여부'].isna())
train_df.loc[mask, '착상 전 유전 검사 사용 여부'] = 1
mask = (test_df['착상 전 유전 진단 사용 여부'] == 1) & (test_df['착상 전 유전 검사 사용 여부'].isna())
test_df.loc[mask, '착상 전 유전 검사 사용 여부'] = 1

train_df.info()

# 모든 행을 표시하도록 설정
pd.set_option('display.max_rows', None)

# NaN 개수와 비율 계산
nan_info = pd.DataFrame({
    'NaN 개수': test_df.isna().sum(),
    'NaN 비율(%)': (test_df.isna().sum() / len(test_df) * 100).round(2)
})

# 내림차순으로 정렬하여 출력
print(nan_info.sort_values('NaN 개수', ascending=False))

# 설정을 다시 원래대로 되돌리기 (필요한 경우)
pd.reset_option('display.max_rows')

mask = (
    (train_df['PGS 시술 여부'].isna()) &
    (train_df['PGD 시술 여부'].isna()) &
    ((train_df['착상 전 유전 검사 사용 여부'] == 0) |
     (train_df['착상 전 유전 진단 사용 여부'] == 0))
)

# 해당 조건에 맞는 행의 PGS, PGD 시술 여부를 0으로 채우기
train_df.loc[mask, 'PGS 시술 여부'] = 0
train_df.loc[mask, 'PGD 시술 여부'] = 0

mask = (
    (test_df['PGS 시술 여부'].isna()) &
    (test_df['PGD 시술 여부'].isna()) &
    ((test_df['착상 전 유전 검사 사용 여부'] == 0) |
     (test_df['착상 전 유전 진단 사용 여부'] == 0))
)

# 해당 조건에 맞는 행의 PGS, PGD 시술 여부를 0으로 채우기
test_df.loc[mask, 'PGS 시술 여부'] = 0
test_df.loc[mask, 'PGD 시술 여부'] = 0

"""<h3> 난자 해동 경과일 채우기"""

# 1. 해동 과정이 필요없는 경우 확인
no_thaw_mask = (
    (train_df['해동 난자 수'] == 0) |
    (train_df['동결 배아 사용 여부'] == 0) |
    (train_df['FER'] == 0)
)

# 2. 해동 경과일이 있는 데이터의 unique한 값 확인
existing_values = train_df['난자 해동 경과일'].dropna().unique()
print("현재 존재하는 해동 경과일 값:", existing_values)  # [0, 1, 5] 확인

# 3. NaN 값 채우기
# 해동 과정이 필요없는 경우 0으로 채우기
train_df.loc[no_thaw_mask & train_df['난자 해동 경과일'].isna(), '난자 해동 경과일'] = 0

# 나머지 NaN 값에 대해서는 [0, 1, 5] 중에서 적절한 값으로 채우기
# 예를 들어, 중앙값이나 가장 빈번한 값으로 채우기
median_thaw_days = train_df['난자 해동 경과일'].median()
train_df.loc[train_df['난자 해동 경과일'].isna(), '난자 해동 경과일'] = median_thaw_days

# 결과 확인
print("\n값 분포 확인:")
print(train_df['난자 해동 경과일'].value_counts(dropna=False))
def fill_thaw_days(row):
    # 해동 과정이 필요없는 경우
    if (row['해동 난자 수'] == 0 or
        row['동결 배아 사용 여부'] == 0 or
        row['FER'] == 0):
        return 0

    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['난자 해동 경과일']):
        return row['난자 해동 경과일']

    # 나머지 NaN 케이스에 대한 처리
    if pd.isna(row['난자 해동 경과일']):
        # 여기서 다른 조건들을 확인하여 0, 1, 5 중 적절한 값 할당
        if row['FER'] == 1:  # FER 시술인 경우
            return 1  # 또는 다른 적절한 값
        else:
            return median_thaw_days  # 전체 데이터의 중앙값 사용

# 적용
train_df['난자 해동 경과일'] = train_df.apply(fill_thaw_days, axis=1)

# 1. 해동 과정이 필요없는 경우 확인
no_thaw_mask = (
    (test_df['해동 난자 수'] == 0) |
    (test_df['동결 배아 사용 여부'] == 0) |
    (test_df['FER'] == 0)
)

# 2. 해동 경과일이 있는 데이터의 unique한 값 확인
existing_values = test_df['난자 해동 경과일'].dropna().unique()
print("현재 존재하는 해동 경과일 값:", existing_values)  # [0, 1, 5] 확인

# 3. NaN 값 채우기
# 해동 과정이 필요없는 경우 0으로 채우기
test_df.loc[no_thaw_mask & test_df['난자 해동 경과일'].isna(), '난자 해동 경과일'] = 0

# 나머지 NaN 값에 대해서는 [0, 1, 5] 중에서 적절한 값으로 채우기
# 예를 들어, 중앙값이나 가장 빈번한 값으로 채우기
median_thaw_days = test_df['난자 해동 경과일'].median()
test_df.loc[test_df['난자 해동 경과일'].isna(), '난자 해동 경과일'] = median_thaw_days

# 결과 확인
print("\n값 분포 확인:")
print(test_df['난자 해동 경과일'].value_counts(dropna=False))
def fill_thaw_days(row):
    # 해동 과정이 필요없는 경우
    if (row['해동 난자 수'] == 0 or
        row['동결 배아 사용 여부'] == 0 or
        row['FER'] == 0):
        return 0

    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['난자 해동 경과일']):
        return row['난자 해동 경과일']

    # 나머지 NaN 케이스에 대한 처리
    if pd.isna(row['난자 해동 경과일']):
        # 여기서 다른 조건들을 확인하여 0, 1, 5 중 적절한 값 할당
        if row['FER'] == 1:  # FER 시술인 경우
            return 1  # 또는 다른 적절한 값
        else:
            return median_thaw_days  # 전체 데이터의 중앙값 사용

# 적용
test_df['난자 해동 경과일'] = test_df.apply(fill_thaw_days, axis=1)

"""<h3>배아 해동 경과일 채워넣기"""

# 1. 배아 해동 과정이 필요없는 경우 확인
no_embryo_thaw_mask = (
    (train_df['해동된 배아 수'] == 0) |
    (train_df['동결 배아 사용 여부'] == 0) |
    (train_df['FER'] == 0) |
    (train_df['신선 배아 사용 여부'] == 1)  # 신선 배아 사용 시 해동 불필요
)

# 2. 해동 경과일이 있는 데이터의 unique한 값 확인
existing_values = train_df['배아 해동 경과일'].dropna().unique()
print("현재 존재하는 배아 해동 경과일 값:", existing_values)

# 3. NaN 값 채우기
# 해동 과정이 필요없는 경우 0으로 채우기
train_df.loc[no_embryo_thaw_mask & train_df['배아 해동 경과일'].isna(), '배아 해동 경과일'] = 0

# 나머지 NaN 값에 대해서는 기존 범위(0~7) 내에서 적절한 값으로 채우기
# 예를 들어, 중앙값으로 채우기
median_embryo_thaw_days = train_df[train_df['배아 해동 경과일'].notna()]['배아 해동 경과일'].median()

# 더 상세한 조건부 채우기를 위한 함수
def fill_embryo_thaw_days(row):
    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['배아 해동 경과일']):
        return row['배아 해동 경과일']

    # 해동 과정이 필요없는 경우
    if (row['해동된 배아 수'] == 0 or
        row['동결 배아 사용 여부'] == 0 or
        row['FER'] == 0 or
        row['신선 배아 사용 여부'] == 1):
        return 0

    # FER 시술이면서 해동된 배아가 있는 경우
    if row['FER'] == 1 and row['해동된 배아 수'] > 0:
        # 배아 이식 경과일이 있는 경우, 그 값에서 1-2일 정도 뺀 값을 사용
        if not pd.isna(row['배아 이식 경과일']):
            return max(0, row['배아 이식 경과일'] - 2)
        return median_embryo_thaw_days

    # 그 외의 경우 중앙값 사용
    return median_embryo_thaw_days

# 적용
train_df['배아 해동 경과일'] = train_df.apply(fill_embryo_thaw_days, axis=1)

# 결과 확인
print("\n값 분포 확인:")
print(train_df['배아 해동 경과일'].value_counts(dropna=False))

# 1. 배아 해동 과정이 필요없는 경우 확인
no_embryo_thaw_mask = (
    (test_df['해동된 배아 수'] == 0) |
    (test_df['동결 배아 사용 여부'] == 0) |
    (test_df['FER'] == 0) |
    (test_df['신선 배아 사용 여부'] == 1)  # 신선 배아 사용 시 해동 불필요
)

# 2. 해동 경과일이 있는 데이터의 unique한 값 확인
existing_values = test_df['배아 해동 경과일'].dropna().unique()
print("현재 존재하는 배아 해동 경과일 값:", existing_values)

# 3. NaN 값 채우기
# 해동 과정이 필요없는 경우 0으로 채우기
test_df.loc[no_embryo_thaw_mask & test_df['배아 해동 경과일'].isna(), '배아 해동 경과일'] = 0

# 나머지 NaN 값에 대해서는 기존 범위(0~7) 내에서 적절한 값으로 채우기
# 예를 들어, 중앙값으로 채우기
median_embryo_thaw_days = test_df[test_df['배아 해동 경과일'].notna()]['배아 해동 경과일'].median()

# 더 상세한 조건부 채우기를 위한 함수
def fill_embryo_thaw_days(row):
    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['배아 해동 경과일']):
        return row['배아 해동 경과일']

    # 해동 과정이 필요없는 경우
    if (row['해동된 배아 수'] == 0 or
        row['동결 배아 사용 여부'] == 0 or
        row['FER'] == 0 or
        row['신선 배아 사용 여부'] == 1):
        return 0

    # FER 시술이면서 해동된 배아가 있는 경우
    if row['FER'] == 1 and row['해동된 배아 수'] > 0:
        # 배아 이식 경과일이 있는 경우, 그 값에서 1-2일 정도 뺀 값을 사용
        if not pd.isna(row['배아 이식 경과일']):
            return max(0, row['배아 이식 경과일'] - 2)
        return median_embryo_thaw_days

    # 그 외의 경우 중앙값 사용
    return median_embryo_thaw_days

# 적용
test_df['배아 해동 경과일'] = test_df.apply(fill_embryo_thaw_days, axis=1)

# 결과 확인
print("\n값 분포 확인:")
print(test_df['배아 해동 경과일'].value_counts(dropna=False))

"""<h3>난자 채취 경과일 제거"""

train_df.drop(['난자 채취 경과일'], axis=1, inplace=True)
test_df.drop(['난자 채취 경과일'], axis=1, inplace=True)

"""<h3>난자 혼합 경과일 채워넣기"""

# 1. 난자 혼합 과정이 필요없는 경우 확인
no_mixing_mask = (
    (train_df['혼합된 난자 수'] == 0) |
    (train_df['파트너 정자와 혼합된 난자 수'] == 0) |
    (train_df['기증자 정자와 혼합된 난자 수'] == 0) |
    (train_df['FER'] == 1)  # 동결 배아 이식의 경우 난자 혼합 과정 불필요
)

# 2. 혼합 경과일이 있는 데이터의 unique한 값 확인
existing_values = train_df['난자 혼합 경과일'].dropna().unique()
print("현재 존재하는 난자 혼합 경과일 값:", existing_values)

# 3. 더 상세한 조건부 채우기를 위한 함수
def fill_mixing_days(row):
    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['난자 혼합 경과일']):
        return row['난자 혼합 경과일']

    # 혼합 과정이 필요없는 경우
    if (row['혼합된 난자 수'] == 0 or
        (row['파트너 정자와 혼합된 난자 수'] == 0 and row['기증자 정자와 혼합된 난자 수'] == 0) or
        row['FER'] == 1):
        return 0

    # 배아 이식이 있는 경우, 그 시점을 기준으로 역산
    if not pd.isna(row['배아 이식 경과일']):
        return max(0, min(7, row['배아 이식 경과일'] - 3))  # 이식 3일 전으로 가정

    # IVF나 ICSI 시술인 경우
    if row['IVF'] == 1 or row['ICSI'] == 1:
        if row['혼합된 난자 수'] > 0:
            return min(7, max(1, row['혼합된 난자 수'] // 5))  # 난자 수에 따라 1~7일 범위 내 값 할당

    # 그 외의 경우 중앙값 사용
    median_mixing_days = train_df[train_df['난자 혼합 경과일'].notna()]['난자 혼합 경과일'].median()
    return median_mixing_days

# 적용
train_df['난자 혼합 경과일'] = train_df.apply(fill_mixing_days, axis=1)

# 결과 확인
print("\n값 분포 확인:")
print(train_df['난자 혼합 경과일'].value_counts(dropna=False))

# 범위 확인 및 조정 (0~7일 범위 보장)
train_df['난자 혼합 경과일'] = train_df['난자 혼합 경과일'].clip(0, 7)

# 1. 난자 혼합 과정이 필요없는 경우 확인
no_mixing_mask = (
    (test_df['혼합된 난자 수'] == 0) |
    (test_df['파트너 정자와 혼합된 난자 수'] == 0) |
    (test_df['기증자 정자와 혼합된 난자 수'] == 0) |
    (test_df['FER'] == 1)  # 동결 배아 이식의 경우 난자 혼합 과정 불필요
)

# 2. 혼합 경과일이 있는 데이터의 unique한 값 확인
existing_values = test_df['난자 혼합 경과일'].dropna().unique()
print("현재 존재하는 난자 혼합 경과일 값:", existing_values)

# 3. 더 상세한 조건부 채우기를 위한 함수
def fill_mixing_days(row):
    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['난자 혼합 경과일']):
        return row['난자 혼합 경과일']

    # 혼합 과정이 필요없는 경우
    if (row['혼합된 난자 수'] == 0 or
        (row['파트너 정자와 혼합된 난자 수'] == 0 and row['기증자 정자와 혼합된 난자 수'] == 0) or
        row['FER'] == 1):
        return 0

    # 배아 이식이 있는 경우, 그 시점을 기준으로 역산
    if not pd.isna(row['배아 이식 경과일']):
        return max(0, min(7, row['배아 이식 경과일'] - 3))  # 이식 3일 전으로 가정

    # IVF나 ICSI 시술인 경우
    if row['IVF'] == 1 or row['ICSI'] == 1:
        if row['혼합된 난자 수'] > 0:
            return min(7, max(1, row['혼합된 난자 수'] // 5))  # 난자 수에 따라 1~7일 범위 내 값 할당

    # 그 외의 경우 중앙값 사용
    median_mixing_days = test_df[test_df['난자 혼합 경과일'].notna()]['난자 혼합 경과일'].median()
    return median_mixing_days

# 적용
test_df['난자 혼합 경과일'] = test_df.apply(fill_mixing_days, axis=1)

# 결과 확인
print("\n값 분포 확인:")
print(test_df['난자 혼합 경과일'].value_counts(dropna=False))

# 범위 확인 및 조정 (0~7일 범위 보장)
test_df['난자 혼합 경과일'] = test_df['난자 혼합 경과일'].clip(0, 7)

"""<h3>배아 이식 경과일 채우기"""

# 1. 배아 이식 과정이 필요없는 경우 확인
no_embryo_transfer_mask = (
    (train_df['이식된 배아 수'] == 0) |  # 이식된 배아가 없는 경우
    (train_df['혼합된 난자 수'] == 0) |   # 혼합된 난자가 없는 경우
    ((train_df['파트너 정자와 혼합된 난자 수'] == 0) & (train_df['기증자 정자와 혼합된 난자 수'] == 0))  # 정자와 혼합되지 않은 경우
)

# 2. 해동 경과일이 있는 데이터의 unique한 값 확인
existing_values = train_df['배아 이식 경과일'].dropna().unique()
print("현재 존재하는 배아 이식 경과일 값:", existing_values)

# 3. 더 상세한 조건부 채우기를 위한 함수
def fill_embryo_transfer_days(row):
    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['배아 이식 경과일']):
        return row['배아 이식 경과일']

    # 이식 과정이 필요없는 경우
    if ((row['이식된 배아 수'] == 0) or
        (row['혼합된 난자 수'] == 0) or
        ((row['파트너 정자와 혼합된 난자 수'] == 0) and (row['기증자 정자와 혼합된 난자 수'] == 0))):
        return 0

    # 난자 혼합 경과일이 있는 경우, 그 값에 기반하여 설정
    if not pd.isna(row['난자 혼합 경과일']):
        return min(7, row['난자 혼합 경과일'] + 2)  # 혼합 후 2일 정도 후에 이식

    # 배아 해동 경과일이 있는 경우
    if not pd.isna(row['배아 해동 경과일']) and row['배아 해동 경과일'] > 0:
        return min(7, row['배아 해동 경과일'] + 1)  # 해동 후 1일 정도 후에 이식

    # 그 외의 경우 중앙값 사용
    median_transfer_days = train_df[train_df['배아 이식 경과일'].notna()]['배아 이식 경과일'].median()
    return median_transfer_days

# 적용
train_df['배아 이식 경과일'] = train_df.apply(fill_embryo_transfer_days, axis=1)

# 4. 값이 0~7 범위를 벗어나는 경우 조정
train_df['배아 이식 경과일'] = train_df['배아 이식 경과일'].clip(0, 7)

# 결과 확인
print("\n값 분포 확인:")
print(train_df['배아 이식 경과일'].value_counts(dropna=False))

# 1. 배아 이식 과정이 필요없는 경우 확인
no_embryo_transfer_mask = (
    (test_df['이식된 배아 수'] == 0) |  # 이식된 배아가 없는 경우
    (test_df['혼합된 난자 수'] == 0) |   # 혼합된 난자가 없는 경우
    ((test_df['파트너 정자와 혼합된 난자 수'] == 0) & (test_df['기증자 정자와 혼합된 난자 수'] == 0))  # 정자와 혼합되지 않은 경우
)

# 2. 해동 경과일이 있는 데이터의 unique한 값 확인
existing_values = test_df['배아 이식 경과일'].dropna().unique()
print("현재 존재하는 배아 이식 경과일 값:", existing_values)

# 3. 더 상세한 조건부 채우기를 위한 함수
def fill_embryo_transfer_days(row):
    # 이미 값이 있는 경우는 그대로 유지
    if not pd.isna(row['배아 이식 경과일']):
        return row['배아 이식 경과일']

    # 이식 과정이 필요없는 경우
    if ((row['이식된 배아 수'] == 0) or
        (row['혼합된 난자 수'] == 0) or
        ((row['파트너 정자와 혼합된 난자 수'] == 0) and (row['기증자 정자와 혼합된 난자 수'] == 0))):
        return 0

    # 난자 혼합 경과일이 있는 경우, 그 값에 기반하여 설정
    if not pd.isna(row['난자 혼합 경과일']):
        return min(7, row['난자 혼합 경과일'] + 2)  # 혼합 후 2일 정도 후에 이식

    # 배아 해동 경과일이 있는 경우
    if not pd.isna(row['배아 해동 경과일']) and row['배아 해동 경과일'] > 0:
        return min(7, row['배아 해동 경과일'] + 1)  # 해동 후 1일 정도 후에 이식

    # 그 외의 경우 중앙값 사용
    median_transfer_days = test_df[test_df['배아 이식 경과일'].notna()]['배아 이식 경과일'].median()
    return median_transfer_days

# 적용
test_df['배아 이식 경과일'] = test_df.apply(fill_embryo_transfer_days, axis=1)

# 4. 값이 0~7 범위를 벗어나는 경우 조정
test_df['배아 이식 경과일'] = test_df['배아 이식 경과일'].clip(0, 7)

# 결과 확인
print("\n값 분포 확인:")
print(test_df['배아 이식 경과일'].value_counts(dropna=False))

"""<h3> PGS, PGD 시술여부 채우기"""

# PGS 시술 여부와 PGD 시술 여부의 NaN을 1로 채우기
train_df['PGS 시술 여부'] = train_df['PGS 시술 여부'].fillna(1)
train_df['PGD 시술 여부'] = train_df['PGD 시술 여부'].fillna(1)

# 결과 확인
print("\nPGS 시술 여부 값 분포:")
print(train_df['PGS 시술 여부'].value_counts(dropna=False))
print("\nPGD 시술 여부 값 분포:")
print(train_df['PGD 시술 여부'].value_counts(dropna=False))

# PGS 시술 여부와 PGD 시술 여부의 NaN을 1로 채우기
test_df['PGS 시술 여부'] = test_df['PGS 시술 여부'].fillna(1)
test_df['PGD 시술 여부'] = test_df['PGD 시술 여부'].fillna(1)

# 결과 확인
print("\nPGS 시술 여부 값 분포:")
print(test_df['PGS 시술 여부'].value_counts(dropna=False))
print("\nPGD 시술 여부 값 분포:")
print(test_df['PGD 시술 여부'].value_counts(dropna=False))

"""<h3>특이 케이스 6291행 채우기"""

# 해당 열들을 리스트로 정의
columns_to_check = [
    '미세주입 후 저장된 배아 수', '이식된 배아 수', '기증 배아 사용 여부',
    '신선 배아 사용 여부', '동결 배아 사용 여부', '총 생성 배아 수',
    '미세주입된 난자 수', '미세주입에서 생성된 배아 수', '미세주입 배아 이식 수',
    '기증자 정자와 혼합된 난자 수', '저장된 배아 수', '해동된 배아 수',
    '해동 난자 수', '수집된 신선 난자 수', '저장된 신선 난자 수',
    '혼합된 난자 수', '대리모 여부', '파트너 정자와 혼합된 난자 수',
    '단일 배아 이식 여부'
]

# 모든 열이 NaN인 행 확인
all_nan_mask = train_df[columns_to_check].isna().all(axis=1)
problematic_rows = train_df[all_nan_mask]

# 해당 행들의 다른 특성 확인
print("문제가 되는 행들의 다른 특성:")
print(problematic_rows[['시술 유형', '임신 성공 여부', 'ICSI', 'IVF', 'IUI', 'BLASTOCYST']].head())
# 1) 시술 유형에 따른 기본값 설정
def fill_based_on_procedure(row):
    if row['IVF'] == 1:
        return {
            '미세주입된 난자 수': 0,
            '이식된 배아 수': 1,  # 기본적으로 1개 이식 가정
            '총 생성 배아 수': 1,
            '단일 배아 이식 여부': 1
        }
    elif row['ICSI'] == 1:
        return {
            '미세주입된 난자 수': 1,
            '이식된 배아 수': 1,
            '총 생성 배아 수': 1,
            '단일 배아 이식 여부': 1
        }
    # 다른 시술 유형에 대한 기본값 추가
    return {}

# 2) 기본값으로 채우기
for idx in problematic_rows.index:
    default_values = fill_based_on_procedure(train_df.loc[idx])
    for col, val in default_values.items():
        train_df.loc[idx, col] = val

    # 나머지 열들은 0으로 채우기
    for col in columns_to_check:
        if pd.isna(train_df.loc[idx, col]):
            train_df.loc[idx, col] = 0

# 3) 검증
print("\n처리 후 NaN 값 개수:")
print(train_df[columns_to_check].isna().sum())

# 해당 열들을 리스트로 정의
columns_to_check = [
    '미세주입 후 저장된 배아 수', '이식된 배아 수', '기증 배아 사용 여부',
    '신선 배아 사용 여부', '동결 배아 사용 여부', '총 생성 배아 수',
    '미세주입된 난자 수', '미세주입에서 생성된 배아 수', '미세주입 배아 이식 수',
    '기증자 정자와 혼합된 난자 수', '저장된 배아 수', '해동된 배아 수',
    '해동 난자 수', '수집된 신선 난자 수', '저장된 신선 난자 수',
    '혼합된 난자 수', '대리모 여부', '파트너 정자와 혼합된 난자 수',
    '단일 배아 이식 여부'
]

# 모든 열이 NaN인 행 확인
all_nan_mask = test_df[columns_to_check].isna().all(axis=1)
problematic_rows = test_df[all_nan_mask]

# 해당 행들의 다른 특성 확인
print("문제가 되는 행들의 다른 특성:")
print(problematic_rows[['시술 유형', 'ICSI', 'IVF', 'IUI', 'BLASTOCYST']].head())
# 1) 시술 유형에 따른 기본값 설정
def fill_based_on_procedure(row):
    if row['IVF'] == 1:
        return {
            '미세주입된 난자 수': 0,
            '이식된 배아 수': 1,  # 기본적으로 1개 이식 가정
            '총 생성 배아 수': 1,
            '단일 배아 이식 여부': 1
        }
    elif row['ICSI'] == 1:
        return {
            '미세주입된 난자 수': 1,
            '이식된 배아 수': 1,
            '총 생성 배아 수': 1,
            '단일 배아 이식 여부': 1
        }
    # 다른 시술 유형에 대한 기본값 추가
    return {}

# 2) 기본값으로 채우기
for idx in problematic_rows.index:
    default_values = fill_based_on_procedure(test_df.loc[idx])
    for col, val in default_values.items():
        test_df.loc[idx, col] = val

    # 나머지 열들은 0으로 채우기
    for col in columns_to_check:
        if pd.isna(test_df.loc[idx, col]):
            test_df.loc[idx, col] = 0

# 3) 검증
print("\n처리 후 NaN 값 개수:")
print(test_df[columns_to_check].isna().sum())

# '알 수 없음'을 NaN으로 대체
train_df = train_df.replace('알 수 없음', np.nan)
test_df = test_df.replace('알 수 없음', np.nan)

# 모든 행을 표시하도록 설정
pd.set_option('display.max_rows', None)

# NaN 개수와 비율 계산
nan_info = pd.DataFrame({
    'NaN 개수': test_df.isna().sum(),
    'NaN 비율(%)': (test_df.isna().sum() / len(test_df) * 100).round(2)
})

# 내림차순으로 정렬하여 출력
print(nan_info.sort_values('NaN 개수', ascending=False))

# 설정을 다시 원래대로 되돌리기 (필요한 경우)
pd.reset_option('display.max_rows')

"""<h3>임신 시도 또는 마지막 임신 경과 연수 채워기"""

# 나이 범주를 숫자로 변환하는 함수
def age_to_numeric(age_range):
    if pd.isna(age_range) or age_range == '알 수 없음':
        return np.nan

    age_map = {
        '만18-34세': 26,  # (18+34)/2
        '만35-37세': 36,  # (35+37)/2
        '만38-39세': 38.5,  # (38+39)/2
        '만40-42세': 41,  # (40+42)/2
        '만43-44세': 43.5,  # (43+44)/2
        '만45-50세': 47.5   # (45+50)/2
    }
    return age_map.get(age_range, np.nan)

# 시술 횟수 문자열을 숫자로 변환하는 함수
def count_to_numeric(count_str):
    if pd.isna(count_str):
        return np.nan
    if count_str == '6회 이상':
        return 6
    return int(count_str.replace('회', ''))

# 나이 열을 숫자로 변환
numeric_age = train_df['시술 당시 나이'].map(age_to_numeric)

# 시술 횟수 관련 열들을 숫자로 변환
numeric_cols = ['총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수',
                'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수',
                'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수']

numeric_features = {}
for col in numeric_cols:
    numeric_features[col] = train_df[col].map(count_to_numeric)

# 불임 원인 관련 열
infertility_cols = ['남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인',
                    '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인',
                    '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애',
                    '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증',
                    '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성',
                    '불임 원인 - 정자 형태']

# 불임 원인이 모두 0인 경우 시도 연수를 0으로 채우기
all_zero_mask = (train_df[infertility_cols] == 0).all(axis=1)
train_df.loc[all_zero_mask & train_df['임신 시도 또는 마지막 임신 경과 연수'].isna(),
             '임신 시도 또는 마지막 임신 경과 연수'] = 0

# 임신 시도 경과 연수와 상관관계가 있는 피처들을 기반으로 결측치 채우기
from sklearn.impute import KNNImputer

# 선택된 피처들로 데이터프레임 생성
imputation_df = pd.DataFrame({
    '시술 당시 나이': numeric_age,
    **numeric_features
})
for col in infertility_cols:
    imputation_df[col] = train_df[col]
imputation_df['임신 시도 또는 마지막 임신 경과 연수'] = train_df['임신 시도 또는 마지막 임신 경과 연수']

# KNN Imputer 적용
imputer = KNNImputer(n_neighbors=5)
imputed_values = imputer.fit_transform(imputation_df)

# 결과를 다시 DataFrame에 적용
train_df.loc[train_df['임신 시도 또는 마지막 임신 경과 연수'].isna(), '임신 시도 또는 마지막 임신 경과 연수'] = \
    imputed_values[:, -1][train_df['임신 시도 또는 마지막 임신 경과 연수'].isna()]

# 값을 0-20 범위로 클리핑하고 반올림
train_df['임신 시도 또는 마지막 임신 경과 연수'] = np.clip(
    train_df['임신 시도 또는 마지막 임신 경과 연수'].round(0), 0, 20
)

# 결과 확인
print("\n채운 후 통계:")
print(train_df['임신 시도 또는 마지막 임신 경과 연수'].describe())

# 나이 범주를 숫자로 변환하는 함수
def age_to_numeric(age_range):
    if pd.isna(age_range) or age_range == '알 수 없음':
        return np.nan

    age_map = {
        '만18-34세': 26,  # (18+34)/2
        '만35-37세': 36,  # (35+37)/2
        '만38-39세': 38.5,  # (38+39)/2
        '만40-42세': 41,  # (40+42)/2
        '만43-44세': 43.5,  # (43+44)/2
        '만45-50세': 47.5   # (45+50)/2
    }
    return age_map.get(age_range, np.nan)

# 시술 횟수 문자열을 숫자로 변환하는 함수
def count_to_numeric(count_str):
    if pd.isna(count_str):
        return np.nan
    if count_str == '6회 이상':
        return 6
    return int(count_str.replace('회', ''))

# 나이 열을 숫자로 변환
numeric_age = test_df['시술 당시 나이'].map(age_to_numeric)

# 시술 횟수 관련 열들을 숫자로 변환
numeric_cols = ['총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수',
                'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수',
                'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수']

numeric_features = {}
for col in numeric_cols:
    numeric_features[col] = test_df[col].map(count_to_numeric)

# 불임 원인 관련 열
infertility_cols = ['남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인',
                    '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인',
                    '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애',
                    '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증',
                    '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성',
                    '불임 원인 - 정자 형태']

# 불임 원인이 모두 0인 경우 시도 연수를 0으로 채우기
all_zero_mask = (test_df[infertility_cols] == 0).all(axis=1)
test_df.loc[all_zero_mask & test_df['임신 시도 또는 마지막 임신 경과 연수'].isna(),
             '임신 시도 또는 마지막 임신 경과 연수'] = 0

# 임신 시도 경과 연수와 상관관계가 있는 피처들을 기반으로 결측치 채우기
from sklearn.impute import KNNImputer

# 선택된 피처들로 데이터프레임 생성
imputation_df = pd.DataFrame({
    '시술 당시 나이': numeric_age,
    **numeric_features
})
for col in infertility_cols:
    imputation_df[col] = test_df[col]
imputation_df['임신 시도 또는 마지막 임신 경과 연수'] = test_df['임신 시도 또는 마지막 임신 경과 연수']

# KNN Imputer 적용
imputer = KNNImputer(n_neighbors=5)
imputed_values = imputer.fit_transform(imputation_df)

# 결과를 다시 DataFrame에 적용
test_df.loc[test_df['임신 시도 또는 마지막 임신 경과 연수'].isna(), '임신 시도 또는 마지막 임신 경과 연수'] = \
    imputed_values[:, -1][test_df['임신 시도 또는 마지막 임신 경과 연수'].isna()]

# 값을 0-20 범위로 클리핑하고 반올림
test_df['임신 시도 또는 마지막 임신 경과 연수'] = np.clip(
    test_df['임신 시도 또는 마지막 임신 경과 연수'].round(0), 0, 20
)

# 결과 확인
print("\n채운 후 통계:")
print(train_df['임신 시도 또는 마지막 임신 경과 연수'].describe())