"""
청소년기 비만 및 생활습관이 성인기 당뇨병 위험에 미치는 영향 분석
당뇨병 위험도 예측 모델
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("XGBoost를 사용할 수 없습니다. 다른 모델을 사용합니다.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM을 사용할 수 없습니다. 다른 모델을 사용합니다.")

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    print("SHAP을 사용할 수 없습니다. 특성 중요도만 표시합니다.")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("청소년기 비만 및 생활습관이 성인기 당뇨병 위험에 미치는 영향 분석")
print("=" * 80)

# 데이터 로드
print("\n[1] 데이터 로드 중...")
df_teen = pd.read_csv('9ch_final_data.csv')
df_adult = pd.read_csv('hn_cleand_data (2).csv')

print(f"청소년 데이터: {len(df_teen):,}개 행")
print(f"성인 데이터: {len(df_adult):,}개 행")

# 성인 데이터 컬럼명 매핑
df_adult = df_adult.rename(columns={
    'year': 'YEAR',
    'age': 'AGE',
    'sex': 'SEX',
    'region': 'REGION',
    'ho_incm5': 'INCOME',
    'HE_ht': 'HT',
    'HE_wt': 'WT',
    'HE_BMI': 'BMI',
    'HE_obe': 'OBESITY',
    'HE_glu': 'GLUCOSE',
    'HE_HbA1c': 'HbA1c',
    'DE1_dg': 'DIABETES',
    'L_BR_FQ': 'BREAKFAST'
})

# 데이터 전처리
print("\n[2] 데이터 전처리 중...")

# 청소년 데이터 전처리
df_teen['BMI'] = df_teen['WT'] / ((df_teen['HT'] / 100) ** 2)
df_teen['AGE_GROUP'] = pd.cut(df_teen['AGE'], bins=[0, 13, 15, 18], labels=['초등', '중등', '고등'])

# 성인 데이터 전처리
df_adult['AGE_GROUP'] = pd.cut(
    df_adult['AGE'],
    bins=[0, 30, 40, 50, 60, 70, 100],
    labels=['20대', '30대', '40대', '50대', '60대', '70대 이상']
)

# 성인 데이터에서 타겟 변수 생성 (당뇨병 여부: 1.0 = 당뇨병, 0.0 = 정상)
df_adult['DIABETES_TARGET'] = (df_adult['DIABETES'] == 1.0).astype(int)

# 성인 데이터에서 고위험군 정의 (혈당 >= 126 또는 당화혈색소 >= 6.5 또는 당뇨병 = 1)
df_adult['HIGH_RISK'] = (
    (df_adult['GLUCOSE'] >= 126) | 
    (df_adult['HbA1c'] >= 6.5) | 
    (df_adult['DIABETES'] == 1.0)
).astype(int)

print(f"성인 데이터 당뇨병 유병률: {df_adult['DIABETES_TARGET'].mean()*100:.2f}%")
print(f"성인 데이터 고위험군 비율: {df_adult['HIGH_RISK'].mean()*100:.2f}%")

# 모델링을 위한 특성 선택
print("\n[3] 특성 선택 중...")

# 성인 데이터에서 사용할 특성
adult_features = [
    'AGE', 'SEX', 'BMI', 'OBESITY', 'GLUCOSE', 'HbA1c',
    'INCOME', 'REGION', 'BREAKFAST'
]

# 청소년 데이터에서 사용할 특성 (성인 데이터와 매핑)
teen_features = [
    'AGE', 'SEX', 'BMI', 'E_SES', 'CTYPE',
    'F_BR', 'F_FRUIT', 'F_VEG', 'F_FASTFOOD', 'SODA_INTAKE', 'Breakfast_Category'
]

# 성인 데이터로 모델 구축 (현재 특성 기반)
print("\n[4] 성인 데이터 기반 모델 구축...")

# 성인 데이터 준비
adult_model_data = df_adult[adult_features + ['DIABETES_TARGET', 'HIGH_RISK']].copy()
adult_model_data = adult_model_data.dropna()

print(f"모델링용 성인 데이터: {len(adult_model_data):,}개 행")

# 범주형 변수 인코딩
adult_model_data_encoded = adult_model_data.copy()

# OBESITY를 원-핫 인코딩
obesity_dummies = pd.get_dummies(adult_model_data_encoded['OBESITY'], prefix='OBESITY')
adult_model_data_encoded = pd.concat([adult_model_data_encoded, obesity_dummies], axis=1)
adult_model_data_encoded = adult_model_data_encoded.drop('OBESITY', axis=1)

# SEX를 원-핫 인코딩
sex_dummies = pd.get_dummies(adult_model_data_encoded['SEX'], prefix='SEX')
adult_model_data_encoded = pd.concat([adult_model_data_encoded, sex_dummies], axis=1)
adult_model_data_encoded = adult_model_data_encoded.drop('SEX', axis=1)

# 특성과 타겟 분리
X_adult = adult_model_data_encoded.drop(['DIABETES_TARGET', 'HIGH_RISK'], axis=1)
y_diabetes = adult_model_data_encoded['DIABETES_TARGET']
y_high_risk = adult_model_data_encoded['HIGH_RISK']

# 학습/테스트 분할
X_train, X_test, y_train_d, y_test_d = train_test_split(
    X_adult, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
)
X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(
    X_adult, y_high_risk, test_size=0.2, random_state=42, stratify=y_high_risk
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_hr_scaled = scaler.fit_transform(X_train_hr)
X_test_hr_scaled = scaler.transform(X_test_hr)

print(f"\n학습 데이터: {len(X_train):,}개")
print(f"테스트 데이터: {len(X_test):,}개")

# 모델 정의
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)

# 결과 저장
results = []

print("\n" + "=" * 80)
print("당뇨병 예측 모델 결과")
print("=" * 80)

# 각 모델 학습 및 평가
for model_name, model in models.items():
    print(f"\n[{model_name}] 학습 중...")
    
    # 당뇨병 예측
    if model_name in ['Logistic Regression']:
        model.fit(X_train_scaled, y_train_d)
        y_pred_d = model.predict(X_test_scaled)
        y_pred_proba_d = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train_d)
        y_pred_d = model.predict(X_test)
        y_pred_proba_d = model.predict_proba(X_test)[:, 1]
    
    # 평가 지표 계산
    acc_d = accuracy_score(y_test_d, y_pred_d)
    prec_d = precision_score(y_test_d, y_pred_d, zero_division=0)
    rec_d = recall_score(y_test_d, y_pred_d, zero_division=0)
    f1_d = f1_score(y_test_d, y_pred_d, zero_division=0)
    auc_d = roc_auc_score(y_test_d, y_pred_proba_d)
    
    results.append({
        'Model': model_name,
        'Target': '당뇨병',
        'Accuracy': acc_d,
        'Precision': prec_d,
        'Recall': rec_d,
        'F1-Score': f1_d,
        'AUC-ROC': auc_d
    })
    
    print(f"  Accuracy: {acc_d:.4f}")
    print(f"  Precision: {prec_d:.4f}")
    print(f"  Recall: {rec_d:.4f}")
    print(f"  F1-Score: {f1_d:.4f}")
    print(f"  AUC-ROC: {auc_d:.4f}")
    
    # 고위험군 예측
    if model_name in ['Logistic Regression']:
        model_hr = type(model)(**model.get_params())
        model_hr.fit(X_train_hr_scaled, y_train_hr)
        y_pred_hr = model_hr.predict(X_test_hr_scaled)
        y_pred_proba_hr = model_hr.predict_proba(X_test_hr_scaled)[:, 1]
    else:
        model_hr = type(model)(**model.get_params())
        model_hr.fit(X_train_hr, y_train_hr)
        y_pred_hr = model_hr.predict(X_test_hr)
        y_pred_proba_hr = model_hr.predict_proba(X_test_hr)[:, 1]
    
    acc_hr = accuracy_score(y_test_hr, y_pred_hr)
    prec_hr = precision_score(y_test_hr, y_pred_hr, zero_division=0)
    rec_hr = recall_score(y_test_hr, y_pred_hr, zero_division=0)
    f1_hr = f1_score(y_test_hr, y_pred_hr, zero_division=0)
    auc_hr = roc_auc_score(y_test_hr, y_pred_proba_hr)
    
    results.append({
        'Model': model_name,
        'Target': '고위험군',
        'Accuracy': acc_hr,
        'Precision': prec_hr,
        'Recall': rec_hr,
        'F1-Score': f1_hr,
        'AUC-ROC': auc_hr
    })
    
    print(f"  [고위험군] Accuracy: {acc_hr:.4f}, F1: {f1_hr:.4f}, AUC: {auc_hr:.4f}")

# 결과 요약
results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("모델 성능 요약")
print("=" * 80)
print(results_df.to_string(index=False))

# 최고 성능 모델 선택
best_diabetes_model = results_df[results_df['Target'] == '당뇨병'].nlargest(1, 'AUC-ROC')
best_hr_model = results_df[results_df['Target'] == '고위험군'].nlargest(1, 'AUC-ROC')

print("\n" + "=" * 80)
print("최고 성능 모델")
print("=" * 80)
print("\n[당뇨병 예측]")
print(best_diabetes_model[['Model', 'Accuracy', 'F1-Score', 'AUC-ROC']].to_string(index=False))
print("\n[고위험군 예측]")
print(best_hr_model[['Model', 'Accuracy', 'F1-Score', 'AUC-ROC']].to_string(index=False))

# SHAP 분석 (최고 성능 모델)
if SHAP_AVAILABLE:
    print("\n" + "=" * 80)
    print("SHAP 분석 (최고 성능 모델)")
    print("=" * 80)
    
    # Random Forest 모델로 SHAP 분석
    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train_d)
    
    # SHAP 값 계산 (샘플링하여 속도 향상)
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_sample)
        print(f"\nSHAP 값 계산 완료 (샘플 수: {sample_size})")
    except Exception as e:
        print(f"SHAP 계산 중 오류 발생: {e}")
        SHAP_AVAILABLE = False
else:
    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train_d)

# 특성 중요도
feature_importance = pd.DataFrame({
    'Feature': X_adult.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n특성 중요도 (Top 10):")
print(feature_importance.head(10).to_string(index=False))

# 결과 저장
results_df.to_csv('model_results.csv', index=False, encoding='utf-8-sig')
feature_importance.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("결과 파일 저장 완료")
print("=" * 80)
print("- model_results.csv: 모델 성능 결과")
print("- feature_importance.csv: 특성 중요도")

print("\n분석 완료!")

