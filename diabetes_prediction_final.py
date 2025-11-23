"""
청소년기 비만 및 생활습관이 성인기 당뇨병 위험에 미치는 영향 분석
최종 버전: 클래스 가중치 조정 및 개선된 모델
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("청소년기 비만 및 생활습관이 성인기 당뇨병 위험에 미치는 영향 분석")
print("(클래스 가중치 조정 및 SMOTE 적용 버전)")
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

# 성인 데이터 전처리
df_adult['DIABETES_TARGET'] = (df_adult['DIABETES'] == 1.0).astype(int)

# BMI 카테고리 생성 (한국 기준)
def categorize_bmi(bmi):
    if pd.isna(bmi):
        return np.nan
    elif bmi < 18.5:
        return 1  # 저체중
    elif bmi < 23:
        return 2  # 정상
    elif bmi < 25:
        return 3  # 과체중
    else:
        return 4  # 비만

df_adult['BMI_CATEGORY'] = df_adult['BMI'].apply(categorize_bmi)

print(f"성인 데이터 당뇨병 유병률: {df_adult['DIABETES_TARGET'].mean()*100:.2f}%")

# 모델링을 위한 특성 선택 (혈당 지표 제외)
print("\n[3] 특성 선택 중... (혈당 지표 제외)")

# 성인 데이터에서 사용할 특성
adult_features = [
    'AGE', 'SEX', 'BMI', 'BMI_CATEGORY', 'OBESITY',
    'INCOME', 'REGION', 'BREAKFAST'
]

# 성인 데이터 준비
adult_model_data = df_adult[adult_features + ['DIABETES_TARGET']].copy()
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

# BMI_CATEGORY를 원-핫 인코딩
bmi_cat_dummies = pd.get_dummies(adult_model_data_encoded['BMI_CATEGORY'], prefix='BMI_CAT')
adult_model_data_encoded = pd.concat([adult_model_data_encoded, bmi_cat_dummies], axis=1)
adult_model_data_encoded = adult_model_data_encoded.drop('BMI_CATEGORY', axis=1)

# 특성과 타겟 분리
X_adult = adult_model_data_encoded.drop(['DIABETES_TARGET'], axis=1)
y_diabetes = adult_model_data_encoded['DIABETES_TARGET']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_adult, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n학습 데이터: {len(X_train):,}개 (당뇨병: {y_train.sum():,}개, {y_train.mean()*100:.2f}%)")
print(f"테스트 데이터: {len(X_test):,}개 (당뇨병: {y_test.sum():,}개, {y_test.mean()*100:.2f}%)")

# 클래스 가중치 계산
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\n클래스 가중치: 정상={class_weight_dict[0]:.3f}, 당뇨병={class_weight_dict[1]:.3f}")

# SMOTE 적용
print("\n[4] SMOTE 오버샘플링 적용 중...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_smote_scaled = scaler.transform(X_test)

print(f"SMOTE 후 학습 데이터: {len(X_train_smote):,}개 (당뇨병: {y_train_smote.sum():,}개, {y_train_smote.mean()*100:.2f}%)")

# 모델 정의
models = {
    'Logistic Regression (Balanced)': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    ),
    'Logistic Regression (SMOTE)': LogisticRegression(
        random_state=42, max_iter=1000
    ),
    'Random Forest (Balanced)': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, 
        max_depth=10, class_weight='balanced'
    ),
    'Random Forest (SMOTE)': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
    ),
}

# 결과 저장
results = []

print("\n" + "=" * 80)
print("당뇨병 예측 모델 결과 (개선 버전)")
print("=" * 80)

# 각 모델 학습 및 평가
for model_name, model in models.items():
    print(f"\n[{model_name}] 학습 중...")
    
    # 모델 학습
    if 'SMOTE' in model_name:
        if 'Logistic' in model_name:
            model.fit(X_train_smote_scaled, y_train_smote)
            y_pred = model.predict(X_test_smote_scaled)
            y_pred_proba = model.predict_proba(X_test_smote_scaled)[:, 1]
        else:
            model.fit(X_train_smote, y_train_smote)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        if 'Logistic' in model_name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 평가 지표 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 교차 검증
    if 'SMOTE' in model_name:
        cv_X = X_train_smote if 'Logistic' not in model_name else X_train_smote_scaled
        cv_y = y_train_smote
    else:
        cv_X = X_train if 'Logistic' not in model_name else X_train_scaled
        cv_y = y_train
    
    cv_scores = cross_val_score(model, cv_X, cv_y, cv=5, scoring='roc_auc')
    
    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'CV_AUC_Mean': cv_scores.mean(),
        'CV_AUC_Std': cv_scores.std()
    })
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  CV AUC (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  혼동 행렬:")
    print(f"    실제 음성 / 예측 음성: {cm[0,0]:,}")
    print(f"    실제 음성 / 예측 양성: {cm[0,1]:,}")
    print(f"    실제 양성 / 예측 음성: {cm[1,0]:,}")
    print(f"    실제 양성 / 예측 양성: {cm[1,1]:,}")

# 결과 요약
results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("모델 성능 요약")
print("=" * 80)
print(results_df.to_string(index=False))

# 최고 성능 모델 선택
best_model_info = results_df.nlargest(1, 'F1-Score')
print("\n" + "=" * 80)
print("최고 성능 모델 (F1-Score 기준)")
print("=" * 80)
print(best_model_info[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].to_string(index=False))

# 특성 중요도 (최고 성능 Random Forest 모델)
print("\n" + "=" * 80)
print("특성 중요도 분석")
print("=" * 80)

# SMOTE 버전 Random Forest로 특성 중요도 계산
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train_smote, y_train_smote)

feature_importance = pd.DataFrame({
    'Feature': X_adult.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n특성 중요도 (Top 15):")
print(feature_importance.head(15).to_string(index=False))

# 시각화
print("\n[5] 시각화 생성 중...")

# 1. 모델 성능 비교
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Accuracy 비교
axes[0, 0].bar(range(len(results_df)), results_df['Accuracy'], color=['skyblue', 'lightblue', 'lightgreen', 'green'])
axes[0, 0].set_xticks(range(len(results_df)))
axes[0, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[0, 0].set_title('모델별 정확도 (Accuracy)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(axis='y', alpha=0.3)

# AUC-ROC 비교
axes[0, 1].bar(range(len(results_df)), results_df['AUC-ROC'], color=['skyblue', 'lightblue', 'lightgreen', 'green'])
axes[0, 1].set_xticks(range(len(results_df)))
axes[0, 1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[0, 1].set_title('모델별 AUC-ROC', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('AUC-ROC')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(axis='y', alpha=0.3)

# F1-Score 비교
axes[0, 2].bar(range(len(results_df)), results_df['F1-Score'], color=['skyblue', 'lightblue', 'lightgreen', 'green'])
axes[0, 2].set_xticks(range(len(results_df)))
axes[0, 2].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[0, 2].set_title('모델별 F1-Score', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('F1-Score')
axes[0, 2].set_ylim([0, max(results_df['F1-Score']) * 1.2])
axes[0, 2].grid(axis='y', alpha=0.3)

# Precision 비교
axes[1, 0].bar(range(len(results_df)), results_df['Precision'], color=['skyblue', 'lightblue', 'lightgreen', 'green'])
axes[1, 0].set_xticks(range(len(results_df)))
axes[1, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[1, 0].set_title('모델별 Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(axis='y', alpha=0.3)

# Recall 비교
axes[1, 1].bar(range(len(results_df)), results_df['Recall'], color=['skyblue', 'lightblue', 'lightgreen', 'green'])
axes[1, 1].set_xticks(range(len(results_df)))
axes[1, 1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[1, 1].set_title('모델별 Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(axis='y', alpha=0.3)

# 특성 중요도 Top 10
top_features = feature_importance.head(10)
axes[1, 2].barh(top_features['Feature'], top_features['Importance'], color='coral')
axes[1, 2].set_title('특성 중요도 Top 10', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Importance')
axes[1, 2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_final.png', dpi=300, bbox_inches='tight')
print("✓ model_performance_final.png 저장 완료")

# 2. ROC Curve
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model in models.items():
    if 'SMOTE' in model_name:
        if 'Logistic' in model_name:
            y_pred_proba = model.predict_proba(X_test_smote_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        if 'Logistic' in model_name:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve 비교', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curve_final.png', dpi=300, bbox_inches='tight')
print("✓ roc_curve_final.png 저장 완료")

# 결과 저장
results_df.to_csv('model_results_final.csv', index=False, encoding='utf-8-sig')
feature_importance.to_csv('feature_importance_final.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("결과 파일 저장 완료")
print("=" * 80)
print("- model_results_final.csv: 모델 성능 결과")
print("- feature_importance_final.csv: 특성 중요도")
print("- model_performance_final.png: 모델 성능 비교 그래프")
print("- roc_curve_final.png: ROC 곡선 그래프")

print("\n분석 완료!")

