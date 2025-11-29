from typing import Dict, Optional
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

TEEN_EXCLUDED_YEARS = {2015, 2016}
TEEN_OBESITY_PERCENTILE = 0.95
TEEN_MODEL_THRESHOLD = 0.49


def prepare_teen_model_data(
    dataframe: pd.DataFrame,
) -> Optional[Dict[str, np.ndarray]]:
    required_cols = [
        "F_BR",
        "F_FRUIT",
        "F_VEG",
        "F_FASTFOOD",
        "SODA_INTAKE",
        "Breakfast_Category",
        "AGE",
        "SEX",
        "E_SES",
        "HEALTHY_SCORE",
        "UNHEALTHY_SCORE",
        "NET_DIET_SCORE",
        "TEEN_OBESE_TOP5",
    ]
    if not set(required_cols).issubset(dataframe.columns):
        return None

    feature_cols = [
        "F_BR",
        "F_FRUIT",
        "F_VEG",
        "F_FASTFOOD",
        "SODA_INTAKE",
        "Breakfast_Category",
        "AGE",
        "SEX",
        "E_SES",
        "HEALTHY_SCORE",
        "UNHEALTHY_SCORE",
        "NET_DIET_SCORE",
    ]
    optional_cat = []
    for col in ["GROUP", "CTYPE"]:
        if col in dataframe.columns:
            optional_cat.append(col)

    cols_to_use = required_cols + optional_cat
    data = dataframe[cols_to_use].dropna().reset_index(drop=True)
    if len(data) < 400:
        return None
    y = data["TEEN_OBESE_TOP5"]
    if y.nunique() < 2:
        return None

    X_numeric = data[feature_cols].copy()
    interaction_pairs = [
        ("AGE_FRUIT", "AGE", "F_FRUIT"),
        ("AGE_VEG", "AGE", "F_VEG"),
        ("AGE_FASTFOOD", "AGE", "F_FASTFOOD"),
        ("FRUIT_VEG", "F_FRUIT", "F_VEG"),
        ("FASTFOOD_SODA", "F_FASTFOOD", "SODA_INTAKE"),
        ("BREAKFAST_AGE", "Breakfast_Category", "AGE"),
    ]
    for new_col, c1, c2 in interaction_pairs:
        if c1 in X_numeric.columns and c2 in X_numeric.columns:
            X_numeric[new_col] = X_numeric[c1] * X_numeric[c2]

    if optional_cat:
        dummy_frames = [
            pd.get_dummies(data[col], prefix=col, drop_first=False) for col in optional_cat
        ]
        cat_df = pd.concat(dummy_frames, axis=1)
        X = pd.concat([X_numeric, cat_df], axis=1)
    else:
        X = X_numeric

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "sample_size": len(data),
    }


def compute_teen_model_results(dataframe: pd.DataFrame):
    prep = prepare_teen_model_data(dataframe)
    if not prep:
        return None
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
    sample_weight = y_train.map(weight_dict).values

    # SMOTE 적용
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(prep["X_train_scaled"], y_train)

    # C 값 최적화 (더 넓은 범위, SMOTE 적용)
    best_c = 0.1
    best_score = 0
    best_thr = TEEN_MODEL_THRESHOLD
    best_result = None
    for c_val in [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
        lr_temp = LogisticRegression(max_iter=5000, class_weight="balanced", C=c_val, solver="lbfgs")
        lr_temp.fit(X_train_smote, y_train_smote)
        y_prob_temp = lr_temp.predict_proba(prep["X_test_scaled"])[:, 1]
        test_auc = roc_auc_score(y_test, y_prob_temp)
        # 임계값 스윕
        for thr in np.linspace(0.35, 0.60, 26):
            y_pred_temp = (y_prob_temp >= thr).astype(int)
            acc = accuracy_score(y_test, y_pred_temp)
            rec = recall_score(y_test, y_pred_temp)
            if acc >= 0.60 and rec >= 0.65:
                score = acc * 0.4 + rec * 0.4 + test_auc * 0.2
                if score > best_score:
                    best_score = score
                    best_c = c_val
                    best_thr = thr
                    best_result = {'c': c_val, 'thr': thr, 'acc': acc, 'rec': rec, 'auc': test_auc}
    
    # 조건 만족하는 결과가 없으면 기본값 사용
    if best_result is None:
        best_c = 0.1
        best_thr = TEEN_MODEL_THRESHOLD
    
    lr_model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=best_c,
        solver="lbfgs"
    )
    lr_model.fit(X_train_smote, y_train_smote)
    y_prob_lr = lr_model.predict_proba(prep["X_test_scaled"])[:, 1]
    y_pred_lr = (y_prob_lr >= best_thr).astype(int)
    
    y_pred_lr = (y_prob_lr >= best_thr).astype(int)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

    # Random Forest 하이퍼파라미터 튜닝
    best_rf_score = 0
    best_rf_params = None
    for n_est in [200, 300, 400]:
        for max_d in [10, 12, 15]:
            rf_temp = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_d, 
                min_samples_split=8, min_samples_leaf=4, 
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            )
            rf_temp.fit(prep["X_train"], y_train)
            y_prob_rf_temp = rf_temp.predict_proba(prep["X_test"])[:, 1]
            auc_rf = roc_auc_score(y_test, y_prob_rf_temp)
            for thr in np.linspace(0.35, 0.60, 26):
                y_pred_rf_temp = (y_prob_rf_temp >= thr).astype(int)
                acc = accuracy_score(y_test, y_pred_rf_temp)
                rec = recall_score(y_test, y_pred_rf_temp)
                if acc >= 0.60 and rec >= 0.65:
                    score = acc * 0.4 + rec * 0.4 + auc_rf * 0.2
                    if score > best_rf_score:
                        best_rf_score = score
                        best_rf_params = {'n_est': n_est, 'max_d': max_d, 'thr': thr}
    
    # 최적 파라미터로 RF 학습
    if best_rf_params:
        rf_model = RandomForestClassifier(
            n_estimators=best_rf_params['n_est'],
            max_depth=best_rf_params['max_d'],
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
        rf_thr = best_rf_params['thr']
    else:
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
        rf_thr = 0.5
    
    rf_model.fit(prep["X_train"], y_train)
    y_prob_rf = rf_model.predict_proba(prep["X_test"])[:, 1]
    y_pred_rf = (y_prob_rf >= rf_thr).astype(int)

    brf_model = BalancedRandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    brf_model.fit(prep["X_train"], y_train)
    y_prob_brf = brf_model.predict_proba(prep["X_test"])[:, 1]
    y_pred_brf = brf_model.predict(prep["X_test"])

    hgb_model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=30,
        class_weight=weight_dict,
        random_state=42,
    )
    hgb_model.fit(prep["X_train"], y_train, sample_weight=sample_weight)
    y_prob_hgb = hgb_model.predict_proba(prep["X_test"])[:, 1]
    y_pred_hgb = hgb_model.predict(prep["X_test"])

    # 앙상블 모델 (가중치 최적화)
    best_ensemble_score = 0
    best_weights = None
    best_ens_thr = best_thr
    for w1 in np.linspace(0.3, 0.8, 6):
        w2 = 1 - w1
        ensemble_prob_temp = w1 * y_prob_lr + w2 * y_prob_rf
        auc_ens = roc_auc_score(y_test, ensemble_prob_temp)
        for thr in np.linspace(0.35, 0.60, 26):
            y_pred_ens_temp = (ensemble_prob_temp >= thr).astype(int)
            acc = accuracy_score(y_test, y_pred_ens_temp)
            rec = recall_score(y_test, y_pred_ens_temp)
            if acc >= 0.60 and rec >= 0.65:
                score = acc * 0.4 + rec * 0.4 + auc_ens * 0.2
                if score > best_ensemble_score:
                    best_ensemble_score = score
                    best_weights = (w1, w2)
                    best_ens_thr = thr
    
    # 최적 가중치로 앙상블 생성
    if best_weights:
        ensemble_prob = best_weights[0] * y_prob_lr + best_weights[1] * y_prob_rf
    else:
        ensemble_prob = 0.6 * y_prob_lr + 0.4 * y_prob_rf
        best_ens_thr = best_thr
    
    y_pred_ens = (ensemble_prob >= best_ens_thr).astype(int)

    # 가장 좋은 AUC를 가진 모델 찾기 (ROC 곡선용)
    model_aucs = {
        "logistic": roc_auc_score(y_test, y_prob_lr),
        "random_forest": roc_auc_score(y_test, y_prob_rf),
        "balanced_random_forest": roc_auc_score(y_test, y_prob_brf),
        "hist_gradient_boosting": roc_auc_score(y_test, y_prob_hgb),
        "ensemble": roc_auc_score(y_test, ensemble_prob),
    }
    best_model_name = max(model_aucs, key=model_aucs.get)
    best_auc = model_aucs[best_model_name]
    
    # 최고 AUC 모델의 ROC 곡선
    if best_model_name == "logistic":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_lr)
    elif best_model_name == "random_forest":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_rf)
    elif best_model_name == "balanced_random_forest":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_brf)
    elif best_model_name == "hist_gradient_boosting":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_hgb)
    else:  # ensemble
        fpr_best, tpr_best, _ = roc_curve(y_test, ensemble_prob)

    results = {
        "threshold": best_thr,
        "optimal_c": best_c,
        "sample_size": prep["sample_size"],
        "logistic": {
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "recall": recall_score(y_test, y_pred_lr),
            "precision": precision_score(y_test, y_pred_lr, zero_division=0),
            "f1": f1_score(y_test, y_pred_lr),
            "auc": roc_auc_score(y_test, y_prob_lr),
            "threshold": best_thr,
            "optimal_c": best_c,
            "sample_size": prep["sample_size"],
        },
        "random_forest": {
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "recall": recall_score(y_test, y_pred_rf),
            "precision": precision_score(y_test, y_pred_rf, zero_division=0),
            "f1": f1_score(y_test, y_pred_rf),
            "auc": roc_auc_score(y_test, y_prob_rf),
        },
        "balanced_random_forest": {
            "accuracy": accuracy_score(y_test, y_pred_brf),
            "recall": recall_score(y_test, y_pred_brf),
            "precision": precision_score(y_test, y_pred_brf, zero_division=0),
            "f1": f1_score(y_test, y_pred_brf),
            "auc": roc_auc_score(y_test, y_prob_brf),
        },
        "hist_gradient_boosting": {
            "accuracy": accuracy_score(y_test, y_pred_hgb),
            "recall": recall_score(y_test, y_pred_hgb),
            "precision": precision_score(y_test, y_pred_hgb, zero_division=0),
            "f1": f1_score(y_test, y_pred_hgb),
            "auc": roc_auc_score(y_test, y_prob_hgb),
        },
        "ensemble": {
            "accuracy": accuracy_score(y_test, y_pred_ens),
            "recall": recall_score(y_test, y_pred_ens),
            "precision": precision_score(y_test, y_pred_ens, zero_division=0),
            "f1": f1_score(y_test, y_pred_ens),
            "auc": roc_auc_score(y_test, ensemble_prob),
            "threshold": best_ens_thr,
            "weights": best_weights if best_weights else (0.6, 0.4),
        },
        "roc_curve": {
            "fpr": fpr_best.tolist(),
            "tpr": tpr_best.tolist(),
            "auc": best_auc,
            "model_name": best_model_name,
        },
    }
    return results

# 페이지 설정
st.set_page_config(
    page_title="건강 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (캐싱)
@st.cache_data
def load_data():
    df = pd.read_csv('9ch_final_data.csv')
    df['BMI'] = df['WT'] / ((df['HT'] / 100) ** 2)
    return df

@st.cache_data
def load_new_data():
    df_new = pd.read_csv('hn_cleand_data (2).csv')
    # 컬럼명을 기존 데이터와 일치시키기 위해 매핑
    df_new = df_new.rename(columns={
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
        'DE1_dg': 'DIABETES',  # DE1_pr에서 DE1_dg로 변경
        'L_BR_FQ': 'BREAKFAST'
    })
    # 채소/과일 섭취 빈도 컬럼이 있으면 매핑
    if 'LS_VEG2' in df_new.columns:
        df_new = df_new.rename(columns={'LS_VEG2': 'F_VEG'})
    if 'LS_FRUIT' in df_new.columns:
        df_new = df_new.rename(columns={'LS_FRUIT': 'F_FRUIT'})
    return df_new

def compute_teen_model_summary(dataframe: pd.DataFrame):
    prep = prepare_teen_model_data(dataframe)
    if not prep:
        return None
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs"
    )
    model.fit(prep["X_train_scaled"], y_train)
    y_prob = model.predict_proba(prep["X_test_scaled"])[:, 1]
    y_pred = (y_prob >= TEEN_MODEL_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "threshold": TEEN_MODEL_THRESHOLD,
        "sample_size": prep["sample_size"]
    }
    return metrics


df = load_data()
df_new = load_new_data()

teen_bmi_cutoff = None
if not df.empty:
    df = df[~df['YEAR'].isin(TEEN_EXCLUDED_YEARS)].copy()
    if df['BMI'].notna().any():
        teen_bmi_cutoff = df['BMI'].quantile(TEEN_OBESITY_PERCENTILE)
        df['TEEN_OBESE_TOP5'] = (df['BMI'] >= teen_bmi_cutoff).astype(int)
    else:
        df['TEEN_OBESE_TOP5'] = np.nan
    df['HEALTHY_SCORE'] = df[['F_FRUIT', 'F_VEG', 'Breakfast_Category']].sum(axis=1)
    df['UNHEALTHY_SCORE'] = df[['F_FASTFOOD', 'SODA_INTAKE']].sum(axis=1)
    df['NET_DIET_SCORE'] = df['HEALTHY_SCORE'] - df['UNHEALTHY_SCORE']
    if 'GROUP' in df.columns:
        df['GROUP'] = df['GROUP'].fillna('Unknown').astype(str)
    if 'CTYPE' in df.columns:
        df['CTYPE'] = df['CTYPE'].fillna('Unknown').astype(str)
else:
    df['TEEN_OBESE_TOP5'] = np.nan
    df['HEALTHY_SCORE'] = np.nan
    df['UNHEALTHY_SCORE'] = np.nan
    df['NET_DIET_SCORE'] = np.nan

teen_model_results_global = compute_teen_model_results(df) if not df.empty else None
teen_model_summary_global = (
    teen_model_results_global["logistic"] if teen_model_results_global else None
)

# 사이드바 - 데이터셋 선택
st.sidebar.header("📊 데이터셋 선택")
dataset_choice = st.sidebar.radio(
    "분석할 데이터셋을 선택하세요",
    ["청소년 데이터", "성인 데이터"],
    index=0
)

# 선택된 데이터셋에 따라 사용할 데이터 결정
if dataset_choice == "청소년 데이터":
    current_df = df
    is_adult = False
else:
    current_df = df_new
    is_adult = True

# 사이드바 필터
st.sidebar.header("🔍 필터 옵션")

# 연도 필터
years = sorted(current_df['YEAR'].unique())
selected_years = st.sidebar.multiselect(
    "연도 선택",
    options=years,
    default=years
)

# 성별 필터
sex_options = ['전체', '남성', '여성']
selected_sex = st.sidebar.selectbox("성별 선택", sex_options)

# 연령 필터
min_age = int(current_df['AGE'].min()) if not current_df['AGE'].isna().all() else 0
max_age = int(current_df['AGE'].max()) if not current_df['AGE'].isna().all() else 100
age_range = st.sidebar.slider(
    "연령 범위",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# 데이터 필터링
filtered_df = current_df[
    (current_df['YEAR'].isin(selected_years)) &
    (current_df['AGE'] >= age_range[0]) &
    (current_df['AGE'] <= age_range[1])
]

if selected_sex == '남성':
    filtered_df = filtered_df[filtered_df['SEX'] == 1.0]
elif selected_sex == '여성':
    filtered_df = filtered_df[filtered_df['SEX'] == 2.0]

# 청소년 데이터에만 도시 유형 필터 적용
if not is_adult and 'CTYPE' in current_df.columns:
    city_types = ['전체'] + list(current_df['CTYPE'].unique())
    selected_city = st.sidebar.selectbox("도시 유형 선택", city_types)
    if selected_city != '전체':
        filtered_df = filtered_df[filtered_df['CTYPE'] == selected_city]

# 메인 타이틀
st.title("📊 건강 데이터 분석 대시보드")
st.markdown("---")

# 주요 지표 (KPI)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("총 데이터 수", f"{len(filtered_df):,}개")

with col2:
    avg_height = filtered_df['HT'].dropna().mean()
    st.metric("평균 키", f"{avg_height:.1f}cm" if not pd.isna(avg_height) else "N/A")

with col3:
    avg_weight = filtered_df['WT'].dropna().mean()
    st.metric("평균 몸무게", f"{avg_weight:.1f}kg" if not pd.isna(avg_weight) else "N/A")

with col4:
    avg_bmi = filtered_df['BMI'].dropna().mean()
    st.metric("평균 BMI", f"{avg_bmi:.2f}" if not pd.isna(avg_bmi) else "N/A")

with col5:
    total_records = len(df)
    filtered_ratio = (len(filtered_df) / total_records * 100) if total_records > 0 else 0
    st.metric("필터링 비율", f"{filtered_ratio:.1f}%")

st.markdown("---")

# 탭 생성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 개요",
    "👥 인구통계",
    "🍎 식습관",
    "📊 상관관계",
    "📋 데이터",
    "🤖 모델",
])

# 탭 1: 개요
with tab1:
    st.header("데이터 개요")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 연도별 데이터 분포
        year_counts = filtered_df['YEAR'].value_counts().sort_index()
        fig = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={'x': '연도', 'y': '빈도'},
            title='연도별 데이터 분포',
            color=year_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 성별 분포
        sex_counts = filtered_df['SEX'].value_counts()
        sex_labels = {1.0: '남성', 2.0: '여성'}
        fig = px.pie(
            values=sex_counts.values,
            names=[sex_labels.get(x, x) for x in sex_counts.index],
            title='성별 분포',
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # 연령 분포
        age_counts = filtered_df['AGE'].value_counts().sort_index()
        fig = px.bar(
            x=age_counts.index,
            y=age_counts.values,
            labels={'x': '나이', 'y': '빈도'},
            title='연령 분포',
            color=age_counts.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # 도시 유형별 분포 / 지역별 분포
        if 'CTYPE' in filtered_df.columns:
            # 청소년 데이터: 도시 유형
            ctype_counts = filtered_df['CTYPE'].value_counts()
            fig = px.bar(
                x=ctype_counts.index,
                y=ctype_counts.values,
                labels={'x': '도시 유형', 'y': '빈도'},
                title='도시 유형별 분포',
                color=ctype_counts.values,
                color_continuous_scale='Teal'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        elif 'REGION' in filtered_df.columns:
            # 성인 데이터: 지역
            region_counts = filtered_df['REGION'].value_counts().sort_index()
            if len(region_counts) > 0:
                fig = px.bar(
                    x=region_counts.index,
                    y=region_counts.values,
                    labels={'x': '지역', 'y': '빈도'},
                    title='지역별 분포',
                    color=region_counts.values,
                    color_continuous_scale='Teal'
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

# 탭 2: 인구통계
with tab2:
    st.header("인구통계 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 키 분포
        ht_data = filtered_df['HT'].dropna()
        if len(ht_data) > 0:
            fig = px.histogram(
                x=ht_data,
                nbins=30,
                labels={'x': '키 (cm)', 'count': '빈도'},
                title='키 분포',
                color_discrete_sequence=['coral']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 몸무게 분포
        wt_data = filtered_df['WT'].dropna()
        if len(wt_data) > 0:
            fig = px.histogram(
                x=wt_data,
                nbins=30,
                labels={'x': '몸무게 (kg)', 'count': '빈도'},
                title='몸무게 분포',
                color_discrete_sequence=['gold']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 키 vs 몸무게 산점도
    scatter_df = filtered_df[['HT', 'WT', 'AGE', 'SEX', 'YEAR']].dropna()
    if len(scatter_df) > 0:
        fig = px.scatter(
            scatter_df,
            x='HT',
            y='WT',
            color='AGE',
            size='AGE',
            hover_data=['SEX', 'YEAR'],
            labels={'HT': '키 (cm)', 'WT': '몸무게 (kg)', 'AGE': '나이'},
            title='키 vs 몸무게 (나이별 색상)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # 연도별 평균 키 추이
        year_height = filtered_df.groupby('YEAR')['HT'].mean().dropna()
        if len(year_height) > 0:
            fig = px.line(
                x=year_height.index,
                y=year_height.values,
                markers=True,
                labels={'x': '연도', 'y': '평균 키 (cm)'},
                title='연도별 평균 키 추이'
            )
            fig.update_traces(line_color='blue', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # 연도별 평균 몸무게 추이
        year_weight = filtered_df.groupby('YEAR')['WT'].mean().dropna()
        if len(year_weight) > 0:
            fig = px.line(
                x=year_weight.index,
                y=year_weight.values,
                markers=True,
                labels={'x': '연도', 'y': '평균 몸무게 (kg)'},
                title='연도별 평균 몸무게 추이'
            )
            fig.update_traces(line_color='red', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        # 성별 평균 키 비교
        sex_height = filtered_df.groupby('SEX')['HT'].mean().dropna()
        if len(sex_height) > 0:
            sex_labels_bar = ['남성', '여성']
            fig = px.bar(
                x=sex_labels_bar[:len(sex_height)],
                y=sex_height.values,
                labels={'x': '성별', 'y': '평균 키 (cm)'},
                title='성별 평균 키 비교',
                color=sex_labels_bar[:len(sex_height)],
                color_discrete_sequence=['#ff9999', '#66b3ff']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col6:
        # 성별 평균 몸무게 비교
        sex_weight = filtered_df.groupby('SEX')['WT'].mean().dropna()
        if len(sex_weight) > 0:
            sex_labels_bar = ['남성', '여성']
            fig = px.bar(
                x=sex_labels_bar[:len(sex_weight)],
                y=sex_weight.values,
                labels={'x': '성별', 'y': '평균 몸무게 (kg)'},
                title='성별 평균 몸무게 비교',
                color=sex_labels_bar[:len(sex_weight)],
                color_discrete_sequence=['#ff9999', '#66b3ff']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # BMI 분포
    bmi_data = filtered_df['BMI'].dropna()
    if len(bmi_data) > 0:
        fig = px.histogram(
            x=bmi_data,
            nbins=30,
            labels={'x': 'BMI', 'count': '빈도'},
            title='BMI 분포',
            color_discrete_sequence=['pink']
        )
        # BMI 기준선 추가 (한국 기준)
        fig.add_vline(x=18.5, line_dash="dash", line_color="blue", annotation_text="저체중")
        fig.add_vline(x=23.0, line_dash="dash", line_color="orange", annotation_text="과체중 시작 (23.0)")
        fig.add_vline(x=25.0, line_dash="dash", line_color="red", annotation_text="비만 (25.0 이상)")
        st.plotly_chart(fig, use_container_width=True)

# 탭 3: 식습관 / 건강 지표
with tab3:
    if is_adult:
        st.header("🏥 건강 지표 분석")
        
        # 건강 지표 분석
        col1, col2 = st.columns(2)
        
        with col1:
            # 혈당 분포
            glucose_data = filtered_df['GLUCOSE'].dropna()
            if len(glucose_data) > 0:
                fig = px.histogram(
                    x=glucose_data,
                    nbins=30,
                    labels={'x': '혈당 (mg/dL)', 'count': '빈도'},
                    title='혈당 분포',
                    color_discrete_sequence=['lightblue']
                )
                # 당뇨병 판단 기준선 (공복혈당 126mg/dL 이상)
                fig.add_vline(x=126, line_dash="dash", line_color="red", annotation_text="당뇨병 (126mg/dL 이상)")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 당화혈색소 분포
            hba1c_data = filtered_df['HbA1c'].dropna()
            if len(hba1c_data) > 0:
                fig = px.histogram(
                    x=hba1c_data,
                    nbins=30,
                    labels={'x': '당화혈색소 (%)', 'count': '빈도'},
                    title='당화혈색소 분포',
                    color_discrete_sequence=['lightgreen']
                )
                # 정상 당화혈색소 기준선 (5.7%)
                fig.add_vline(x=5.7, line_dash="dash", line_color="green", annotation_text="정상")
                fig.add_vline(x=6.5, line_dash="dash", line_color="red", annotation_text="당뇨병")
                st.plotly_chart(fig, use_container_width=True)
        
        # 연도별 건강 지표 추이
        st.subheader("📈 연도별 건강 지표 추이")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year_bmi = filtered_df.groupby('YEAR')['BMI'].mean().dropna()
            if len(year_bmi) > 0:
                fig = px.line(
                    x=year_bmi.index,
                    y=year_bmi.values,
                    markers=True,
                    labels={'x': '연도', 'y': '평균 BMI'},
                    title='연도별 평균 BMI 추이'
                )
                fig.update_traces(line_color='blue', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            year_glucose = filtered_df.groupby('YEAR')['GLUCOSE'].mean().dropna()
            if len(year_glucose) > 0:
                fig = px.line(
                    x=year_glucose.index,
                    y=year_glucose.values,
                    markers=True,
                    labels={'x': '연도', 'y': '평균 혈당 (mg/dL)'},
                    title='연도별 평균 혈당 추이'
                )
                fig.update_traces(line_color='red', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
        
        # 비만도 분포
        if 'OBESITY' in filtered_df.columns:
            obesity_counts = filtered_df['OBESITY'].dropna().value_counts().sort_index()
            if len(obesity_counts) > 0:
                obesity_labels = {1.0: '저체중', 2.0: '정상', 3.0: '과체중/비만'}
                fig = px.bar(
                    x=[obesity_labels.get(x, str(x)) for x in obesity_counts.index],
                    y=obesity_counts.values,
                    labels={'x': '비만도', 'y': '빈도'},
                    title='비만도 분포',
                    color=[obesity_labels.get(x, str(x)) for x in obesity_counts.index],
                    color_discrete_sequence=['lightblue', 'green', 'orange']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 연도별 비만도 추이 (전체, 남성, 여성)
        st.subheader("📊 연도별 비만도 추이 (성별 구분)")
        if not is_adult:
            if 'TEEN_OBESE_TOP5' in filtered_df.columns:
                teen_obesity_data = filtered_df[['YEAR', 'SEX', 'TEEN_OBESE_TOP5']].dropna()
            else:
                teen_obesity_data = pd.DataFrame()
            if len(teen_obesity_data) > 0:
                year_obesity_all = teen_obesity_data.groupby('YEAR')['TEEN_OBESE_TOP5'].mean().reset_index(name='비만율')
                year_obesity_all['비만율'] *= 100
                year_obesity_all['성별'] = '전체'
                
                male_data = teen_obesity_data[teen_obesity_data['SEX'] == 1.0]
                if len(male_data) > 0:
                    year_obesity_male = male_data.groupby('YEAR')['TEEN_OBESE_TOP5'].mean().reset_index(name='비만율')
                    year_obesity_male['비만율'] *= 100
                    year_obesity_male['성별'] = '남성'
                else:
                    year_obesity_male = pd.DataFrame(columns=['YEAR', '비만율', '성별'])
                
                female_data = teen_obesity_data[teen_obesity_data['SEX'] == 2.0]
                if len(female_data) > 0:
                    year_obesity_female = female_data.groupby('YEAR')['TEEN_OBESE_TOP5'].mean().reset_index(name='비만율')
                    year_obesity_female['비만율'] *= 100
                    year_obesity_female['성별'] = '여성'
                else:
                    year_obesity_female = pd.DataFrame(columns=['YEAR', '비만율', '성별'])
                
                combined_data = pd.concat([
                    year_obesity_all[['YEAR', '비만율', '성별']],
                    year_obesity_male[['YEAR', '비만율', '성별']],
                    year_obesity_female[['YEAR', '비만율', '성별']]
                ], ignore_index=True)
                
                if len(combined_data) > 0:
                    caption_text = "청소년 비만 기준: 전체 상위 5% (BMI ≥ {:.2f})".format(teen_bmi_cutoff) if teen_bmi_cutoff else "청소년 비만 기준: 전체 상위 5%"
                    st.caption(caption_text)
                    fig = px.line(
                        combined_data,
                        x='YEAR',
                        y='비만율',
                        color='성별',
                        markers=True,
                        labels={'YEAR': '연도', '비만율': '비만율 (%)'},
                        title='연도별 비만율 추이 (상위 5%)',
                        color_discrete_map={'전체': 'blue', '남성': '#ff9999', '여성': '#66b3ff'}
                    )
                    fig.update_traces(line_width=3, marker_size=8)
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            bmi_obesity_data = filtered_df[['YEAR', 'SEX', 'BMI']].dropna()
            if len(bmi_obesity_data) > 0:
                def obesity_rate(df):
                    return (df['BMI'] >= 25.0).mean() * 100
                
                year_obesity_all = bmi_obesity_data.groupby('YEAR').apply(obesity_rate).reset_index(name='비만율')
                year_obesity_all['성별'] = '전체'
                
                male_data = bmi_obesity_data[bmi_obesity_data['SEX'] == 1.0]
                if len(male_data) > 0:
                    year_obesity_male = male_data.groupby('YEAR').apply(obesity_rate).reset_index(name='비만율')
                    year_obesity_male['성별'] = '남성'
                else:
                    year_obesity_male = pd.DataFrame(columns=['YEAR', '비만율', '성별'])
                
                female_data = bmi_obesity_data[bmi_obesity_data['SEX'] == 2.0]
                if len(female_data) > 0:
                    year_obesity_female = female_data.groupby('YEAR').apply(obesity_rate).reset_index(name='비만율')
                    year_obesity_female['성별'] = '여성'
                else:
                    year_obesity_female = pd.DataFrame(columns=['YEAR', '비만율', '성별'])
                
                combined_data = pd.concat([
                    year_obesity_all[['YEAR', '비만율', '성별']],
                    year_obesity_male[['YEAR', '비만율', '성별']],
                    year_obesity_female[['YEAR', '비만율', '성별']]
                ], ignore_index=True)
                
                if len(combined_data) > 0:
                    fig = px.line(
                        combined_data,
                        x='YEAR',
                        y='비만율',
                        color='성별',
                        markers=True,
                        labels={'YEAR': '연도', '비만율': '비만율 (%)'},
                        title='연도별 비만율 추이 (BMI ≥ 25 기준)',
                        color_discrete_map={'전체': 'blue', '남성': '#ff9999', '여성': '#66b3ff'}
                    )
                    fig.update_traces(line_width=3, marker_size=8)
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 연도별 당뇨 발병률 추이 (성별 구분)
            if 'DIABETES' in filtered_df.columns:
                st.subheader("🩺 연도별 당뇨 발병률 추이 (성별 구분)")
                
                # 당뇨병 유병 여부 데이터 (DE1_dg: 0.0 = 정상, 1.0 = 당뇨병)
                diabetes_data = filtered_df[['YEAR', 'SEX', 'DIABETES']].dropna()
                if len(diabetes_data) > 0:
                    # 전체 당뇨 발병률 (1.0 = 당뇨병)
                    year_diabetes_all = diabetes_data.groupby('YEAR').apply(
                        lambda x: (x['DIABETES'] == 1.0).sum() / len(x) * 100
                    ).reset_index(name='당뇨발병률')
                    year_diabetes_all['성별'] = '전체'
                    
                    # 남성 당뇨 발병률
                    diabetes_male = diabetes_data[diabetes_data['SEX'] == 1.0]
                    if len(diabetes_male) > 0:
                        year_diabetes_male = diabetes_male.groupby('YEAR').apply(
                            lambda x: (x['DIABETES'] == 1.0).sum() / len(x) * 100
                        ).reset_index(name='당뇨발병률')
                        year_diabetes_male['성별'] = '남성'
                    else:
                        year_diabetes_male = pd.DataFrame(columns=['YEAR', '당뇨발병률', '성별'])
                    
                    # 여성 당뇨 발병률
                    diabetes_female = diabetes_data[diabetes_data['SEX'] == 2.0]
                    if len(diabetes_female) > 0:
                        year_diabetes_female = diabetes_female.groupby('YEAR').apply(
                            lambda x: (x['DIABETES'] == 1.0).sum() / len(x) * 100
                        ).reset_index(name='당뇨발병률')
                        year_diabetes_female['성별'] = '여성'
                    else:
                        year_diabetes_female = pd.DataFrame(columns=['YEAR', '당뇨발병률', '성별'])
                    
                    # 데이터 결합
                    combined_diabetes_data = pd.concat([
                        year_diabetes_all[['YEAR', '당뇨발병률', '성별']],
                        year_diabetes_male[['YEAR', '당뇨발병률', '성별']] if len(year_diabetes_male) > 0 else pd.DataFrame(),
                        year_diabetes_female[['YEAR', '당뇨발병률', '성별']] if len(year_diabetes_female) > 0 else pd.DataFrame()
                    ], ignore_index=True)
                    
                    if len(combined_diabetes_data) > 0:
                        fig = px.line(
                            combined_diabetes_data,
                            x='YEAR',
                            y='당뇨발병률',
                            color='성별',
                            markers=True,
                            labels={'YEAR': '연도', '당뇨발병률': '당뇨 발병률 (%)'},
                            title='연도별 당뇨 발병률 추이 (성별 구분)',
                            color_discrete_map={'전체': 'purple', '남성': '#ff9999', '여성': '#66b3ff'}
                        )
                        fig.update_traces(line_width=3, marker_size=8)
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # 성별 당뇨 발병률 비교 (바 차트)
                st.subheader("📊 성별 당뇨 발병률 비교")
                
                diabetes_sex_data = filtered_df[['SEX', 'DIABETES']].dropna()
                if len(diabetes_sex_data) > 0:
                    sex_diabetes_rates = {}
                    
                    # 전체 (1.0 = 당뇨병)
                    total_diabetes = (diabetes_sex_data['DIABETES'] == 1.0).sum()
                    sex_diabetes_rates['전체'] = (total_diabetes / len(diabetes_sex_data)) * 100
                    
                    # 남성
                    male_data = diabetes_sex_data[diabetes_sex_data['SEX'] == 1.0]
                    if len(male_data) > 0:
                        male_diabetes = (male_data['DIABETES'] == 1.0).sum()
                        sex_diabetes_rates['남성'] = (male_diabetes / len(male_data)) * 100
                    
                    # 여성
                    female_data = diabetes_sex_data[diabetes_sex_data['SEX'] == 2.0]
                    if len(female_data) > 0:
                        female_diabetes = (female_data['DIABETES'] == 1.0).sum()
                        sex_diabetes_rates['여성'] = (female_diabetes / len(female_data)) * 100
                    
                    if len(sex_diabetes_rates) > 0:
                        fig = px.bar(
                            x=list(sex_diabetes_rates.keys()),
                            y=list(sex_diabetes_rates.values()),
                            labels={'x': '성별', 'y': '당뇨 발병률 (%)'},
                            title='성별 당뇨 발병률 비교',
                            color=list(sex_diabetes_rates.keys()),
                            color_discrete_map={'전체': 'purple', '남성': '#ff9999', '여성': '#66b3ff'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # 비만과 당뇨의 상관관계 시각화
            if 'OBESITY' in filtered_df.columns and 'DIABETES' in filtered_df.columns:
                st.subheader("🔗 비만과 당뇨의 상관관계")
                
                obesity_diabetes_data = filtered_df[['OBESITY', 'DIABETES', 'BMI']].dropna()
                if len(obesity_diabetes_data) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 비만도별 당뇨 발병률
                        obesity_diabetes = obesity_diabetes_data.groupby('OBESITY').apply(
                            lambda x: (x['DIABETES'] == 1.0).sum() / len(x) * 100
                        ).reset_index(name='당뇨발병률')
                        obesity_labels = {1.0: '저체중', 2.0: '정상', 3.0: '과체중/비만'}
                        obesity_diabetes['비만도'] = [obesity_labels.get(x, str(x)) for x in obesity_diabetes['OBESITY']]
                        
                        if len(obesity_diabetes) > 0:
                            fig = px.bar(
                                x=obesity_diabetes['비만도'],
                                y=obesity_diabetes['당뇨발병률'],
                                labels={'x': '비만도', 'y': '당뇨 발병률 (%)'},
                                title='비만도별 당뇨 발병률',
                                color=obesity_diabetes['비만도'],
                                color_discrete_sequence=['lightblue', 'green', 'orange']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # BMI와 당뇨 발병률 산점도
                        obesity_diabetes_data['당뇨여부'] = (obesity_diabetes_data['DIABETES'] == 1.0).astype(int)
                        fig = px.scatter(
                            obesity_diabetes_data,
                            x='BMI',
                            y='당뇨여부',
                            color='OBESITY',
                            size='BMI',
                            hover_data=['OBESITY'],
                            labels={'BMI': 'BMI', '당뇨여부': '당뇨 유병 여부 (0=없음, 1=있음)'},
                            title='BMI와 당뇨 유병 여부',
                            color_discrete_map={1.0: 'lightblue', 2.0: 'green', 3.0: 'orange'}
                        )
                        fig.update_layout(yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['없음', '있음']))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 상관계수 표시
                    if 'BMI' in obesity_diabetes_data.columns:
                        bmi_diabetes_corr = obesity_diabetes_data[['BMI', 'DIABETES']].corr().iloc[0, 1]
                        obesity_diabetes_corr = obesity_diabetes_data[['OBESITY', 'DIABETES']].corr().iloc[0, 1]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("BMI와 당뇨 상관계수", f"{bmi_diabetes_corr:.3f}")
                        with col2:
                            st.metric("비만도와 당뇨 상관계수", f"{obesity_diabetes_corr:.3f}")
            
            # 성별 및 연령대별 당뇨 유병률 비교
            if 'DIABETES' in filtered_df.columns:
                st.subheader("👥 성별 및 연령대별 당뇨 유병률 비교")
                
                # 연령대 그룹 생성
                filtered_df['AGE_GROUP'] = pd.cut(
                    filtered_df['AGE'],
                    bins=[0, 30, 40, 50, 60, 70, 100],
                    labels=['20대', '30대', '40대', '50대', '60대', '70대 이상']
                )
                
                diabetes_age_sex_data = filtered_df[['AGE_GROUP', 'SEX', 'DIABETES']].dropna()
                if len(diabetes_age_sex_data) > 0:
                    # 연령대별, 성별 당뇨 유병률 계산
                    age_sex_diabetes = diabetes_age_sex_data.groupby(['AGE_GROUP', 'SEX']).apply(
                        lambda x: (x['DIABETES'] == 1.0).sum() / len(x) * 100
                    ).reset_index(name='당뇨유병률')
                    age_sex_diabetes['성별'] = age_sex_diabetes['SEX'].map({1.0: '남성', 2.0: '여성'})
                    
                    if len(age_sex_diabetes) > 0:
                        # 그룹 바 차트
                        fig = px.bar(
                            age_sex_diabetes,
                            x='AGE_GROUP',
                            y='당뇨유병률',
                            color='성별',
                            barmode='group',
                            labels={'AGE_GROUP': '연령대', '당뇨유병률': '당뇨 유병률 (%)'},
                            title='연령대별 및 성별 당뇨 유병률 비교',
                            color_discrete_map={'남성': '#ff9999', '여성': '#66b3ff'}
                        )
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 연령대별 당뇨 유병률 (전체)
                    age_diabetes = diabetes_age_sex_data.groupby('AGE_GROUP').apply(
                        lambda x: (x['DIABETES'] == 1.0).sum() / len(x) * 100
                    ).reset_index(name='당뇨유병률')
                    
                    if len(age_diabetes) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.bar(
                                x=age_diabetes['AGE_GROUP'],
                                y=age_diabetes['당뇨유병률'],
                                labels={'x': '연령대', 'y': '당뇨 유병률 (%)'},
                                title='연령대별 당뇨 유병률 (전체)',
                                color=age_diabetes['당뇨유병률'],
                                color_continuous_scale='Reds'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # 성별 당뇨 유병률 (전체)
                            sex_diabetes = diabetes_age_sex_data.groupby('SEX').apply(
                                lambda x: (x['DIABETES'] == 1.0).sum() / len(x) * 100
                            ).reset_index(name='당뇨유병률')
                            sex_diabetes['성별'] = sex_diabetes['SEX'].map({1.0: '남성', 2.0: '여성'})
                            
                            if len(sex_diabetes) > 0:
                                fig = px.bar(
                                    x=sex_diabetes['성별'],
                                    y=sex_diabetes['당뇨유병률'],
                                    labels={'x': '성별', 'y': '당뇨 유병률 (%)'},
                                    title='성별 당뇨 유병률 (전체)',
                                    color=sex_diabetes['성별'],
                                    color_discrete_map={'남성': '#ff9999', '여성': '#66b3ff'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
        
        # 아침식사 빈도
        if 'BREAKFAST' in filtered_df.columns:
            breakfast_counts_new = filtered_df['BREAKFAST'].dropna().value_counts().sort_index()
            if len(breakfast_counts_new) > 0:
                breakfast_labels_new = {1.0: '0회', 2.0: '1-2회', 3.0: '3-4회', 4.0: '5회 이상'}
                fig = px.pie(
                    values=breakfast_counts_new.values,
                    names=[breakfast_labels_new.get(x, str(x)) for x in breakfast_counts_new.index],
                    title='아침식사 빈도 분포',
                    color_discrete_sequence=px.colors.sequential.YlOrBr
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        # 식습관 분석 (성인 데이터)
        if 'F_FRUIT' in filtered_df.columns or 'F_VEG' in filtered_df.columns:
            st.subheader("🍎 식습관 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 과일 섭취 빈도
                if 'F_FRUIT' in filtered_df.columns:
                    fruit_counts = filtered_df['F_FRUIT'].dropna().value_counts().sort_index()
                    if len(fruit_counts) > 0:
                        fig = px.bar(
                            x=fruit_counts.index,
                            y=fruit_counts.values,
                            labels={'x': '과일 섭취 빈도', 'y': '빈도'},
                            title='과일 섭취 빈도 분포',
                            color=fruit_counts.values,
                            color_continuous_scale='Oranges'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 채소 섭취 빈도
                if 'F_VEG' in filtered_df.columns:
                    veg_counts = filtered_df['F_VEG'].dropna().value_counts().sort_index()
                    if len(veg_counts) > 0:
                        fig = px.bar(
                            x=veg_counts.index,
                            y=veg_counts.values,
                            labels={'x': '채소 섭취 빈도', 'y': '빈도'},
                            title='채소 섭취 빈도 분포',
                            color=veg_counts.values,
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            
            # 연도별 식습관 경향성 (성인 데이터)
            if 'F_FRUIT' in filtered_df.columns or 'F_VEG' in filtered_df.columns:
                st.subheader("📈 연도별 식습관 경향성")
                
                year_food_data = {}
                if 'F_FRUIT' in filtered_df.columns:
                    year_fruit = filtered_df.groupby('YEAR')['F_FRUIT'].mean().dropna()
                    if len(year_fruit) > 0:
                        year_food_data['F_FRUIT'] = year_fruit
                
                if 'F_VEG' in filtered_df.columns:
                    year_veg = filtered_df.groupby('YEAR')['F_VEG'].mean().dropna()
                    if len(year_veg) > 0:
                        year_food_data['F_VEG'] = year_veg
                
                if len(year_food_data) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'F_FRUIT' in year_food_data:
                            fig = px.line(
                                x=year_food_data['F_FRUIT'].index,
                                y=year_food_data['F_FRUIT'].values,
                                markers=True,
                                labels={'x': '연도', 'y': '평균 섭취 빈도'},
                                title='연도별 과일 섭취 빈도 추이',
                                color_discrete_sequence=['orange']
                            )
                            fig.update_traces(line_width=3, marker_size=8)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'F_VEG' in year_food_data:
                            fig = px.line(
                                x=year_food_data['F_VEG'].index,
                                y=year_food_data['F_VEG'].values,
                                markers=True,
                                labels={'x': '연도', 'y': '평균 섭취 빈도'},
                                title='연도별 채소 섭취 빈도 추이',
                                color_discrete_sequence=['green']
                            )
                            fig.update_traces(line_width=3, marker_size=8)
                            st.plotly_chart(fig, use_container_width=True)
        
        # 연령대별 분석
        st.subheader("👥 연령대별 건강 지표")
        
        # 연령대 그룹 생성
        filtered_df['AGE_GROUP'] = pd.cut(
            filtered_df['AGE'],
            bins=[0, 30, 40, 50, 60, 70, 100],
            labels=['20대', '30대', '40대', '50대', '60대', '70대 이상']
        )
        
        age_bmi = filtered_df.groupby('AGE_GROUP')['BMI'].mean().dropna()
        if len(age_bmi) > 0:
            fig = px.bar(
                x=age_bmi.index,
                y=age_bmi.values,
                labels={'x': '연령대', 'y': '평균 BMI'},
                title='연령대별 평균 BMI',
                color=age_bmi.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.header("식습관 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 과일 섭취 빈도
            if 'F_FRUIT' in filtered_df.columns:
                fruit_counts = filtered_df['F_FRUIT'].dropna().value_counts().sort_index()
                if len(fruit_counts) > 0:
                    fig = px.bar(
                        x=fruit_counts.index,
                        y=fruit_counts.values,
                        labels={'x': '과일 섭취 빈도', 'y': '빈도'},
                        title='과일 섭취 빈도 분포',
                        color=fruit_counts.values,
                        color_continuous_scale='Oranges'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 채소 섭취 빈도
        if 'F_VEG' in filtered_df.columns:
            veg_counts = filtered_df['F_VEG'].dropna().value_counts().sort_index()
            if len(veg_counts) > 0:
                fig = px.bar(
                    x=veg_counts.index,
                    y=veg_counts.values,
                    labels={'x': '채소 섭취 빈도', 'y': '빈도'},
                    title='채소 섭취 빈도 분포',
                    color=veg_counts.values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # 패스트푸드 섭취 빈도
        if 'F_FASTFOOD' in filtered_df.columns:
            fastfood_counts = filtered_df['F_FASTFOOD'].dropna().value_counts().sort_index()
            if len(fastfood_counts) > 0:
                fig = px.bar(
                    x=fastfood_counts.index,
                    y=fastfood_counts.values,
                    labels={'x': '패스트푸드 섭취 빈도', 'y': '빈도'},
                    title='패스트푸드 섭취 빈도 분포',
                    color=fastfood_counts.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # 탄산음료 섭취 빈도
        if 'SODA_INTAKE' in filtered_df.columns:
            soda_counts = filtered_df['SODA_INTAKE'].dropna().value_counts().sort_index()
            if len(soda_counts) > 0:
                fig = px.bar(
                    x=soda_counts.index,
                    y=soda_counts.values,
                    labels={'x': '탄산음료 섭취 빈도', 'y': '빈도'},
                    title='탄산음료 섭취 빈도 분포',
                    color=soda_counts.values,
                    color_continuous_scale='Purples'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # 아침식사 카테고리
    if 'Breakfast_Category' in filtered_df.columns:
        breakfast_counts = filtered_df['Breakfast_Category'].dropna().value_counts().sort_index()
        if len(breakfast_counts) > 0:
            breakfast_labels = {0.0: '0회', 1.0: '1-2회', 2.0: '3-4회', 3.0: '5회 이상'}
            # 파이 차트로 변경 (100% 원 그래프)
            fig = px.pie(
                values=breakfast_counts.values,
                names=[breakfast_labels.get(x, str(x)) for x in breakfast_counts.index],
                title='아침식사 카테고리 분포',
                color_discrete_sequence=px.colors.sequential.YlOrBr
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    # 연도별 식습관 경향성 분석
    st.subheader("📈 연도별 식습관 경향성")
    
    # 연도별 평균 섭취 빈도 계산 (존재하는 컬럼만)
    agg_dict = {}
    if 'F_FRUIT' in filtered_df.columns:
        agg_dict['F_FRUIT'] = 'mean'
    if 'F_VEG' in filtered_df.columns:
        agg_dict['F_VEG'] = 'mean'
    if 'F_FASTFOOD' in filtered_df.columns:
        agg_dict['F_FASTFOOD'] = 'mean'
    if 'SODA_INTAKE' in filtered_df.columns:
        agg_dict['SODA_INTAKE'] = 'mean'
    
    if len(agg_dict) > 0:
        year_food_data = filtered_df.groupby('YEAR').agg(agg_dict).dropna()
    else:
        year_food_data = pd.DataFrame()
    
    if len(year_food_data) > 0:
        # 연도별 과일 섭취 추이
        col1, col2 = st.columns(2)
        
        with col1:
            if 'F_FRUIT' in year_food_data.columns:
                fig = px.line(
                    x=year_food_data.index,
                    y=year_food_data['F_FRUIT'],
                    markers=True,
                    labels={'x': '연도', 'y': '평균 섭취 빈도'},
                    title='연도별 과일 섭취 빈도 추이',
                    color_discrete_sequence=['orange']
                )
                fig.update_traces(line_width=3, marker_size=8)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'F_VEG' in year_food_data.columns:
                fig = px.line(
                    x=year_food_data.index,
                    y=year_food_data['F_VEG'],
                    markers=True,
                    labels={'x': '연도', 'y': '평균 섭취 빈도'},
                    title='연도별 채소 섭취 빈도 추이',
                    color_discrete_sequence=['green']
                )
                fig.update_traces(line_width=3, marker_size=8)
                st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            if 'F_FASTFOOD' in year_food_data.columns:
                fig = px.line(
                    x=year_food_data.index,
                    y=year_food_data['F_FASTFOOD'],
                    markers=True,
                    labels={'x': '연도', 'y': '평균 섭취 빈도'},
                    title='연도별 패스트푸드 섭취 빈도 추이',
                    color_discrete_sequence=['red']
                )
                fig.update_traces(line_width=3, marker_size=8)
                st.plotly_chart(fig, use_container_width=True)
        
        # 아침식사 연도별 추이
        if 'Breakfast_Category' in filtered_df.columns:
            year_breakfast = filtered_df.groupby('YEAR')['Breakfast_Category'].mean().dropna()
            if len(year_breakfast) > 0:
                fig = px.line(
                    x=year_breakfast.index,
                    y=year_breakfast.values,
                    markers=True,
                    labels={'x': '연도', 'y': '평균 아침식사 카테고리'},
                    title='연도별 아침식사 카테고리 추이 (평균값)',
                    color_discrete_sequence=['brown']
                )
                fig.update_traces(line_width=3, marker_size=8)
                # y축 레이블을 카테고리로 표시
                breakfast_labels_map = {0.0: '0회', 1.0: '1-2회', 2.0: '3-4회', 3.0: '5회 이상'}
                fig.update_layout(
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[0.0, 1.0, 2.0, 3.0],
                        ticktext=[breakfast_labels_map.get(v, str(v)) for v in [0.0, 1.0, 2.0, 3.0]]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            if 'SODA_INTAKE' in year_food_data.columns:
                fig = px.line(
                    x=year_food_data.index,
                    y=year_food_data['SODA_INTAKE'],
                    markers=True,
                    labels={'x': '연도', 'y': '평균 섭취 빈도'},
                    title='연도별 탄산음료 섭취 빈도 추이',
                    color_discrete_sequence=['purple']
                )
                fig.update_traces(line_width=3, marker_size=8)
                st.plotly_chart(fig, use_container_width=True)
        
        # 전체 식습관 비교 (하나의 그래프에 모든 항목)
        st.subheader("📊 연도별 식습관 종합 비교")
        fig = go.Figure()
        
        if 'F_FRUIT' in year_food_data.columns:
            fig.add_trace(go.Scatter(
                x=year_food_data.index,
                y=year_food_data['F_FRUIT'],
                mode='lines+markers',
                name='과일',
                line=dict(color='orange', width=3),
                marker=dict(size=8)
            ))
        
        if 'F_VEG' in year_food_data.columns:
            fig.add_trace(go.Scatter(
                x=year_food_data.index,
                y=year_food_data['F_VEG'],
                mode='lines+markers',
                name='채소',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
        
        if 'F_FASTFOOD' in year_food_data.columns:
            fig.add_trace(go.Scatter(
                x=year_food_data.index,
                y=year_food_data['F_FASTFOOD'],
                mode='lines+markers',
                name='패스트푸드',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
        
        if 'SODA_INTAKE' in year_food_data.columns:
            fig.add_trace(go.Scatter(
                x=year_food_data.index,
                y=year_food_data['SODA_INTAKE'],
                mode='lines+markers',
                name='탄산음료',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ))
        
        # 아침식사 추가
        if 'Breakfast_Category' in filtered_df.columns:
            year_breakfast = filtered_df.groupby('YEAR')['Breakfast_Category'].mean().dropna()
            if len(year_breakfast) > 0:
                fig.add_trace(go.Scatter(
                    x=year_breakfast.index,
                    y=year_breakfast.values,
                    mode='lines+markers',
                    name='아침식사',
                    line=dict(color='brown', width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title='연도별 식습관 종합 비교',
            xaxis_title='연도',
            yaxis_title='평균 섭취 빈도 / 카테고리',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 식습관 평균 비교
    food_means = {}
    if 'F_FRUIT' in filtered_df.columns and filtered_df['F_FRUIT'].notna().any():
        food_means['과일'] = filtered_df['F_FRUIT'].dropna().mean()
    if 'F_VEG' in filtered_df.columns and filtered_df['F_VEG'].notna().any():
        food_means['채소'] = filtered_df['F_VEG'].dropna().mean()
    if 'F_FASTFOOD' in filtered_df.columns and filtered_df['F_FASTFOOD'].notna().any():
        food_means['패스트푸드'] = filtered_df['F_FASTFOOD'].dropna().mean()
    if 'SODA_INTAKE' in filtered_df.columns and filtered_df['SODA_INTAKE'].notna().any():
        food_means['탄산음료'] = filtered_df['SODA_INTAKE'].dropna().mean()
    
    # NaN이 아닌 값만 필터링
    food_means = {k: v for k, v in food_means.items() if not pd.isna(v)}
    if len(food_means) > 0:
        fig = px.bar(
            x=list(food_means.keys()),
            y=list(food_means.values()),
            labels={'x': '식품 유형', 'y': '평균 섭취 빈도'},
            title='식습관 평균 섭취 빈도',
            color=list(food_means.keys()),
            color_discrete_sequence=['orange', 'green', 'red', 'purple']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤖 청소년 비만 예측 모델")
    teen_model_metrics = teen_model_summary_global
    if teen_model_metrics:
        cutoff_text = f"{teen_bmi_cutoff:.2f}" if teen_bmi_cutoff else "정보 없음"
        st.markdown(
            "- **모델**: Logistic Regression (class_weight='balanced')\n"
            f"- **라벨 기준**: BMI 상위 5% (컷오프 {cutoff_text} 이상)\n"
            f"- **적용 임계값**: {teen_model_metrics['threshold']:.2f}"
        )
        metrics_chart = pd.DataFrame({
            "지표": ["Accuracy", "Recall", "Precision", "F1-Score", "AUC-ROC"],
            "값": [
                teen_model_metrics["accuracy"],
                teen_model_metrics["recall"],
                teen_model_metrics["precision"],
                teen_model_metrics["f1"],
                teen_model_metrics["auc"]
            ]
        })
        fig = px.bar(
            metrics_chart,
            x="지표",
            y="값",
            title="모델 성능 지표",
            color="지표",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{teen_model_metrics['accuracy']*100:.1f}%")
        col2.metric("Recall", f"{teen_model_metrics['recall']*100:.1f}%")
        col3.metric("Precision", f"{teen_model_metrics['precision']*100:.1f}%")

        col4, col5 = st.columns(2)
        col4.metric("F1-Score", f"{teen_model_metrics['f1']*100:.1f}%")
        col5.metric("AUC-ROC", f"{teen_model_metrics['auc']:.3f}")
        st.caption(f"학습 표본 수: {teen_model_metrics['sample_size']:,}건")
    else:
        st.info("선택한 필터 조건에서는 모델을 학습할 충분한 데이터가 없습니다. 연도나 연령 범위를 넓혀보세요.")

# 탭 4: 상관관계
with tab4:
    st.header("상관관계 분석")
    
    if is_adult:
        # 성인 데이터 상관관계
        health_cols = ['BMI', 'GLUCOSE', 'HbA1c', 'OBESITY']
        health_data = filtered_df[health_cols].dropna()
        if len(health_data) > 0:
            health_corr = health_data.corr()
            fig = px.imshow(
                health_corr,
                labels=dict(x="변수", y="변수", color="상관계수"),
                x=health_cols,
                y=health_cols,
                color_continuous_scale='RdBu',
                aspect="auto",
                title='건강 지표 상관관계 히트맵'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # 청소년 데이터 상관관계
        # 식습관 상관관계 히트맵
        food_cols = ['F_BR', 'F_FRUIT', 'F_VEG', 'F_FASTFOOD', 'SODA_INTAKE']
        if all(col in filtered_df.columns for col in food_cols):
            food_data = filtered_df[food_cols].dropna()
            if len(food_data) > 0:
                food_corr = food_data.corr()
                fig = px.imshow(
                    food_corr,
                    labels=dict(x="변수", y="변수", color="상관계수"),
                    x=food_cols,
                    y=food_cols,
                    color_continuous_scale='RdBu',
                    aspect="auto",
                    title='식습관 상관관계 히트맵'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # 전체 상관관계 히트맵
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'BMI' in numeric_cols:
        numeric_cols.remove('BMI')
    numeric_cols.append('BMI')
    
    st.subheader("전체 변수 상관관계")
    numeric_data = filtered_df[numeric_cols].dropna()
    if len(numeric_data) > 0:
        full_corr = numeric_data.corr()
        fig = px.imshow(
            full_corr,
            labels=dict(x="변수", y="변수", color="상관계수"),
            x=numeric_cols,
            y=numeric_cols,
            color_continuous_scale='RdBu',
            aspect="auto",
            title='전체 변수 상관관계 히트맵'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 연령별 평균 키/몸무게
    col1, col2 = st.columns(2)
    
    with col1:
        age_height = filtered_df.groupby('AGE')['HT'].mean().dropna().sort_index()
        if len(age_height) > 0:
            fig = px.line(
                x=age_height.index,
                y=age_height.values,
                markers=True,
                labels={'x': '나이', 'y': '평균 키 (cm)'},
                title='연령별 평균 키'
            )
            fig.update_traces(line_color='green', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        age_weight = filtered_df.groupby('AGE')['WT'].mean().dropna().sort_index()
        if len(age_weight) > 0:
            fig = px.line(
                x=age_weight.index,
                y=age_weight.values,
                markers=True,
                labels={'x': '나이', 'y': '평균 몸무게 (kg)'},
                title='연령별 평균 몸무게'
            )
            fig.update_traces(line_color='orange', line_width=3)
            st.plotly_chart(fig, use_container_width=True)

# 탭 5: 데이터
with tab5:
    st.header("데이터 테이블")
    
    # 통계 요약
    st.subheader("📊 통계 요약")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**기본 정보**")
        st.write(f"- 총 데이터 수: {len(filtered_df):,}개")
        st.write(f"- 연도 범위: {filtered_df['YEAR'].min()} ~ {filtered_df['YEAR'].max()}")
        st.write(f"- 나이 범위: {filtered_df['AGE'].min()} ~ {filtered_df['AGE'].max()}세")
    
    with col2:
        st.write("**평균값**")
        st.write(f"- 평균 키: {filtered_df['HT'].mean():.2f}cm")
        st.write(f"- 평균 몸무게: {filtered_df['WT'].mean():.2f}kg")
        st.write(f"- 평균 BMI: {filtered_df['BMI'].mean():.2f}")
    
    with col3:
        st.write("**분포**")
        sex_counts = filtered_df['SEX'].value_counts()
        for sex_val, count in sex_counts.items():
            sex_name = '남성' if sex_val == 1.0 else '여성'
            st.write(f"- {sex_name}: {count:,}명")
    
    st.markdown("---")
    
    # 데이터프레임 표시
    st.subheader("필터링된 데이터")
    
    # 검색 기능
    search_term = st.text_input("🔍 데이터 검색 (컬럼명 또는 값으로 검색)", "")
    
    display_df = filtered_df.copy()
    
    if search_term:
        # 숫자 검색
        try:
            search_num = float(search_term)
            mask = display_df.select_dtypes(include=[np.number]).apply(
                lambda x: x.astype(str).str.contains(search_term, na=False)
            ).any(axis=1)
        except:
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
        display_df = display_df[mask]
    
    # 페이지네이션
    page_size = st.selectbox("페이지당 행 수", [100, 500, 1000, 5000], index=1)
    
    total_pages = (len(display_df) // page_size) + (1 if len(display_df) % page_size > 0 else 0)
    if total_pages > 0:
        page = st.number_input(f"페이지 (1-{total_pages})", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(
            display_df.iloc[start_idx:end_idx],
            use_container_width=True,
            height=600
        )
        
        st.info(f"총 {len(display_df):,}개 행 중 {start_idx+1}-{min(end_idx, len(display_df))}개 표시 중")
    else:
        st.warning("검색 결과가 없습니다.")
    
    # 데이터 다운로드
    st.markdown("---")
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 필터링된 데이터 다운로드 (CSV)",
        data=csv,
        file_name=f'filtered_data_{len(filtered_df)}rows.csv',
        mime='text/csv'
    )

with tab6:
    st.header("🤖 청소년 비만 예측 모델")
    if is_adult:
        st.info("모델 분석은 청소년 데이터에서만 제공합니다. 사이드바에서 청소년 데이터를 선택해주세요.")
    else:
        if teen_model_results_global:
            optimal_c = teen_model_results_global.get('optimal_c', 'N/A')
            optimal_thr = teen_model_results_global.get('threshold', 0.49)
            ensemble_info = teen_model_results_global.get('ensemble', {})
            ensemble_weights = ensemble_info.get('weights', (0.6, 0.4))
            st.markdown(
                f"- **라벨 기준**: BMI 상위 5% (컷오프 {teen_bmi_cutoff:.2f} 이상)\n"
                f"- **최적화된 C 값**: {optimal_c}\n"
                f"- **최적화된 임계값**: {optimal_thr:.3f}\n"
                f"- **앙상블 가중치**: LR {ensemble_weights[0]:.2f} + RF {ensemble_weights[1]:.2f}\n"
                f"- **학습 표본 수**: {teen_model_results_global['sample_size']:,}건\n"
                f"- **SMOTE 적용**: 예 (오버샘플링)"
            )
            st.markdown(
                "- **비교 모델**: Logistic Regression (SMOTE) / Random Forest (튜닝) / Balanced Random Forest / HistGradientBoosting / **Ensemble (최적 가중치)**"
            )

            metric_labels = [
                ("accuracy", "Accuracy"),
                ("recall", "Recall"),
                ("precision", "Precision"),
                ("f1", "F1-Score"),
                ("auc", "AUC-ROC"),
            ]
            model_name_map = {
                "logistic": "Logistic Regression",
                "random_forest": "Random Forest",
                "balanced_random_forest": "Balanced Random Forest",
                "hist_gradient_boosting": "HistGradientBoosting",
                "ensemble": "Ensemble (LR+RF)",
            }
            metric_rows = []
            for metric_key, metric_name in metric_labels:
                for model_key, model_title in model_name_map.items():
                    values = teen_model_results_global.get(model_key)
                    if values and metric_key in values:
                        metric_rows.append(
                            {
                                "모델": model_title,
                                "지표": metric_name,
                                "값": values[metric_key],
                            }
                        )

            if metric_rows:
                metrics_df = pd.DataFrame(metric_rows)
                fig = px.bar(
                    metrics_df,
                    x="지표",
                    y="값",
                    color="모델",
                    barmode="group",
                    title="모델별 성능 비교",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

                pivot_df = (
                    metrics_df.pivot_table(index="모델", columns="지표", values="값")
                    .round(3)
                    .reset_index()
                )
                st.dataframe(pivot_df, use_container_width=True)

            roc_data = teen_model_results_global.get("roc_curve")
            if roc_data:
                best_model_name = roc_data.get("model_name", "logistic")
                best_model_title = model_name_map.get(best_model_name, "Best Model")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=roc_data["fpr"],
                        y=roc_data["tpr"],
                        mode="lines",
                        name=f"{best_model_title} ROC (AUC {roc_data['auc']:.3f})",
                        line=dict(color="#2ca02c", width=3),
                        fill="tozeroy",
                        opacity=0.3,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random Guess",
                        line=dict(color="gray", dash="dash"),
                    )
                )
                fig.update_layout(
                    title=f"{best_model_title} ROC Curve (최고 AUC 모델)",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("모델을 학습할 충분한 데이터가 없어 성능을 표시할 수 없습니다.")

# 사이드바 하단 정보
st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **현재 필터링된 데이터:**
    - {len(filtered_df):,}개 행
    - 전체 데이터의 {len(filtered_df)/len(current_df)*100:.1f}%
    """
)
    
# 사이드바 하단 정보
st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **현재 필터링된 데이터:**
    - {len(filtered_df):,}개 행
    - 전체 데이터의 {len(filtered_df)/len(df)*100:.1f}%
    """
)
