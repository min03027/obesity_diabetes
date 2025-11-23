# 건강 데이터 분석 대시보드

청소년 및 성인 건강 데이터를 분석하는 Streamlit 대시보드 및 당뇨병 위험도 예측 모델입니다.

## 프로젝트 개요

**청소년기 비만 및 생활습관이 성인기 당뇨병 위험에 미치는 영향 분석**

## 주요 기능

### 📊 Streamlit 대시보드
- 📊 **데이터셋 선택**: 청소년 데이터 / 성인 데이터 선택
- 📈 **개요 분석**: 연도별, 성별, 연령별, 지역별 데이터 분포
- 👥 **인구통계 분석**: 키, 몸무게, BMI 분포 및 추이
- 🍎 **식습관 분석**: 과일, 채소, 패스트푸드, 탄산음료 섭취 빈도 및 연도별 추이
- 🏥 **건강 지표 분석**: 혈당, 당화혈색소, 비만도, 당뇨 발병률 분석
- 📊 **상관관계 분석**: 변수 간 상관관계 히트맵
- 📋 **데이터 탐색**: 필터링, 검색, 다운로드 기능

### 🤖 머신러닝 모델
- **당뇨병 위험도 예측 모델**: Logistic Regression, Random Forest
- **클래스 가중치 조정 및 SMOTE 적용**: 불균형 데이터 문제 해결
- **특성 중요도 분석**: SHAP 기반 위험도 설명
- **모델 성능 평가**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

## 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd 데이터분석

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

## 실행 방법

### 대시보드 실행
```bash
streamlit run veiw.py
```

브라우저에서 `http://localhost:8501`로 접속하면 대시보드를 확인할 수 있습니다.

### 머신러닝 모델 실행
```bash
# 기본 모델 (혈당 지표 포함)
python diabetes_prediction.py

# 개선된 모델 (혈당 지표 제외)
python diabetes_prediction_improved.py

# 최종 모델 (클래스 가중치 조정 및 SMOTE)
python diabetes_prediction_final.py
```

## 모델 성능 결과

### 최고 성능 모델: Logistic Regression (Balanced)
- **Accuracy**: 66.51%
- **Precision**: 23.65%
- **Recall**: 74.24% ⭐
- **F1-Score**: 35.88%
- **AUC-ROC**: 76.56%

### 주요 특성 중요도
1. **AGE (연령)**: 29.04%
2. **INCOME (소득)**: 17.24%
3. **BREAKFAST (아침식사)**: 11.77%
4. **BMI**: 8.78%
5. **SEX (성별)**: 15.70%

자세한 결과는 `model_summary.md`를 참고하세요.

## Streamlit Cloud 배포 방법

1. **GitHub에 저장소 업로드**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Streamlit Cloud에서 배포**
   - [Streamlit Cloud](https://streamlit.io/cloud)에 접속
   - GitHub 계정으로 로그인
   - "New app" 클릭
   - GitHub 저장소 선택
   - Main file path: `veiw.py`
   - "Deploy!" 클릭

3. **배포 완료**
   - 몇 분 후 자동으로 배포됩니다
   - 공유 가능한 URL이 생성됩니다

## 데이터 파일

- `9ch_final_data.csv`: 청소년 건강 데이터
- `hn_cleand_data (2).csv`: 성인 건강 데이터

## 기술 스택

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- NumPy

## 라이선스

MIT License

