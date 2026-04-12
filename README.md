# 설명 가능한 AI 기반 SaaS 고객 이탈 예측 및 고객 유지 전략 제안 | J.Son

이 프로젝트는 **RavenStack Synthetic SaaS Dataset**을 활용하여  
고객 이탈(Churn)을 예측하고, **설명 가능한 AI(XAI)** 를 통해 주요 이탈 요인을 해석한 뒤,  
실질적인 **고객 유지 전략(Retention Strategy)** 까지 연결하는 개인 프로젝트이다.

단순히 고객이 이탈할지를 예측하는 데서 멈추지 않고,  
**왜 이탈 위험이 높게 판단되었는지**를 설명하고,  
그 결과를 바탕으로 **실행 가능한 대응 방향**을 제시하는 것을 목표로 했다.

---

## 📌 프로젝트 개요

- **주제**: 설명 가능한 SaaS 고객 이탈 예측 및 고객 유지 전략 분석
- **데이터셋**: RavenStack Synthetic SaaS Dataset
- **분석 단위**: account 기준
- **핵심 목표**
  - SaaS 고객 이탈 예측
  - 주요 이탈 요인 해석
  - threshold tuning을 통한 운영 기준 탐색
  - 고객 유지 전략 제안
  - Streamlit 대시보드 구현

---

## 🧭 프로젝트 흐름

<p align="center">
  <img src="assets/images/pipeline-diagram.png" width="850">
</p>

<p align="center">
  데이터 파악 → EDA → ML/Modeling → DL → XAI → Retention Strategy
</p>

---

## ❓ 문제 정의

SaaS 환경에서는 이미 이탈한 고객을 사후적으로 분석하는 것보다,  
**이탈 가능성이 높은 고객을 미리 탐지하고 선제적으로 대응하는 것**이 더 중요하다.

이 프로젝트는 account 단위 데이터를 통합하여 churn을 예측하고,  
예측 결과를 SHAP 기반 설명과 연결하여  
최종적으로는 **실무적으로 활용 가능한 retention 전략**까지 제시하는 구조로 설계하였다.

---

## ⚙️ 분석 과정

### 1. 데이터 파악
분석에 들어가기 전, 각 테이블의 구조와 변수 의미를 먼저 확인하였다.  
고객 계정, 구독, 사용량, 지원 이력 등으로 나뉜 데이터를 account 기준으로 연결할 수 있도록 정리하고,  
타깃 변수와 예측에 사용할 수 있는 입력 변수를 구분하였다.

### 2. EDA
churn 고객과 non-churn 고객의 차이를 비교하여,  
어떤 특성이 이탈과 연결되는지 탐색하였다.

- 전체 churn 분포 확인
- churn 여부별 평균 차이 비교
- 사용량 및 health score 관련 변수 확인
- 주요 변수 간 상관관계 점검

### 3. ML / Modeling
머신러닝 모델을 활용해 기본 churn 예측 성능을 비교하였다.

- Logistic Regression
- Random Forest

또한 threshold를 고정값 0.5로 두지 않고,  
precision / recall / F1 변화를 함께 비교하여  
실제 운영에 더 적합한 기준을 탐색하였다.

### 4. DL
딥러닝 MLP 모델을 추가로 실험하여  
비선형 패턴과 변수 간 복합 상호작용을 반영할 수 있는지 확인하였다.

이번 프로젝트에서 DL은 성능 자체보다도,  
ML과 비교했을 때 어떤 차이를 보이는지 확인하는 확장 실험의 의미가 컸다.

### 5. XAI
SHAP 기반 해석을 통해  
모델이 고객을 왜 이탈 위험이 높다고 판단했는지 분석하였다.

주요 해석 포인트:
- 사용량 감소
- 활성 구독 비율 저하
- 오류 경험 증가
- 지원 응답 관련 변수 영향

### 6. Retention Strategy
예측과 해석 결과를 바탕으로  
실제 고객 유지 전략으로 연결하였다.

- 사용량 저하 고객 선제 케어
- 오류 경험 고객 우선 대응
- 고위험 고객군 세분화 운영
- threshold 기반 운영 기준 수립

---

## 🖥️ 대시보드 구성

Streamlit 대시보드는 다음 흐름으로 구성하였다.

- **프로젝트 개요**
- **EDA: 이탈 고객 특성 탐색**
- **모델 성능 및 운영 기준**
- **이탈 요인 해석 및 고객 유지 전략**
- **고객별 예측 및 액션 제안**

이를 통해  
단순 예측 결과뿐 아니라,  
**왜 그런 결과가 나왔는지**,  
그리고 **어떻게 대응할 수 있는지**까지 한 흐름 안에서 확인할 수 있도록 설계하였다.

---

## 🛠️ 사용 기술

- **Language**: Python
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Modeling**: scikit-learn
- **Deep Learning**: PyTorch
- **Explainable AI**: SHAP
- **Dashboard**: Streamlit

---

## 🎯 프로젝트 의의

이 프로젝트는 단순한 churn classification 실습이 아니라,  
**예측 → 해석 → 전략 제안**으로 이어지는 실무형 분석 흐름을 구현하는 데 의미가 있다.

특히 다음 두 가지에 초점을 두었다.

- **설명 가능한 예측**
- **실행 가능한 고객 유지 전략 연결**

즉, 모델 성능 자체보다도  
**실제 운영 환경에서 어떻게 활용할 수 있을지**를 함께 고민한 프로젝트라고 볼 수 있다.

---

## 🚀 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Streamlit 실행

```bash
streamlit run src/app/streamlit_app.py
```

## 📝 참고

이 저장소는 개인 프로젝트 정리용 레포지토리이며,
분석 과정과 대시보드 구현 흐름을 기록하고 정리하기 위해 작성하였다.
