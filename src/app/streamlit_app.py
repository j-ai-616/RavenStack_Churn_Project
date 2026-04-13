from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.app.sections.overview_section import render as render_overview
from src.app.sections.eda_section import render as render_eda
from src.app.sections.model_section import render as render_model
from src.app.sections.xai_section import render as render_xai
from src.app.sections.prediction_section import render as render_prediction
from src.utils.plot_utils import set_korean_font

set_korean_font()

st.set_page_config(
    page_title="설명 가능한 AI 기반 SaaS 고객 이탈 예측 및 고객 유지 전략 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.2rem;
            padding-bottom: 2.2rem;
            max-width: 1200px;
        }

        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 300px;
        }

        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }

        .hero-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
            color: white;
            padding: 2rem 2rem 1.6rem 2rem;
            border-radius: 22px;
            margin_top: 0.6rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
        }

        .hero-eyebrow {
            font-size: 0.95rem;
            opacity: 0.82;
            margin-bottom: 0.4rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.3;
            margin-bottom: 0.65rem;
        }

        .hero-desc {
            font-size: 1.02rem;
            opacity: 0.92;
            line-height: 1.7;
        }

        .section-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .mini-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        }

        .mini-label {
            color: #475569;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }

        .mini-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
        }

        .mini-sub {
            color: #64748b;
            font-size: 0.9rem;
        }

        .sidebar-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .sidebar-sub {
            color: #64748b;
            font-size: 0.88rem;
            margin-bottom: 1rem;
            line-height: 1.5;
        }

        .flow-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-top: 1rem;
            line-height: 1.8;
            color: #334155;
            font-size: 0.92rem;
        }

        .stRadio > div {
            gap: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-eyebrow">J.Son 머신러닝/딥러닝 프로젝트 · RavenStack Synthetic SaaS Dataset</div>
            <div class="hero-title">설명 가능한 SaaS 고객 이탈 예측 및 고객 유지 전략 분석</div>
            <div class="hero-desc">
                단순히 churn을 맞히는 데서 멈추지 않고, 
                <b>어떤 고객이 왜 이탈 위험이 높은지</b>를 해석하고 
                그 결과를 바탕으로 <b>실행 가능한 고객 유지 전략</b>까지 연결하는 것을 목표로 한 대시보드이다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Dashboard Navigation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-sub">프로젝트 흐름에 따라 핵심 결과를 순서대로 확인할 수 있도록 구성했다.</div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "페이지 이동",
            [
                "프로젝트 개요",
                "EDA",
                "모델 성능",
                "이탈 요인 & 유지 전략",
                "고객별 예측",
            ],
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="flow-box">
            <b>분석 흐름</b><br>
            ① 문제 정의<br>
            ② 이탈 고객 특성 파악<br>
            ③ 예측 모델 비교 및 threshold 조정<br>
            ④ XAI 기반 주요 이탈 신호 해석<br>
            ⑤ 고객 유지 전략 제안
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


def main() -> None:
    inject_custom_css()
    render_header()
    page = render_sidebar()

    if page == "프로젝트 개요":
        render_overview()
    elif page == "EDA":
        render_eda()
    elif page == "모델 성능":
        render_model()
    elif page == "이탈 요인 & 유지 전략":
        render_xai()
    elif page == "고객별 예측":
        render_prediction()
    else:
        st.error("알 수 없는 페이지입니다.")


if __name__ == "__main__":
    main()
