from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import streamlit as st

# ------------------------------------------------------
# 프로젝트 루트 경로를 sys.path에 추가
# - 로컬 / GitHub / Streamlit Cloud 환경에서 import 안정성 확보
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------
# 섹션 import
# ------------------------------------------------------
from src.app.sections.eda_section import render as render_eda
from src.app.sections.model_section import render as render_model
from src.app.sections.overview_section import render as render_overview
from src.app.sections.prediction_section import render as render_prediction
from src.app.sections.xai_section import render as render_xai

# 한글 폰트 설정 함수가 있다면 사용
try:
    from src.utils.plot_utils import set_korean_font
except ImportError:
    set_korean_font = None


# ------------------------------------------------------
# 페이지 기본 설정
# ------------------------------------------------------
st.set_page_config(
    page_title="손지은",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_global_settings() -> None:
    """앱 시작 시 전역 설정 적용"""
    if set_korean_font is not None:
        try:
            set_korean_font()
        except Exception:
            pass


def inject_custom_css() -> None:
    """앱 전역 CSS"""
    st.markdown(
        dedent(
            """
            <style>
            /* --------------------------------------------------
               전체 기본
            -------------------------------------------------- */
            html, body, [class*="css"] {
                word-break: keep-all;
            }

            .stApp {
                background: #ffffff;
            }

            .block-container {
                padding-top: 2.65rem;
                padding-bottom: 2.2rem;
                max-width: 1240px;
                background: #ffffff;
            }

            h1, h2, h3, h4 {
                color: #2b3137;
                letter-spacing: -0.01em;
                font-weight: 800;
            }

            p, li, div {
                color: #4d6377;
            }

            /* --------------------------------------------------
               사이드바
            -------------------------------------------------- */
            [data-testid="stSidebar"] {
                min-width: 300px;
                max-width: 300px;
                background:
                    linear-gradient(180deg, #eef5fa 0%, #e7f1f8 55%, #e3eef6 100%);
                border-right: 1px solid #d5e3ee;
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 2.15rem;
                padding-bottom: 1.5rem;
                background: transparent;
            }

            /* --------------------------------------------------
               Hero
            -------------------------------------------------- */
            .hero-box {
                position: relative;
                overflow: hidden;
                background:
                    linear-gradient(135deg, #8eb7d1 0%, #a4c4db 42%, #c8dcea 100%);
                color: #ffffff;
                padding: 2.2rem 2rem 1.8rem 2rem;
                border-radius: 28px;
                margin-top: 0.85rem;
                margin-bottom: 1.45rem;
                box-shadow: 0 18px 40px rgba(123, 164, 195, 0.20);
                border: 1px solid rgba(255, 255, 255, 0.42);
            }

            .hero-box::before {
                content: "";
                position: absolute;
                top: -38px;
                right: -12px;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0.02) 70%);
                pointer-events: none;
            }

            .hero-box::after {
                content: "";
                position: absolute;
                left: 0;
                right: 0;
                bottom: 0;
                height: 7px;
                background: linear-gradient(90deg, rgba(255,255,255,0.00), rgba(255,255,255,0.55), rgba(255,255,255,0.00));
                opacity: 0.75;
            }

            .hero-eyebrow {
                font-size: 0.92rem;
                color: #edf6fb;
                opacity: 0.95;
                margin-bottom: 0.45rem;
                line-height: 1.5;
                font-weight: 700;
                letter-spacing: 0.01em;
            }

            .hero-title {
                position: relative;
                z-index: 1;
                font-size: 2.18rem;
                font-weight: 900;
                line-height: 1.28;
                margin-bottom: 0.7rem;
                letter-spacing: -0.025em;
                color: #ffffff;
            }

            .hero-desc {
                position: relative;
                z-index: 1;
                font-size: 1.01rem;
                color: #f5fafd;
                opacity: 0.98;
                line-height: 1.82;
                margin: 0;
                max-width: 920px;
            }

            /* --------------------------------------------------
               공통 카드
            -------------------------------------------------- */
            .section-card {
                background: linear-gradient(180deg, #fbfdff 0%, #f8fbfd 100%);
                border: 1px solid #dce8f1;
                border-radius: 20px;
                padding: 1.15rem 1.2rem;
                margin-bottom: 1rem;
                box-sizing: border-box;
                overflow: hidden;
                box-shadow: 0 8px 20px rgba(154, 183, 204, 0.08);
            }

            .mini-card {
                background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
                border: 1px solid #dde8f1;
                border-radius: 18px;
                padding: 1rem 1rem 0.85rem 1rem;
                box-shadow: 0 8px 18px rgba(140, 172, 195, 0.10);
                box-sizing: border-box;
                margin-bottom: 1rem;
                overflow: hidden;
                transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            }

            .mini-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 14px 26px rgba(140, 172, 195, 0.16);
                border-color: #cfddea;
            }

            .equal-card {
                min-height: 220px;
            }

            .mini-label {
                color: #7a97ad;
                font-size: 0.89rem;
                font-weight: 700;
                margin-bottom: 0.28rem;
                line-height: 1.45;
            }

            .mini-value {
                font-size: 2rem;
                font-weight: 900;
                color: #2c2f33;
                line-height: 1.12;
                margin-bottom: 0.22rem;
                letter-spacing: -0.025em;
            }

            .mini-sub {
                color: #7f99ae;
                font-size: 0.9rem;
                line-height: 1.5;
            }

            .feature-card-title {
                font-size: 1.14rem;
                font-weight: 800;
                color: #2f3338;
                line-height: 1.35;
                margin-top: 0.55rem;
                word-break: break-word;
                overflow-wrap: anywhere;
            }

            /* --------------------------------------------------
               사이드바 텍스트
            -------------------------------------------------- */
            .sidebar-title {
                font-size: 1.05rem;
                font-weight: 900;
                color: #5f92b8;
                margin-bottom: 0.35rem;
                letter-spacing: -0.01em;
            }

            .sidebar-sub {
                color: #7b96ab;
                font-size: 0.89rem;
                line-height: 1.58;
                margin-bottom: 1rem;
            }

            .flow-box {
                background: rgba(255, 255, 255, 0.68);
                border: 1px solid #d4e3ed;
                border-radius: 16px;
                padding: 0.95rem 1rem;
                margin-top: 1rem;
                line-height: 1.88;
                color: #6d8ca5;
                font-size: 0.92rem;
                box-shadow:
                    inset 0 1px 0 rgba(255,255,255,0.38),
                    0 6px 18px rgba(151, 181, 203, 0.08);
                backdrop-filter: blur(4px);
            }

            .flow-box b {
                color: #4d6f8c;
            }

            /* --------------------------------------------------
               사이드바 라디오 네비게이션
            -------------------------------------------------- */
            .stRadio > div {
                gap: 0.7rem;
            }

            div[role="radiogroup"] {
                display: flex;
                flex-direction: column;
                gap: 0.3rem;
            }

            div[role="radiogroup"] label {
                padding: 0 !important;
                margin: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
                border-radius: 0 !important;
            }

            div[role="radiogroup"] label:hover {
                background: transparent !important;
            }

            div[role="radiogroup"] input[type="radio"] {
                position: absolute !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }

            div[role="radiogroup"] label > div {
                background: transparent !important;
                box-shadow: none !important;
                border-radius: 12px !important;
                padding: 0.05rem 0 !important;
                transition: all 0.18s ease;
            }

            div[role="radiogroup"] [data-testid="stMarkdownContainer"] {
                background: transparent !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            div[role="radiogroup"] p {
                display: inline-block !important;
                margin: 0 !important;
                padding: 0.42rem 0.78rem !important;
                border-radius: 11px !important;
                background: transparent !important;
                color: #6f90a9 !important;
                font-weight: 800 !important;
                line-height: 1.4 !important;
                transition: all 0.18s ease;
                border: 1px solid transparent !important;
            }

            div[role="radiogroup"] label:hover p {
                background: rgba(143, 181, 210, 0.11) !important;
                color: #5f92b8 !important;
                border: 1px solid rgba(143, 181, 210, 0.12) !important;
            }

            div[role="radiogroup"] label:has(input:checked) p {
                background: linear-gradient(135deg, #8bb5d1 0%, #9ec2db 100%) !important;
                color: #ffffff !important;
                box-shadow: 0 6px 14px rgba(143, 181, 210, 0.20) !important;
                border: 1px solid rgba(255,255,255,0.20) !important;
            }

            div[role="radiogroup"] label:has(input:checked) p::before {
                content: "●";
                display: inline-block;
                margin-right: 0.45rem;
                font-size: 0.72rem;
                vertical-align: middle;
                color: #eef6fb;
            }

            /* --------------------------------------------------
               버튼 / 입력 / 선택 UI
            -------------------------------------------------- */
            .stButton > button {
                border-radius: 12px;
                border: 1px solid #d7e4ee;
                background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
                color: #4f6d89;
                font-weight: 700;
                box-shadow: 0 4px 12px rgba(145, 175, 198, 0.08);
            }

            .stButton > button:hover {
                border-color: #b9d0e0;
                color: #3f6483;
                box-shadow: 0 8px 18px rgba(145, 175, 198, 0.14);
            }

            .stSelectbox > div > div,
            .stMultiSelect > div > div,
            .stTextInput > div > div > input,
            .stNumberInput input,
            .stTextArea textarea {
                border-radius: 12px !important;
                border-color: #d7e4ee !important;
            }

            /* --------------------------------------------------
               Tabs / DataFrame / Expander / Alert
            -------------------------------------------------- */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.42rem;
                padding-bottom: 0.15rem;
            }

            .stTabs [data-baseweb="tab"] {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                border-radius: 11px 11px 0 0;
                color: #6f8ea7;
                font-weight: 700;
            }

            .stTabs [aria-selected="true"] {
                color: #4f6f8d !important;
                background: rgba(157, 193, 218, 0.12) !important;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid #dee9f2;
                box-shadow: 0 6px 16px rgba(151, 181, 203, 0.08);
            }

            div[data-testid="stExpander"] {
                border-radius: 14px;
                border: 1px solid #dce8f1;
                overflow: hidden;
                box-shadow: 0 6px 16px rgba(151, 181, 203, 0.07);
            }

            div[data-testid="stExpander"] summary {
                background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
            }

            div[data-testid="stInfo"] {
                border-radius: 14px;
                border: 1px solid #d7e5ef;
                background: #f5f9fc;
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
                border: 1px solid #dce8f1;
                padding: 0.75rem 0.9rem;
                border-radius: 16px;
                box-shadow: 0 8px 18px rgba(151, 181, 203, 0.08);
            }

            /* --------------------------------------------------
               Plotly / Charts 주변 여백 정리
            -------------------------------------------------- */
            [data-testid="stPlotlyChart"] {
                background: #ffffff;
                border: 1px solid #dce8f1;
                border-radius: 18px;
                padding: 0.45rem 0.45rem 0.2rem 0.45rem;
                box-shadow: 0 8px 18px rgba(151, 181, 203, 0.08);
            }

            /* --------------------------------------------------
               구분선
            -------------------------------------------------- */
            hr {
                border: none;
                height: 1px;
                background: linear-gradient(90deg, transparent, #d7e4ee, transparent);
                margin-top: 1.2rem;
                margin-bottom: 1.2rem;
            }

            /* --------------------------------------------------
               반응형
            -------------------------------------------------- */
            @media (max-width: 1100px) {
                .block-container {
                    max-width: 100%;
                    padding-top: 2.2rem;
                }

                .hero-title {
                    font-size: 1.9rem;
                }

                .hero-desc {
                    max-width: 100%;
                }
            }

            @media (max-width: 900px) {
                [data-testid="stSidebar"] {
                    min-width: 260px;
                    max-width: 260px;
                }

                .hero-box {
                    padding: 1.7rem 1.4rem 1.45rem 1.4rem;
                    border-radius: 22px;
                }

                .hero-title {
                    font-size: 1.65rem;
                }

                .hero-desc {
                    font-size: 0.98rem;
                }

                .equal-card {
                    min-height: unset;
                }
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """상단 히어로 영역"""
    st.markdown(
        dedent(
            """
            <div class="hero-box">
                <div class="hero-title">
                    Kaggle 데이터를 활용한 XAI 기반 고객 이탈 예측 프로젝트
                </div>
                <p class="hero-desc">
                    단순히 churn을 맞히는 데서 멈추지 않고,
                    어떤 고객이 왜 이탈 위험이 높은지를 해석하고
                    그 결과를 바탕으로 실행 가능한 고객 유지 전략까지 연결하는 것을 목표로 한 대시보드이다.
                </p>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """사이드바 네비게이션"""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">Dashboard Navigation</div>',
            unsafe_allow_html=True,
        )
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
            dedent(
                """
                <div class="flow-box">
                    <b>분석 흐름</b><br>
                    ① 문제 정의<br>
                    ② 이탈 고객 특성 파악<br>
                    ③ 예측 모델 비교 및 threshold 조정<br>
                    ④ XAI 기반 주요 이탈 신호 해석<br>
                    ⑤ 고객 유지 전략 제안
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    return page


def render_page(page: str) -> None:
    """선택된 페이지 렌더링"""
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


def main() -> None:
    apply_global_settings()
    inject_custom_css()
    render_header()
    selected_page = render_sidebar()
    render_page(selected_page)


if __name__ == "__main__":
    main()