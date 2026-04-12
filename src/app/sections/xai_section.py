from __future__ import annotations

import streamlit as st
from src.app.utils.load_data import load_xai_summary
from src.config.paths import XAI_OUTPUT_DIR


def _feature_card(label: str, value: str) -> str:
    return f"""
    <div class="mini-card" style="height: 120px;">
        <div class="mini-label">{label}</div>
        <div style="
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.35;
            margin-top: 0.55rem;
            word-break: break-word;
            overflow-wrap: anywhere;
        ">
            {value}
        </div>
    </div>
    """


def _center_note(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            text-align: center;
            color: #000000;
            font-size: 1rem;
            line-height: 1.7;
            margin-top: 0.35rem;
            margin-bottom: 1.1rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_centered_image_with_note(path, caption: str, note: str, width_ratio: float = 0.72) -> None:
    if not path.exists():
        return

    if width_ratio >= 0.95:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        side = (1 - width_ratio) / 2
        left, center, right = st.columns([side, width_ratio, side])
        with center:
            st.image(str(path), caption=caption, use_container_width=True)

    _center_note(note)


def render() -> None:
    st.markdown("## 이탈 요인 해석 및 고객 유지 전략")

    summary = load_xai_summary()

    top_features = []
    if not summary.empty and "feature" in summary.columns:
        top_features = summary["feature"].head(5).tolist()

    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top:0;">핵심 메시지</h4>
            <p style="line-height:1.8; margin-bottom:0;">
            설명 가능한 AI(XAI)는 모델이 높은 churn 확률을 부여한 이유를 해석하게 해준다.
            본 프로젝트에서는 주요 이탈 신호를 확인한 뒤, 이를 제품 사용성, 장애 경험,
            고객 지원 품질, 활성도 저하 같은 실무 개입 포인트와 연결했다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            _feature_card("대표 이탈 신호 1", top_features[0] if len(top_features) > 0 else "N/A"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _feature_card("대표 이탈 신호 2", top_features[1] if len(top_features) > 1 else "N/A"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _feature_card("대표 이탈 신호 3", top_features[2] if len(top_features) > 2 else "N/A"),
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["XAI 요약표", "SHAP 시각화", "유지 전략 제안"])

    with tab1:
        st.subheader("주요 이탈 신호")
        if summary.empty:
            st.info("xai_summary_report.csv 파일이 없습니다.")
        else:
            st.dataframe(summary.head(15), use_container_width=True)
            _center_note(
                "이 표는 모델이 churn 여부를 판단할 때 중요하게 반영한 변수를 정리한 결과이다. "
                "상위 변수일수록 단순 상관관계를 넘어 실제 예측 과정에서 더 큰 영향을 준 신호로 해석할 수 있다."
            )

    with tab2:
        st.subheader("SHAP 기반 해석")

        summary_path = XAI_OUTPUT_DIR / "shap_summary.png"
        bar_path = XAI_OUTPUT_DIR / "shap_bar.png"

        _show_centered_image_with_note(
            summary_path,
            "전체 변수 영향 방향과 크기",
            "이 그래프는 각 변수가 churn 확률을 높이는 방향인지 낮추는 방향인지, 그리고 그 영향의 크기가 어느 정도인지를 함께 보여준다. "
            "즉, 모델이 어떤 조건에서 고객을 더 위험하게 판단하는지 구조적으로 해석할 수 있게 해준다.",
            width_ratio=0.72,
        )

        _show_centered_image_with_note(
            bar_path,
            "평균 영향도 기준 상위 변수",
            "이 그래프는 전체 고객을 기준으로 평균 영향력이 큰 변수를 요약한 결과이다. "
            "따라서 개별 사례를 넘어서, 어떤 변수들이 전반적으로 churn 예측에 핵심적인 역할을 했는지 파악하는 데 적합하다.",
            width_ratio=0.62,
        )

    with tab3:
        st.subheader("고객 유지 전략 제안")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div class="section-card">
                    <h4 style="margin-top:0;">전략 1. 사용량 저하 고객 선제 케어</h4>
                    <p style="line-height:1.8; margin-bottom:0;">
                    최근 사용량 감소, 마지막 사용일 증가, 활성 구독 비율 저하는
                    이탈 전조로 해석할 수 있다.
                    따라서 로그인 빈도, 사용시간, 핵심 기능 사용량이 감소한 고객에게는
                    튜토리얼, 리마인드 메일, 기능 재활성화 캠페인을 우선 적용할 수 있다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="section-card">
                    <h4 style="margin-top:0;">전략 2. 오류 경험 고객 즉시 대응</h4>
                    <p style="line-height:1.8; margin-bottom:0;">
                    error_rate와 응답 지연은 서비스 품질에 대한 불만을 키울 수 있다.
                    따라서 장애 경험이 잦은 고객에게는
                    우선 대응 SLA, 전담 지원, 문제 복구 후 후속 안내가 필요하다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="section-card">
                    <h4 style="margin-top:0;">전략 3. 고위험 고객군 세분화 운영</h4>
                    <p style="line-height:1.8; margin-bottom:0;">
                    모든 churn 위험 고객을 동일하게 볼 수는 없다.
                    제품 사용 저하형, 장애 불만형, 지원 응답 불만형처럼
                    위험 원인을 유형화하면 훨씬 정교한 retention 액션 설계가 가능하다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="section-card">
                    <h4 style="margin-top:0;">핵심 메시지</h4>
                    <p style="line-height:1.7; margin-bottom:0;">
                    모델이 고객을 왜 이탈 위험으로 판단했는지를 설명하는 단계이다.<br>
                    주요 이탈 신호를 해석하고, 이를 실제 고객 유지 전략으로 연결하는 것이 목적이다.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
)