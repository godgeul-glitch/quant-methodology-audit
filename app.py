"""몽글 리포트 - Streamlit 대시보드 (화면 그리는 부분).

데이터 수집·모델·통계 검정은 core.py에 있습니다.
"""
from core import *          # noqa: F401,F403  (상수/함수 일괄 반입)
from core import _W         # 언더스코어 이름은 import * 로 안 넘어옴


# ==========================================
# 4. 웹 대시보드 및 전문가 UI
# ==========================================
st.set_page_config(page_title="몽글 리포트", layout="wide")

# ⭐ 신규: Claude Design에서 만든 "솜사탕 미니멀" 디자인 시스템을 그대로 이식.
# Streamlit 기본 다크 테마를 CSS로 완전히 덮어써서 크림/화이트 톤으로 전환합니다.
# (원본 cottoncandy-overrides.css의 색상 토큰을 Streamlit 컴포넌트 셀렉터에 맞게 적용)
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Quicksand:wght@500;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.css');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {BG_CREAM} !important;
    font-family: "Pretendard Variable", "Quicksand", sans-serif !important;
    color: {TEXT_DARK} !important;
}}
[data-testid="stHeader"] {{ background-color: transparent !important; }}
[data-testid="stSidebar"] {{
    background-color: {CARD_WHITE} !important;
    border-right: 1px solid {GRID_LIGHT};
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: "Pretendard Variable", "Poppins", sans-serif !important;
    color: {TEXT_DARK} !important;
    font-weight: 700 !important;
}}
p, span, div, label {{ color: {TEXT_DARK}; }}

/* 카드처럼 보이도록 - 컬럼 안 요소 여백/모서리 */
[data-testid="stMetric"] {{
    background: {CARD_WHITE};
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(74, 68, 88, 0.06);
}}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; font-size: 12px !important; }}
[data-testid="stMetricValue"] {{ color: {TEXT_DARK} !important; font-family: "Pretendard Variable", sans-serif !important; }}

/* 버튼: 필(pill) 형태 */
.stButton > button, .stFormSubmitButton > button {{
    border-radius: 999px !important;
    background: {ACCENT_PINK} !important;
    color: {ACCENT_PINK_DARK} !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    background: #FFB8D8 !important;
}}

/* 인풋: 항상 보이는 둥근 테두리
   Streamlit 1.4x+ 가 제공하는 공식 test id를 사용합니다.
   실제 DOM 구조: stTextInput > stTextInputRootElement > stTextInputField(<input>)
   테두리를 그리는 주체는 RootElement이므로 거기에 지정하고, 안쪽 input과
   BaseWeb 래퍼는 완전히 투명하게 두어 선이 겹치거나 잘리지 않게 합니다. */
[data-testid="stTextInputRootElement"],
[data-testid="stTextInput"] div[data-baseweb="input"] {{
    border: 1.5px solid #E7D9D1 !important;
    border-radius: 999px !important;
    background-color: {CARD_WHITE} !important;
    box-shadow: none !important;
}}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
    border-color: {ACCENT_PINK} !important;
    box-shadow: 0 0 0 3px rgba(255, 214, 232, 0.45) !important;
}}
[data-testid="stTextInput"] div[data-baseweb="base-input"] {{
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}
[data-testid="stTextInputField"],
[data-testid="stTextInput"] input {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background-color: transparent !important;
    color: {TEXT_DARK} !important;
    padding: 10px 16px !important;
}}
/* 폼 안에서 뜨는 "Press Enter to submit form" 안내 문구 숨김 */
[data-testid="stTextInput"] [data-testid="InputInstructions"],
[data-testid="InputInstructions"] {{ display: none !important; }}

/* 탭 → 필 형태 네비게이션으로 */
[data-testid="stTabs"] [role="tablist"] {{ gap: 6px; border-bottom: none !important; }}
[data-testid="stTabs"] [role="tab"] {{
    border-radius: 999px !important;
    background: transparent;
    color: {TEXT_BODY} !important;
    border: none !important;
    font-weight: 600;
    padding: 8px 18px;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: #F7F1FF !important;
    color: {ACCENT_LAVENDER_DARK} !important;
}}

/* 표 카드화 */
[data-testid="stDataFrame"] {{
    border-radius: 18px !important;
    overflow: hidden;
    border: 1px solid {GRID_LIGHT} !important;
}}

/* 알림 박스(성공/경고/에러/정보) - 파스텔 배경 + 진한 텍스트로 가독성 확보 */
div[data-testid="stAlertContainer"] {{ border-radius: 18px !important; border: none !important; }}
div[data-testid="stAlertContainer"]:has([data-testid="stIconMaterial"]) p {{ color: {TEXT_DARK} !important; }}
div.stSuccess, div[data-baseweb="notification"][kind="positive"] {{ background-color: {MINT_BG} !important; }}
div.stWarning, div[data-baseweb="notification"][kind="warning"] {{ background-color: {WATCH_BG} !important; }}
div.stError, div[data-baseweb="notification"][kind="negative"] {{ background-color: {UP_COLOR_BG} !important; }}
div.stInfo, div[data-baseweb="notification"][kind="info"] {{ background-color: #F7F1FF !important; }}

/* 캡션(부가 설명) 색상 - ⭐ [사용자 요청] 크림색 배경에 연보라(TEXT_MUTED)가
   묻혀 안 읽힌다는 피드백으로 더 진한 톤 + 굵게로 조정 */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color: {TEXT_BODY} !important;
    font-weight: 600 !important;
}}

/* 확장 패널(expander) - 카드 스타일로 */
[data-testid="stExpander"] {{
    border-radius: 18px !important;
    border: 1px solid {GRID_LIGHT} !important;
    background: {CARD_WHITE} !important;
}}
[data-testid="stExpander"] summary {{ color: {TEXT_DARK} !important; font-weight: 600 !important; }}

/* 구분선 */
hr {{ border-color: {GRID_LIGHT} !important; }}

/* 표 래퍼: 헤더 폰트/색만 조정 가능 (내부는 캔버스 렌더링이라 셀 폰트는 제한적) */
[data-testid="stDataFrame"] [role="columnheader"] {{
    background-color: {NEUTRAL_100} !important;
    color: {TEXT_BODY} !important;
    font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)

url_ticker = st.query_params.get("ticker", "NVDA")
if "selected_ticker" not in st.session_state or st.session_state["selected_ticker"] != url_ticker:
    st.session_state["selected_ticker"] = url_ticker

st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">'
    f'<div style="width:36px;height:36px;border-radius:999px;background:{ACCENT_PINK};'
    f'display:flex;align-items:center;justify-content:center;font-size:18px;">🍬</div>'
    f'<div style="font-family:\'Pretendard Variable\',sans-serif;font-weight:700;font-size:20px;'
    f'color:{TEXT_DARK};">몽글 리포트</div></div>',
    unsafe_allow_html=True
)

# ⭐ 전역 분석 설정 제거: 알고리즘이 검증된 최적값을 자동 적용합니다.
start_date = (pd.Timestamp.today() - pd.DateOffset(years=AUTO_YEARS)).strftime("%Y-%m-%d")
prob_threshold = AUTO_PROB_THRESHOLD
half_kelly_frac = AUTO_HALF_KELLY
user_kelly_cap = AUTO_KELLY_CAP

st.sidebar.markdown(
    f'<div style="font-size:12px;font-weight:600;color:{TEXT_MUTED};padding-left:4px;margin-bottom:4px;">종목 검색</div>',
    unsafe_allow_html=True
)
with st.sidebar.form(key="search_form", border=False):
    user_input = st.text_input(
        "종목명 또는 티커를 입력하세요",
        st.session_state["selected_ticker"],
        help="예) 애플, AAPL, 삼성전자, 005930",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button(label="분석하기")

if submit_button:
    clean_input = user_input.strip().upper()
    if clean_input.isdigit() and len(clean_input) == 6:
        clean_input += ".KS"
    reverse_map = {v: k for k, v in company_names.items()}
    mapped_ticker = reverse_map.get(clean_input, clean_input)
    st.session_state["selected_ticker"] = mapped_ticker
    st.query_params["ticker"] = mapped_ticker
    st.rerun()

# ⭐ 디자인의 사이드바 하단 안내 박스: 상단의 큰 배너 대신 옅은 박스로 은은하게 고지
st.sidebar.markdown(
    f'<div style="margin-top:24px;padding:14px;border-radius:18px;background:{NEUTRAL_100};'
    f'font-size:11.5px;line-height:1.6;color:{TEXT_MUTED};">'
    f'이 결과는 투자 권유가 아니며, 참고용 자료입니다.</div>',
    unsafe_allow_html=True
)

def _summary_card(bg, icon, label, count, suffix="개"):
    """요약 카드 HTML. tab2(종목 개수)와 tab3(AUC/표본수) 양쪽에서 쓰이므로
    모듈 레벨에 정의하고 접미사를 인자로 받습니다."""
    return f"""
<div style="background:{bg};border-radius:24px;padding:20px 22px;display:flex;align-items:center;gap:14px;">
  <div style="font-size:24px;">{icon}</div>
  <div>
    <div style="font-size:12.5px;color:{TEXT_BODY};">{label}</div>
    <div style="font-family:'Pretendard Variable',sans-serif;font-weight:700;font-size:22px;color:{TEXT_DARK};">{count}{suffix}</div>
  </div>
</div>"""

df_all = download_all_data(tuple(ALL_TICKERS), start_date)
macro_df = download_macro_data(start_date)
macro_prepared = prepare_macro(macro_df)

# ⭐ 시장 국면 감지 및 전역 배너
_regime = compute_market_regime(macro_prepared)
_regime_mult = _regime.get("threshold_mult", 1.0) if _regime else 1.0
if _regime:
    _rg_bg = {"평온": MINT_BG, "주의": WATCH_BG, "위기": UP_COLOR_BG}.get(_regime["state"], NEUTRAL_100)
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;background:{_rg_bg};border-radius:24px;
     padding:14px 20px;margin-bottom:14px;">
  <div style="font-size:24px;line-height:1;">{_regime['emoji']}</div>
  <div style="flex:1;">
    <div style="font-family:'Pretendard Variable',sans-serif;font-weight:700;font-size:15px;color:{TEXT_DARK};">
      시장 국면: {_regime['state']}
    </div>
    <div style="font-size:12.5px;color:{TEXT_BODY};margin-top:2px;">{_regime['desc']}</div>
  </div>
  <div style="text-align:right;font-size:11.5px;color:{TEXT_MUTED};white-space:nowrap;">
    VIX {_regime['vix']:.0f} · 고점대비 {_regime['drawdown']:+.1f}%<br>변동성 {_regime['volatility']:.0f}%
  </div>
</div>
""", unsafe_allow_html=True)
    if _regime["state"] != "평온":
        st.caption(
            f"ℹ️ 지금은 '{_regime['state']}' 국면이라 관심 신호 기준을 평소의 {_regime_mult:.1f}배로 "
            f"자동으로 높였습니다. 외부 충격은 예측할 수 없지만, 충격이 진행 중인지는 감지할 수 있어요."
        )

tab1, tab2, tab3 = st.tabs(["📄 종목 리포트", "📊 단기 랭킹", "💎 가치투자 랭킹"])

with tab1:
    ticker = st.session_state["selected_ticker"]
    if df_all.empty and ticker in ALL_TICKERS:
        st.warning("데이터 서버 통신 지연으로 일괄 조회가 불가능합니다. 개별 조회를 수행합니다.")
        
    with st.spinner(f"{ticker} 추론 및 OOS 자본곡선 백테스팅 중..."):
        try:
            df_ticker, is_fallback = process_ticker_data(df_all, macro_prepared, ticker, start_date)
            r = run_model_pipeline(
                df_ticker, ticker, kelly_cap=user_kelly_cap, is_macro_fallback=is_fallback,
                prob_threshold=prob_threshold, half_kelly_frac=half_kelly_frac
            ) if not df_ticker.empty else None

            if not r:
                st.error(f"⚠️ '{ticker}' 종목의 데이터를 가져오지 못했습니다. 종목명이나 티커를 다시 확인해 주세요.")
            else:
                cv = r["cv"]
                # 개별 분석 시 p-value 기준 1차 필터링
                is_stat_sig = cv["auc_pvalue"] <= 0.10 if not pd.isna(cv["auc_pvalue"]) else False
                signal = classify_signal(r["ev"], r["prob_profit"], r["prob_loss"], r["atr_pct"], r["vix"], r["market_relative"], is_fdr_passed=is_stat_sig, is_fallback=is_fallback, prob_threshold_pct=prob_threshold * 100.0, regime_mult=_regime_mult)

                kor_name = company_names.get(ticker, ticker)
                display_name = kor_name if kor_name != ticker else ticker
                market_tag = "국내" if str(ticker).endswith(".KS") else "해외"

                if "관심" in signal:
                    banner_bg, banner_icon = MINT_BG, "🔵"
                elif "조건 미달" in signal:
                    banner_bg, banner_icon = NEUTRAL_100, "🔴"
                else:
                    banner_bg, banner_icon = WATCH_BG, "🟡"

                # ⭐ 직관성 개선 + Claude Design 반영: 헤더 행(이름/티커/구분/가격) + 결론 배너 카드
                _stab = f"최근 {r['stability_window']}일 중 {r['stable_days']}일 동일 신호"
                if r["stable_days"] <= 1 and "관심" in signal:
                    _stab += " · 오늘 처음 나온 신호이니 며칠 더 지켜보세요"
                _headline = signal.split(" ", 1)[-1] if " " in signal else signal
                _fund = get_fundamentals(ticker)
                _fund_txt = format_per_pbr(_fund)

                st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
  <div style="font-family:'Pretendard Variable',sans-serif;font-size:26px;font-weight:700;color:{TEXT_DARK};">{display_name}</div>
  <div style="font-size:13px;color:{TEXT_MUTED};">{ticker}</div>
  <div style="background:#F7F1FF;color:{ACCENT_LAVENDER_DARK};padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;">{market_tag}</div>
  <div style="background:{NEUTRAL_200};color:{TEXT_BODY};padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;">PER/PBR {_fund_txt}</div>
  <div style="margin-left:auto;font-family:'Pretendard Variable',sans-serif;font-size:22px;font-weight:700;color:{TEXT_DARK};">{format_price(r['current_price'], ticker)}</div>
</div>
<div style="font-size:13px;color:{TEXT_MUTED};margin-bottom:14px;">{company_descriptions.get(ticker, '기업 정보가 등록되지 않았습니다.')}</div>
<div style="display:flex;align-items:center;gap:14px;background:{banner_bg};border-radius:24px;padding:18px 22px;box-shadow:0 2px 8px rgba(74,68,88,0.06);margin-bottom:16px;">
  <div style="font-size:26px;line-height:1;">{banner_icon}</div>
  <div style="display:flex;flex-direction:column;gap:2px;">
    <div style="font-family:'Pretendard Variable',sans-serif;font-weight:700;font-size:16px;color:{TEXT_DARK};">{_headline} — {r['easy_verdict']}</div>
    <div style="font-size:13px;color:{TEXT_BODY};">{_stab}</div>
  </div>
</div>
""", unsafe_allow_html=True)

                if is_fallback:
                    st.caption("※ 시장 지표 서버 지연으로 기본값이 적용되어 다소 보수적으로 계산되었습니다.")

                # ⭐ Claude Design 반영: 반원형 Plotly 게이지 → 카드 안 원형 링 게이지(SVG)로 교체
                def _ring_gauge(value, color, label, desc):
                    dash = f"{(max(0, min(100, value)) / 100) * 314:.1f} 314"
                    return f"""
<div style="background:{CARD_WHITE};border-radius:24px;padding:20px 22px;box-shadow:0 2px 8px rgba(74,68,88,0.06);display:flex;align-items:center;gap:20px;">
  <div style="position:relative;width:104px;height:104px;flex:none;">
    <svg width="104" height="104" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="50" fill="none" stroke="{NEUTRAL_200}" stroke-width="12"></circle>
      <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"
        transform="rotate(-90 60 60)" stroke-dasharray="{dash}"></circle>
    </svg>
    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
      font-family:'Pretendard Variable',sans-serif;font-size:24px;font-weight:700;color:{TEXT_DARK};">{value:.0f}</div>
  </div>
  <div>
    <div style="font-family:'Pretendard Variable',sans-serif;font-weight:700;font-size:15px;color:{TEXT_DARK};">{label}</div>
    <div style="font-size:12.5px;color:{TEXT_MUTED};margin-top:4px;">{desc}</div>
  </div>
</div>"""

                # ⭐ [사용자 요청] RSI 설명과 같은 형식으로 통일: 라벨에 현재 상태(이모지 포함)를
                # 괄호로 붙이고, 아래 설명은 방향성 서술 대신 기준선(20/80) 숫자로.
                gauge_col1, gauge_col2 = st.columns(2)
                with gauge_col1:
                    st.markdown(_ring_gauge(r['stoch_k'], DOWN_COLOR, f"스토캐스틱 %K({r['market_state']})", "단기 매수·매도 압력 지표"), unsafe_allow_html=True)
                    st.caption("단기 매수·매도 압력을 보는 지표입니다. 20 이하면 과매도, 80 이상이면 과매수로 봅니다.")
                with gauge_col2:
                    st.markdown(_ring_gauge(r['rsi'], ACCENT_LAVENDER, "RSI", "과매수·과매도 판단 지표"), unsafe_allow_html=True)
                    st.caption("가격이 과열됐는지 보는 지표입니다. 70 이상이면 과열, 30 이하면 침체로 봅니다.")


                fmt_price = format_price(r['current_price'], ticker)
                fmt_target = format_price(r['target_price'], ticker)
                fmt_loss = format_price(r['stop_loss_price'], ticker)

                def _metric_card(label, value, sub=None, progress=None, value_color=TEXT_DARK):
                    sub_html = f'<div style="font-size:11.5px;color:{TEXT_MUTED};margin-top:4px;">{sub}</div>' if sub else ""
                    bar_html = ""
                    if progress is not None:
                        bar_html = f"""
<div style="margin-top:10px;height:6px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
  <div style="height:100%;border-radius:999px;background:{ACCENT_PINK};width:{progress:.0f}%;"></div>
</div>"""
                    return f"""
<div style="background:{CARD_WHITE};border-radius:24px;padding:20px 22px;box-shadow:0 2px 8px rgba(74,68,88,0.06);height:100%;">
  <div style="font-size:12px;color:{TEXT_MUTED};margin-bottom:8px;">{label}</div>
  <div style="font-family:'Pretendard Variable',sans-serif;font-size:26px;font-weight:700;color:{value_color};">{value}</div>
  {sub_html}{bar_html}
</div>"""

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(_metric_card("현재 주가", fmt_price), unsafe_allow_html=True)
                with m2:
                    st.markdown(_metric_card("오를 가능성", f"{r['prob_profit']:.0f}%",
                                              f"내릴 가능성 {r['prob_loss']:.0f}%",
                                              progress=r['prob_profit'], value_color=ACCENT_PINK_DARK), unsafe_allow_html=True)
                with m3:
                    st.markdown(_metric_card("🎯 목표가 (여기까지 오르면 익절)", fmt_target,
                                              f"+{r['profit_pct']:.1f}%"), unsafe_allow_html=True)
                with m4:
                    st.markdown(_metric_card("🛑 손절가 (여기까지 내리면 손절)", fmt_loss,
                                              f"-{r['loss_pct']:.1f}%"), unsafe_allow_html=True)

                st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                st.caption(
                    f"※ 목표가·손절가는 이 종목의 평소 하루 변동폭을 기준으로 자동 계산됩니다. "
                    f"실제 매수는 다음 거래일 시작 가격 기준이며, 밤사이 가격이 평균 {r['avg_gap_5']:.1f}% 정도 튀는 종목입니다."
                )
                st.divider()

                # XAI 섹션 (초보자용)
                st.subheader("🧠 AI가 이렇게 판단한 이유")

                # ⭐ Claude Design 반영: 텍스트+별도 막대그래프 2단 레이아웃 →
                # 한 카드 안에 설명 + 라벨형 막대 리스트로 통합. 디자인 목업의
                # "외국인 순매수 전환" 같은 항목은 실제 모델에 없는 예시 데이터라
                # 형식(라벨+막대)만 가져오고, 값은 실제 계산된 피처 중요도를 사용합니다.
                reason_rows_html = ""
                if r["feature_importances"]:
                    fi_sorted = sorted(r["feature_importances"].items(), key=lambda x: x[1], reverse=True)[:5]
                    for i, (feat, val) in enumerate(fi_sorted):
                        color = REASON_COLORS[i % len(REASON_COLORS)]
                        label = FEATURE_EASY_NAMES.get(feat, feat)
                        pct = max(0.0, val * 100)
                        reason_rows_html += f"""
<div style="display:flex;align-items:center;gap:12px;">
  <div style="width:150px;flex:none;font-size:13px;color:{TEXT_BODY};">{label}</div>
  <div style="flex:1;height:10px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
    <div style="height:100%;border-radius:999px;background:{color};width:{pct:.0f}%;"></div>
  </div>
  <div style="width:34px;text-align:right;font-size:12.5px;color:{TEXT_MUTED};">{pct:.0f}%</div>
</div>"""

                st.markdown(f"""
<div style="background:{CARD_WHITE};border-radius:24px;padding:20px 22px;box-shadow:0 2px 8px rgba(74,68,88,0.06);">
{r['xai_text']}
<div style="display:flex;flex-direction:column;gap:12px;margin-top:16px;">
{reason_rows_html}
</div>
</div>
""", unsafe_allow_html=True)
                st.caption("막대가 길수록 AI가 그 정보를 더 많이 보고 판단했다는 뜻입니다. (실제 모델이 계산한 값입니다)")

                # 백테스트 및 표본 투명성
                with st.expander("🧪 이 AI가 과거에 얼마나 잘 맞췄는지 보기"):
                    if pd.isna(cv["pooled_auc"]): 
                        st.warning("데이터가 부족해서 과거 성적을 계산할 수 없습니다.")
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("예측 실력 점수", f"{cv['pooled_auc']:.2f}",
                                  help="0.5는 동전 던지기 수준, 1.0에 가까울수록 잘 맞춘다는 뜻입니다.")
                        c2.metric("과거 적중률", f"{cv['pooled_precision']*100:.0f}%",
                                  help="AI가 '오른다'고 했을 때 실제로 목표가에 먼저 닿은 비율입니다.")
                        c3.metric("과거 누적 성과", f"{cv['cum_ret']:+.1f}%",
                                  help="과거 데이터로 이 신호만 따라 투자했다면 나왔을 결과입니다.")
                        c4.metric("적정 투자 비중", f"{r['kelly_pct']:.1f}%",
                                  help="이 종목만 투자할 때의 이론적 상한입니다. 여러 종목을 동시에 담을 때는 랭킹 탭의 '포트폴리오 배분'을 확인하세요.")

                        if not cv["equity_df"].empty:
                            st.line_chart(cv["equity_df"]["Equity"], height=200, **_W)
                            st.caption(
                                "📈 위 그래프는 과거에 이 AI 신호만 따라 투자했을 때 원금이 어떻게 변했을지 보여줍니다 "
                                "(수수료와 체결 오차 포함). 과거 성적이 미래를 보장하지는 않습니다."
                            )

                st.subheader("📈 최근 주가 흐름")

                # ⭐ 차트 기간/단위 선택
                ch_c1, ch_c2 = st.columns([1.3, 1])
                with ch_c1:
                    _period = st.radio(
                        "기간", ["1개월", "3개월", "6개월", "1년", "전체"],
                        index=2, horizontal=True, label_visibility="collapsed", key="chart_period"
                    )
                with ch_c2:
                    _interval = st.radio(
                        "단위", ["일", "주", "월"],
                        index=0, horizontal=True, label_visibility="collapsed", key="chart_interval"
                    )

                _full = r["df_tail"]
                _pdays = {"1개월": 22, "3개월": 66, "6개월": 126, "1년": 252, "전체": len(_full)}[_period]
                chart_data = _full.tail(_pdays)

                # 주/월 단위는 OHLC 규칙에 맞게 재집계 (시가=첫값, 고가=최댓값, 저가=최솟값, 종가=마지막값)
                if _interval in ("주", "월"):
                    rule = "W" if _interval == "주" else "ME"
                    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
                    for extra in ["SMA_20", "BB_Upper", "BB_Lower", "ATR_14", "MACD_Hist"]:
                        if extra in chart_data.columns:
                            agg[extra] = "last"
                    chart_data = chart_data.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
                st.markdown(f"""
<div style="display:flex;gap:16px;font-size:11.5px;color:{TEXT_MUTED};margin-bottom:6px;">
  <span><span style="display:inline-block;width:9px;height:9px;border-radius:999px;background:{UP_COLOR};margin-right:5px;"></span>상승</span>
  <span><span style="display:inline-block;width:9px;height:9px;border-radius:999px;background:{DOWN_COLOR};margin-right:5px;"></span>하락</span>
  <span><span style="display:inline-block;width:9px;height:9px;border-radius:999px;background:{ACCENT_LINE};margin-right:5px;"></span>이동평균</span>
  <span><span style="display:inline-block;width:9px;height:9px;border-radius:999px;background:{ACCENT_LAVENDER};margin-right:5px;"></span>변동범위</span>
</div>
""", unsafe_allow_html=True)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28])

                # ⭐ 눈이 편한 디자인으로 전면 교체:
                # - 볼린저 밴드: 회색 점선 2개 → 라벤더 반투명 음영 하나로 (덜 어지러움)
                # - 캔들: 원색 빨강/파랑 → 부드러운 코랄/하늘색
                # - 배경: 다크 네이비 → 화이트 (앱 전체의 크림/화이트 톤과 통일)
                # - 목표가/손절가 점선: 얇고 옅게, 후버 시에만 두드러지도록
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['BB_Upper'], mode='lines',
                    line=dict(width=0), showlegend=False, hoverinfo='skip'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['BB_Lower'], mode='lines',
                    line=dict(width=0), fill='tonexty', fillcolor=BAND_FILL,
                    name='평소 변동 범위', hoverinfo='skip'
                ), row=1, col=1)

                fig.add_trace(go.Candlestick(
                    x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
                    low=chart_data['Low'], close=chart_data['Close'],
                    increasing_line_color=UP_COLOR, increasing_fillcolor=UP_COLOR,
                    decreasing_line_color=DOWN_COLOR, decreasing_fillcolor=DOWN_COLOR,
                    line=dict(width=1), name='주가'
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['SMA_20'],
                    line=dict(color=ACCENT_LINE, width=1.8), name='20일 평균'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['Close'] + (chart_data['ATR_14'] * 1.5),
                    line=dict(color=UP_COLOR, width=1, dash='dot'), name='목표가', opacity=0.6
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=chart_data['Close'] - (chart_data['ATR_14'] * 1.5),
                    line=dict(color=DOWN_COLOR, width=1, dash='dot'), name='손절가', opacity=0.6
                ), row=1, col=1)

                macd_colors = [UP_COLOR if val >= 0 else DOWN_COLOR for val in chart_data['MACD_Hist']]
                fig.add_trace(go.Bar(
                    x=chart_data.index, y=chart_data['MACD_Hist'],
                    marker_color=macd_colors, marker_line_width=0, name='MACD'
                ), row=2, col=1)

                fig.update_layout(
                    height=560, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor=CARD_WHITE, plot_bgcolor=CARD_WHITE,
                    font=dict(color=TEXT_DARK, size=12),
                    hovermode='x unified',
                )
                fig.update_xaxes(showgrid=False, showline=False, zeroline=False)
                fig.update_yaxes(showgrid=True, gridcolor=GRID_LIGHT, gridwidth=1, zeroline=False)
                st.plotly_chart(fig, **_W)

                sma_diff = (r['current_price'] - r['sma_20']) / r['sma_20'] * 100
                st.caption(
                    f"💡 **차트 보는 법:** 코랄색 봉은 오른 날, 하늘색 봉은 내린 날입니다. "
                    f"금색 선은 최근 20일 평균 가격이고, 지금 주가는 그 평균보다 `{sma_diff:+.1f}%` 위치에 있습니다. "
                    f"연보라 음영을 벗어나면 평소보다 크게 움직인 구간, 옅은 점선은 목표가·손절가입니다. "
                    f"이 종목은 하루에 보통 `{r['atr_pct']:.1f}%` 정도 움직입니다."
                )
        except Exception as e:
            st.error("분석 중 문제가 발생했습니다. 다른 종목으로 다시 시도해 주세요.")
            with st.expander("기술적 오류 내용 보기"):
                st.code(traceback.format_exc())

with tab2:
    st.subheader("🏆 전체 종목 한눈에 비교")

    st.caption(
        "🔵 관심 · 🟡 지켜보기 · 🔴 조건 미달 — 종목명을 누르면 자세한 분석으로 이동합니다."
    )

    run_ranking = st.button("전체 종목 분석하기")

    if run_ranking:
        if df_all.empty:
            st.error("주가 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            my_bar = st.progress(0, text="분석을 준비하고 있습니다...")
            raw_results = []

            # ⭐ Fix 2: ThreadPoolExecutor 워커 스레드는 기본적으로 Streamlit의
            # ScriptRunContext를 갖고 있지 않습니다. 메인 스레드의 컨텍스트를
            # 미리 받아와 각 워커 스레드에 명시적으로 전파합니다.
            _main_ctx = get_script_run_ctx() if get_script_run_ctx else None
            _panels = {}   # 횡단면 모델용 종목별 피처 패널 수집

            def run_analysis(tk):
                if _main_ctx is not None and add_script_run_ctx is not None:
                    add_script_run_ctx(threading.current_thread(), _main_ctx)
                try:
                    # ⭐ 최적화: fast_mode로 불필요한 네트워크 재시도 제거
                    df_ticker, is_fb = process_ticker_data(df_all, macro_prepared, tk, start_date, fast_mode=True)
                    if df_ticker.empty:
                        return None
                    _panels[tk] = df_ticker.assign(Ticker=tk)
                    r = run_model_pipeline(
                        df_ticker, tk, kelly_cap=user_kelly_cap, is_macro_fallback=is_fb,
                        prob_threshold=prob_threshold, half_kelly_frac=half_kelly_frac
                    )
                    if not r:
                        return None
                    
                    kor_name = company_names.get(tk, tk) 
                    fund = get_fundamentals(tk)
                    return {
                        "종목명": f"/?ticker={tk}&name={kor_name}",
                        "Ticker": tk,
                        "ev": r["ev"],
                        "prob_profit": r["prob_profit"],
                        "prob_loss": r["prob_loss"],
                        "atr_pct": r["atr_pct"],
                        "vix": r["vix"],
                        "market_relative": r["market_relative"],
                        "is_fb": is_fb,
                        "오를 가능성": round(r["prob_profit"]),
                        "신호 유지일": f"{r['stable_days']}/{r['stability_window']}일",
                        "현재 상태": r["market_state"],
                        "PER/PBR": format_per_pbr(fund),
                        "1회 기댓값": round(r["ev"], 2),
                        "위험 대비 효율": round(r["reward_score"], 2),
                        "적정 비중": round(r["kelly_pct"], 1),
                        "AI 예측 실력": round(r["cv"]["pooled_auc"], 2) if not pd.isna(r["cv"]["pooled_auc"]) else np.nan,
                        "auc_pvalue": r["cv"]["auc_pvalue"],
                        "과거 누적 성과": f"{r['cv']['cum_ret']:+.1f}%",
                        "현재가": format_price(r["current_price"], tk),
                    }
                except Exception as e:
                    logging.warning(f"Analysis failed for ticker {tk}: {e}")
                    return None

            # ⭐ 최적화: 워커 수를 늘려 네트워크/연산 대기 시간을 겹쳐서 처리
            _workers = min(12, max(4, len(ALL_TICKERS) // 4))
            completed = 0
            _t0 = time.time()
            with ThreadPoolExecutor(max_workers=_workers) as executor:
                future_to_ticker = {executor.submit(run_analysis, tk): tk for tk in ALL_TICKERS}
                for future in as_completed(future_to_ticker):
                    res = future.result()
                    if res:
                        raw_results.append(res)
                    completed += 1
                    _pct = completed / len(ALL_TICKERS)
                    _eta = (time.time() - _t0) / max(_pct, 0.01) * (1 - _pct)
                    my_bar.progress(
                        int(_pct * 100),
                        text=f"분석 중... {completed}/{len(ALL_TICKERS)} 종목 (약 {int(_eta)}초 남음)"
                    )
            my_bar.empty()

            if raw_results:
                result_df = pd.DataFrame(raw_results)

                # ⭐ 횡단면 모델 점수 산출 (종목별 모델의 표본 부족 문제 보완)
                _cs = None
                try:
                    if len(_panels) >= 5:
                        _cs_panel = pd.concat(_panels.values())
                        with st.spinner("종목 간 비교 모델 학습 중..."):
                            _cs = run_cross_sectional_model(_cs_panel)
                except Exception as e:
                    logging.warning(f"CS model step failed: {e}")
                if _cs:
                    result_df["종목 비교 점수"] = result_df["Ticker"].map(_cs["scores"]).round(0)
                    st.session_state["cs_meta"] = {"auc": _cs["auc"], "pvalue": _cs["pvalue"], "n_eval": _cs["n_eval"]}
                else:
                    st.session_state["cs_meta"] = None
                
                # ⭐ Step 1: FDR Step-up 보정 수행 ⭐
                valid_df = result_df.dropna(subset=["auc_pvalue"]).sort_values("auc_pvalue").reset_index()
                m = len(valid_df)
                q_thresh = 0.10
                max_passed_rank = 0

                for rank, row in valid_df.iterrows():
                    i = rank + 1
                    if row["auc_pvalue"] <= (i / m) * q_thresh:
                        max_passed_rank = i

                fdr_passed_tickers = set(valid_df.iloc[:max_passed_rank]["Ticker"]) if max_passed_rank > 0 else set()

                # ⭐ Step 2: FDR 통과 여부를 classify_signal에 직접 전달하여 의사결정 결합 ⭐
                result_df["신뢰도"] = result_df["Ticker"].apply(
                    lambda tk: "✅ 검증됨" if tk in fdr_passed_tickers else "⚠️ 낮음"
                )
                result_df["상태"] = result_df.apply(
                    lambda row: classify_signal(
                        row["ev"], row["prob_profit"], row["prob_loss"], row["atr_pct"],
                        row["vix"], row["market_relative"], is_fdr_passed=(row["Ticker"] in fdr_passed_tickers),
                        is_fallback=row["is_fb"], prob_threshold_pct=prob_threshold * 100.0, regime_mult=_regime_mult
                    ), axis=1
                )
                
                result_df["AI 예측 실력"] = result_df["AI 예측 실력"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

                # ⭐ 신호 색깔 원(dot)을 별도 컬럼이 아니라 순번(인덱스) 자체에 합쳐서 표시
                result_df["신호"] = result_df["상태"].str.split(" ").str[0]

                # ⭐ Fix 11: 순위를 '오늘 점수' 하나로 매기면 매일 1위가 36위로
                # 떨어지는 일이 생깁니다. 스윙 투자에 맞게 최근 며칠간의 점수를
                # 평균해서 순위를 산정하고, 어제 대비 변동도 함께 보여줍니다.
                _hist = load_history()
                _today = date.today().isoformat()
                _today_scores = dict(zip(result_df["Ticker"], result_df["위험 대비 효율"]))

                # 과거 날짜(오늘 제외)를 최신순으로 정렬
                _past_dates = sorted([d for d in _hist.keys() if d != _today], reverse=True)
                _use_dates = _past_dates[:max(0, RANK_HISTORY_DAYS - 1)]

                def _avg_score(tk):
                    vals = [_today_scores.get(tk, 0.0)]
                    for d in _use_dates:
                        rec = _hist.get(d, {}).get(tk)
                        if isinstance(rec, dict) and rec.get("score") is not None:
                            vals.append(float(rec["score"]))
                    return float(np.mean(vals))

                result_df["평균 점수"] = result_df["Ticker"].apply(_avg_score).round(2)

                # 직전 실행 시점의 순위(있으면)와 비교
                _prev_rank = {}
                if _past_dates:
                    _prev_rank = {
                        tk: rec.get("rank")
                        for tk, rec in _hist.get(_past_dates[0], {}).items()
                        if isinstance(rec, dict)
                    }

                # ⭐ Step 3: FDR 검증 통과 종목을 최우선 정렬 (2차 기준은 며칠 평균 점수) ⭐
                priority_map = {"🔵 관심": 1, "🟡 지켜보기": 2, "🟡 지켜보기 (확률 부족)": 3, "🟡 지켜보기 (검증 부족)": 4, "🔴 조건 미달": 5}
                result_df["우선순위"] = result_df["상태"].map(lambda x: priority_map.get(x, 3))
                
                result_df = result_df.sort_values(by=["우선순위", "평균 점수"], ascending=[True, False]).reset_index(drop=True)

                def _rank_delta(tk, new_rank):
                    old = _prev_rank.get(tk)
                    if not isinstance(old, (int, float)):
                        return "신규"
                    diff = int(old) - int(new_rank)
                    if diff > 0:
                        return f"▲ {diff}"
                    if diff < 0:
                        return f"▼ {abs(diff)}"
                    return "-"

                result_df["순위 변동"] = [
                    _rank_delta(tk, i + 1) for i, tk in enumerate(result_df["Ticker"])
                ]

                # ⭐ Fix 12 [안전 문제]: 종목별 켈리 비중을 그대로 나열하면 관심 종목이
                # 5개일 때 25% x 5 = 125%가 되어 과도한 레버리지를 유발합니다.
                # (코드 주석과 안내문에는 '단순 합산 금지'라고만 적혀 있었고 실제 조정은
                #  없었습니다.) 관심 종목들의 비중 합이 TOTAL_PORTFOLIO_CAP을 넘으면
                # 비율을 유지한 채 비례 축소해 실제로 실행 가능한 배분을 제시합니다.
                _is_pick = result_df["상태"].str.contains("관심")
                _raw_w = result_df["적정 비중"].where(_is_pick, 0.0).astype(float)
                _tot = float(_raw_w.sum())
                if _tot > TOTAL_PORTFOLIO_CAP and _tot > 0:
                    _scaled = _raw_w * (TOTAL_PORTFOLIO_CAP / _tot)
                else:
                    _scaled = _raw_w
                result_df["포트폴리오 배분"] = _scaled.round(1)
                st.session_state["ranking_alloc_total"] = round(float(_scaled.sum()), 1)
                st.session_state["ranking_alloc_raw_total"] = round(_tot, 1)

                # 오늘 점수/순위를 이력에 저장 (같은 날 재실행 시 덮어씀)
                _hist[_today] = {
                    tk: {"score": float(sc), "rank": int(i + 1)}
                    for i, (tk, sc) in enumerate(zip(result_df["Ticker"], result_df["위험 대비 효율"]))
                }
                save_history(_hist)
                _hist_days = len(_use_dates) + 1
                # ⭐ 순위 변동을 괄호로 묶어 가독성 개선 (🟡 2 (▲20) 형태 - 이전엔 공백만 있어 붙어 보였음)
                _deltas = list(result_df["순위 변동"])
                def _idx_suffix(d):
                    if d == "-":
                        return ""
                    if d == "신규":
                        return " ✨신규"
                    return f" ({d})"
                result_df.index = [
                    f"{sig} {i}{_idx_suffix(d)}"
                    for i, sig, d in zip(range(1, len(result_df) + 1), result_df["신호"], _deltas)
                ]
                # (참고: 현재 상태 컬럼에 과매수/과매도/정상 범위 값이 이미 들어있음)
                drop_cols = ["우선순위", "상태", "신호", "auc_pvalue", "ev", "prob_profit", "prob_loss", "atr_pct", "vix", "market_relative", "is_fb"]
                result_df.drop(columns=[c for c in drop_cols if c in result_df.columns], inplace=True)


                # ⭐ Fix 1: 결과를 세션에 저장 (다른 탭으로 이동하거나 tab1에서
                # 재검색해도 이 결과가 사라지지 않도록)
                st.session_state["ranking_result_df"] = result_df
                st.session_state["ranking_updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["ranking_hist_days"] = _hist_days
            else:
                st.session_state["ranking_result_df"] = None
                st.info("분석 가능한 데이터가 없습니다.")

    # ⭐ Fix 1: 버튼을 누른 그 rerun에서만이 아니라, 세션에 저장된 결과가
    # 있으면 항상(탭 전환·다른 위젯 조작 이후에도) 표시합니다.
    if st.session_state.get("ranking_result_df") is not None:
        _df = st.session_state["ranking_result_df"]

        # ⭐ Claude Design 반영: 3개 색상 카드(관심=민트, 지켜보기=크림, 조건미달=뉴트럴)
        _signals = [str(i).split(" ")[0] for i in _df.index]
        _n_watch, _n_hold, _n_fail = _signals.count('🔵'), _signals.count('🟡'), _signals.count('🔴')

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(_summary_card(MINT_BG, "🔵", "관심", _n_watch), unsafe_allow_html=True)
        with s2:
            st.markdown(_summary_card(WATCH_BG, "🟡", "지켜보기", _n_hold), unsafe_allow_html=True)
        with s3:
            st.markdown(_summary_card(NEUTRAL_100, "🔴", "조건 미달", _n_fail), unsafe_allow_html=True)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.caption(
            "💡 대부분의 종목이 '조건 미달'로 나오는 것은 정상입니다. 이 도구는 수수료와 체결 오차를 "
            "모두 뺀 뒤에도 기대수익이 남는 경우만 골라내기 때문에, 평상시에는 관심 종목이 0~3개인 것이 자연스럽습니다. "
            "관심 종목이 없다는 것은 '오늘은 살 만한 게 없다'는 뜻이며, 억지로 사지 않는 것도 하나의 판단입니다."
        )

        _hd = st.session_state.get("ranking_hist_days", 1)
        if _hd > 1:
            st.caption(
                f"🕒 마지막 갱신: {st.session_state.get('ranking_updated_at', '-')} · "
                f"순위는 최근 **{_hd}일치 점수 평균**으로 매겨집니다 (하루 등락에 순위가 급변하지 않도록)."
            )
        else:
            st.caption(
                f"🕒 마지막 갱신: {st.session_state.get('ranking_updated_at', '-')} · "
                f"오늘이 첫 분석입니다. 며칠간 반복 실행하면 순위가 여러 날 평균으로 안정화됩니다."
            )

        # ⭐ Fix 13: 컬럼이 너무 많아 옆 스크롤이 심하고 초보자가 이해하기 어렵다는
        # 피드백 반영 — 핵심 5개만 기본 표로 보여주고, 나머지 분석 지표는
        # 아래 '더 자세히 보기'에서 필요할 때만 열어보게 분리했습니다.
        # 동시에 st.dataframe의 LinkColumn은 항상 새 탭(target=_blank)으로만 열리고
        # 캔버스 렌더링이라 배경색도 못 바꾸는 두 가지 문제가 있어, 기본 표는
        # 순수 HTML로 직접 그려서 같은 창 이동 + 팔레트 색상을 모두 확보합니다.
        _core_cols = ["종목명", "오를 가능성", "현재 상태", "위험 대비 효율", "신호 유지일", "PER/PBR"]
        if "종목 비교 점수" in _df.columns:
            _core_cols.insert(2, "종목 비교 점수")
        _core_labels = {"종목명": "종목명", "오를 가능성": "오를 가능성", "현재 상태": "과매수/과매도",
                        "위험 대비 효율": "위험 대비 효율", "신호 유지일": "신호 유지일", "PER/PBR": "PER / PBR",
                        "종목 비교 점수": "종목 비교 점수"}

        header_html = "<th></th>" + "".join(f"<th>{_core_labels[c]}</th>" for c in _core_cols)
        rows_html = []
        for idx, row in _df.iterrows():
            cells = [f"<td style='color:{TEXT_MUTED};'>{idx}</td>"]
            for c in _core_cols:
                if c == "종목명":
                    url = str(row[c])
                    name = url.split("name=")[-1] if "name=" in url else url
                    price_txt = str(row["현재가"]) if "현재가" in _df.columns else ""
                    cells.append(
                        f"<td><a href='{url}' target='_self' "
                        f"style='color:{ACCENT_LAVENDER_DARK};font-weight:600;text-decoration:none;'>{name}</a>"
                        f"<span style='color:{TEXT_MUTED};font-size:12px;margin-left:6px;'>({price_txt})</span></td>"
                    )
                elif c == "오를 가능성":
                    pct = float(row[c])
                    cells.append(f"""<td style="min-width:130px;">
<div style="display:flex;align-items:center;gap:8px;">
  <div style="flex:1;height:8px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
    <div style="height:100%;border-radius:999px;background:{ACCENT_PINK};width:{pct:.0f}%;"></div>
  </div>
  <span style="font-size:12.5px;color:{TEXT_DARK};">{pct:.0f}%</span>
</div></td>""")
                elif c == "종목 비교 점수":
                    _v = row[c]
                    if pd.isna(_v):
                        cells.append(f"<td style='color:{TEXT_MUTED};'>-</td>")
                    else:
                        _v = float(_v)
                        cells.append(f"""<td style="min-width:120px;">
<div style="display:flex;align-items:center;gap:8px;">
  <div style="flex:1;height:8px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
    <div style="height:100%;border-radius:999px;background:{ACCENT_LAVENDER};width:{_v:.0f}%;"></div>
  </div>
  <span style="font-size:12.5px;color:{TEXT_DARK};">{_v:.0f}</span>
</div></td>""")
                elif c == "현재 상태":
                    state_text = str(row[c])
                    if "과매수" in state_text:
                        badge_bg, badge_color = UP_COLOR_BG, "#93375F"
                    elif "과매도" in state_text:
                        badge_bg, badge_color = DOWN_COLOR_BG, "#0C447C"
                    else:
                        badge_bg, badge_color = NEUTRAL_200, TEXT_MUTED
                    cells.append(
                        f"<td><span style='background:{badge_bg};color:{badge_color};"
                        f"padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;"
                        f"white-space:nowrap;'>{state_text}</span></td>"
                    )
                elif c == "PER/PBR":
                    val = str(row[c])
                    # ⭐ 적자/고평가 종목은 경고색으로 표시.
                    # 순위 점수 자체는 단기 기술적 모델이라 PER을 반영하지 않으므로,
                    # 상위권 종목이라도 가치평가상 주의가 필요하면 눈에 띄게 알려줍니다.
                    if "적자" in val:
                        cells.append(
                            f"<td><span style='background:{UP_COLOR_BG};color:#93375F;"
                            f"padding:4px 10px;border-radius:999px;font-size:11.5px;font-weight:600;"
                            f"white-space:nowrap;'>⚠️ {val}</span></td>"
                        )
                    elif val == "정보 없음":
                        cells.append(f"<td style='color:{TEXT_MUTED};'>{val}</td>")
                    else:
                        cells.append(f"<td style='color:{TEXT_DARK};'>{val}</td>")
                else:
                    cells.append(f"<td style='color:{TEXT_DARK};'>{row[c]}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")

        st.markdown(f"""
<style>
.mg-table-wrap {{ overflow-x:auto; border-radius:20px; border:1px solid {GRID_LIGHT}; background:{CARD_WHITE}; }}
.mg-table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-family:'Pretendard Variable',sans-serif; }}
.mg-table th, .mg-table td {{ padding:10px 16px; border-bottom:1px solid {GRID_LIGHT}; text-align:left; white-space:nowrap; }}
.mg-table th {{ color:{TEXT_MUTED}; font-weight:600; background:{NEUTRAL_100}; }}
.mg-table tr:last-child td {{ border-bottom:none; }}
.mg-table tr:hover td {{ background:{BG_CREAM}; }}
/* ⭐ [사용자 요청] 가치투자 랭킹이 500개 종목까지 늘어나서, 표 하나가 페이지를
   끝없이 늘리지 않도록 세로 스크롤 영역으로 감쌈. 헤더는 스크롤해도 안 밀리게 고정. */
.mg-table-wrap.mg-scroll {{ max-height:640px; overflow-y:auto; }}
.mg-table-wrap.mg-scroll thead th {{ position:sticky; top:0; z-index:1; }}
</style>
<div class="mg-table-wrap">
<table class="mg-table">
<thead><tr>{header_html}</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

        _csm = st.session_state.get("cs_meta")
        if _csm:
            _sig_txt = "통계적으로 유의함" if _csm["pvalue"] <= 0.10 else "통계적으로 불충분"
            st.caption(
                f"🆚 **종목 비교 점수**: 전 종목을 한 모델로 묶어 '같은 날 다른 종목보다 잘할까'를 예측한 값입니다 "
                f"(0~100, 높을수록 상대적으로 유리). 종목별 개별 모델의 표본 부족 문제를 보완합니다. "
                f"— 검증 AUC {_csm['auc']:.2f} ({_sig_txt}, 평가 {_csm['n_eval']}건)"
            )

        st.caption(
            "⚠️ **순위는 단기(10일) 기술적 신호만으로 매겨집니다.** PER/PBR 같은 가치평가 지표는 "
            "순위에 반영되지 않은 참고 정보예요. 상위권이라도 ⚠️ 표시(적자)가 있으면 "
            "기업이 이익을 못 내고 있다는 뜻이니, 장기 보유를 생각한다면 따로 확인해보세요."
        )

        with st.expander("📚 더 자세한 분석 지표 보기 (AI 검증 점수, 과거 성과 등)"):
            st.dataframe(_df, column_config={
                "종목명": st.column_config.LinkColumn("종목명", display_text=r"name=(.+)$", pinned=True),
                "현재 상태": st.column_config.TextColumn(
                    "현재 상태", help="스토캐스틱 기준 🔥 과매수 / 🧊 과매도 / ⚖️ 정상 범위 중 하나입니다."
                ),
                "PER/PBR": st.column_config.TextColumn(
                    "PER / PBR",
                    help="PER(주가수익비율)은 낮을수록, PBR(주가순자산비율)은 1에 가깝거나 낮을수록 "
                         "이익·자산 대비 저평가됐다고 보는 경향이 있습니다. 업종마다 정상 범위가 다르니 "
                         "같은 업종 종목끼리 비교할 때 더 의미가 있습니다."
                ),
                "순위 변동": st.column_config.TextColumn(
                    "순위 변동", help="직전 분석 대비 순위 변화입니다. ▲는 상승, ▼는 하락입니다."
                ),
                "평균 점수": st.column_config.NumberColumn(
                    "평균 점수", format="%.2f",
                    help="최근 며칠간 점수를 평균한 값입니다. 순위는 이 값으로 매겨져 하루 등락에 덜 흔들립니다."
                ),
                "오를 가능성": st.column_config.ProgressColumn(
                    "오를 가능성", format="%d%%", min_value=0, max_value=100,
                    help="목표가에 먼저 닿을 확률입니다."
                ),
                "신호 유지일": st.column_config.TextColumn(
                    "신호 유지일",
                    help=f"최근 {STABILITY_WINDOW}거래일 중 며칠이나 같은 신호였는지입니다. 숫자가 클수록 안정적인 신호입니다."
                ),
                "1회 기댓값": st.column_config.NumberColumn(
                    "1회 기댓값", format="%.2f%%",
                    help="한 번 투자할 때 기대되는 평균 수익률(수수료 포함)입니다."
                ),
                "위험 대비 효율": st.column_config.NumberColumn(
                    "위험 대비 효율", format="%.2f",
                    help="1보다 크면 기대 수익이 위험보다 크다는 뜻입니다."
                ),
                "적정 비중": st.column_config.NumberColumn(
                    "단독 기준 비중", format="%.1f%%",
                    help="이 종목만 투자할 때의 이론적 상한입니다. 여러 종목을 동시에 살 때는 '포트폴리오 배분'을 보세요."
                ),
                "포트폴리오 배분": st.column_config.NumberColumn(
                    "포트폴리오 배분", format="%.1f%%",
                    help=f"관심 종목을 동시에 담을 때 실제로 넣을 비율입니다. 0%는 '관심' 상태가 아니라는 뜻입니다. 합계가 전체 투자금의 {TOTAL_PORTFOLIO_CAP:.0f}%를 넘지 않게 자동 조정됩니다."
                ),
                "AI 예측 실력": st.column_config.TextColumn(
                    "AI 예측 실력", help="0.5는 동전 던지기 수준, 높을수록 잘 맞춥니다."
                ),
                "신뢰도": st.column_config.TextColumn(
                    "신뢰도", help="여러 종목을 동시에 검사할 때 생기는 우연의 효과를 통계적으로 걸러낸 결과입니다."
                ),
            }, **_W)

        _at = st.session_state.get("ranking_alloc_total", 0.0)
        _art = st.session_state.get("ranking_alloc_raw_total", 0.0)
        if _at > 0:
            if _art > TOTAL_PORTFOLIO_CAP:
                st.caption(
                    f"💰 **포트폴리오 배분 합계 {_at:.1f}%** — 단독 기준 비중을 그대로 더하면 {_art:.1f}%로 "
                    f"과도해지므로, 비율을 유지한 채 {TOTAL_PORTFOLIO_CAP:.0f}% 한도로 자동 축소했습니다. "
                    f"나머지는 현금으로 두는 것이 안전합니다."
                )
            else:
                st.caption(f"💰 **포트폴리오 배분 합계 {_at:.1f}%** — 나머지는 현금으로 두는 것이 안전합니다.")
        else:
            st.caption("💰 오늘은 배분할 관심 종목이 없습니다. 현금 보유도 하나의 선택입니다.")

# ==========================================
# 5. 가치투자 랭킹 탭 (횡단면 6개월 모델)
# ==========================================
with tab3:
    st.subheader("💎 가치투자 랭킹")
    st.caption(
        "재무제표(이익·자산·수익성)를 기준으로 **앞으로 6개월 동안 다른 종목보다 잘할 가능성**을 매깁니다. "
        "단기 랭킹과 달리 며칠 단위 가격 흐름이 아니라 기업의 실적을 봅니다."
    )

    run_value = st.button("가치투자 분석 시작")

    if run_value:
        # ⭐ [사용자 요청] 종목 하나당 재무제표 데이터 포인트가 10개 안팎이라
        # 표본이 너무 작습니다. S&P500 유니버스(VALUE_UNIVERSE_TICKERS)를
        # 기존 41개 종목과 합쳐서 학습 표본을 늘립니다. 이 유니버스는 이
        # 탭에서만 쓰고, 단기 랭킹 등 다른 기능은 원래 ALL_TICKERS(41개)를
        # 그대로 씁니다 — 안 그러면 다른 탭까지 다 느려집니다.
        _value_tickers = sorted(set(VALUE_UNIVERSE_TICKERS) | set(ALL_TICKERS))
        _v_bar0 = st.progress(0, text="가격 데이터를 불러오는 중...")
        df_all_value = download_all_data(tuple(_value_tickers), start_date)
        _v_bar0.empty()
        if df_all_value.empty:
            st.error("주가 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        else:
            v_bar = st.progress(0, text="재무제표를 불러오는 중...")
            panels = []
            _vctx = get_script_run_ctx() if get_script_run_ctx else None

            def build_one(tk):
                if _vctx is not None and add_script_run_ctx is not None:
                    add_script_run_ctx(threading.current_thread(), _vctx)
                try:
                    fh = get_fundamental_history(tk)
                    if fh.empty:
                        return None
                    return build_value_panel(df_all_value, fh, macro_prepared, tk)
                except Exception as e:
                    logging.warning(f"Value panel build failed for {tk}: {e}")
                    return None

            done = 0
            _t0v = time.time()
            # ⭐ [사용자 요청] 종목이 500개대로 늘면서 느려짐. 이 작업은 순수
            # 네트워크 대기(yfinance 호출)라 CPU 병목이 아니므로 동시 요청 수를
            # 늘리는 게 안전하게 먹히는 최적화입니다(너무 높이면 Yahoo 쪽에서
            # 차단당할 수 있어 20 정도로 절충). 어차피 하루(ttl=86400) 캐시라
            # 같은 날 다시 누르면 이미 받은 종목은 즉시 반환됩니다.
            with ThreadPoolExecutor(max_workers=20) as ex:
                futs = {ex.submit(build_one, tk): tk for tk in _value_tickers}
                for fut in as_completed(futs):
                    p = fut.result()
                    if p is not None and not p.empty:
                        panels.append(p)
                    done += 1
                    pct = done / len(_value_tickers)
                    eta = (time.time() - _t0v) / max(pct, 0.01) * (1 - pct)
                    v_bar.progress(int(pct * 100),
                                   text=f"재무 데이터 수집 중... {done}/{len(_value_tickers)} (약 {int(eta)}초 남음)")
            v_bar.empty()

            if not panels:
                st.session_state["value_result"] = None
                st.warning(
                    "재무제표 데이터를 가져오지 못했습니다. 무료 데이터 제공처가 분기 재무를 주지 않거나 "
                    "일시적으로 차단된 경우일 수 있습니다. 잠시 후 다시 시도해 주세요."
                )
            else:
                panel = pd.concat(panels)
                with st.spinner("횡단면 모델 학습 중..."):
                    vres = run_value_model(panel)
                if vres is None or "error" in vres:
                    st.session_state["value_result"] = None
                    # ⭐ 모델이 실제로 멈춘 지점의 사유를 그대로 표시합니다
                    # (이전에는 UI가 원인을 다시 추측해서 실제와 다른 안내가 나갔음)
                    _reason = vres.get("error") if vres else "알 수 없는 오류"
                    _avail = [c for c in VALUE_FEATURES if c in panel.columns]
                    _cov = {c: float(panel[c].notna().mean()) for c in _avail}
                    st.warning(f"가치투자 모델을 학습하지 못했습니다.\n\n**원인:** {_reason}")
                    with st.expander("자세한 진단 정보"):
                        _lines = [f"- 수집된 종목 수: {len(panels)}개",
                                  f"- 패널 전체 행수: {len(panel):,}행",
                                  f"- 날짜 수: {panel.index.nunique()}일", "",
                                  "**지표별 데이터 확보율:**"]
                        for c in _avail:
                            _lines.append(f"  - {VALUE_FEATURE_EASY.get(c, c)}: {_cov[c]*100:.0f}%")
                        st.markdown("\n".join(_lines))
                    st.caption(
                        "무료 데이터 제공처가 과거 재무를 충분히 제공하지 않으면 이런 현상이 생깁니다. "
                        "시간이 지나 데이터가 채워지면 해결될 수 있습니다."
                    )
                else:
                    st.session_state["value_result"] = vres
                    st.session_state["value_updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

                    # ⭐ [사용자 요청] 벤치마크 4개가 전부 무작위(0.5) 수준으로 나온 뒤,
                    # "종목(breadth) 대신 검증 시점(depth)이 문제일 수 있다"는 가설을
                    # 검증하기 위한 실험입니다. 예측 기간을 6개월→3개월→1개월로 줄이면
                    # 같은 데이터에서 겹치지 않는 검증 구간을 더 많이 뽑을 수 있어서,
                    # 호라이즌이 짧을 때 신호가 살아나는지 비교합니다. 재무제표는 이미
                    # panel에 다 있어 재수집 없이 재계산만 하면 되므로 빠릅니다.
                    with st.spinner("호라이즌(예측 기간)별 예측력 비교 중..."):
                        _sweep = {"6개월(기본)": vres}
                        for h, h_label in [(21, "1개월"), (63, "3개월")]:
                            _r = run_value_model(panel, horizon_override=h)
                            if _r and "error" not in _r:
                                _sweep[h_label] = _r
                        st.session_state["value_horizon_sweep"] = _sweep

    _vres = st.session_state.get("value_result")
    if _vres is not None:
        _auc = _vres["auc"]; _pv = _vres["pvalue"]
        # ⭐ 검증 자체가 불가능했던 경우(validated=False)를 유의성 판정과 분리해서 다룹니다.
        # 이전에는 이 값을 반환만 하고 화면에서 쓰지 않아, 검증되지 않은 결과가
        # 검증된 것처럼 보일 수 있었습니다.
        _validated = bool(_vres.get("validated", True)) and np.isfinite(_auc)
        _is_sig = _validated and np.isfinite(_pv) and _pv <= 0.10

        # ⭐⭐ [중대 버그 수정] 유의성은 Fama-MacBeth 검정 결과를 씁니다.
        # 예전에는 행(종목×날짜) 수를 독립 표본으로 세는 바람에, 사실상 무작위
        # 수준인 결과도 'p<0.05 유의함'으로 표시됐습니다(표준오차 20배 이상 과소평가).
        # 이제 독립 단위는 '평가 날짜 수'이고, 그만큼 검정력이 낮다는 사실이
        # 화면에 그대로 드러납니다.
        _fm_T = int(_vres.get("fm_n_dates", 0) or 0)
        _fm_t = _vres.get("fm_tstat", float("nan"))
        _fm_auc = _vres.get("fm_auc", float("nan"))
        # 날짜가 최소 2개는 있어야 시계열 분산을 추정할 수 있음
        _testable = _validated and _fm_T >= 2 and np.isfinite(_pv)
        _is_sig = _testable and _pv <= 0.05

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            _auc_txt = f"{_auc:.2f}" if _validated else "측정 불가"
            st.markdown(_summary_card(MINT_BG if _is_sig else (NEUTRAL_100 if _validated else WATCH_BG),
                                      "🎯", "예측 실력(AUC)", _auc_txt, suffix=""),
                        unsafe_allow_html=True)
        with vc2:
            # ⭐ 학습 행수(수만 개)를 그대로 보여주면 표본이 충분한 것처럼 오해됩니다.
            # 실제 검정의 자유도를 좌우하는 것은 '검정에 쓰인 평가 날짜 수'입니다.
            st.markdown(_summary_card(MINT_BG if _fm_T >= 20 else WATCH_BG, "🔢",
                                      "유효 표본(평가 날짜)", f"{_fm_T}", suffix="개"),
                        unsafe_allow_html=True)
        with vc3:
            if not _validated:
                _sig_label = "검증 불가"
            elif not _testable:
                _sig_label = "검정 불가"
            elif _is_sig:
                _sig_label = "유의함"
            else:
                _sig_label = "유의하지 않음"
            st.markdown(_summary_card(MINT_BG if _is_sig else WATCH_BG, "📐", "통계적 유의성",
                                      _sig_label, suffix=""),
                        unsafe_allow_html=True)

        if _testable:
            st.caption(
                f"📐 **Fama-MacBeth + Newey-West 검정:** 날짜별 AUC 평균 {_fm_auc:.3f} · "
                f"t = {_fm_t:.2f} (자유도 {_fm_T - 1}) · p = {_pv:.3f}. "
                f"같은 날짜 안의 종목들은 서로 독립이 아니므로 행 수가 아니라 "
                f"**평가 날짜 {_fm_T}개**가 표본 크기이고, 예측 기간이 겹쳐 생기는 "
                f"자기상관은 Newey-West로 보정했습니다."
            )
        elif _validated:
            st.caption(
                f"📐 독립 평가 날짜가 {_fm_T}개뿐이라 시계열 분산을 추정할 수 없어 "
                "유의성 검정 자체가 불가능합니다. AUC는 참고용 점추정일 뿐입니다."
            )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # ⭐ [사용자 요청] "예측력이 경쟁력"이라는 주장을 하려면 무작위·단순
        # 규칙보다 실제로 나은지 숫자로 보여줘야 합니다. 같은 평가 구간(OOF)에서
        # 계산한 4가지 방식의 AUC를 나란히 비교합니다.
        _bench = _vres.get("bench_auc", {}) if _validated else {}
        _bench_fm = _vres.get("bench_fm", {}) if _validated else {}
        if _bench:
            # ⭐ AUC 점추정만 보면 "0.52 vs 0.50"이 의미 있어 보이지만, t값을 같이
            # 봐야 그 차이가 표본 노이즈 범위인지 알 수 있습니다(|t| > 2가 대략 기준).
            def _auc_row(name, key, pooled_val):
                txt = f"{pooled_val:.2f}" if np.isfinite(pooled_val) else "측정 불가"
                fm = _bench_fm.get(key)
                if fm and np.isfinite(fm[1]):
                    t_txt = f"{fm[1]:+.2f}"
                    p_txt = f"{fm[2]:.3f}"
                    verdict = "유의" if fm[2] <= 0.05 else "무의미"
                    vcol = "#1C4E3E" if fm[2] <= 0.05 else TEXT_MUTED
                else:
                    t_txt, p_txt, verdict, vcol = "-", "-", "검정 불가", TEXT_MUTED
                return (f"<tr><td>{name}</td>"
                        f"<td style='text-align:right;font-weight:700;'>{txt}</td>"
                        f"<td style='text-align:right;'>{t_txt}</td>"
                        f"<td style='text-align:right;'>{p_txt}</td>"
                        f"<td style='text-align:right;color:{vcol};font-weight:600;'>{verdict}</td></tr>")
            _rows_html = "".join([
                _auc_row("🎲 무작위 선택", "random", _bench.get("random", float("nan"))),
                _auc_row("💰 단순 저PER", "simple_per", _bench.get("simple_per", float("nan"))),
                _auc_row("📈 린치식 PEG", "lynch_peg", _bench.get("lynch_peg", float("nan"))),
                _auc_row("🤖 이 모델(RandomForest)", "model", _auc),
            ])
            st.markdown(f"""
<div class="mg-table-wrap">
<table class="mg-table">
<thead><tr><th>방식</th><th style="text-align:right;">AUC</th><th style="text-align:right;">t값</th><th style="text-align:right;">p값</th><th style="text-align:right;">판정</th></tr></thead>
<tbody>{_rows_html}</tbody>
</table>
</div>
""", unsafe_allow_html=True)
            st.caption(
                "ℹ️ 같은 검증 구간에서 네 가지 방식을 비교했습니다. AUC는 점추정이고, "
                "실제 판단은 **t값·p값**으로 해야 합니다(Fama-MacBeth + Newey-West, 날짜 단위 검정). "
                "AUC가 0.52라도 t값이 작으면 무작위와 구분되지 않는다는 뜻이에요."
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # ⭐ [사용자 요청] "종목 수(breadth)가 아니라 검증 시점 수(depth)가
        # 병목일 수 있다"는 가설 검증용. 예측 기간을 줄이면 같은 데이터로도
        # 독립 검증 구간을 더 확보할 수 있어, 호라이즌별로 AUC가 달라지는지 봅니다.
        _sweep = st.session_state.get("value_horizon_sweep", {})
        if _sweep:
            def _sweep_row(label, r):
                _a = r.get("auc", float("nan"))
                _n = r.get("n_eval_dates", 0)
                _a_txt = f"{_a:.2f}" if np.isfinite(_a) else "측정 불가"
                return (f"<tr><td>{label}</td><td style='text-align:right;font-weight:700;'>{_a_txt}</td>"
                        f"<td style='text-align:right;color:{TEXT_MUTED};'>{_n}개</td></tr>")
            _sweep_rows = "".join(_sweep_row(lbl, r) for lbl, r in _sweep.items())
            st.markdown(f"""
<div class="mg-table-wrap">
<table class="mg-table">
<thead><tr><th>예측 기간</th><th style="text-align:right;">AUC(RandomForest)</th><th style="text-align:right;">독립 검증 시점</th></tr></thead>
<tbody>{_sweep_rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)
            st.caption(
                "ℹ️ 예측 기간을 줄이면 같은 데이터로 검증 구간을 더 많이 뽑을 수 있어, "
                "종목 수(breadth) 대신 검증 시점 수(depth)가 병목이었는지 확인하는 실험입니다."
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        if not _validated:
            st.warning(
                "⚠️ **이 순위는 검증되지 않았습니다.** 재무 이력이 짧아 미래 정보 누출 없이 "
                "성능을 측정할 구간을 만들지 못했습니다. 순위는 참고용으로만 보시고, "
                "투자 판단의 근거로 삼지 마세요."
            )
        elif not _testable:
            st.warning(
                f"⚠️ **유의성을 검정할 수 없습니다.** 독립 평가 날짜가 {_fm_T}개뿐이라 "
                "시계열 분산을 추정할 수 없습니다. 아래 순위는 참고만 하세요."
            )
        elif not _is_sig:
            st.warning(
                f"⚠️ 이 모델의 예측력이 통계적으로 입증되지 않았습니다 "
                f"(Fama-MacBeth t={_fm_t:.2f}, p={_pv:.3f}, 독립 날짜 {_fm_T}개). "
                "아래 순위는 참고만 하시고, 투자 근거로 삼지 마세요."
            )
        else:
            st.success(
                f"✅ 예측력이 통계적으로 유의합니다 "
                f"(Fama-MacBeth t={_fm_t:.2f}, p={_pv:.3f}, 독립 날짜 {_fm_T}개). "
                "다만 과거 기준이며 미래를 보장하진 않습니다."
            )

        st.caption(
            "ℹ️ 이 점수는 **'자산·이익 대비 주가가 다른 종목보다 싼가'**를 비교한 것입니다. "
            "실적이 꾸준히 좋은지, 앞으로도 돈을 벌 수 있는지는 별도로 보지 않으니 "
            "PER·PBR·ROE 숫자를 같이 확인하며 참고해 주세요."
        )

        _dropped = _vres.get("dropped_feats", [])
        if _dropped:
            _dn = ", ".join(VALUE_FEATURE_EASY.get(c, c) for c in _dropped)
            st.caption(f"ℹ️ 데이터가 부족한 지표는 분석에서 제외했습니다: {_dn}")

        _sc = _vres["scores"].sort_values("점수", ascending=False).reset_index(drop=True)

        # ⭐ [버그 수정] PER/PBR/ROE를 종목마다 순차적으로 실시간 조회하면
        # (get_fundamentals가 종목당 네트워크 호출 1건) 종목 수만큼 직렬로
        # 기다리게 되어 표가 한참 동안 아예 안 뜨는 것처럼 보입니다.
        # 위 "전체 종목 분석하기"와 동일하게 스레드풀로 병렬 조회합니다.
        _fund_ctx = get_script_run_ctx() if get_script_run_ctx else None
        _fund_map = {}

        def _fetch_fund(tk):
            if _fund_ctx is not None and add_script_run_ctx is not None:
                add_script_run_ctx(threading.current_thread(), _fund_ctx)
            return tk, get_fundamentals(tk)

        with st.spinner("최신 PER/PBR/ROE를 불러오는 중..."):
            with ThreadPoolExecutor(max_workers=20) as ex:
                for tk, fund in ex.map(_fetch_fund, _sc["Ticker"].unique()):
                    _fund_map[tk] = fund

        v_rows = []
        _n_unprofitable = 0
        for i, row in _sc.iterrows():
            tk = row["Ticker"]
            nm = company_names.get(tk, tk)
            score = row["점수"]

            # ⭐ [버그 수정] 표에 보여주는 PER/PBR/ROE는 재구성한 과거 재무
            # 패널(분기 공시 지연 보정용) 대신, 실시간 yfinance 값(get_fundamentals,
            # 단일 종목 상세 화면과 동일한 소스)을 우선 씁니다. 패널은 백테스트용
            # 시계열이라 회사마다 다른 자본구조(전환사채, 자사주 등)에서 오차가
            # 누적되기 쉬운데, 실시간 값은 Yahoo가 이미 계산해둔 걸 그대로 쓰니
            # 더 정확하고, 하루 단위로 자동 갱신도 됩니다. 값이 없을 때만
            # 재구성 패널로 대체합니다.
            _fund = _fund_map.get(tk, {})
            per, pbr, roe, is_loss = _fund.get("per"), _fund.get("pbr"), _fund.get("roe"), _fund.get("is_loss", False)
            if per is None or pbr is None or roe is None:
                ey = row.get("Earnings_Yield", np.nan)
                bp = row.get("Book_to_Price", np.nan)
                if per is None and pd.notna(ey) and ey != 0:
                    per = 100 / ey
                    is_loss = is_loss or ey <= 0
                if pbr is None and pd.notna(bp) and bp > 0:
                    pbr = 100 / bp
                if roe is None:
                    _panel_roe = row.get("ROE", np.nan)
                    roe = float(_panel_roe) if pd.notna(_panel_roe) else None

            is_neg_roe = roe is not None and roe < 0
            if is_loss or is_neg_roe:
                _n_unprofitable += 1

            # ⭐ 적자/마이너스 ROE 경고 배지 (단기 랭킹 표와 동일한 스타일로 통일).
            # 이 모델은 '자산 대비 싼가'를 보되 '돈을 벌고 있는가'는 걸러내지
            # 않으므로, 저평가처럼 보이지만 실제로는 가치 함정(value trap)일
            # 수 있는 종목을 반드시 눈에 띄게 표시해야 합니다.
            #
            # ⭐ [사용자 요청] PER이 크거나(수백~수천 배) 적자일 때 "이익률 X%"로
            # 바꿔 보여주던 걸 없앴습니다. 사용자가 가공 없이 실제 PER 숫자
            # 그대로 보길 원해서, 적자 여부만 경고 배지로 표시하고 값은 그대로 씁니다.
            if per is None:
                per_txt = "-"
            elif per <= 0:
                per_txt = (f"<span style='background:{UP_COLOR_BG};color:#93375F;padding:2px 8px;border-radius:999px;"
                          f"font-size:11.5px;font-weight:600;'>⚠️ {per:.1f}</span>")
            else:
                per_txt = f"{per:.1f}"
            pbr_txt = f"{pbr:.1f}" if pbr is not None else "-"
            if roe is not None:
                roe_txt = (f"<span style='background:{UP_COLOR_BG};color:#93375F;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600;'>⚠️ {roe:.1f}%</span>"
                          if is_neg_roe else f"{roe:.1f}%")
            else:
                roe_txt = "-"
            v_rows.append(f"""<tr>
<td style="color:{TEXT_MUTED};">{i+1}</td>
<td><a href='/?ticker={tk}&name={nm}' target='_self' style='color:{ACCENT_LAVENDER_DARK};font-weight:600;text-decoration:none;'>{nm}</a></td>
<td style="min-width:130px;">
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="flex:1;height:8px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
      <div style="height:100%;border-radius:999px;background:{ACCENT_LAVENDER};width:{score:.0f}%;"></div>
    </div>
    <span style="font-size:12.5px;color:{TEXT_DARK};">{score:.0f}</span>
  </div>
</td>
<td style="color:{TEXT_DARK};">{per_txt}</td>
<td style="color:{TEXT_DARK};">{pbr_txt}</td>
<td style="color:{TEXT_DARK};">{roe_txt}</td>
</tr>""")

        if _n_unprofitable > 0:
            st.warning(
                f"⚠️ 상위권에 적자이거나 ROE가 마이너스인 종목이 {_n_unprofitable}개 있습니다. "
                "이 순위는 **'자산·이익 대비 주가가 싼가'만 보고 '실제로 돈을 벌고 있는가'는 걸러내지 않습니다.** "
                "자산 대비 극도로 싸 보이는데 알고 보니 적자가 지속되는 경우(가치 함정, value trap)일 수 있으니, "
                "⚠️ 표시가 있는 종목은 재무제표를 직접 한 번 더 확인해보세요."
            )

        st.markdown(f"""
<div class="mg-table-wrap mg-scroll">
<table class="mg-table">
<thead><tr><th></th><th>종목명</th><th>가치 점수</th><th>PER</th><th>PBR</th><th>ROE</th></tr></thead>
<tbody>{''.join(v_rows)}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

        _hz = _vres.get("horizon", VALUE_HORIZON)
        _hz_txt = f"약 {_hz//21}개월" if _hz >= 21 else f"{_hz}일"
        _updated_at = st.session_state.get("value_updated_at")
        _update_line = (
            f"🕒 마지막 갱신: {_updated_at or '-'} · "
            f"기준일 {_vres['latest_date'].strftime('%Y-%m-%d')} · 종목 {_vres['n_tickers']}개 · "
            f"예측 기간 {_hz_txt}"
            + ("" if _hz >= VALUE_HORIZON else " (재무 이력이 짧아 기간을 자동 단축했습니다)")
        )
        # ⭐ 이 결과는 버튼을 누른 시점의 스냅샷이라 세션이 오래 유지되면
        # (탭을 계속 켜두는 등) 주가가 많이 움직였는데도 옛날 PER/PBR이
        # 그대로 표시될 수 있습니다. 눈에 띄게 경고해서 재실행을 유도합니다.
        _stale = _updated_at and (pd.Timestamp.now() - pd.Timestamp(_updated_at)) > pd.Timedelta(hours=24)
        if _stale:
            st.warning(f"⚠️ 이 결과는 하루 이상 지난 스냅샷입니다. {_update_line}\n\n"
                       "주가가 그 사이 많이 움직였을 수 있으니 '가치투자 분석 시작'을 다시 눌러 갱신하세요.")
        else:
            st.caption(_update_line)

        # 어떤 재무 지표를 많이 봤는지
        imp_rows = ""
        for i, (feat, val) in enumerate(sorted(_vres["importances"].items(), key=lambda x: -x[1])[:5]):
            color = REASON_COLORS[i % len(REASON_COLORS)]
            label = VALUE_FEATURE_EASY.get(feat, feat)
            pct = val * 100
            imp_rows += f"""
<div style="display:flex;align-items:center;gap:12px;">
  <div style="width:160px;flex:none;font-size:13px;color:{TEXT_BODY};">{label}</div>
  <div style="flex:1;height:10px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
    <div style="height:100%;border-radius:999px;background:{color};width:{pct:.0f}%;"></div>
  </div>
  <div style="width:34px;text-align:right;font-size:12.5px;color:{TEXT_MUTED};">{pct:.0f}%</div>
</div>"""
        st.markdown(f"""
<div style="background:{CARD_WHITE};border-radius:24px;padding:20px 22px;box-shadow:0 2px 8px rgba(74,68,88,0.06);margin-top:16px;">
  <div style="font-family:'Pretendard Variable',sans-serif;font-weight:700;font-size:15px;color:{TEXT_DARK};margin-bottom:12px;">AI가 중요하게 본 재무 지표</div>
  <div style="display:flex;flex-direction:column;gap:12px;">{imp_rows}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander("📌 이 모델의 한계 (반드시 읽어주세요)"):
            st.markdown(f"""
- **소급 수정 문제:** 무료 데이터는 '지금 시점에서 본 과거 재무'라, 당시 실제 발표치와 다를 수 있습니다. 이 때문에 성과가 실제보다 좋게 나올 여지가 있습니다.
- **공시 지연 반영:** 분기 실적을 그 즉시 알 수 없으므로 **{REPORT_LAG_DAYS}일 뒤부터** 사용 가능한 것으로 처리했습니다.
- **생존 편향:** 분석 대상이 현재 살아남은 대형주 위주라, 과거에 상장폐지된 기업이 빠져 있습니다.
- **표본 한계:** 학습에는 {_vres['n_samples']:,}개 행을 쓰지만, 라벨이 6개월을 참조하므로 서로 겹치지 않는 **독립 검증 시점은 {_vres.get('n_eval_dates', 0)}개**뿐입니다. AUC의 신뢰구간이 매우 넓으니 수치를 과신하지 마세요.
- **유의성 검정 방식:** 같은 날짜의 종목들은 (1) 같은 시장 환경을 공유하고 (2) 라벨이 '그날 중앙값 초과'라 절반이 1로 강제되어 서로 독립이 아닙니다. 따라서 행 수가 아니라 **날짜별 AUC의 시계열을 t검정하는 Fama-MacBeth 방식**을 씁니다(평가 날짜 {_fm_T}개, 자유도 {max(_fm_T - 1, 0)}).
- **겹치는 예측 구간 보정:** 예측 기간이 {_hz_txt}이므로 인접한 평가일의 라벨은 대부분 겹칩니다. 겹치는 날짜를 버리면 데이터의 극히 일부만 쓰게 되어 검정력이 사라지므로, 전부 사용하되 그로 인한 자기상관을 **Newey-West**로 표준오차에서 보정했습니다. 다만 '유효 독립 정보량'은 여전히 (데이터 기간 ÷ 예측 기간)에 묶여 있어, 이 검정도 다소 낙관적일 수 있습니다.
- **가치 ≠ 단기 수익:** 저평가 종목이 6개월 안에 오른다는 보장은 없으며, 몇 년간 저평가가 지속되기도 합니다.
""")

    # ⭐ 단기(스윙) + 장기(가치) 종합 뷰
    # 두 모델은 보는 기간과 근거가 완전히 달라서(10일 기술적 vs 6개월 재무),
    # 둘 다 좋다고 하는 종목은 서로 다른 근거가 같은 결론을 가리킨다는 의미가 있습니다.
    # 다만 '두 모델이 일치하면 더 정확하다'는 것은 별도 검증이 필요한 가설이므로,
    # 점수를 합산해 새 순위를 만들지 않고 나란히 비교만 제시합니다.
    _short = st.session_state.get("ranking_result_df")
    if _vres is not None and _short is not None:
        st.divider()
        st.subheader("🔗 단기 + 장기 종합 비교")

        _v = _vres["scores"][["Ticker", "점수"]].rename(columns={"점수": "가치점수"})
        _s = _short.copy()
        _s = _s[["Ticker", "오를 가능성", "현재 상태"]].rename(columns={"오를 가능성": "단기점수"})
        _m = _s.merge(_v, on="Ticker", how="inner")

        if _m.empty:
            st.caption("두 모델의 공통 종목이 없어 비교할 수 없습니다.")
        else:
            _m["종합"] = (_m["단기점수"] + _m["가치점수"]) / 2
            _m = _m.sort_values("종합", ascending=False).reset_index(drop=True)

            _both = _m[(_m["단기점수"] >= 40) & (_m["가치점수"] >= 55)]
            if not _both.empty:
                _names = ", ".join(company_names.get(t, t) for t in _both["Ticker"].head(5))
                st.success(f"✅ **단기·장기 모두 긍정적인 종목:** {_names}")
            else:
                st.info("ℹ️ 지금은 단기와 장기가 동시에 긍정적인 종목이 없습니다.")

            c_rows = []
            for i, row in _m.head(15).iterrows():
                tk = row["Ticker"]; nm = company_names.get(tk, tk)
                ss, vs = float(row["단기점수"]), float(row["가치점수"])
                if ss >= 40 and vs >= 55:
                    verdict, vbg, vcol = "둘 다 긍정", MINT_BG, "#1C4E3E"
                elif ss >= 40:
                    verdict, vbg, vcol = "단기만 긍정", WATCH_BG, "#5A470D"
                elif vs >= 55:
                    verdict, vbg, vcol = "장기만 긍정", "#F7F1FF", ACCENT_LAVENDER_DARK
                else:
                    verdict, vbg, vcol = "둘 다 미흡", NEUTRAL_200, TEXT_MUTED
                c_rows.append(f"""<tr>
<td style="color:{TEXT_MUTED};">{i+1}</td>
<td><a href='/?ticker={tk}&name={nm}' target='_self' style='color:{ACCENT_LAVENDER_DARK};font-weight:600;text-decoration:none;'>{nm}</a></td>
<td style="min-width:110px;"><div style="display:flex;align-items:center;gap:8px;">
  <div style="flex:1;height:8px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
    <div style="height:100%;border-radius:999px;background:{ACCENT_PINK};width:{ss:.0f}%;"></div></div>
  <span style="font-size:12.5px;color:{TEXT_DARK};">{ss:.0f}</span></div></td>
<td style="min-width:110px;"><div style="display:flex;align-items:center;gap:8px;">
  <div style="flex:1;height:8px;border-radius:999px;background:{NEUTRAL_200};overflow:hidden;">
    <div style="height:100%;border-radius:999px;background:{ACCENT_LAVENDER};width:{vs:.0f}%;"></div></div>
  <span style="font-size:12.5px;color:{TEXT_DARK};">{vs:.0f}</span></div></td>
<td><span style='background:{vbg};color:{vcol};padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap;'>{verdict}</span></td>
</tr>""")

            st.markdown(f"""
<div class="mg-table-wrap">
<table class="mg-table">
<thead><tr><th></th><th>종목명</th><th>단기 점수</th><th>가치 점수</th><th>종합 판단</th></tr></thead>
<tbody>{''.join(c_rows)}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

            st.caption(
                "💡 **단기 점수**는 10일 기술적 신호, **가치 점수**는 6개월 재무 기반으로 서로 다른 근거입니다. "
                "둘이 엇갈리는 것은 이상한 게 아니라 자연스러운 현상이에요 "
                "(예: 적자 성장주는 단기 급등해도 가치 점수는 낮음). "
                "두 점수를 합한 순위가 더 정확하다는 것은 아직 검증되지 않았으므로, 참고용 비교로만 봐주세요."
            )
    elif _vres is not None:
        st.divider()
        st.caption("🔗 '📊 단기 랭킹' 탭에서도 분석을 실행하면, 단기와 장기를 함께 비교해서 볼 수 있어요.")