"""가치투자 모델의 순수 파이프라인 (Streamlit UI 없음).

app.py에서 분리한 이유: 논문용 재현 스크립트(ablation.py)가 이 파이프라인을
임포트해서 돌려야 하는데, Streamlit 스크립트는 임포트만 해도 UI가 전부
실행돼 버려서 불가능했습니다. 여기에는 데이터 수집·피처 생성·모델·검정만
두고, 화면에 그리는 코드는 app.py에 남깁니다.

주의: @st.cache_data 데코레이터는 그대로 두었습니다. Streamlit 런타임 밖에서
호출해도 동작하며(캐시 없이 그냥 실행), app.py에서 쓸 때는 캐시가 살아납니다.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
import random
import threading
import traceback
import json
import os
import inspect
from datetime import date
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from stats_utils import fama_macbeth_auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 스트림릿 버전에 따라 경로가 다를 수 있어 방어적으로 임포트
# (ThreadPoolExecutor 워커 스레드에 Streamlit 컨텍스트를 전파하기 위해 필요 - Fix #2)
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    add_script_run_ctx = None
    get_script_run_ctx = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# 🔧 코드 리뷰 / 수정 내역 요약
# ------------------------------------------------------------
# 1) [버그] 랭킹 탭(tab2) 분석 결과가 다른 위젯을 조작하면(탭 이동,
#    개별 분석 재검색 등) 사라지던 문제 → st.session_state에 결과를
#    저장해 무관한 rerun에도 유지되도록 수정
# 2) [버그] ThreadPoolExecutor 워커 스레드에 Streamlit
#    ScriptRunContext가 전달되지 않아 발생할 수 있는 경고/불안정성
#    → add_script_run_ctx로 컨텍스트를 명시적으로 전파
# 3) [로직 오류] Reward-to-Risk(하방 리스크) 계산이 '타임아웃 시나리오의
#    평균 수익률이 음수인 경우'를 반영하지 못해 리스크를 과소평가하던
#    부분 수정
# 4) [개선] AI 신호 확률 임계값 / 켈리 축소 계수 / 데이터 학습기간을
#    하드코딩 대신 사이드바에서 조절 가능하도록 파라미터화 (표본 수·
#    민감도를 직접 튜닝하며 성능을 실험할 수 있음)
# 5) [개선] 피처 중요도(XAI)를 RF 단일 모델이 아닌 RF+GB 평균으로 산출
#    해 특정 서브모델 편향을 줄임
# 6) [안정성] yfinance 다운로드 재시도 로직에 지터(jitter) 추가로
#    스레드 4개 동시 재시도 시 Yahoo 쪽 레이트리밋에 몰리는 것을 완화,
#    캐시 TTL을 5분→15분으로 늘려 불필요한 재다운로드 축소
# ============================================================

# ==========================================
# 0. 전역 상수 및 설정
# ==========================================
LOOKAHEAD = 10
FEATURES = [
    "RSI", "MACD_Hist", "Momentum_5", "Volume_Change",
    "Stoch_K", "Volatility", "VIX_Close", "TNX_Close", "Market_Relative",
    # ⭐ 지정학적 충격 반영 (전쟁·분쟁 진행 중 상태를 관측 가능한 변수로 포착)
    "Oil_Change_20",   # 유가 20일 변화율 - 지금 유가 충격이 진행 중인지
    "Oil_Beta_60",     # 이 종목이 유가에 얼마나 민감한지 (정유주 +, 항공주 -)
]

FEATURE_KR_NAMES = {
    "RSI": "RSI(상대강도지수)",
    "MACD_Hist": "MACD 히스토그램",
    "Momentum_5": "5일 단기 모멘텀",
    "Volume_Change": "5일 거래량 변화율",
    "Stoch_K": "스토캐스틱 %K",
    "Volatility": "10일 변동성",
    "VIX_Close": "VIX(시장 공포지수)",
    "TNX_Close": "미 국채 10년물 금리",
    "Market_Relative": "S&P500 대비 상대강도",
    "Oil_Change_20": "유가 20일 변화율",
    "Oil_Beta_60": "유가 민감도(60일 베타)"
}

# ⭐ 초보자용 쉬운 지표 이름 (전문 용어를 일상어로 번역)
FEATURE_EASY_NAMES = {
    "RSI": "가격 과열도",
    "MACD_Hist": "상승하는 힘",
    "Momentum_5": "최근 5일 흐름",
    "Volume_Change": "거래 활발한 정도",
    "Stoch_K": "단기 가격 위치",
    "Volatility": "가격 출렁임 정도",
    "VIX_Close": "시장 전체 불안감",
    "TNX_Close": "미국 금리 수준",
    "Market_Relative": "시장 평균과 비교한 성적",
    "Oil_Change_20": "최근 유가 흐름",
    "Oil_Beta_60": "유가에 민감한 정도"
}

# ⭐ 알고리즘 자동 최적 설정 (사용자가 만질 필요 없도록 내부 고정)
# 초보자가 잘못 조정해 모델을 망가뜨리는 것을 막기 위해 검증된 기본값을 사용합니다.
AUTO_YEARS = 5           # 표본 수와 최신 시장 반영의 균형점
# ⭐ 논문용 스윙(가격 기반) 연구 전용. 재무제표를 쓰지 않으므로 yfinance의
# ~4년 재무 이력 제약이 없어 가격 데이터가 허용하는 만큼 기간을 늘릴 수 있다.
# 라이브 앱(AUTO_YEARS)과는 별개 상수로, 앱의 기존 동작에는 영향을 주지 않는다.
SWING_YEARS = 15
SWING_HORIZON = 21  # ~1개월. 4.4절에서 유의성이 몰려 있던 구간.
AUTO_PROB_THRESHOLD = 0.38   # 상승 신호로 인정하는 최소 확률
AUTO_HALF_KELLY = 0.5        # 하프 켈리(권장 안전 배수)
AUTO_KELLY_CAP = 25.0        # 한 종목 최대 투자 비중 상한(%)
TOTAL_PORTFOLIO_CAP = 40.0   # 동시 보유 종목 합계 최대 비중 상한(%)

# ==========================================
# 가치투자 모델 설정
# ==========================================
# 단기 기술적 모델(10일)과 별개로, 재무 지표 기반 6개월 예측 모델을 운영합니다.
VALUE_HORIZON = 126          # 예측 기간(거래일) ≈ 6개월
# ⭐⭐ 실적 공시 지연 (룩어헤드 방지의 핵심 파라미터)
#
# 결산일 당일엔 아무도 그 실적을 모릅니다. 결산일에 재무데이터를 바로 붙이면
# "그 시점에 알 수 없던 정보"로 과거를 예측하는 셈이 되므로 지연을 둬야 합니다.
#
# [중요] 이 값은 '오늘 화면 숫자가 증권앱과 맞는지'를 보고 튜닝하면 안 됩니다.
# 그건 제품 편의를 위해 논문의 룩어헤드 방지를 깨는 행위입니다. 반드시 SEC
# 제출 기한에서 사전에(a priori) 정합니다. 분기와 연간은 기한이 다릅니다:
#   - 10-Q(분기): 대형가속신고자 40일 / 그 외 45일  -> 45일
#   - 10-K(연간): 대형가속신고자 60일 / 가속 75일 / 비가속 90일 -> 90일
# 예전에는 둘 다 45일을 썼는데, 연간 재무는 실제 공시 전에 쓰이는
# 룩어헤드 누수였습니다.
REPORT_LAG_QUARTERLY = 45
REPORT_LAG_ANNUAL = 90
# 하위 호환용(기존 코드/화면 문구가 참조). 분기 기준값을 가리킵니다.
REPORT_LAG_DAYS = REPORT_LAG_QUARTERLY
VALUE_FEATURES = [
    "Earnings_Yield",    # 이익수익률 (EPS/주가) - PER의 역수, 적자도 음수로 자연 표현
    "Book_to_Price",     # 순자산/주가 - PBR의 역수
    "ROE",               # 자기자본이익률
    "Profit_Margin",     # 순이익률
    "Earnings_Growth",   # 이익 성장률(전년 동기 대비)
    "PEG_Inv",           # ⭐ [사용자 요청] 린치식 PEG의 역수(클수록 좋음) - 성장 대비 가격
    "Momentum_126",      # 6개월 가격 모멘텀 (가치 + 모멘텀 결합)
    "Volatility_60",     # 60일 변동성
    "Market_Relative",   # 시장 대비 상대강도
]
# 재무제표에서 오는 지표 (주가만으로 계산되는 지표와 구분).
# 이들이 빠지면 '가치투자'가 아니라 사실상 주가 모멘텀 모델이 되므로,
# 최소 2개는 반드시 확보되어야 분석을 진행합니다.
FUNDAMENTAL_FEATURES = ["Earnings_Yield", "Book_to_Price", "ROE", "Profit_Margin", "Earnings_Growth", "PEG_Inv"]
# ⭐ 논문용 스윙 연구 전용: 가격만으로 계산되는 지표만 남긴 부분집합.
# run_value_model(..., require_fundamentals=False)와 함께 쓰면 재무제표
# 없이(=4년 이력 제약 없이) 순수 가격 기반 횡단면 모델을 돌릴 수 있다.
SWING_FEATURES = ["Momentum_126", "Volatility_60", "Market_Relative"]

VALUE_FEATURE_EASY = {
    "Earnings_Yield": "이익 대비 주가 매력도",
    "Book_to_Price": "자산 대비 주가 매력도",
    "ROE": "자본 활용 효율",
    "Profit_Margin": "이익을 남기는 정도",
    "Earnings_Growth": "이익 성장 속도",
    "PEG_Inv": "성장 대비 가격 매력도(PEG)",
    "Momentum_126": "6개월 주가 흐름",
    "Volatility_60": "가격 출렁임 정도",
    "Market_Relative": "시장 평균과 비교한 성적",
}

# ⭐ 솜사탕 미니멀 팔레트 — Claude Design에서 내보낸 cottoncandy-overrides.css의 정확한 토큰
BG_CREAM = "#FFF9F5"
CARD_WHITE = "#FFFFFF"
TEXT_DARK = "#4A4458"
# ⭐ [사용자 요청] 원래 연보라(#9992A8)라 크림색 배경에서 잘 안 읽힌다는
# 피드백으로 TEXT_BODY와 같은 진한 톤으로 조정. 이 상수를 쓰는 모든 캡션/부가
# 설명 글씨(사이드바 안내문 등 포함)가 한 번에 같이 진해집니다.
TEXT_MUTED = "#5A526A"
TEXT_BODY = "#5A526A"
GRID_LIGHT = "#F0E6E0"          # --color-divider
UP_COLOR = "#FF8FA3"            # --color-rise (상승/코랄)
UP_COLOR_BG = "#FFD1DC"         # --color-rise-bg
DOWN_COLOR = "#7EC4EF"          # --color-fall (하락/하늘색)
DOWN_COLOR_BG = "#CDEBFF"       # --color-fall-bg
MINT_BG = "#E6FBF5"             # --color-mint-bg (관심 카드용)
WATCH_BG = "#FFF6DC"            # --color-watch-bg (지켜보기 카드용)
NEUTRAL_100 = "#FFFDFB"
NEUTRAL_200 = "#FBF3EE"
ACCENT_LINE = "#E8B84B"         # --color-gold (이동평균선)
BAND_FILL = "rgba(228, 212, 255, 0.30)"   # 변동 범위(라벤더 음영, --color-accent-200 계열)
ACCENT_LAVENDER = "#E4D4FF"     # --color-accent / --color-accent-200
ACCENT_LAVENDER_DARK = "#513A78"  # --color-accent-800 (라벤더 배경 위 텍스트용)
ACCENT_PINK = "#FFD6E8"         # --color-accent-2 / --color-accent-2-200
ACCENT_PINK_DARK = "#93375F"    # --color-accent-2-800
REASON_COLORS = [ACCENT_PINK, ACCENT_LAVENDER, "#C8F4E8", "#FFE9A8"]  # AI 판단 근거 막대 4색 순환

# ⭐ Streamlit 호환: use_container_width가 deprecated(2025-12-31 이후 제거 예정)라
# 신버전에서는 width="stretch"를 쓰고, 구버전에서는 기존 인자로 자동 폴백합니다.
def _stretch_kwargs():
    try:
        if "width" in inspect.signature(st.plotly_chart).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}

_W = _stretch_kwargs()

# ⭐ 스윙 투자(수일~수주 보유) 기준 설정
# 단타가 아니라 스윙이므로, 하루치 신호에 반응하지 않고 며칠간의 추세를 봅니다.
SMOOTH_DAYS = 5          # 예측 확률을 평균할 일수
STABILITY_WINDOW = 10    # 신호 유지일을 세는 구간
RANK_HISTORY_DAYS = 5    # 순위 산정에 사용할 과거 점수 일수
HISTORY_KEEP_DAYS = 40   # 이력 파일에 보관할 최대 일수

try:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _APP_DIR = os.getcwd()
HISTORY_PATH = os.path.join(_APP_DIR, "signal_history.json")

def load_history() -> dict:
    """⭐ Fix 11: 순위가 매일 크게 요동치는 문제를 해결하기 위해 과거 점수를
    파일에 저장합니다. 스윙 투자에서는 '오늘의 1위'보다 '며칠간 꾸준히 상위권'이
    훨씬 신뢰할 수 있는 정보이기 때문입니다."""
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                h = json.load(f)
                return h if isinstance(h, dict) else {}
    except Exception as e:
        logging.warning(f"History load failed: {e}")
    return {}

def save_history(hist: dict) -> None:
    try:
        # 오래된 이력은 잘라내 파일이 무한히 커지지 않도록 합니다.
        for k in sorted(hist.keys())[:-HISTORY_KEEP_DAYS]:
            hist.pop(k, None)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"History save failed: {e}")

company_names = {
    "AAPL": "애플", "MSFT": "마이크로소프트", "NVDA": "엔비디아", "GOOGL": "구글(알파벳)",
    "AMZN": "아마존", "META": "메타", "TSLA": "테슬라", "AVGO": "브로드컴",
    "COST": "코스트코", "PEP": "펩시코", "CSCO": "시스코", "TMUS": "티모바일",
    "ADBE": "어도비", "TXN": "텍사스 인스트루먼트", "NFLX": "넷플릭스", "AMD": "AMD",
    "QCOM": "퀄컴", "INTU": "인튜이트", "AMGN": "암젠", "HON": "허니웰",
    "INTC": "인텔", "ISRG": "인튜이티브 서지컬", "SBUX": "스타벅스", "BKNG": "부킹홀딩스",
    "GILD": "길리어드 사이언스", "MDLZ": "몬델레즈", "VRTX": "버텍스", "ADP": "ADP",
    "ADI": "아나로그디바이스", "PANW": "팔로알토 네트웍스", "MSTR": "마이크로스트레티지",
    "APP": "앱러빈", "CRWD": "크라우드스트라이크", "PLTR": "팔란티어", "ARM": "ARM",
    "UBER": "우버", "SNOW": "스노우플레이크", "ROKU": "로쿠", "ZM": "줌", "COIN": "코인베이스",
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스"
}

company_descriptions = {
    "AAPL": "아이폰 생태계를 기반으로 한 세계 최대의 글로벌 IT 기업",
    "MSFT": "윈도우, 클라우드(Azure), AI(OpenAI)를 선도하는 소프트웨어 제국",
    "NVDA": "글로벌 AI 인프라 확장을 주도하는 GPU 및 AI 반도체 절대 강자",
    "GOOGL": "전 세계 검색 엔진 1위이자 유튜브, 안드로이드를 지배하는 플랫폼 기업",
    "AMZN": "세계 최대의 전자상거래 및 클라우드(AWS) 인프라 제공 기업",
    "META": "페이스북, 인스타그램을 운영하는 글로벌 1위 소셜 미디어 플랫폼",
    "TSLA": "전기차 대중화를 이끌고 자율주행 및 에너지 혁신을 주도하는 기업",
    "AVGO": "통신, 데이터센터용 맞춤형 AI 반도체 및 인프라 소프트웨어 선도 기업",
    "COST": "강력한 멤버십 충성도를 자랑하는 글로벌 창고형 할인매장 1위",
    "PEP": "펩시, 레이즈 등 강력한 브랜드 파워를 가진 글로벌 식음료 거인",
    "CSCO": "기업용 네트워크 장비 및 보안 솔루션을 제공하는 IT 통신 장비 1위 기업",
    "TMUS": "미국 내 5G 네트워크 커버리지 1위를 자랑하는 초고속 통신사",
    "ADBE": "포토샵 등 크리에이티브 소프트웨어 시장의 독점적 지배자",
    "TXN": "아날로그 반도체 및 임베디드 프로세서 분야의 세계 1위 기업",
    "NFLX": "오리지널 콘텐츠 IP 경쟁력을 앞세운 글로벌 1위 OTT 스트리밍 서비스",
    "AMD": "엔비디아와 인텔의 강력한 경쟁자이자 CPU/GPU 양도류 반도체 설계사",
    "QCOM": "모바일 스마트폰의 두뇌인 AP와 무선 통신 칩셋 시장의 최강자",
    "INTU": "터보택스 등 미국 세무/회계 및 재무 관리 소프트웨어 시장 독점 기업",
    "AMGN": "자가면역질환 및 항암제 분야에서 탁월한 파이프라인을 가진 거대 바이오테크",
    "HON": "항공우주, 건축 제어, 신소재 등 다양한 산업을 아우르는 복합 기업",
    "INTC": "오랜 기간 PC 및 서버용 CPU 시장을 호령해 온 종합 반도체 제조사(IDM)",
    "ISRG": "'다빈치' 시스템으로 전 세계 수술용 로봇 시장을 독점 중인 헬스케어 기업",
    "SBUX": "전 세계 수만 개의 매장을 운영하는 독보적인 1위 글로벌 커피 체인",
    "BKNG": "부킹닷컴, 아고다를 소유한 전 세계 1위 온라인 여행 예약 플랫폼",
    "GILD": "HIV, 항바이러스제 및 간 질환 치료제 분야의 선두 바이오 제약사",
    "MDLZ": "오레오, 리츠 등 전 세계적인 스낵 브랜드를 다수 보유한 글로벌 제과업체",
    "VRTX": "낭포성 섬유증 등 희귀 난치성 질환 치료제 개발에 특화된 바이오 기업",
    "ADP": "미국 민간 기업 고용 지표의 기준이 되는 세계 최대의 급여/인사 관리 기업",
    "ADI": "산업, 자동차, 통신용 고성능 아날로그 및 혼합 신호 반도체 강자",
    "PANW": "방화벽 및 클라우드 보안, AI 기반 사이버 보안 솔루션을 선도하는 1위 기업",
    "MSTR": "세계에서 비트코인을 가장 많이 보유한 상장사이자 엔터프라이즈 분석 기업",
    "APP": "AI 기반 모바일 앱 마케팅 및 수익화 플랫폼 시장의 신흥 강자",
    "CRWD": "클라우드 기반 엔드포인트 사이버 보안 시장을 주도하는 차세대 리더",
    "PLTR": "미국 국방부 및 대기업을 위한 빅데이터/AI 분석 플랫폼 제공 기업",
    "ARM": "전 세계 모바일 프로세서 설계 및 아키텍처 라이선스를 독점하는 핵심 기업",
    "UBER": "승차 공유 및 배달(우버이츠) 네트워크를 통해 모빌리티 혁신을 이끄는 플랫폼",
    "SNOW": "데이터 클라우드 및 강력한 데이터 공유 아키텍처를 제공하는 차세대 SaaS 기업",
    "ROKU": "미국 1위 스마트 TV OS 및 스트리밍 플랫폼을 운영하는 미디어 허브 기업",
    "ZM": "하이브리드 워크 환경의 필수재가 된 글로벌 화상 회의 및 통신 플랫폼",
    "COIN": "글로벌 가상자산 규제 준수를 선도하는 미국 최대의 암호화폐 거래소",
    "005930.KS": "글로벌 메모리 반도체 1위 및 스마트폰, 가전 시장을 이끄는 한국 대표 기업",
    "000660.KS": "HBM 등 차세대 AI 메모리 시장을 선도하는 글로벌 D램/낸드플래시 톱티어"
}

ALL_TICKERS = list(company_names.keys())

# ⭐ [사용자 요청] 가치투자 랭킹(횡단면 모델) 전용 확장 유니버스.
# 분기 재무제표는 종목당 데이터 포인트가 10개 안팎이라 종목 하나로는 학습이
# 안 되고, 여러 종목을 모아 표본을 늘려야 합니다(41개→503개, S&P500 기준,
# 2026-09-18 위키피디아 스냅샷). 단기 랭킹·종목 검색 등 나머지 기능은 원래
# ALL_TICKERS(41개, 한글명 보유)를 그대로 쓰고, 이 유니버스는 가치투자 랭킹
# 계산에만 씁니다 — 다른 탭까지 503개로 늘리면 속도/API 호출량이 불필요하게
# 커집니다. 한글명이 없는 종목은 표에 티커 그대로 표시됩니다.
VALUE_UNIVERSE_TICKERS = sorted(set([
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
    "ADSK", "ADP", "AZO", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX",
    "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BNY", "BA", "BKNG",
    "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP", "CHRW",
    "CDNS", "CPT", "COF", "CAH", "CCL", "CARR", "CVNA", "CASY", "CAT", "CBOE",
    "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX",
    "CMG", "CB", "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO", "C", "CFG",
    "CLX", "CME", "CMS", "KO", "CTSH", "COHR", "COIN", "CL", "CMCSA", "FIX",
    "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP",
    "COST", "CRH", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG",
    "DVA", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG",
    "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD",
    "ETN", "EBAY", "ECL", "EIX", "EW", "ELV", "EME", "EMR", "ETR",
    "EOG", "EQT", "EFX", "EQIX", "ERIE", "ESS", "EL", "EG", "EVRG", "ES",
    "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST",
    "FRT", "FDX", "FERG", "FIS", "FITB", "FSLR", "FE", "FLEX",
    "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE",
    "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN",
    "GL", "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY",
    "HPE", "HLT", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB",
    "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW", "INCY", "IR", "PODD",
    "INTC", "IBKR", "ICE", "IFF", "IP", "INTU", "ISRG", "IVZ", "INVH", "IQV",
    "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "KVUE", "KDP",
    "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX",
    "LH", "LRCX", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV", "LMT",
    "L", "LOW", "LULU", "LITE", "LYB", "MTB", "MPC", "MAR", "MLM",
    "MRVL", "MAS", "MA", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET",
    "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "TAP", "MDLZ", "MPWR",
    "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM",
    "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH",
    "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON",
    "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PH", "PAYX",
    "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "PPG",
    "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM",
    "PWR", "QCOM", "DGX", "RL", "RJF", "RDDT", "RTX", "O", "REG",
    "REGN", "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL", "ROP", "ROST",
    "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "NOW", "SHW",
    "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX",
    "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW",
    "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN", "TPL",
    "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG", "TRV", "TRMB",
    "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS",
    "URI", "UNH", "UHS", "VLO", "VEEV", "VTR", "VLTO", "VRSN", "VRSK", "VZ",
    "VRTX", "VRT", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW",
    "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST",
    "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM",
    "ZBRA", "ZBH", "ZTS",
]))
# ⭐ 지정학적 충격(전쟁 등)이 주가로 전달되는 경로를 관측 가능한 변수로 추가.
# "전쟁이 났다"는 뉴스 자체는 과거 시점 데이터를 구할 수 없어 피처로 쓸 수 없지만,
# 전쟁이 주가에 영향을 주는 통로인 유가·금·달러는 매일 관측되고 과거 이력도 완전해
# 백테스트가 가능합니다. (중동 전쟁 -> 유가 급등 -> 정유주↑/항공주↓ 같은 경로)
MACRO_TICKERS = ["^VIX", "^TNX", "^GSPC", "CL=F", "GC=F", "DX-Y.NYB"]

def format_price(price, ticker):
    if str(ticker).endswith(".KS"):
        return f"₩{price:,.0f}"
    return f"${price:,.2f}"

def standardize_ohlcv(df_input: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    if df_input is None or df_input.empty:
        return pd.DataFrame()
    df = df_input.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker and ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0).copy()
        elif ticker and ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1).copy()
        else:
            df.columns = [col[0] for col in df.columns]
    
    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in req_cols):
        return pd.DataFrame()
    return df[req_cols]

def get_calibrated_cv(base_model, cv):
    try:
        return CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=cv)

# ==========================================
# 1. 데이터 수집 및 피처 엔지니어링 / 라벨링
# ==========================================
@st.cache_data(show_spinner=False, ttl=900)
def download_all_data(tickers: tuple, start_date: str):
    for attempt in range(3):
        try:
            df = yf.download(list(tickers), start=start_date, group_by='ticker', auto_adjust=True, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            logging.error(f"Batch download failed (attempt {attempt+1}/3): {e}")
        time.sleep(1.0 * (attempt + 1))
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=900)
def download_macro_data(start_date: str):
    for attempt in range(3):
        try:
            df = yf.download(MACRO_TICKERS, start=start_date, group_by='ticker', auto_adjust=True, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            logging.error(f"Macro download failed (attempt {attempt+1}/3): {e}")
        time.sleep(1.0 * (attempt + 1))
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=900)
def compute_market_regime(macro_prepared: pd.DataFrame) -> dict:
    """⭐ 시장 국면(regime) 감지.

    코로나·전쟁 같은 외부 충격은 '예측'이 불가능합니다. 발생 시점을 미리 알 수
    있다는 주장은 검증할 수 없고, 과거 뉴스를 붙이면 룩어헤드 편향만 생깁니다.
    대신 이 함수는 **이미 관측 가능한 시장 데이터만으로 지금이 어떤 국면인지
    판정**합니다. 즉 위기를 맞히는 것이 아니라, 위기 상태임을 감지해
    시스템이 자동으로 보수적으로 전환하도록 하는 장치입니다.

    판정 근거 (모두 과거 관측치):
      - VIX 수준: 시장이 예상하는 향후 변동성
      - S&P500 고점 대비 낙폭(drawdown): 실제로 얼마나 빠졌는지
      - 최근 실현 변동성: 실제 출렁임 정도
    """
    try:
        if macro_prepared is None or macro_prepared.empty:
            return {}
        sp = macro_prepared["SP500_Close"].dropna()
        vix_s = macro_prepared["VIX_Close"].dropna()
        if sp.empty or vix_s.empty:
            return {}

        roll_max = sp.rolling(252, min_periods=60).max()
        dd_s = (sp / roll_max - 1.0) * 100
        vol_s = sp.pct_change().rolling(20).std() * np.sqrt(252) * 100

        vix = float(vix_s.iloc[-1])
        dd = float(dd_s.iloc[-1]) if not dd_s.dropna().empty else 0.0
        vol = float(vol_s.iloc[-1]) if not vol_s.dropna().empty else 0.0

        if vix >= 30.0 or dd <= -15.0:
            state, emoji = "위기", "🌊"
            desc = "시장 전체가 크게 흔들리는 구간입니다. 개별 종목 분석보다 시장 전체 흐름이 주가를 좌우합니다."
            mult = 2.0
        elif vix >= 22.0 or dd <= -7.0:
            state, emoji = "주의", "⛅"
            desc = "변동성이 평소보다 높습니다. 평소보다 신중하게 접근하는 편이 좋습니다."
            mult = 1.4
        else:
            state, emoji = "평온", "☀️"
            desc = "시장이 비교적 안정적입니다. 개별 종목의 특성이 상대적으로 잘 드러나는 구간입니다."
            mult = 1.0

        return {
            "state": state, "emoji": emoji, "desc": desc, "threshold_mult": mult,
            "vix": vix, "drawdown": dd, "volatility": vol,
            "as_of": sp.index[-1],
        }
    except Exception as e:
        logging.warning(f"Market regime calc failed: {e}")
        return {}

@st.cache_data(show_spinner=False, ttl=900)
def prepare_macro(macro_df: pd.DataFrame) -> pd.DataFrame:
    """⭐ 최적화: 거시지표(VIX/금리/S&P500) 파싱을 종목마다 반복하지 않고
    한 번만 수행해 캐싱합니다. (기존에는 41개 종목마다 같은 작업을 반복)"""
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()
    try:
        m_df = macro_df.copy()
        m_df.index = pd.to_datetime(m_df.index).tz_localize(None).normalize()
        out = pd.DataFrame(index=m_df.index)
        out["VIX_Close"] = standardize_ohlcv(m_df, "^VIX")["Close"]
        out["TNX_Close"] = standardize_ohlcv(m_df, "^TNX")["Close"]
        out["SP500_Close"] = standardize_ohlcv(m_df, "^GSPC")["Close"]

        # ⭐ 지정학적 충격 전달 경로. 개별 시리즈가 실패해도 전체가 무너지지 않도록
        # 각각 따로 감싸고, 실패 시 NaN으로 두어 이후 단계에서 자연스럽게 제외되게 합니다.
        for col, sym in [("Oil_Close", "CL=F"), ("Gold_Close", "GC=F"), ("Dollar_Close", "DX-Y.NYB")]:
            try:
                s = standardize_ohlcv(m_df, sym)
                out[col] = s["Close"] if not s.empty else np.nan
            except Exception:
                out[col] = np.nan
        return out
    except Exception as e:
        logging.warning(f"Macro prepare failed: {e}")
        return pd.DataFrame()

def compute_features(df_raw: pd.DataFrame, macro_prepared: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, bool]:
    try:
        df = standardize_ohlcv(df_raw, ticker)
        if df.empty or df.isna().all().all():
            return pd.DataFrame(), False
            
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.ffill(inplace=True)

        is_macro_fallback = False
        if macro_prepared is not None and not macro_prepared.empty:
            # 한국 종목은 미국 장 마감 시점 차이로 전일 매크로를 참조해야 함(정보 누출 방지)
            shift = 1 if str(ticker).endswith(".KS") else 0
            try:
                m = macro_prepared.shift(shift)
                df = df.join(m)
                df.ffill(inplace=True)
                df["Market_Relative"] = (df["Close"].pct_change(10) - df["SP500_Close"].pct_change(10)) * 100

                # ⭐ 지정학적 충격 반영 지표
                # Oil_Change_20: 지금 유가 충격이 진행 중인지 (시장 전체 공통)
                # Oil_Beta_60: 이 종목이 유가 변동에 얼마나 민감한지 (종목별로 다름)
                if "Oil_Close" in df.columns and df["Oil_Close"].notna().sum() > 60:
                    df["Oil_Change_20"] = df["Oil_Close"].pct_change(20) * 100
                    _stock_ret = df["Close"].pct_change()
                    _oil_ret = df["Oil_Close"].pct_change()
                    _cov = _stock_ret.rolling(60).cov(_oil_ret)
                    _var = _oil_ret.rolling(60).var()
                    df["Oil_Beta_60"] = (_cov / _var.replace(0, np.nan)).clip(-3, 3)
                else:
                    df["Oil_Change_20"] = 0.0
                    df["Oil_Beta_60"] = 0.0
            except Exception as e:
                logging.warning(f"Macro join failed for {ticker}: {e}")
                is_macro_fallback = True
                df["VIX_Close"] = 20.0
                df["TNX_Close"] = 42.0
                df["Market_Relative"] = 0.0
                df["Oil_Change_20"] = 0.0
                df["Oil_Beta_60"] = 0.0
        else:
            is_macro_fallback = True
            df["VIX_Close"] = 20.0
            df["TNX_Close"] = 42.0
            df["Market_Relative"] = 0.0
            df["Oil_Change_20"] = 0.0
            df["Oil_Beta_60"] = 0.0

        # 보조지표 연산
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["STD_20"] = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["SMA_20"] + (df["STD_20"] * 2)
        df["BB_Lower"] = df["SMA_20"] - (df["STD_20"] * 2)
        
        # 오버나이트 갭 리스크 (시가 vs 전일 종가 괴리율 추적)
        df["Overnight_Gap"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)).abs() * 100
        
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI"] = df["RSI"].fillna(50)
        
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD_Hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

        for _c in ["Oil_Change_20", "Oil_Beta_60"]:
            if _c not in df.columns:
                df[_c] = 0.0
            df[_c] = df[_c].fillna(0.0)

        df["Momentum_5"] = df["Close"].pct_change(5) * 100
        df["Volume_Change"] = df["Volume"].pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        df["Volatility"] = df["Close"].pct_change().rolling(10).std() * 100

        min_low = df['Low'].rolling(window=5).min()
        max_high = df['High'].rolling(window=5).max()
        df['Stoch_K'] = ((df['Close'] - min_low) / (max_high - min_low + 1e-8) * 100).rolling(window=3).mean()
        
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()
        df["ATR_Pct"] = (df["ATR_14"] / df["Close"]) * 100

        df.dropna(inplace=True)

        # 익일 시가 체결 포인트-인-타임 라벨링
        df["Next_Open"] = df["Open"].shift(-1)
        
        n = len(df)
        next_open = df["Next_Open"].to_numpy()
        atr = df["ATR_14"].to_numpy()
        
        target = np.zeros(n)
        realized_rets = np.zeros(n)
        resolved = np.zeros(n, dtype=bool)

        for j in range(0, LOOKAHEAD):
            idx = j + 1
            tp = next_open + (atr * 1.5)
            sl = next_open - (atr * 1.5)
            
            future_low = df["Low"].shift(-idx).to_numpy()
            future_high = df["High"].shift(-idx).to_numpy()
            
            loss_now = (~resolved) & (future_low <= sl)
            profit_now = (~resolved) & (~loss_now) & (future_high >= tp)
            
            realized_rets[loss_now] = ((sl - next_open) / (next_open + 1e-8))[loss_now]
            realized_rets[profit_now] = ((tp - next_open) / (next_open + 1e-8))[profit_now]
            
            target[loss_now] = -1
            target[profit_now] = 1
            resolved = resolved | loss_now | profit_now

        expired = (~resolved)
        realized_rets[expired] = (df["Close"].shift(-LOOKAHEAD) / (next_open + 1e-8) - 1.0)[expired]

        target[n - LOOKAHEAD:] = np.nan
        realized_rets[n - LOOKAHEAD:] = np.nan
        
        df["Target"] = target
        df["Realized_Ret"] = realized_rets
        return df, is_macro_fallback
    except Exception as e:
        logging.error(f"Error computing features for {ticker}: {e}")
        return pd.DataFrame(), False

@st.cache_data(show_spinner=False, ttl=86400)
def get_fundamentals(ticker: str) -> dict:
    """⭐ PER(주가수익비율)/PBR(주가순자산비율) 조회.
    yfinance의 .info는 시세 다운로드(.history)보다 느리고 실패가 잦아서,
    하루 단위로 캐싱하고 실패 시 None을 반환해 파이프라인이 멈추지 않게 합니다.
    (재무 데이터는 하루에도 여러 번 바뀌지 않으므로 TTL을 길게 잡아도 무방합니다)"""
    try:
        info = yf.Ticker(ticker).info
        per = info.get("trailingPE")
        pbr = info.get("priceToBook")

        # ⭐ yfinance는 적자 기업의 trailingPE를 아예 반환하지 않습니다(None).
        # 그러면 "데이터 없음"과 "적자라서 PER이 음수"가 화면에서 구분되지 않으므로,
        # EPS로 직접 계산해 음수 PER도 표시하고 적자 여부를 따로 알려줍니다.
        eps = info.get("trailingEps")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        is_loss = False
        if per is None and eps is not None and price is not None and eps != 0:
            per = float(price) / float(eps)
        if eps is not None and float(eps) < 0:
            is_loss = True

        roe = info.get("returnOnEquity")

        return {
            "per": float(per) if per is not None else None,
            "pbr": float(pbr) if pbr is not None else None,
            "roe": float(roe) * 100 if roe is not None else None,
            "is_loss": is_loss,
        }
    except Exception as e:
        logging.warning(f"Fundamentals fetch failed for {ticker}: {e}")
        return {"per": None, "pbr": None, "roe": None, "is_loss": False}

def format_per_pbr(fund: dict) -> str:
    per, pbr = fund.get("per"), fund.get("pbr")
    pbr_txt = f"{pbr:.1f}" if pbr is not None else "-"
    if per is None:
        per_txt = "-"
    elif per < 0:
        # ⭐ [사용자 요청] 가공 없이 실제 PER 숫자를 그대로 보여줍니다(적자 표시만 유지).
        per_txt = f"{per:.1f}(적자)"
    else:
        per_txt = f"{per:.1f}"
    if per is None and pbr is None:
        return "정보 없음"
    return f"{per_txt} / {pbr_txt}"

def _pick_row(df: pd.DataFrame, candidates: list):
    """yfinance 재무제표의 행 이름이 종목/버전마다 달라서(예: 'Net Income' vs
    'Net Income Common Stockholders') 후보 목록 중 존재하는 첫 행을 반환합니다."""
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    for name in candidates:
        matches = [i for i in df.index if name.lower() in str(i).lower()]
        if matches:
            return df.loc[matches[0]]
    return None

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_raw_statements(ticker: str):
    """yfinance 원본 재무제표 4종(연간 손익/재무상태, 분기 손익/재무상태)만 가져옵니다.

    ⭐ 공시 지연·성장률 공식 같은 '가공 방식'과 '네트워크 수집'을 분리하기 위한
    함수입니다. ablation 연구에서 지연일수나 공식을 바꿔가며 여러 번 돌려야
    하는데, 이게 합쳐져 있으면 설정 하나 바꿀 때마다 500종목을 다시 받아야
    합니다(종목당 4회 호출 × 500종목 = 2000회). 원본만 따로 캐싱해두면
    가공 방식을 바꿔도 네트워크 호출이 0회입니다.
    """
    tk = yf.Ticker(ticker)
    def _safe(getter):
        try:
            v = getter()
            return v if v is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return (_safe(lambda: tk.income_stmt),
            _safe(lambda: tk.balance_sheet),
            _safe(lambda: tk.quarterly_income_stmt),
            _safe(lambda: tk.quarterly_balance_sheet))

@st.cache_data(show_spinner=False, ttl=86400)
def get_fundamental_history(ticker: str,
                            lag_quarterly: int = None,
                            lag_annual: int = None,
                            growth_formula: str = "abs_base") -> pd.DataFrame:
    """⭐ 가치투자 모델용 과거 재무 시계열 복원 (연간 + 분기 결합).

    [왜 연간까지 쓰는가]
    yfinance 무료 분기 재무제표는 보통 4~5분기(약 1년)치만 제공합니다.
    TTM(4분기 합계)을 만들면 유효 시점이 1~2개만 남고, 그마저 최근 날짜라
    6개월 뒤 수익률(정답 라벨)을 만들 수 없어 학습 표본이 0이 됩니다.
    그래서 4년치를 주는 **연간 재무제표를 기본 축**으로 쓰고, 최근 구간만
    분기 TTM으로 보강해 백테스트 가능한 기간을 확보합니다.

    [ablation 파라미터] - 논문의 '결함별 기여도 분해'용. 기본값은 항상 올바른 설정.
      lag_quarterly / lag_annual : 공시 지연 일수. 둘 다 None이면 SEC 기한 기반
          기본값(45 / 90)을 씁니다. 예전처럼 둘 다 45로 주면 연간 재무에
          룩어헤드 누수가 생기고, 그 효과를 측정할 수 있습니다.
      growth_formula : "abs_base"(정상) | "naive"(부호 뒤집히는 원래 공식)

    [한계]
      - 소급 수정: yfinance 값은 '현재 시점에서 본 과거 재무'라 당시 발표치와
        다를 수 있음 (무료 데이터의 구조적 한계, 논문에 명시 필요)
    """
    lag_q = REPORT_LAG_QUARTERLY if lag_quarterly is None else lag_quarterly
    lag_a = REPORT_LAG_ANNUAL if lag_annual is None else lag_annual

    def _extract(inc, bs, is_annual: bool) -> pd.DataFrame:
        if inc is None or inc.empty or bs is None or bs.empty:
            return pd.DataFrame()
        net_income = _pick_row(inc, ["Net Income", "Net Income Common Stockholders",
                                     "Net Income From Continuing Operation Net Minority Interest"])
        revenue = _pick_row(inc, ["Total Revenue", "Operating Revenue"])
        eps_row = _pick_row(inc, ["Diluted EPS", "Basic EPS"])
        equity = _pick_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])
        # ⭐ 주식수는 재무상태표에 없는 경우가 잦아 손익계산서의 평균주식수까지 폭넓게 탐색.
        # (이 값이 없으면 BPS -> Book_to_Price가 전부 결측이 되어 학습이 통째로 무너짐)
        shares = _pick_row(bs, ["Ordinary Shares Number", "Share Issued",
                                "Common Stock Shares Outstanding", "Shares Outstanding"])
        if shares is None:
            shares = _pick_row(inc, ["Diluted Average Shares", "Basic Average Shares",
                                     "Weighted Average Shares Diluted", "Weighted Average Shares"])
        if net_income is None or equity is None:
            return pd.DataFrame()

        d = pd.DataFrame({
            "NetIncome": pd.to_numeric(net_income, errors="coerce"),
            "Equity": pd.to_numeric(equity, errors="coerce"),
        })
        if revenue is not None:
            d["Revenue"] = pd.to_numeric(revenue, errors="coerce")
        if eps_row is not None:
            d["EPS_P"] = pd.to_numeric(eps_row, errors="coerce")
        if shares is not None:
            d["Shares"] = pd.to_numeric(shares, errors="coerce")
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d = d.sort_index()
        if d.empty:
            return pd.DataFrame()

        if "EPS_P" not in d.columns and "Shares" in d.columns:
            d["EPS_P"] = d["NetIncome"] / d["Shares"].replace(0, np.nan)

        if is_annual:
            # 연간 수치는 그 자체가 1년치이므로 그대로 TTM으로 사용
            d["EPS_TTM"] = d.get("EPS_P", np.nan)
            d["NetIncome_TTM"] = d["NetIncome"]
            d["Revenue_TTM"] = d["Revenue"] if "Revenue" in d.columns else np.nan
            growth_lag = 1
        else:
            if len(d) < 4:
                return pd.DataFrame()
            d["EPS_TTM"] = d["EPS_P"].rolling(4).sum() if "EPS_P" in d.columns else np.nan
            d["NetIncome_TTM"] = d["NetIncome"].rolling(4).sum()
            d["Revenue_TTM"] = d["Revenue"].rolling(4).sum() if "Revenue" in d.columns else np.nan
            growth_lag = 4

        d["Profit_Margin"] = (d["NetIncome_TTM"] / d["Revenue_TTM"].replace(0, np.nan)) * 100 \
            if "Revenue_TTM" in d.columns else np.nan
        d["ROE"] = (d["NetIncome_TTM"] / d["Equity"].replace(0, np.nan)) * 100
        d["BPS"] = d["Equity"] / d["Shares"].replace(0, np.nan) if "Shares" in d.columns else np.nan

        # ⭐⭐ [버그 수정] 이익 성장률: 직전 순이익이 음수면 pct_change의 부호가
        # 뒤집힙니다. 분모가 음수라서 그렇습니다:
        #   적자 -200 -> 흑자 +20 (개선)  : (20-(-200))/(-200) = -110%  ← 최악으로 기록
        #   적자  -50 -> 적자 -200 (악화)  : (-200-(-50))/(-50) = +300%  ← 고성장으로 기록
        # 즉 모델에게 "적자가 커지는 회사가 고성장"이라고 가르치고 있었습니다.
        # 분모를 절댓값으로 바꾸면 모든 경우에 방향이 맞고, 흑자 구간 동작은
        # 기존과 완전히 동일합니다(분모가 양수면 |x| = x 이므로).
        _ni = d["NetIncome_TTM"]
        if growth_formula == "naive":
            # ⚠️ ablation 전용: 부호가 뒤집히는 원래(잘못된) 공식
            d["Earnings_Growth"] = _ni.pct_change(growth_lag) * 100
        else:
            _prev = _ni.shift(growth_lag)
            d["Earnings_Growth"] = (_ni - _prev) / _prev.abs().replace(0, np.nan) * 100

        out = d[["EPS_TTM", "BPS", "ROE", "Profit_Margin", "Earnings_Growth"]].copy()
        return out.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    try:
        inc_a, bs_a, inc_q, bs_q = fetch_raw_statements(ticker)
        parts = []
        # ⭐⭐ [버그 수정] 공시 지연을 '합치기 전에, 재무제표 종류별로' 적용합니다.
        # 예전에는 연간·분기를 먼저 합친 뒤 똑같이 45일을 더했는데, 10-K(연간)의
        # 실제 SEC 제출 기한은 60~90일이라 연간 재무를 실제 공시 전에 쓰는
        # 룩어헤드 누수였습니다. 무료 데이터는 과거 구간일수록 연간에 의존하고,
        # 하필 그 구간이 백테스트 학습 구간이라 영향이 큽니다.

        # 연간(긴 이력)을 기본 축으로
        try:
            a = _extract(inc_a, bs_a, is_annual=True)
            if not a.empty:
                a.index = a.index + pd.Timedelta(days=lag_a)
                parts.append(a)
        except Exception as e:
            logging.warning(f"Annual financials failed for {ticker}: {e}")
        # 분기(최근 구간 보강)
        try:
            q = _extract(inc_q, bs_q, is_annual=False)
            if not q.empty:
                q.index = q.index + pd.Timedelta(days=lag_q)
                parts.append(q)
        except Exception as e:
            logging.warning(f"Quarterly financials failed for {ticker}: {e}")

        if not parts:
            return pd.DataFrame()
        # 같은 사용가능일에 연간·분기가 겹치면 더 최신인 분기 값을 우선(뒤에 붙였으므로 last)
        out = pd.concat(parts).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        if out.empty:
            return pd.DataFrame()
        return out
    except Exception as e:
        logging.warning(f"Fundamental history failed for {ticker}: {e}")
        return pd.DataFrame()

def build_value_panel(price_df: pd.DataFrame, fund_df: pd.DataFrame,
                      macro_prepared: pd.DataFrame, ticker: str,
                      include_fundamentals: bool = True) -> pd.DataFrame:
    """일별 주가(+ 재무 시계열)를 결합해 지표 패널을 만듭니다.

    ⭐ PER/PBR 대신 그 역수인 이익수익률(E/P)·순자산비율(B/P)을 씁니다.
    PER은 적자일 때 음수가 되면서 '아주 저평가'처럼 보이는 역전이 생기고,
    이익이 0에 가까우면 무한대로 발산합니다. 역수를 쓰면 적자는 자연스럽게
    음수(=나쁨), 고평가는 0에 가까운 값(=나쁨)으로 단조롭게 정렬됩니다.

    include_fundamentals=False (논문용 스윙 연구 전용): 재무제표를 아예
    조회·결합하지 않고 가격만으로 계산되는 지표(SWING_FEATURES)만 만든다.
    yfinance 무료 재무제표의 ~4년 이력 제약에서 벗어나 가격 데이터가
    허용하는 만큼(SWING_YEARS) 기간을 늘릴 수 있다. 기본값(True)은 기존
    동작 그대로라 app.py의 가치투자 랭킹 탭에는 영향이 없다.
    """
    try:
        px = standardize_ohlcv(price_df, ticker)
        if px.empty:
            return pd.DataFrame()
        px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
        px = px.ffill()

        if not include_fundamentals:
            d = px.copy()
            d["Momentum_126"] = d["Close"].pct_change(126) * 100
            d["Volatility_60"] = d["Close"].pct_change().rolling(60).std() * 100
            if macro_prepared is not None and not macro_prepared.empty:
                shift = 1 if str(ticker).endswith(".KS") else 0
                m = macro_prepared.shift(shift)
                d = d.join(m[["SP500_Close"]])
                d["Market_Relative"] = (d["Close"].pct_change(126) - d["SP500_Close"].pct_change(126)) * 100
            else:
                d["Market_Relative"] = 0.0
            d = d.replace([np.inf, -np.inf], np.nan)
            d["Fwd_Return"] = (d["Close"].shift(-VALUE_HORIZON) / d["Close"] - 1.0) * 100
            d["Ticker"] = ticker
            keep = SWING_FEATURES + ["Close", "Fwd_Return", "Ticker"]
            keep = [c for c in dict.fromkeys(keep) if c in d.columns]
            return d[keep]

        if fund_df is None or fund_df.empty:
            return pd.DataFrame()

        f = fund_df.copy()
        f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
        f = f[~f.index.duplicated(keep="last")].sort_index()
        # 발표 시점 이후로만 값이 유효하도록 전방 채움
        f = f.reindex(px.index.union(f.index)).ffill().reindex(px.index)

        d = px.join(f)
        d["Earnings_Yield"] = (d["EPS_TTM"] / d["Close"]) * 100
        d["Book_to_Price"] = (d["BPS"] / d["Close"]) * 100
        d["Momentum_126"] = d["Close"].pct_change(126) * 100
        d["Volatility_60"] = d["Close"].pct_change().rolling(60).std() * 100

        # ⭐ [사용자 요청] 3단계(피처 확장) - 린치식 PEG를 벤치마크뿐 아니라
        # 모델 입력 피처로도 넣습니다. 값이 클수록(=PEG 작을수록) 좋음.
        # 적자거나 성장이 없으면(PEG 정의 불가) 결측으로 두고, 다른 피처처럼
        # 같은 날짜 안에서 랭킹화될 때 중립(0.5)으로 채워집니다.
        with np.errstate(divide="ignore", invalid="ignore"):
            _peg = (100.0 / d["Earnings_Yield"]) / d["Earnings_Growth"]
        _peg_valid = (d["Earnings_Yield"] > 0) & (d["Earnings_Growth"] > 0) & (_peg > 0)
        d["PEG_Inv"] = np.where(_peg_valid, 1.0 / _peg, np.nan)

        if macro_prepared is not None and not macro_prepared.empty:
            shift = 1 if str(ticker).endswith(".KS") else 0
            m = macro_prepared.shift(shift)
            d = d.join(m[["SP500_Close"]])
            d["Market_Relative"] = (d["Close"].pct_change(126) - d["SP500_Close"].pct_change(126)) * 100
        else:
            d["Market_Relative"] = 0.0

        d = d.replace([np.inf, -np.inf], np.nan)

        # 6개월 뒤 수익률(정답 라벨의 재료). 시장 대비 초과수익 여부로 판정하기 위해 보관.
        d["Fwd_Return"] = (d["Close"].shift(-VALUE_HORIZON) / d["Close"] - 1.0) * 100
        d["Ticker"] = ticker
        keep = VALUE_FEATURES + ["Close", "Fwd_Return", "Ticker", "EPS_TTM", "BPS", "ROE"]
        keep = [c for c in dict.fromkeys(keep) if c in d.columns]
        return d[keep]
    except Exception as e:
        logging.warning(f"Value panel failed for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600, max_entries=10)
def run_value_model(panel: pd.DataFrame, horizon_override: int = None,
                    pvalue_method: str = "fama_macbeth",
                    eval_mode: str = "overlap_nw",
                    train_tickers=None, eval_tickers=None,
                    feature_subset=None, rf_params=None,
                    require_fundamentals: bool = True):
    """⭐ 횡단면(cross-sectional) 가치투자 모델.

    단기 모델과 결정적으로 다른 점:
      - 종목마다 별도 모델(표본 8개)이 아니라 전 종목을 한 모델로 통합 학습(표본 300+)
      - 예측 대상이 '오를까?'가 아니라 '같은 날 다른 종목들보다 잘할까?'(상대 성과)
        → 시장 전체 등락(분산의 대부분)이 상쇄되어 신호 대 잡음비가 개선됨
      - 검증은 날짜 기준으로 분할해 미래 정보가 과거로 새지 않도록 함
    """
    try:
        # ⭐ 피처 단위 내성:
        # 무료 데이터는 종목/항목마다 결측이 제각각이라, 모든 피처를 한꺼번에
        # dropna 하면 항목 하나(예: 주식수 누락 -> Book_to_Price)만 비어도
        # 전체 행이 0이 되어 학습이 불가능해집니다.
        # 그래서 '쓸 수 있는 피처만 골라서' 학습하도록 합니다.
        # ⭐ [실험용] 하이퍼파라미터를 한 곳에서 정의하고 폴드/최종 모델이 같이 씁니다.
        # rf_params로 덮어쓸 수 있게 한 이유: 다중검정 연구에서 '연구자가 설정을
        # 여러 개 시도해보는' 행위를 재현해야 하기 때문입니다. 기본값은 기존과 동일.
        _rf_kwargs = dict(n_estimators=120, max_depth=3, min_samples_leaf=10,
                          max_features="sqrt", class_weight="balanced",
                          n_jobs=1, random_state=42)
        if rf_params:
            _rf_kwargs.update(rf_params)

        avail = [c for c in VALUE_FEATURES if c in panel.columns]

        if require_fundamentals:
            fund_avail = [c for c in FUNDAMENTAL_FEATURES if c in panel.columns]
            if not fund_avail:
                return {"error": "재무 지표를 전혀 가져오지 못했습니다"}

            # ⭐ [중요] 확보율은 '재무 데이터가 존재하는 구간' 기준으로 계산합니다.
            # 전체 기간 기준으로 계산하면, 재무 이력이 최근 일부에만 있을 때
            # 모든 재무 지표가 확보율 미달로 탈락하고 주가 지표만 남아
            # '가치투자'라는 이름으로 사실상 모멘텀 모델이 학습되는 문제가 있었습니다.
            _has_fund = panel[fund_avail].notna().any(axis=1)
            _base = panel[_has_fund]
            if _base.empty:
                return {"error": "재무 지표가 있는 구간이 없습니다"}
            # 재무 데이터가 있는 구간만 분석 대상으로 삼습니다.
            panel = _base

        coverage = {c: float(panel[c].notna().mean()) for c in avail}
        used_feats = [c for c in avail if coverage[c] >= 0.30]
        # ⭐ [실험용] 다중검정 연구에서 '연구자가 피처 조합을 바꿔보는' 행위를
        # 재현하기 위한 손잡이. 기본값(None)이면 기존 동작 그대로입니다.
        if feature_subset is not None:
            used_feats = [c for c in used_feats if c in set(feature_subset)]
        if require_fundamentals:
            used_fund = [c for c in used_feats if c in FUNDAMENTAL_FEATURES]
            if len(used_fund) < 2:
                return {"error": f"쓸 수 있는 재무 지표가 {len(used_fund)}개뿐입니다 (최소 2개 필요). "
                                 f"주가 지표만으로는 가치투자 분석이라 할 수 없어 중단했습니다."}
        _min_used = 3 if require_fundamentals else 2
        if len(used_feats) < _min_used:
            return {"error": f"쓸 수 있는 지표가 {len(used_feats)}개뿐입니다 (최소 {_min_used}개 필요)"}

        # ⭐ 결측 처리 전략 [중요]:
        # 지표별 확보율이 각각 48~94%여도, 8개를 '동시에' 갖춘 행만 남기면
        # 교집합이 급격히 줄어 학습 구간이 사라집니다(이전 실패 원인).
        # 횡단면 순위 모델에서는 결측 행을 버리는 대신, 같은 날짜 안에서
        # 순위를 매긴 뒤 빠진 값만 '중립(0.5)'으로 채우는 것이 표준적이고
        # 정보 손실도 훨씬 적습니다. 정답(Fwd_Return)이 없는 행만 제외합니다.
        df = panel.copy()
        # 최소한의 근거는 있어야 하므로, 지표가 너무 많이 빈 행은 제외
        _min_feats = max(2, len(used_feats) // 2)
        df = df[df[used_feats].notna().sum(axis=1) >= _min_feats]
        if df.empty or "Close" not in df.columns:
            return {"error": "재무 지표를 가진 데이터가 없습니다"}

        # ⭐ 예측 기간(horizon)을 데이터 길이에 맞춰 먼저 정한 뒤,
        # 그 기간으로 정답(미래 수익률)을 다시 계산합니다.
        # (기간만 줄이고 라벨은 6개월치를 쓰면, 학습/평가 간격보다 라벨이 더 멀리
        #  내다보게 되어 정보 누출이 발생하므로 반드시 함께 맞춰야 합니다.)
        _feat_dates = df.index.nunique()
        _target_horizon = horizon_override if horizon_override else VALUE_HORIZON
        horizon = _target_horizon
        if _feat_dates < _target_horizon * 2 + 30:
            horizon = max(21, int((_feat_dates - 20) // 3))
        if horizon < 21:
            return {"error": f"재무 데이터 기간이 {_feat_dates}일뿐이라 미래 수익률을 만들 수 없습니다 (최소 약 80일 필요)"}

        df = df.sort_index()
        # ⭐ [버그 수정] yfinance가 주는 데이터의 인덱스 이름이 "Date"인데,
        # 여기서 같은 이름의 컬럼을 또 만들면 groupby("Date")가
        # "인덱스인가 컬럼인가"로 모호해져 ValueError가 납니다.
        # (합성 데이터는 인덱스에 이름이 없어 테스트에서 재현되지 않았던 버그)
        df = df.rename_axis(None)
        df.index = df.index.rename(None)
        df["Date"] = df.index

        # 같은 날짜 안에서 피처를 순위화(0~1) 후, 결측은 중립값 0.5로 채움.
        # ⭐ [버그 수정] 이 순위화는 Fwd_Return(미래 수익률) dropna보다 먼저
        # 해야 합니다. Fwd_Return은 정의상 가장 최근 horizon(~6개월)치
        # 날짜엔 존재할 수 없는데, dropna를 먼저 하면 '오늘 기준 현재 점수'를
        # 낼 때도 그 최근 구간이 통째로 사라져서 몇 달 전 주가로 계산한
        # PER/PBR이 마치 최신인 것처럼 표시되는 문제가 있었습니다.
        # all_df(라벨 없이도 되는 전체 패널)는 '지금 점수'용으로 따로 보관하고,
        # 학습/검증용 df만 아래에서 라벨 dropna를 적용합니다.
        for c in used_feats:
            df[c + "_rank"] = df.groupby("Date")[c].rank(pct=True).fillna(0.5)
        rank_feats = [c + "_rank" for c in used_feats]
        all_df = df.copy()

        # ⭐ [사용자 요청] "예측력이 경쟁력"이라는 주장을 하려면 무작위·단순 규칙
        # 대비 실제로 더 나은지 숫자로 보여줘야 합니다. RandomForest와 같은
        # 평가 구간(OOF)에서 비교할 두 가지 규칙 기반 벤치마크 점수를 미리
        # 계산해둡니다. 학습이 필요 없는 고정 규칙이라 폴드마다 다시 만들
        # 필요 없이 한 번만 계산하면 됩니다.
        #   - 단순 저PER: Earnings_Yield(=100/PER)가 높을수록(=PER 낮을수록) 좋음
        #   - 린치 PEG: PER을 성장률로 나눈 값이 낮을수록 좋음. 적자거나
        #     성장이 없으면(PEG 정의 불가) 린치라면 사지 않을 종목이므로
        #     최하 점수로 처리합니다(중립이 아니라 "나쁨"으로 취급).
        if "Earnings_Yield" in df.columns:
            df["_bench_simple_per"] = df["Earnings_Yield"]
        else:
            df["_bench_simple_per"] = np.nan
        if "Earnings_Yield" in df.columns and "Earnings_Growth" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                _peg = (100.0 / df["Earnings_Yield"]) / df["Earnings_Growth"]
            _peg_valid = (df["Earnings_Yield"] > 0) & (df["Earnings_Growth"] > 0)
            df["_bench_lynch_peg"] = np.where(_peg_valid, -_peg, -1e6)
        else:
            df["_bench_lynch_peg"] = np.nan

        df["Fwd_Return"] = (
            df.groupby("Ticker")["Close"].shift(-horizon) / df["Close"] - 1.0
        ) * 100
        df = df.dropna(subset=["Fwd_Return"])
        if len(df) < 150 or df["Ticker"].nunique() < 5:
            return {"error": f"분석 가능한 데이터가 {len(df)}행 / {df['Ticker'].nunique() if len(df) else 0}종목뿐입니다 (최소 150행·5종목 필요)"}
        # ⭐ 같은 날짜 안에서 '시장 평균 대비 초과수익 여부'로 라벨 생성 (횡단면)
        med = df.groupby("Date")["Fwd_Return"].transform("median")
        df["Target"] = (df["Fwd_Return"] > med).astype(int)

        # ⭐ 퍼징 전략 (데이터 양에 맞춰 적응):
        # 라벨이 미래 horizon일을 참조하므로, 학습 구간과 평가 구간 사이에
        # 반드시 그만큼의 간격이 있어야 미래 정보가 과거로 새지 않습니다.
        #
        # ⭐⭐ [핵심 수정] 평가일 처리 방식 (eval_mode)
        # 예전에는 평가일을 horizon 간격으로 솎아내(test_dates[::horizon]) 라벨이
        # 겹치지 않게 만들었습니다. 통계적으로는 정직하지만 **가용 데이터의
        # 1/horizon만 쓰는** 셈이라, 독립 평가일이 3개까지 줄어 어떤 효과도
        # 검출할 수 없었습니다(검정력 사실상 0).
        #   - "overlap_nw"(기본): 평가일을 전부 쓰고, 겹침이 만드는 자기상관은
        #     Newey-West로 표준오차에서 보정합니다. 자산가격 문헌의 표준이며
        #     평가일이 3개 -> 수백 개로 늘어납니다.
        #   - "nonoverlap_iid": 예전 방식(솎아내기 + iid 가정). ablation 비교용.
        uniq_dates = np.array(sorted(df["Date"].unique()))
        n_dates = len(uniq_dates)
        if n_dates < horizon + 20:
            return {"error": f"검증 가능한 날짜가 {n_dates}일뿐입니다 (예측기간 {horizon}일 + 여유 20일 필요)"}

        max_folds = max(1, (n_dates - horizon) // horizon)
        # 솎아내기 방식에선 폴드 수가 곧 평가일 수라 상한이 치명적이었지만,
        # 겹침 허용 방식에선 폴드당 평가일이 많아 3폴드로도 충분합니다.
        n_splits = int(min(3, max_folds))
        nw_lag = horizon if eval_mode == "overlap_nw" else 0

        oof_true, oof_prob = [], []
        oof_simple_per, oof_lynch_peg = [], []  # ⭐ 벤치마크용, RF와 동일한 OOF 구간
        oof_eval_dates = []   # ⭐ Fama-MacBeth: 행마다 어느 평가일인지 기록
        n_eval_dates = 0
        for k in range(1, n_splits + 1):
            train_end = int(n_dates * k / (n_splits + 1))
            test_start = train_end + horizon                # 누출 차단 간격
            if test_start >= n_dates:
                continue
            test_end = int(n_dates * (k + 1) / (n_splits + 1)) if k < n_splits else n_dates
            test_end = max(test_end, test_start + 1)

            train_dates = uniq_dates[:train_end]
            test_dates = list(uniq_dates[test_start:test_end])
            if eval_mode == "nonoverlap_iid":
                # 예전 방식: 라벨이 겹치지 않도록 솎아냄 (데이터의 1/horizon만 사용)
                test_dates = test_dates[::horizon] if len(test_dates) > horizon else test_dates[:1]
            # overlap_nw: 솎아내지 않고 전부 사용. 겹침은 아래 Newey-West로 보정.
            if not test_dates:
                continue

            tr = df[df["Date"].isin(train_dates)]
            te = df[df["Date"].isin(test_dates)]
            # ⭐ [실험용] 종목 단위 분할. 날짜로만 나누면 같은 종목이 학습·평가에
            # 모두 등장해서, 모델이 일반적인 팩터 관계 대신 "이 종목은 대체로
            # 평균을 이긴다"는 종목별 습성을 외울 수 있습니다. 학습 종목과 평가
            # 종목을 분리하면 그 부분을 걸러낸 성능을 볼 수 있습니다.
            # (피처 순위와 라벨은 전체 유니버스 기준으로 계산된 값을 그대로 써서,
            #  두 조건 사이에 '평가 종목이 학습에 있었는지'만 달라지도록 합니다)
            if train_tickers is not None:
                tr = tr[tr["Ticker"].isin(train_tickers)]
            if eval_tickers is not None:
                te = te[te["Ticker"].isin(eval_tickers)]
            if len(tr) < 40 or len(te) < 10 or tr["Target"].nunique() < 2:
                continue
            n_eval_dates += len(test_dates)
            m = RandomForestClassifier(**_rf_kwargs)
            m.fit(tr[rank_feats], tr["Target"])
            oof_prob.extend(m.predict_proba(te[rank_feats])[:, 1])
            oof_true.extend(te["Target"].values)
            # 벤치마크는 학습이 필요 없으니 같은 평가 구간(te)에 규칙만 적용
            oof_simple_per.extend(te["_bench_simple_per"].values)
            oof_lynch_peg.extend(te["_bench_lynch_peg"].values)
            oof_eval_dates.extend(te["Date"].values)

        # ⭐ 학습 가능 여부와 검증 가능 여부를 분리합니다.
        # 무료 재무데이터는 이력이 짧아 '학습은 되는데 누출 없는 검증 구간은
        # 못 만드는' 경우가 흔합니다. 예전에는 이때 아무것도 보여주지 않았지만,
        # 그보다는 순위를 제공하되 '검증되지 않음'을 분명히 경고하는 편이 낫습니다.
        def _bench_auc(scores):
            """⭐ [사용자 요청] 무작위/단순 저PER/린치 PEG 대비 우리 모델이
            실제로 더 나은지 같은 OOF 구간에서 AUC로 비교합니다. 결측 점수는
            그 행만 제외하고(짝을 맞춰) 계산합니다."""
            y = np.asarray(oof_true, dtype=float)
            s = np.asarray(scores, dtype=float)
            mask = ~np.isnan(s)
            y, s = y[mask], s[mask]
            if len(set(y)) < 2 or len(y) < 10:
                return float("nan")
            return float(roc_auc_score(y, s))

        # ⭐⭐ [중대 버그 수정] Fama-MacBeth 방식으로 유의성 검정
        #
        # [기존 방식의 문제]
        # 예전에는 모든 평가 행(종목×날짜)을 한 덩어리로 모아 AUC를 구하고,
        # Hanley-McNeil 공식에 n_pos/n_neg로 '행 수'를 그대로 넣었습니다.
        # 그런데 그 공식은 관측치가 서로 독립이라고 가정하는데, 여기 행들은
        # 전혀 독립이 아닙니다:
        #   1) 같은 날짜의 종목들은 같은 시장 환경을 공유하고,
        #   2) 라벨이 '그날 중앙값 초과'라서 날짜마다 정확히 절반이 1로 강제되며,
        #   3) 같은 종목이 여러 평가일에 반복 등장합니다.
        # 실제 독립 단위는 '평가 날짜 수'(보통 3~10개)인데 수백~수천 행을
        # 독립 표본으로 세는 바람에, 표준오차가 20배 이상 과소평가되어
        # 사실상 무작위 수준인 결과도 'p<0.05 유의함'으로 표시됐습니다.
        #
        # [수정 방식 - 횡단면 자산가격 연구의 표준]
        # 날짜별로 따로 AUC를 구한 뒤, 그 '날짜별 AUC의 시계열'이 0.5와
        # 다른지 t검정합니다(자유도 = 날짜수-1). 날짜 안의 종속성은 날짜별
        # 통계량 하나로 압축되므로 문제가 사라지고, 검정력이 실제 확보한
        # 독립 시점 수에 정직하게 좌우됩니다.
        # 구현과 자체 검증은 stats_utils.py에 있습니다(`py stats_utils.py` 실행 시
        # 합성 데이터로 "기존 방식은 오탐 45%, 이 방식은 5%"를 재현해 확인합니다).
        def _fama_macbeth(scores):
            """날짜별 AUC -> 그 시계열의 t검정. (평균AUC, t값, p값, 유효날짜수) 반환.
            nw_lag > 0 이면 겹치는 평가구간의 자기상관을 Newey-West로 보정합니다."""
            return fama_macbeth_auc(oof_true, scores, oof_eval_dates, nw_lag=nw_lag)

        if len(set(oof_true)) >= 2:
            # 전체를 모아 구한 AUC는 '판별력의 점추정'으로는 여전히 유효하므로 표시용으로 유지
            auc = float(roc_auc_score(oof_true, oof_prob))
            fm_auc, fm_t, pval, fm_T = _fama_macbeth(oof_prob)
            if pvalue_method == "iid":
                # ⚠️ ablation 전용: 행(종목×날짜) 수를 독립 표본으로 세는 원래(잘못된) 방식.
                # 같은 날짜 종목들의 종속성을 무시해 표준오차를 크게 과소평가합니다.
                _np_ = int(sum(oof_true)); _nn_ = len(oof_true) - _np_
                pval = float(calc_auc_pvalue(auc, _np_, _nn_))
            _rng = np.random.RandomState(42)
            _rand_scores = _rng.rand(len(oof_true))
            bench_auc = {
                "random": _bench_auc(_rand_scores),
                "simple_per": _bench_auc(oof_simple_per),
                "lynch_peg": _bench_auc(oof_lynch_peg),
            }
            # 벤치마크도 같은 방식으로 t검정해서 나란히 비교 가능하게 함
            bench_fm = {
                "random": _fama_macbeth(_rand_scores),
                "simple_per": _fama_macbeth(oof_simple_per),
                "lynch_peg": _fama_macbeth(oof_lynch_peg),
                "model": (fm_auc, fm_t, pval, fm_T),
            }
            validated = fm_T >= 2
        else:
            auc, pval, validated = float("nan"), float("nan"), False
            fm_auc, fm_t, fm_T = float("nan"), float("nan"), 0
            bench_auc = {"random": float("nan"), "simple_per": float("nan"), "lynch_peg": float("nan")}
            bench_fm = {}

        # 최종 모델: 전체 표본으로 학습
        final = RandomForestClassifier(**_rf_kwargs)
        final.fit(df[rank_feats], df["Target"])
        importances = dict(zip(used_feats, final.feature_importances_))

        # 가장 최근 날짜 기준으로 현재 점수 산출
        # ⭐ 라벨(Fwd_Return)이 없어도 되는 all_df를 씁니다 (위 버그 수정 참고).
        # df만 썼다면 최근 horizon(~6개월)치가 dropna로 빠져 있어 몇 달 전
        # 주가 기준 PER/PBR이 '현재' 점수인 것처럼 나왔습니다.
        latest_date = all_df["Date"].max()
        cur = all_df[all_df["Date"] == latest_date].copy()
        if cur.empty:
            return {"error": "가장 최근 시점의 재무 데이터를 찾지 못했습니다"}
        cur["점수"] = final.predict_proba(cur[rank_feats])[:, 1] * 100

        return {
            "scores": cur,
            "auc": auc,               # 전체를 모아 구한 AUC (점추정용)
            "pvalue": pval,           # ⭐ Fama-MacBeth 기반 p값 (날짜 수 기준)
            "fm_auc": fm_auc,         # 날짜별 AUC의 평균
            "fm_tstat": fm_t,         # 날짜별 AUC 시계열의 t값
            "fm_n_dates": fm_T,       # 검정에 실제 쓰인 독립 날짜 수 (= 유효 표본)
            "bench_fm": bench_fm,     # 벤치마크별 (평균AUC, t, p, 날짜수)
            "validated": validated,
            "bench_auc": bench_auc,
            "n_samples": int(len(df)),
            "n_eval_dates": int(n_eval_dates),
            "horizon": int(horizon),
            "used_feats": used_feats,
            "dropped_feats": [c for c in avail if c not in used_feats],
            "n_tickers": int(cur["Ticker"].nunique()),
            "importances": importances,
            "latest_date": pd.Timestamp(latest_date),
        }
    except Exception as e:
        logging.error(f"Value model error: {e}\n{traceback.format_exc()}")
        return {"error": f"계산 중 오류: {type(e).__name__} - {str(e)[:200]}"}

def process_ticker_data(df_all: pd.DataFrame, macro_prepared: pd.DataFrame, ticker: str, start_date: str,
                        fast_mode: bool = False) -> tuple[pd.DataFrame, bool]:
    """fast_mode=True(전체 랭킹 분석)일 때는 개별 재다운로드 재시도를 1회로 줄입니다.
    ⭐ 최적화 핵심: 랭킹 분석이 오래 걸리던 진짜 원인은 모델 학습이 아니라,
    일괄 다운로드에서 누락된 종목마다 최대 3회씩 네트워크 재요청을 하던 부분입니다."""
    try:
        df_raw = standardize_ohlcv(df_all, ticker)
        max_retries = 1 if fast_mode else 3
        if df_raw.empty or len(df_raw) < 100:
            for attempt in range(max_retries):
                try:
                    df_raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
                    df_raw = standardize_ohlcv(df_raw, ticker)
                    if not df_raw.empty and len(df_raw) >= 100:
                        break
                    time.sleep(0.3 * (attempt + 1) + random.uniform(0, 0.3))
                except Exception:
                    time.sleep(0.5 + random.uniform(0, 0.3))
                
        if df_raw.empty or len(df_raw) < 100:
            return pd.DataFrame(), False
        return compute_features(df_raw, macro_prepared, ticker)
    except Exception as e:
        logging.error(f"Error processing ticker {ticker}: {e}")
        return pd.DataFrame(), False

# ==========================================
# 2. 이산 로그-켈리 최적화 및 모델 파이프라인
# ==========================================
def solve_discrete_kelly(p_profit, p_loss, p_time, r_profit, r_loss, r_time):
    def obj(f):
        term_win = p_profit * np.log(max(1e-6, 1.0 + f * r_profit))
        term_loss = p_loss * np.log(max(1e-6, 1.0 - f * r_loss))
        term_time = p_time * np.log(max(1e-6, 1.0 + f * r_time))
        return -(term_win + term_loss + term_time)
    
    res = minimize_scalar(obj, bounds=(0.0, 0.5), method='bounded')
    return max(0.0, float(res.x)) if res.success else 0.0

def calc_auc_pvalue(auc, n_pos, n_neg):
    if pd.isna(auc) or n_pos < 5 or n_neg < 5:
        return 1.0
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    se = np.sqrt((auc * (1.0 - auc) + (n_pos - 1.0) * (q1 - auc**2) + (n_neg - 1.0) * (q2 - auc**2)) / (n_pos * n_neg))
    if se <= 0:
        return 1.0
    z = (auc - 0.5) / se
    return 1.0 - norm.cdf(z)

def estimate_slippage(atr_pct: float, vix: float, base_fee_dec: float) -> float:
    """⭐ Fix 8 [중대 버그]: 기존 슬리피지 공식이 거래비용을 과대 계상했습니다.
    기존: (ATR% x 0.10) + 수수료 + (VIX/100 x 0.05)
      -> VIX가 평범한 20일 때도 VIX 항만으로 1.0%p를 부과.
         대형 우량주 실제 체결 비용(0.01~0.05%)의 20~100배 수준이며,
         이 때문에 모든 종목의 기댓값(EV)이 구조적으로 -1.35%에서 시작해
         사실상 전 종목이 '조건 미달'로 강제 분류되고 있었습니다.

    수정: 실제 시장 미시구조에 맞춰 세 항목으로 재구성
      - 수수료/세금 (시장별 고정)
      - 시장충격: 변동성(ATR)에 비례하되 계수를 0.10 -> 0.02로 현실화
      - 변동성 프리미엄: VIX가 평시(20)를 초과한 만큼만 소액 가산
    """
    impact = (atr_pct / 100.0) * 0.02
    vix_extra = max(0.0, (vix - 20.0) / 100.0) * 0.01
    return base_fee_dec + impact + vix_extra

@st.cache_data(show_spinner=False, ttl=900, max_entries=5)
def run_cross_sectional_model(panel: pd.DataFrame):
    """⭐ 단기 신호의 횡단면(cross-sectional) 강화 모델.

    [왜 필요한가]
    종목마다 따로 모델을 학습하면 퍼징 후 독립 표본이 종목당 약 120개뿐이라
    9~11개 피처를 학습하기에 턱없이 부족합니다(AUC가 0.5 근처에 머무는 주원인).
    전 종목을 하나로 묶고 '같은 날 다른 종목보다 잘할까?'를 예측하면
      - 학습 표본이 수십 배로 늘고
      - 시장 전체 등락(분산의 대부분)이 상쇄되어 신호 대 잡음비가 개선됩니다.

    [검증 결과] 합성 데이터 실험에서 종목별 모델 대비 AUC +0.07 개선,
    신호가 없는 대조군에서는 0.5 근처를 유지해 정보 누출이 없음을 확인했습니다.
    """
    try:
        df = panel.dropna(subset=["Realized_Ret"]).copy()
        if len(df) < 500 or df["Ticker"].nunique() < 5:
            return None
        # ⭐ [버그 수정] 위와 동일: 인덱스 이름이 "Date"면 groupby가 모호해짐
        df = df.rename_axis(None)
        df.index = df.index.rename(None)  # ⭐ pandas 3.x 호환: 인덱스 이름이 "Date"와
        # 겹치면 groupby("Date")가 "인덱스 레벨인지 컬럼인지 모호하다"는 ValueError를 던짐
        df["Date"] = df.index
        med = df.groupby("Date")["Realized_Ret"].transform("median")
        df["CS_Target"] = (df["Realized_Ret"] > med).astype(int)

        feats = [c for c in FEATURES if c in df.columns]
        for c in feats:
            df[c + "_r"] = df.groupby("Date")[c].rank(pct=True).fillna(0.5)
        rank_feats = [c + "_r" for c in feats]

        dates = np.array(sorted(df["Date"].unique()))
        if len(dates) < LOOKAHEAD * 5:
            return None

        # 학습은 전체 날짜, 평가는 간격(gap) 확보 + 퍼징 -> 정직한 OOS 추정
        n_splits = 4
        edges = np.array_split(dates, n_splits + 1)
        yt, yp = [], []
        for k in range(1, len(edges)):
            tr_d = np.concatenate(edges[:k])
            cutoff = pd.Timestamp(tr_d[-1]) + pd.Timedelta(days=int(LOOKAHEAD * 1.5))
            te_d = [d for d in edges[k] if pd.Timestamp(d) >= cutoff][::LOOKAHEAD]
            if not te_d:
                continue
            tr = df[df["Date"].isin(tr_d)]
            te = df[df["Date"].isin(te_d)]
            if len(tr) < 100 or len(te) < 20 or tr["CS_Target"].nunique() < 2:
                continue
            m = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=20,
                                       max_features="sqrt", class_weight="balanced",
                                       n_jobs=1, random_state=42)
            m.fit(tr[rank_feats], tr["CS_Target"])
            yp.extend(m.predict_proba(te[rank_feats])[:, 1])
            yt.extend(te["CS_Target"].values)

        if len(set(yt)) < 2:
            return None
        auc = roc_auc_score(yt, yp)
        n_pos = int(sum(yt)); n_neg = len(yt) - n_pos
        pval = calc_auc_pvalue(auc, n_pos, n_neg)

        final = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=20,
                                       max_features="sqrt", class_weight="balanced",
                                       n_jobs=1, random_state=42)
        final.fit(df[rank_feats], df["CS_Target"])

        # 가장 최근 날짜 기준 현재 점수
        latest = df["Date"].max()
        cur = df[df["Date"] == latest].copy()
        if cur.empty:
            return None
        cur["cs_score"] = final.predict_proba(cur[rank_feats])[:, 1] * 100
        return {
            "scores": dict(zip(cur["Ticker"], cur["cs_score"])),
            "auc": float(auc), "pvalue": float(pval),
            "n_eval": int(len(yt)),
        }
    except Exception as e:
        logging.warning(f"Cross-sectional model failed: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=900, max_entries=100)
def run_model_pipeline(df: pd.DataFrame, ticker: str, kelly_cap: float = 25.0, is_macro_fallback: bool = False,
                        prob_threshold: float = 0.38, half_kelly_frac: float = 0.5):
    try:
        train_df = df.dropna(subset=["Target", "Realized_Ret"]).copy()
        if len(train_df) < 100:
            return None

        X = train_df[FEATURES]
        y = train_df["Target"].astype(int)

        X_purged_for_info = X[::LOOKAHEAD]
        y_purged_for_info = y[::LOOKAHEAD]
        class_counts = y_purged_for_info.value_counts().to_dict()
        total_samples = len(y_purged_for_info)
        dist_info = {k: f"{(v/total_samples)*100:.1f}%" for k, v in class_counts.items()}

        def build_model():
            # ⭐ Fix 7: CalibratedClassifierCV가 VotingClassifier를 감쌀 때 sklearn이
            # sample_weight를 전달하지 못하고 조용히 무시합니다(sklearn #21134).
            # 그 결과 확률 보정 단계에서 클래스 불균형 보정이 사라지는 문제가 있었습니다.
            # RandomForest에 class_weight='balanced'를 직접 지정해, sample_weight가
            # 누락되더라도 불균형 보정이 유지되도록 이중으로 안전장치를 둡니다.
            rf = RandomForestClassifier(n_estimators=35, max_depth=2, min_samples_leaf=3,
                                        max_features='sqrt', class_weight='balanced',
                                        n_jobs=1, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=35, learning_rate=0.03, max_depth=2, subsample=0.7, random_state=42)
            return VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")

        n_splits = 5 if len(X) > 300 else 3
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=LOOKAHEAD + 1)
        
        oof_y_true, oof_y_prob, oof_y_pred = [], [], []
        backtest_dates, backtest_trade_rets = [], []
        oof_class_0_rets = []
        valid_folds = 0

        base_fee_dec = 0.0015 if str(ticker).endswith(".KS") else 0.0005

        for train_idx, test_idx in tscv.split(X):
            X_train, y_train = X.iloc[train_idx][::LOOKAHEAD], y.iloc[train_idx][::LOOKAHEAD]
            X_test_purged = X.iloc[test_idx][::LOOKAHEAD]
            y_test_purged = y.iloc[test_idx][::LOOKAHEAD]
            test_dates_purged = train_df.iloc[test_idx][::LOOKAHEAD].index
            realized_rets_purged = train_df.iloc[test_idx][::LOOKAHEAD]['Realized_Ret'].values
            atr_pct_purged = train_df.iloc[test_idx][::LOOKAHEAD]['ATR_Pct'].values
            vix_purged = train_df.iloc[test_idx][::LOOKAHEAD]['VIX_Close'].values if 'VIX_Close' in train_df.columns else np.zeros(len(test_dates_purged))

            if len(X_train) < 15 or len(y_test_purged) < 3:
                continue

            if y_train.nunique() < 2 or (y_train == 1).sum() < 3 or (y_train == -1).sum() < 3:
                for d, r_real, y_val in zip(test_dates_purged, realized_rets_purged, y_test_purged):
                    backtest_dates.append(d)
                    backtest_trade_rets.append(0.0)
                    if y_val == 0:
                        oof_class_0_rets.append(r_real)
                continue
            
            weights = compute_sample_weight(class_weight='balanced', y=y_train)
            model = build_model()
            model.fit(X_train, y_train, sample_weight=weights)

            if 1 in model.classes_:
                idx_1 = np.where(model.classes_ == 1)[0][0]
                proba_all = model.predict_proba(X_test_purged)
                proba_1 = proba_all[:, idx_1]
                
                is_max_class = np.argmax(proba_all, axis=1) == idx_1
                pred_1 = ((proba_1 >= prob_threshold) & is_max_class).astype(int)
                y_test_bin = (y_test_purged == 1).astype(int)

                oof_y_true.extend(y_test_bin)
                oof_y_prob.extend(proba_1)
                oof_y_pred.extend(pred_1)
                valid_folds += 1

                for d, p, r_real, a_pct, vx, y_val in zip(test_dates_purged, pred_1, realized_rets_purged, atr_pct_purged, vix_purged, y_test_purged):
                    backtest_dates.append(d)
                    fold_slippage = estimate_slippage(a_pct, vx, base_fee_dec)
                    trade_ret = (r_real - fold_slippage) if p == 1 else 0.0
                    backtest_trade_rets.append(trade_ret)
                    if y_val == 0:
                        oof_class_0_rets.append(r_real)
            else:
                for d, r_real, y_val in zip(test_dates_purged, realized_rets_purged, y_test_purged):
                    backtest_dates.append(d)
                    backtest_trade_rets.append(0.0)
                    if y_val == 0:
                        oof_class_0_rets.append(r_real)

        try:
            pooled_auc = roc_auc_score(oof_y_true, oof_y_prob) if len(np.unique(oof_y_true)) > 1 else np.nan
            pooled_precision = precision_score(oof_y_true, oof_y_pred, zero_division=0)
            n_pos = sum(oof_y_true)
            n_neg = len(oof_y_true) - n_pos
            auc_pvalue = calc_auc_pvalue(pooled_auc, n_pos, n_neg)
        except Exception:
            pooled_auc, pooled_precision, auc_pvalue = np.nan, np.nan, 1.0

        if len(backtest_trade_rets) > 0:
            equity_curve = np.cumprod(1.0 + np.array(backtest_trade_rets))
            cum_ret = (equity_curve[-1] - 1.0) * 100.0
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / (peak + 1e-8)
            mdd = np.min(drawdown) * 100.0
            equity_df = pd.DataFrame({"Date": backtest_dates, "Equity": equity_curve}).set_index("Date")
        else:
            cum_ret, mdd = 0.0, 0.0
            equity_df = pd.DataFrame()

        cv_metrics = {
            "pooled_auc": float(pooled_auc),
            "pooled_precision": float(pooled_precision),
            "auc_pvalue": float(auc_pvalue),
            "class_dist": dist_info,
            "total_samples": total_samples,
            "valid_folds": valid_folds,
            "total_folds": n_splits,
            "cum_ret": cum_ret,
            "mdd": mdd,
            "equity_df": equity_df
        }

        # 최종 배포 모델 학습
        X_final_train = X[::LOOKAHEAD]
        y_final_train = y[::LOOKAHEAD]
        base_model = build_model()
        final_weights = compute_sample_weight(class_weight='balanced', y=y_final_train)
        
        is_calibrated = True
        try:
            if y_final_train.nunique() >= 2:
                tscv_calib = TimeSeriesSplit(n_splits=3, gap=1)
                calibrated_model = get_calibrated_cv(base_model, tscv_calib)
                calibrated_model.fit(X_final_train, y_final_train, sample_weight=final_weights)
            else:
                raise ValueError("Insufficient classes")
        except Exception:
            is_calibrated = False
            base_model.fit(X_final_train, y_final_train, sample_weight=final_weights)
            calibrated_model = base_model

        if is_calibrated:
            try:
                base_model.fit(X_final_train, y_final_train, sample_weight=final_weights)
            except Exception:
                pass

        feature_importance_dict = {}
        try:
            # ⭐ Fix 5: RF 단독이 아닌 RF+GB 평균 중요도로 산출하여 특정 서브모델 편향 완화
            rf_importances = base_model.named_estimators_['rf'].feature_importances_
            gb_importances = base_model.named_estimators_['gb'].feature_importances_
            avg_importances = (rf_importances + gb_importances) / 2.0
            feature_importance_dict = dict(zip(FEATURES, avg_importances))
            top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            top_features_str = ", ".join([f"{FEATURE_EASY_NAMES.get(f[0], f[0])} ({f[1]*100:.0f}%)" for f in top_features])
        except Exception:
            top_features_str = "산출 불가"

        is_oof_fallback = False
        if len(oof_class_0_rets) > 0:
            class_0_mean_ret = float(np.mean(oof_class_0_rets))
        else:
            is_oof_fallback = True
            class_0_mean_ret = 0.0

        today_data = df.iloc[-1:]

        # ⭐ Fix 9: 신호가 매일 바뀌어 사용자가 혼란스러운 문제 해결
        # 하루치 스냅샷만 쓰면 확률이 일별 노이즈에 크게 흔들립니다.
        # 스윙 투자 기준(SMOOTH_DAYS일)으로 예측 확률을 평균해 분산을 줄입니다.
        # (없는 엣지를 만들어내는 것이 아니라, 같은 모델의 추정 분산만 낮추는 조치)
        recent_features = df[FEATURES].iloc[-SMOOTH_DAYS:]
        proba_recent = calibrated_model.predict_proba(recent_features)
        proba_output = proba_recent.mean(axis=0)
        prob_dict = dict(zip(calibrated_model.classes_, proba_output))
        
        p_loss = prob_dict.get(-1, 0.0)
        p_time = prob_dict.get(0, 0.0)
        p_profit = prob_dict.get(1, 0.0)

        current_price = today_data["Close"].values[0]
        atr = today_data["ATR_14"].values[0]
        atr_pct = today_data["ATR_Pct"].values[0]
        avg_gap_5 = df["Overnight_Gap"].tail(5).mean()
        
        target_price = current_price + (atr * 1.5)
        stop_loss_price = current_price - (atr * 1.5)
        
        stoch_k = today_data["Stoch_K"].values[0]
        rsi_val = today_data["RSI"].values[0]
        macd_hist_val = today_data["MACD_Hist"].values[0]
        mom_5 = today_data["Momentum_5"].values[0]
        sma_20 = today_data["SMA_20"].values[0]
        bb_upper = today_data["BB_Upper"].values[0]
        bb_lower = today_data["BB_Lower"].values[0]

        vix_current = today_data["VIX_Close"].values[0] if "VIX_Close" in today_data.columns else 0.0
        relative_strength = today_data["Market_Relative"].values[0] if "Market_Relative" in today_data.columns else 0.0

        raw_profit_dec = (atr * 1.5) / current_price
        raw_loss_dec = (atr * 1.5) / current_price
        
        slippage_dec = estimate_slippage(atr_pct, vix_current, base_fee_dec)
        
        net_profit_dec = raw_profit_dec - slippage_dec
        net_loss_dec = raw_loss_dec + slippage_dec
        ev_dec = (p_profit * net_profit_dec) - (p_loss * net_loss_dec) + (p_time * (class_0_mean_ret - slippage_dec))

        # ⭐ Fix 10 [중대 안전 문제]: 기존에는 EV>0이면 무조건 켈리 비중을 계산했습니다.
        # 그 결과 예측력이 전혀 없는 모델(AUC 0.49)에서도 '횡보 구간 평균수익'이 (+)이면
        # 켈리가 상한(25%)까지 치솟았습니다. 이는 모델의 예측 엣지가 아니라 단순한
        # 주가 우상향 드리프트에 최대 베팅하는 것이며, 실제로 손실 위험이 큽니다.
        # 따라서 (1) 통계적으로 유의한 예측력이 확인되고 (2) 상승확률이 하락확률보다
        # 높을 때만 비중을 제시하고, 그 외에는 0%로 둡니다.
        has_edge = (not pd.isna(auc_pvalue)) and (auc_pvalue <= 0.10) and (p_profit > p_loss)

        if ev_dec > 0 and has_edge:
            full_kelly = solve_discrete_kelly(p_profit, p_loss, p_time, net_profit_dec, net_loss_dec, (class_0_mean_ret - slippage_dec))
            kelly_pct = min(kelly_cap, (full_kelly * half_kelly_frac) * 100.0)
        else:
            kelly_pct = 0.0

        # ⭐ Fix 3: 기존에는 타임아웃(class 0) 시나리오에서 슬리피지만 하방 리스크로 잡고,
        # class_0_mean_ret 자체가 음수(즉 횡보/타임아웃 구간에서도 실제로 손실이 나는 경우)인
        # 상황을 반영하지 못해 리스크를 과소평가했습니다. 타임아웃 시나리오의 순 기대수익이
        # 음수일 때 그 손실분까지 하방 리스크에 더하도록 수정했습니다.
        timeout_net_ret = class_0_mean_ret - slippage_dec
        timeout_downside = max(0.0, -timeout_net_ret)
        # ⭐ Fix 9(계속): 신호 안정성 측정 — 최근 며칠간 같은 방향이었는지 계산해
        # "오늘만 반짝 뜬 신호"와 "며칠째 유지되는 신호"를 사용자가 구분할 수 있게 합니다.
        stable_days = 0
        try:
            hist = df.iloc[-STABILITY_WINDOW:]
            hist_proba = calibrated_model.predict_proba(hist[FEATURES])
            cls = list(calibrated_model.classes_)
            i_p = cls.index(1) if 1 in cls else None
            i_l = cls.index(-1) if -1 in cls else None
            i_t = cls.index(0) if 0 in cls else None
            for k in range(len(hist)):
                d_pp = hist_proba[k, i_p] if i_p is not None else 0.0
                d_pl = hist_proba[k, i_l] if i_l is not None else 0.0
                d_pt = hist_proba[k, i_t] if i_t is not None else 0.0
                d_atr_pct = hist["ATR_Pct"].iloc[k]
                d_vix = hist["VIX_Close"].iloc[k] if "VIX_Close" in hist.columns else 20.0
                d_slip = estimate_slippage(d_atr_pct, d_vix, base_fee_dec)
                d_raw = (d_atr_pct * 1.5) / 100.0
                d_ev = (d_pp * (d_raw - d_slip)) - (d_pl * (d_raw + d_slip)) \
                       + (d_pt * (class_0_mean_ret - d_slip))
                if d_ev > 0 and d_pp > d_pl:
                    stable_days += 1
        except Exception as e:
            logging.warning(f"Stability calc failed for {ticker}: {e}")
            stable_days = 0

        downside_risk = (p_loss * net_loss_dec) + (p_time * timeout_downside)
        reward_score = (p_profit * net_profit_dec) / (downside_risk if downside_risk > 0 else 0.0001)

        if stoch_k >= 80:
            market_state = "🔥 과매수"
        elif stoch_k <= 20:
            market_state = "🧊 과매도"
        else:
            market_state = "⚖️ 정상 범위"

        # ⭐ 초보자용 XAI: 전문 용어 대신 일상어로, "그래서 뭘 뜻하는지"까지 설명
        explanations = []
        if stoch_k <= 20:
            explanations.append(
                f"**① 지금 가격 위치 — 많이 내려온 상태**  \n"
                f"최근 5일 가격 범위에서 아래쪽 {stoch_k:.0f}% 지점에 있습니다. "
                f"많이 빠진 상태라 잠깐 반등할 수도 있지만, 떨어지는 중일 수도 있어 신중해야 합니다."
            )
        elif stoch_k >= 80:
            explanations.append(
                f"**① 지금 가격 위치 — 많이 올라온 상태**  \n"
                f"최근 5일 가격 범위에서 위쪽 {stoch_k:.0f}% 지점에 있습니다. "
                f"단기간에 많이 올라서, 이미 오른 가격에 사는 것일 수 있으니 주의가 필요합니다."
            )
        else:
            explanations.append(
                f"**① 지금 가격 위치 — 보통 수준**  \n"
                f"최근 5일 가격 범위의 중간({stoch_k:.0f}%) 부근입니다. "
                f"과열도 침체도 아닌, 특별히 서두를 이유가 없는 평범한 구간입니다."
            )

        if macd_hist_val > 0:
            explanations.append(
                f"**② 상승하는 힘 — 붙어 있음**  \n"
                f"최근 5일간 {mom_5:+.1f}% 움직였고, 오르는 방향으로 힘이 실려 있습니다. "
                f"단기적으로는 흐름이 나쁘지 않다는 뜻입니다."
            )
        else:
            explanations.append(
                f"**② 상승하는 힘 — 아직 약함**  \n"
                f"최근 5일간 {mom_5:+.1f}% 움직였지만, 전체적인 힘은 아직 내리는 쪽입니다. "
                f"잠깐 올라도 다시 밀릴 가능성이 남아 있습니다."
            )

        if relative_strength > 0:
            explanations.append(
                f"**③ 시장 평균과 비교 — 더 잘하고 있음**  \n"
                f"미국 시장 평균(S&P 500)보다 {relative_strength:+.1f}%p 앞서고 있습니다. "
                f"같은 기간 다른 주식들보다 성적이 좋다는 의미입니다."
            )
        else:
            explanations.append(
                f"**③ 시장 평균과 비교 — 뒤처지고 있음**  \n"
                f"미국 시장 평균(S&P 500)보다 {relative_strength:+.1f}%p 뒤처져 있습니다. "
                f"시장이 오를 때 이 종목은 덜 오르고 있다는 의미입니다."
            )

        # 초보자용 한 줄 결론 (확률 비교를 짧게 요약)
        if p_profit > p_loss * 1.2:
            easy_verdict = "오를 가능성이 내릴 가능성보다 조금 높습니다."
        elif p_loss > p_profit * 1.2:
            easy_verdict = "내릴 가능성이 더 높아 서두를 필요가 없습니다."
        else:
            easy_verdict = "오를 가능성과 내릴 가능성이 비슷합니다."

        return {
            "prob_profit": p_profit * 100.0, "prob_loss": p_loss * 100.0, "prob_time": p_time * 100.0,
            "current_price": current_price, "target_price": target_price, "stop_loss_price": stop_loss_price,
            "profit_pct": raw_profit_dec * 100.0, "loss_pct": raw_loss_dec * 100.0, "ev": ev_dec * 100.0,
            "reward_score": reward_score, "avg_gap_5": avg_gap_5,
            "stoch_k": stoch_k, "rsi": rsi_val, "macd_hist": macd_hist_val, "sma_20": sma_20,
            "bb_upper": bb_upper, "bb_lower": bb_lower,
            "market_state": market_state, "cv": cv_metrics,
            "slippage": slippage_dec * 100.0, "atr_pct": atr_pct, "atr_val": atr,
            "vix": vix_current, "market_relative": relative_strength,
            "class_0_ret": class_0_mean_ret * 100.0,
            "df_tail": df.tail(1300),
            "is_calibrated": is_calibrated,
            "is_oof_fallback": is_oof_fallback,
            "top_features": top_features_str,
            "feature_importances": feature_importance_dict,
            "xai_text": "\n\n".join(explanations),
            "easy_verdict": easy_verdict,
            "stable_days": stable_days,
            "stability_window": STABILITY_WINDOW,
            "kelly_pct": kelly_pct
        }
    except Exception as e:
        logging.error(f"Model Pipeline Error on {ticker}: {e}\n{traceback.format_exc()}")
        return None

# ==========================================
# 3. 실무형 신호 분류 엔진 (FDR 통계 유의성 연동)
# ==========================================
def classify_signal(ev, p_profit, p_loss, atr_pct, vix, market_relative, is_fdr_passed=True, is_fallback=False, prob_threshold_pct=38.0, regime_mult=1.0):
    ev_threshold = max(0.1, atr_pct * 0.05) 
    
    if is_fallback:
        ev_threshold *= 1.3
    if vix > 30.0: 
        ev_threshold *= 1.5 
    if market_relative < -5.0: 
        ev_threshold *= 2.0 
    # ⭐ 시장 국면 반영: 위기 국면에서는 관심 신호 기준을 자동으로 높여
    # 시스템이 스스로 보수적으로 전환합니다 (평온 1.0 / 주의 1.4 / 위기 2.0배)
    ev_threshold *= regime_mult

    is_high_winrate = p_profit >= prob_threshold_pct
    p_time = 100.0 - (p_profit + p_loss)
    is_too_stuck = p_time > 50.0

    # ⭐ Fix 1: FDR 통계 검증을 통과하지 못한 종목은 '추천' 획득 불가 (강제 강등) ⭐
    if ev > ev_threshold and p_profit > p_loss and is_high_winrate: 
        if not is_fdr_passed:
            return "🟡 지켜보기 (검증 부족)"
        return "🔵 관심"
    elif ev < 0.0 or p_loss >= (p_profit + 0.1): 
        return "🔴 조건 미달"
    elif ev > 0.0 and (not is_high_winrate or is_too_stuck):
        return "🟡 지켜보기 (확률 부족)"
        
    return "🟡 지켜보기"
