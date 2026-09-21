# 무료 데이터 기반 개인 퀀트 파이프라인의 방법론적 취약성 분석

논문 *「무료 공개 데이터 기반 개인 투자자용 퀀트 파이프라인의 방법론적 취약성 분석: 결함별 기여도 분해」* 의 재현용 코드와 실험 산출물입니다.

## 무엇을 하는 연구인가

새로운 주가 예측 알고리즘을 제안하지 않습니다. 대신 **실제로 작동하는 퀀트 파이프라인에서 방법론적 결함을 하나씩 켜고 끄며**, 각 결함이 얼마만큼의 '없는 예측력'을 만들어내는지 측정합니다.

초기 구현은 재무 지표(PER·PBR 등) 기반의 '가치투자' 모델이었으나, 예비 실험(4.4절)에서 유의성이 재무 지표가 아니라 가격 변동성에서 나온다는 사실이 드러났고, 무료 재무제표는 이력이 ~4년으로 제한돼 검정력을 늘릴 수도 없었습니다. 그래서 본 연구는 재무제표를 완전히 배제하고 **가격만으로 계산되는 지표(모멘텀·변동성·시장 대비 상대강도)**, 예측 기간 1개월, 15년치 가격 이력으로 파이프라인을 재구성했습니다.

핵심 결과:

| 설정 | AUC | p값 | 판정 |
|---|---:|---:|---|
| 결함을 모두 포함 (초기 구현) | [TODO] | [TODO] | [TODO] |
| 결함을 모두 교정 | [TODO] | [TODO] | [TODO] |

같은 데이터, 같은 모델입니다. 바뀐 것은 방법론뿐입니다.

## 빠른 시작

```bash
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python stats_utils.py              # 통계 검정 자체 검증 (네트워크 불필요, 수 초)
python simulate_falsepositive.py   # 검정 방식별 오탐률 (네트워크 불필요, 약 1분)
```

위 두 개는 **외부 데이터 없이** 돌아가므로 핵심 주장을 가장 빠르게 확인할 수 있습니다.

## 재현 스크립트

| 스크립트 | 논문 절 | 소요 | 산출물 |
|---|---|---|---|
| `stats_utils.py` | 3.1 | 초 | (자체 검증 출력) |
| `simulate_falsepositive.py` | 4.2 | ~1분 | `simulate_falsepositive_results.csv` |
| `ablation.py` | 4.1 | ~15분 | `ablation_results.csv` |
| `experiment_universe.py` | 4.3 | ~15분 | `experiment_universe_results.csv` |
| `experiment_multipletesting.py` | 4.4 | ~15분 | `experiment_multipletesting_results.csv` |
| `experiment_survivorship.py` | 4.5 | ~10분 | `experiment_survivorship_results.csv` |
| `experiment_stocksplit.py` | 4.6 | ~10분 | `experiment_stocksplit_results.csv` |

네트워크를 쓰는 스크립트는 첫 실행에서 종목별 재무·가격 데이터를 수집하므로 오래 걸립니다. 원본 재무제표는 캐싱되어 같은 프로세스 안에서 재사용됩니다.

## 구조

| 파일 | 역할 |
|---|---|
| `core.py` | 데이터 수집·피처 생성·모델·검정 (순수 파이프라인) |
| `stats_utils.py` | Fama-MacBeth + Newey-West 검정, 자체 검증 포함 |
| `app.py` | Streamlit 대시보드 (연구 대상이 된 원 시스템) |
| `paper_draft.md` | 논문 원고 |

`app.py`는 Streamlit 스크립트라 임포트만 해도 UI가 실행됩니다. 그래서 파이프라인을 `core.py`로 분리했고, 모든 실험 스크립트는 `core.py`만 임포트합니다.

## 결함 토글

모든 결함은 `core.run_value_model()` 의 인자로 켜고 끌 수 있습니다.

| 인자 | 기본값(교정) | 결함 재현 |
|---|---|---|
| `pvalue_method` | `"fama_macbeth"` | `"iid"` |
| `eval_mode` | `"overlap_nw"` | `"nonoverlap_iid"` |

`require_fundamentals=False` / `horizon_override=core.SWING_HORIZON`(=21)은 본 연구가 쓰는 가격 기반(스윙) 모드를 켭니다. 재무제표 기반 결함(연간 공시 지연, 이익성장률 부호 오류)은 재무제표 자체를 더 이상 쓰지 않으므로 해당 없음입니다 — 자세한 경위는 3장 참조.

## 데이터 출처

모두 공개 출처에서 자동 수집되며 별도 구매가 필요 없습니다. 본 연구의 재현 스크립트는 **가격 데이터만** 씁니다(재무제표 미사용) — 이유는 3장 참조.

| 데이터 | 출처 |
|---|---|
| 가격(일별 수정주가) | Yahoo Finance (`yfinance`) |
| 시점별 S&P500 구성종목 | [fja05680/sp500](https://github.com/fja05680/sp500) |
| 현재 구성종목·편입일 | Wikipedia |

## 재현성

- 무작위 추출에는 고정 시드(42)를 사용합니다.
- 병렬 수집 결과는 티커 순으로 정렬해 결합하므로, 실행 순서가 달라져도 결과가 동일합니다.
- 다만 Yahoo Finance의 데이터는 시간이 지나면 갱신되므로, **실행 시점에 따라 수치가 달라질 수 있습니다.** 논문의 수치는 2026-09-21 기준입니다.

## 한계

무료 데이터로는 해결할 수 없는 한계가 있으며, 논문 6장에 상세히 기술했습니다. 요약하면:

- 시점 고정(point-in-time) 가격 데이터 부재 (소급 수정 문제 — 다만 재무제표를 쓰지 않으므로 4.5절의 생존편향 교정에서는 가격 데이터의 긴 이력 덕분에 탈락 종목 복원율이 개선됨)
- Newey-West 보정 후에도 오탐률이 명목 5%가 아닌 10%로 남음

## 라이선스

[TODO: 라이선스 결정 — 연구 코드는 보통 MIT 또는 Apache-2.0]
