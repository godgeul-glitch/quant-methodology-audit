# 무료 데이터 기반 개인 퀀트 파이프라인의 방법론적 취약성 분석

논문 *「무료 공개 데이터 기반 개인 투자자용 퀀트 파이프라인의 방법론적 취약성 분석: 결함별 기여도 분해」* 의 재현용 코드와 실험 산출물입니다.

## 무엇을 하는 연구인가

새로운 주가 예측 알고리즘을 제안하지 않습니다. 대신 **실제로 작동하는 퀀트 파이프라인에서 방법론적 결함을 하나씩 켜고 끄며**, 각 결함이 얼마만큼의 '없는 예측력'을 만들어내는지 측정합니다.

연구 대상은 저자가 만든 주식 분석 대시보드(`app.py`)의 핵심 기능인 **가격 기반 단기 분석**을 단순화한 파이프라인입니다. S&P500 501종목, 15년(2011-09-21 ~ 2026-09-21) 일별 가격, 가격 지표 3종(모멘텀·변동성·시장 대비 상대강도), 예측 기간 1개월을 씁니다. 앱에는 재무 지표 기반 '가치투자 랭킹' 탭도 있지만 부가 기능이며, 본 연구는 재무제표를 쓰지 않습니다(논문 부록 B).

핵심 결과 (검정 교정 후, 양측 5%):

| 항목 | 결과 |
|---|---|
| 기준선 (생존편향 미교정) | AUC 0.511, p=0.095 — **처음부터 유의하지 않음** |
| ① 행 수 기반 p값 | 같은 AUC인데 p < 0.0001 — **추정은 그대로, 추론만 왜곡** |
| ⑥ 48개 설정 중 최선만 보고 | p=0.020 → 시도 횟수 보정 시 0.98 |
| ②·⑤·⑧ (소표본·솎아내기·생존편향) | AUC를 예상 방향으로 움직이나 대응 검정에서 0과 구별 안 됨 |
| 미사용 기간(1996–2011) 1회 검증 | AUC 0.505, p=0.485 — 유의하지 않음 |
| 검정 자체의 오탐률 (참효과 0) | 행 수 기반 34–39% · 표준 NW 8.7% · **NW 3h + fixed-b 5.7%** |

2026-09-23 개정에서 이전 초고의 오류(낙관적 검정, 탈락 종목 '100% 복원' 집계 오류, p값 유의/비유의 비교 등)를 바로잡았습니다. 논문 5.2절에 정리돼 있습니다.

## 빠른 시작

```bash
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python stats_utils.py              # 통계 검정 자체 검증 (네트워크 불필요, ~20초)
python simulate_falsepositive.py   # 검정 방식별 오탐률 (네트워크 불필요, ~30분)
```

위 두 개는 **외부 데이터 없이** 돌아가므로 핵심 주장(검정 방식에 따라 오탐률이 달라진다)을 가장 먼저 확인할 수 있습니다. `python simulate_falsepositive.py --n 200 --n3 100`으로 줄이면 몇 분 안에 끝납니다.

## 재현 스크립트

| 스크립트 | 논문 절 | 소요* | 산출물 |
|---|---|---|---|
| `stats_utils.py` | 2.3 | ~20초 | (자체 검증 출력) |
| `simulate_falsepositive.py` | 4.1 | ~30분 | `simulate_falsepositive_results.csv` |
| `experiment_survivorship.py` | 4.5 | ~30분 (첫 실행, 가격 다운로드 포함) | `experiment_survivorship_results.csv`, `_diffs.csv`, `_coverage.csv` |
| `ablation.py` | 4.2 | ~5분 | `ablation_results.csv` |
| `experiment_multipletesting.py` | 4.4 | ~50분 | `experiment_multipletesting_results.csv`, `experiment_feature_direction.csv` |
| `experiment_universe.py` | 4.3 | ~45분 | `experiment_universe_results.csv` |
| `experiment_stocksplit.py` | 4.6 | ~25분 | `experiment_stocksplit_results.csv` |
| `experiment_holdout.py` | 4.8 | ~30분 (다운로드 포함) | `experiment_holdout_results.csv`, `_coverage.csv` |

\* 22코어 PC, `QUANT_RF_JOBS=-1` 기준. 기본값(1)이면 RandomForest가 단일 코어로 돌아 약 6배 느립니다. `random_state`가 고정돼 있어 병렬 수와 무관하게 결과는 같습니다.

```bash
set QUANT_RF_JOBS=-1               # Windows (macOS/Linux: export QUANT_RF_JOBS=-1)
python experiment_survivorship.py  # 탈락 종목까지 가격을 받으므로 가장 먼저 돌리면 이후가 빠름
```

**가격 스냅샷**: 모든 실험은 고정 기간(`core.SWING_START`~`core.SWING_END`)의 가격을 쓰며, 처음 받은 가격을 `data_cache/`에 저장해 재사용합니다. 그래서 스크립트마다 수치가 어긋나지 않습니다. 새로 받고 싶으면 `data_cache/`를 지우세요.

## 구조

| 파일 | 역할 |
|---|---|
| `core.py` | 데이터 수집·가격 스냅샷·피처 생성·모델·검정 (순수 파이프라인) |
| `stats_utils.py` | Fama-MacBeth + Newey-West(fixed-b) 검정, 대응 차이 검정, 사전 검정력, 자체 검증 |
| `app.py` | Streamlit 대시보드 (연구 대상이 된 원 시스템) |
| `paper_draft.md` | 논문 원고 |

`app.py`는 Streamlit 스크립트라 임포트만 해도 UI가 실행됩니다. 그래서 파이프라인을 `core.py`로 분리했고, 모든 실험 스크립트는 `core.py`만 임포트합니다. `core.py`의 캐시는 Streamlit 런타임 안에서만 동작하고, 실험 스크립트에서는 건너뜁니다.

## 결함 토글

모든 결함은 `core.run_value_model()` 의 인자로 켜고 끌 수 있습니다.

| 인자 | 기본값(교정) | 결함 재현 |
|---|---|---|
| `pvalue_method` | `"fama_macbeth"` | `"iid"` (①) |
| `eval_mode` | `"overlap_nw"` | `"nonoverlap_iid"` (⑤) |
| `nw_lag_mult`, `fixed_b` | `3`, `True` | `1`, `False` (이전 기본 검정) |

`require_fundamentals=False` / `horizon_override=core.SWING_HORIZON`(=21)은 본 연구의 가격 기반 모드입니다. 결과 dict에는 `pvalue`(단측), `pvalue_two_sided`, `per_date_auc`(조건 간 대응 검정용), `feature_fm`(지표별 단독 AUC)이 들어 있습니다.

## 데이터 출처

모두 공개 출처에서 자동 수집되며 별도 구매가 필요 없습니다.

| 데이터 | 출처 |
|---|---|
| 가격(일별 수정주가) | Yahoo Finance (`yfinance`) |
| 시점별 S&P500 구성종목 | [fja05680/sp500](https://github.com/fja05680/sp500) |

## 재현성

- 분석 기간과 가격 스냅샷을 고정합니다(위 참조). 논문 수치는 2026-09-23에 받은 스냅샷 기준입니다. Yahoo 데이터는 수시로 갱신·소급 수정되므로, 스냅샷을 새로 받으면 수치가 조금 달라질 수 있습니다.
- 무작위 추출에는 고정 시드(42)를, fixed-b 귀무분포 시뮬레이션에는 고정 시드(12345)를 씁니다.
- 병렬 수집 결과는 티커 순으로 정렬해 결합하므로, 실행 순서가 달라져도 결과가 동일합니다.

## 한계

논문 6장에 상세히 기술했습니다. 요약하면:

- 상장폐지 종목은 yfinance에 가격이 없어 생존편향을 부분적으로만 교정합니다(탈락 종목 329개 중 115개 복원, 종목-일 기준 36.3%).
- 검정 교정(NW 3h + fixed-b)은 정규 오차 합성 데이터에서 한 것이라, 실제 수익률의 두꺼운 꼬리 등에서는 오탐률이 다를 수 있습니다.
- 이 데이터로 검출 가능한 최소 효과가 ΔAUC 0.011–0.029라, 그보다 작은 결함 효과는 식별할 수 없습니다.

## 라이선스

[TODO: 라이선스 결정 — 연구 코드는 보통 MIT 또는 Apache-2.0]
