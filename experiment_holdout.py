"""논문 실험(신규): 한 번도 보지 않은 기간(1996~2011)에서의 사전 등록 검증.

[왜 필요한가]
본 연구는 예측 기간 1개월을 '앞선 실행에서 유의성이 몰려 있던 구간'이라는
관찰로 골랐습니다(논문 6장 한계 1). 같은 데이터로 고르고 같은 데이터로 검증하면
그 선택 자체가 다중검정이 됩니다. 이를 해소하는 표준적인 방법은, 설정을 고르는
데 한 번도 쓰지 않은 데이터로 **설정을 바꾸지 않고 딱 한 번** 검증하는 것입니다.

본 연구의 모든 분석은 2011-09-21 이후 가격만 썼습니다. 시점별 구성종목 데이터는
1996년부터 있으므로, 1996-01-02 ~ 2011-09-20 구간이 미사용 표본외 데이터입니다.

[사전 등록한 설정 - 이 실험을 돌리기 전에 고정, 결과를 보고 바꾸지 않음]
  - 지표: SWING_FEATURES 3종 전체 / 예측 기간 21거래일 / RF(깊이3, 잎10, 트리120)
  - 유니버스: 시점별 S&P500 구성종목 (4.5절 조건 C와 같은 방식, 라벨은 필터 전 계산)
  - 검정: Fama-MacBeth + NW(lag 3h) + fixed-b, 양측 5%
  - 가격은 2011-09-20까지만 받는다 → 라벨이 본 분석 기간의 가격을 보지 않는다.

[한계]
  이 기간의 탈락 종목은 상장폐지된 경우가 많아 가격 복원율이 본 기간보다 낮습니다.
  복원율을 함께 보고합니다.

실행: py experiment_holdout.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time

import numpy as np
import pandas as pd

import core
from experiment_survivorship import (load_historical_membership, apply_membership,
                                     coverage_table)

HOLDOUT_START = "1996-01-02"
HOLDOUT_END = core.SWING_START      # 배타적: 본 분석 기간 첫날 전날까지


def main():
    start_ts, end_ts = pd.Timestamp(HOLDOUT_START), pd.Timestamp(HOLDOUT_END)
    print(f"홀드아웃 기간: {HOLDOUT_START} ~ {HOLDOUT_END} (배타적)\n")

    hist = load_historical_membership()
    window = hist[(hist["date"] >= start_ts) & (hist["date"] < end_ts)]
    ever = set()
    for s in window["tickers"]:
        ever |= {t.strip().replace(".", "-") for t in str(s).split(",")}
    ever = sorted(ever)
    print(f"기간 내 한 번이라도 편입된 종목: {len(ever)}개")

    print("가격·매크로 로드 (스냅샷)...")
    t0 = time.time()
    prices, macro_prepared = core.load_swing_data(ever, HOLDOUT_START, HOLDOUT_END)
    print(f"  완료 ({time.time() - t0:.0f}초)")

    calendar = macro_prepared.index[(macro_prepared.index >= start_ts)
                                    & (macro_prepared.index < end_ts)]
    cov = coverage_table(ever, prices, hist, calendar)
    cov.to_csv("experiment_holdout_coverage.csv", index=False, encoding="utf-8-sig")
    counts = cov["분류"].value_counts()
    tot_mem, tot_hit = cov["편입일수"].sum(), cov["가격있는편입일수"].sum()
    print("\n구성종목 가격 복원 현황 (편입 기간 기준)")
    for k in ["복원", "부분 복원", "재사용 의심", "가격 없음"]:
        print(f"  {k:<8}: {int(counts.get(k, 0)):>4}개")
    print(f"  종목-일 기준 복원율: {tot_hit:,}/{tot_mem:,} ({tot_hit / max(tot_mem, 1) * 100:.1f}%)\n")

    print("패널 생성 중...")
    t0 = time.time()
    panel = core.build_swing_panel(ever, prices, macro_prepared)
    panel = core.add_forward_returns(panel, [core.SWING_HORIZON])
    pc = apply_membership(panel, hist)
    print(f"  완료 ({time.time() - t0:.0f}초) · 필터 후 {len(pc):,}행 · "
          f"{pc['Ticker'].nunique()}종목\n")

    res = core.run_value_model(pc, horizon_override=core.SWING_HORIZON,
                               require_fundamentals=False)
    if not res or "error" in res:
        print(f"실패: {res.get('error') if res else '?'}")
        return
    verdict = "유의" if res["pvalue_two_sided"] <= 0.05 else "유의하지 않음"
    print("=" * 84)
    print("사전 등록 설정 · 홀드아웃 1회 검증")
    print("=" * 84)
    print(f"  {core.fmt_result(res)} · {verdict}")
    for k, (a, t, p1, T) in res["feature_fm"].items():
        print(f"    지표 단독 {k:<16} AUC {a:.4f} · t {t:+.2f}")

    pd.DataFrame([{
        "기간": f"{HOLDOUT_START}~{HOLDOUT_END}", "AUC": res["fm_auc"], "t": res["fm_tstat"],
        "p단측": res["pvalue"], "p양측": res["pvalue_two_sided"], "날짜": res["fm_n_dates"],
        "종목-일 복원율": tot_hit / max(tot_mem, 1), "판정": verdict,
        **{f"{k}_단독AUC": v[0] for k, v in res["feature_fm"].items()},
    }]).to_csv("experiment_holdout_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: experiment_holdout_results.csv, experiment_holdout_coverage.csv")


if __name__ == "__main__":
    main()
