"""논문 4.2절: 검정 방식별 오탐률을 충분한 반복으로 정밀 추정.

stats_utils.py의 자체 점검은 개발용이라 반복이 40회로 적습니다(오탐률 45%의
95% 신뢰구간이 [30%, 61%]로 매우 넓음). 논문에 싣는 수치는 반복을 크게 늘려
신뢰구간을 좁히고, 두 검정의 차이가 유의한지도 함께 보고합니다.

합성 데이터이므로 비용이 거의 들지 않습니다.

실행: py simulate_falsepositive.py           (기본 1000회)
      py simulate_falsepositive.py --n 200
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, fisher_exact
from sklearn.metrics import roc_auc_score

from stats_utils import (fama_macbeth_auc, hanley_mcneil_pvalue,
                         _make_panel, _make_overlapping_panel)


def clopper_pearson(k, n, alpha=0.05):
    """이항 비율의 정확 신뢰구간 (Clopper-Pearson)."""
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def fmt(k, n):
    lo, hi = clopper_pearson(k, n)
    return f"{k/n*100:5.1f}%  [{lo*100:4.1f}%, {hi*100:4.1f}%]   ({k}/{n})"


def run_case1(trials):
    """참효과 0 · 평가일 3개 · 종목 300개 (초기 구현이 처했던 상황)."""
    k_iid = k_fm = 0
    for seed in range(trials):
        y, s, d = _make_panel(n_dates=3, n_stocks=300, signal_strength=0.0,
                              seed=seed, date_effect_sd=0.35)
        pooled = roc_auc_score(y, s)
        n_pos = int(y.sum())
        p_old = hanley_mcneil_pvalue(pooled, n_pos, len(y) - n_pos)
        _, _, p_new, _ = fama_macbeth_auc(y, s, d)
        if p_old <= 0.05:
            k_iid += 1
        if np.isfinite(p_new) and p_new <= 0.05:
            k_fm += 1
    return k_iid, k_fm


def run_case2(trials, horizon=20):
    """참효과 0 · 겹치는 예측구간(평가일 260개) - Newey-West 보정 효과."""
    k_iid = k_nw = 0
    for seed in range(trials):
        y, s, d = _make_overlapping_panel(n_days=260, n_stocks=120,
                                          horizon=horizon, seed=seed)
        _, _, p_iid, _ = fama_macbeth_auc(y, s, d, nw_lag=0)
        _, _, p_nw, _ = fama_macbeth_auc(y, s, d, nw_lag=horizon)
        if np.isfinite(p_iid) and p_iid <= 0.05:
            k_iid += 1
        if np.isfinite(p_nw) and p_nw <= 0.05:
            k_nw += 1
    return k_iid, k_nw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    n = args.n

    print(f"반복 {n}회 · 유의수준 5% · 참예측력 0 (구간은 Clopper-Pearson 95% CI)\n")

    print("=" * 78)
    print("사례 1) 횡단면 종속성 - 평가일 3개 × 종목 300개")
    print("=" * 78)
    k_iid, k_fm = run_case1(n)
    print(f"  행 수를 표본 수로 사용 (기존) : {fmt(k_iid, n)}")
    print(f"  Fama-MacBeth                : {fmt(k_fm, n)}")
    odds, pv = fisher_exact([[k_iid, n - k_iid], [k_fm, n - k_fm]])
    print(f"  두 방식의 차이 (Fisher 정확검정) : p = {pv:.3g}")

    print()
    print("=" * 78)
    print("사례 2) 겹치는 예측구간 - 평가일 260개, 예측기간 20일")
    print("=" * 78)
    k_iid2, k_nw = run_case2(n)
    print(f"  겹침 무시 (iid)             : {fmt(k_iid2, n)}")
    print(f"  Newey-West 보정             : {fmt(k_nw, n)}")
    odds2, pv2 = fisher_exact([[k_iid2, n - k_iid2], [k_nw, n - k_nw]])
    print(f"  두 방식의 차이 (Fisher 정확검정) : p = {pv2:.3g}")

    rows = [
        {"사례": "횡단면 종속성", "검정": "행 수 기반(기존)", "오탐": k_iid, "반복": n},
        {"사례": "횡단면 종속성", "검정": "Fama-MacBeth", "오탐": k_fm, "반복": n},
        {"사례": "겹치는 예측구간", "검정": "iid", "오탐": k_iid2, "반복": n},
        {"사례": "겹치는 예측구간", "검정": "Newey-West", "오탐": k_nw, "반복": n},
    ]
    out = pd.DataFrame(rows)
    out["오탐률(%)"] = (out["오탐"] / out["반복"] * 100).round(1)
    ci = out.apply(lambda r: clopper_pearson(r["오탐"], r["반복"]), axis=1)
    out["CI하한(%)"] = [round(c[0] * 100, 1) for c in ci]
    out["CI상한(%)"] = [round(c[1] * 100, 1) for c in ci]
    out.to_csv("simulate_falsepositive_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: simulate_falsepositive_results.csv")


if __name__ == "__main__":
    main()
