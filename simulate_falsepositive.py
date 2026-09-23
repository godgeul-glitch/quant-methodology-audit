"""논문 4.2절: 검정 방식별 오탐률을 충분한 반복으로 정밀 추정.

stats_utils.py의 자체 점검은 개발용이라 반복이 40회로 적습니다(오탐률 45%의
95% 신뢰구간이 [30%, 61%]로 매우 넓음). 논문에 싣는 수치는 반복을 크게 늘려
신뢰구간을 좁히고, 두 검정의 차이가 유의한지도 함께 보고합니다.

합성 데이터이므로 네트워크가 필요 없습니다.

[개정] 두 가지를 추가했습니다.
  - 사례 1 민감도: 오탐률 35.5%는 '날짜별 신호 강도의 흔들림(date_effect_sd=0.35)'과
    '평가일 3개'라는 설정값에 좌우되는 숫자입니다. 설정을 바꿔 가며 표로 보여줍니다.
  - 사례 3: 실제 데이터 규모(평가일 2,742개·예측기간 21일)에서 검정 방식별 오탐률.
    본 연구의 기본 검정(NW 대역폭 3h + fixed-b)을 고른 근거입니다.

실행: py simulate_falsepositive.py           (기본 1000회)
      py simulate_falsepositive.py --n 200
"""
import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")

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


def run_case1(trials, n_dates=3, date_effect_sd=0.35):
    """참효과 0 · 평가일 n_dates개 · 종목 300개 (기본값 = 초기 구현이 처했던 상황)."""
    k_iid = k_fm = 0
    for seed in range(trials):
        y, s, d = _make_panel(n_dates=n_dates, n_stocks=300, signal_strength=0.0,
                              seed=seed, date_effect_sd=date_effect_sd)
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
    k_iid = k_nw = k_fb = 0
    for seed in range(trials):
        y, s, d = _make_overlapping_panel(n_days=260, n_stocks=120,
                                          horizon=horizon, seed=seed)
        _, _, p_iid, _ = fama_macbeth_auc(y, s, d, nw_lag=0)
        _, _, p_nw, _ = fama_macbeth_auc(y, s, d, nw_lag=horizon)
        _, _, p_fb, _ = fama_macbeth_auc(y, s, d, nw_lag=3 * horizon, fixed_b=True)
        if np.isfinite(p_iid) and p_iid <= 0.05:
            k_iid += 1
        if np.isfinite(p_nw) and p_nw <= 0.05:
            k_nw += 1
        if np.isfinite(p_fb) and p_fb <= 0.05:
            k_fb += 1
    return k_iid, k_nw, k_fb


def run_case3(trials, n_days=2742, n_stocks=60, horizon=21):
    """참효과 0 · 실제 데이터 규모. 검정 방식별 오탐 수 dict."""
    methods = {
        "iid (겹침 무시)": dict(nw_lag=0),
        "NW lag=h, t분포 (구 기본값)": dict(nw_lag=horizon),
        "NW lag=2h, t분포": dict(nw_lag=2 * horizon),
        "NW lag=3h, t분포": dict(nw_lag=3 * horizon),
        "NW lag=h, fixed-b": dict(nw_lag=horizon, fixed_b=True),
        "NW lag=3h, fixed-b (신 기본값)": dict(nw_lag=3 * horizon, fixed_b=True),
    }
    k = {m: 0 for m in methods}
    for seed in range(trials):
        y, s, d = _make_overlapping_panel(n_days=n_days, n_stocks=n_stocks,
                                          horizon=horizon, seed=10000 + seed)
        for m, kw in methods.items():
            _, _, p, _ = fama_macbeth_auc(y, s, d, **kw)
            if np.isfinite(p) and p <= 0.05:
                k[m] += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n3", type=int, default=600, help="사례 3(실제 규모) 반복 수")
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
    k_iid2, k_nw, k_fb = run_case2(n)
    print(f"  겹침 무시 (iid)             : {fmt(k_iid2, n)}")
    print(f"  Newey-West 보정 (lag=h)     : {fmt(k_nw, n)}")
    print(f"  NW lag=3h + fixed-b         : {fmt(k_fb, n)}")
    odds2, pv2 = fisher_exact([[k_iid2, n - k_iid2], [k_nw, n - k_nw]])
    print(f"  iid vs NW 차이 (Fisher 정확검정) : p = {pv2:.3g}")

    rows = [
        {"사례": "횡단면 종속성", "설정": "날짜3·sd0.35", "검정": "행 수 기반(기존)", "오탐": k_iid, "반복": n},
        {"사례": "횡단면 종속성", "설정": "날짜3·sd0.35", "검정": "Fama-MacBeth", "오탐": k_fm, "반복": n},
        {"사례": "겹치는 예측구간", "설정": "날짜260·h20", "검정": "iid", "오탐": k_iid2, "반복": n},
        {"사례": "겹치는 예측구간", "설정": "날짜260·h20", "검정": "Newey-West", "오탐": k_nw, "반복": n},
        {"사례": "겹치는 예측구간", "설정": "날짜260·h20", "검정": "NW lag=3h + fixed-b", "오탐": k_fb, "반복": n},
    ]

    print()
    print("=" * 78)
    print("사례 1 민감도) 날짜 수 × 날짜별 신호 흔들림(date_effect_sd)")
    print("=" * 78)
    print(f"  {'날짜':>4} {'sd':>5}   {'행 수 기반':>12}   {'Fama-MacBeth':>12}")
    for nd in (3, 10, 30):
        for sd in (0.0, 0.1, 0.2, 0.35, 0.5):
            if nd == 3 and sd == 0.35:
                a, b = k_iid, k_fm           # 위에서 이미 계산한 기본 설정
            else:
                a, b = run_case1(n, n_dates=nd, date_effect_sd=sd)
            print(f"  {nd:>4} {sd:>5.2f}   {a/n*100:>11.1f}%   {b/n*100:>11.1f}%", flush=True)
            rows.append({"사례": "횡단면 종속성(민감도)", "설정": f"날짜{nd}·sd{sd}",
                         "검정": "행 수 기반(기존)", "오탐": a, "반복": n})
            rows.append({"사례": "횡단면 종속성(민감도)", "설정": f"날짜{nd}·sd{sd}",
                         "검정": "Fama-MacBeth", "오탐": b, "반복": n})

    n3 = args.n3
    print()
    print("=" * 78)
    print(f"사례 3) 실제 데이터 규모 - 평가일 2,742개, 예측기간 21일, {n3}회")
    print("=" * 78)
    k3 = run_case3(n3)
    for m, kk in k3.items():
        print(f"  {m:<30}: {fmt(kk, n3)}")
        rows.append({"사례": "실제 규모 겹침", "설정": "날짜2742·h21", "검정": m, "오탐": kk, "반복": n3})
    out = pd.DataFrame(rows)
    out["오탐률(%)"] = (out["오탐"] / out["반복"] * 100).round(1)
    ci = out.apply(lambda r: clopper_pearson(r["오탐"], r["반복"]), axis=1)
    out["CI하한(%)"] = [round(c[0] * 100, 1) for c in ci]
    out["CI상한(%)"] = [round(c[1] * 100, 1) for c in ci]
    out.to_csv("simulate_falsepositive_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: simulate_falsepositive_results.csv")


if __name__ == "__main__":
    main()
