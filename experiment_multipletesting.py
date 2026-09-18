"""논문 실험: 여러 설정을 시도하고 '제일 좋은 것'만 보고하면 어떻게 되는가?

[배경]
연구자는 보통 설정 하나만 돌려보고 끝내지 않습니다. 피처를 빼봤다 넣어봤다,
트리 깊이를 바꿔봤다, 예측 기간을 3개월로 해봤다가 6개월로 해봤다가...
그리고 **제일 잘 나온 설정을 논문에 싣습니다.** 이 과정에서 보고되는 p값은
더 이상 5% 유의수준을 의미하지 않습니다(다중검정 문제).

앞선 실험에서 이 데이터의 검정력은 약 25%로 추정됐습니다. 즉 한 번 돌려서
유의하게 나올 확률이 원래 25% 정도라는 뜻인데, 여러 번 시도하면 그중 하나가
유의할 확률은 훨씬 높아집니다.

[방법]
연구자가 실제로 만질 법한 손잡이들을 조합해 N개 설정을 돌리고
  - 각 설정의 p값 분포
  - '제일 좋은 설정만 보고'했을 때의 p값 (= min p)
  - 몇 개 설정이 p<0.05를 주는지
를 봅니다. 데이터도, 파이프라인도 전부 동일합니다. 바뀌는 건 '연구자의 선택'뿐.

실행: py experiment_multipletesting.py
"""
import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import core


def build_panel(tickers, df_all, macro_prepared, workers=20):
    results = {}

    def one(tk):
        try:
            fh = core.get_fundamental_history(tk)
            if fh.empty:
                return tk, None
            return tk, core.build_value_panel(df_all, fh, macro_prepared, tk)
        except Exception:
            return tk, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tk, p in ex.map(one, tickers):
            if p is not None and not p.empty:
                results[tk] = p
    if not results:
        return pd.DataFrame()
    panel = pd.concat([results[t] for t in sorted(results)])
    return panel.sort_values("Ticker", kind="mergesort").sort_index(kind="mergesort")


def make_variants():
    """연구자가 실제로 시도해볼 법한 설정들 (researcher degrees of freedom)."""
    all_feats = list(core.VALUE_FEATURES)
    price_feats = ["Momentum_126", "Volatility_60", "Market_Relative"]
    fund_feats = [f for f in all_feats if f not in price_feats]

    feature_choices = [
        ("전체 지표", None),
        ("재무 지표만", fund_feats),
        ("PEG 제외", [f for f in all_feats if f != "PEG_Inv"]),
        ("변동성 제외", [f for f in all_feats if f != "Volatility_60"]),
        ("성장률 제외", [f for f in all_feats if f != "Earnings_Growth"]),
    ]
    horizon_choices = [("6개월", None), ("3개월", 63), ("1개월", 21)]
    rf_choices = [
        ("깊이3", None),
        ("깊이2", {"max_depth": 2}),
        ("깊이5", {"max_depth": 5}),
        ("잎20", {"min_samples_leaf": 20}),
    ]

    variants = []
    for (fname, feats), (hname, hz), (rname, rf) in itertools.product(
            feature_choices, horizon_choices, rf_choices):
        variants.append({
            "label": f"{fname} · {hname} · {rname}",
            "feature_subset": feats,
            "horizon_override": hz,
            "rf_params": rf,
        })
    return variants


def main():
    pool = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=core.AUTO_YEARS)).strftime("%Y-%m-%d")

    print("가격·매크로 수집 중...")
    t0 = time.time()
    df_all = core.download_all_data(tuple(pool), start_date)
    macro_prepared = core.prepare_macro(core.download_macro_data(start_date))
    print(f"  완료 ({time.time() - t0:.0f}초)")

    print("패널 생성 중 (한 번만)...")
    t0 = time.time()
    panel = build_panel(pool, df_all, macro_prepared)
    print(f"  완료 ({time.time() - t0:.0f}초) · {len(panel):,}행 · "
          f"{panel['Ticker'].nunique()}종목\n")

    variants = make_variants()
    print(f"설정 {len(variants)}개를 순서대로 시도합니다 "
          f"(데이터·파이프라인 동일, 연구자의 선택만 다름)\n")

    rows = []
    t0 = time.time()
    for i, v in enumerate(variants, 1):
        res = core.run_value_model(
            panel,
            horizon_override=v["horizon_override"],
            feature_subset=v["feature_subset"],
            rf_params=v["rf_params"],
        )
        if not res or "error" in res:
            print(f"  [{i:2d}/{len(variants)}] {v['label']:<34} 실패")
            continue
        rows.append({"label": v["label"], "auc": res["auc"],
                     "t": res.get("fm_tstat", np.nan), "p": res["pvalue"],
                     "T": res.get("fm_n_dates", 0)})
        star = " ***" if res["pvalue"] <= 0.05 else ""
        print(f"  [{i:2d}/{len(variants)}] {v['label']:<34} "
              f"AUC {res['auc']:.3f} · p {res['pvalue']:.4f}{star}", flush=True)

    if not rows:
        print("유효한 결과가 없습니다.")
        return

    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    df.to_csv("experiment_multipletesting_results.csv", index=False, encoding="utf-8-sig")

    p = df["p"]
    n_sig = int((p <= 0.05).sum())
    best = df.iloc[0]
    honest = df[df["label"].str.startswith("전체 지표 · 6개월 · 깊이3")]

    print()
    print("=" * 80)
    print(f"설정 {len(df)}개 시도 결과")
    print("=" * 80)
    if len(honest):
        h = honest.iloc[0]
        print(f"  사전에 정한 설정 하나만 보고했다면 : "
              f"AUC {h['auc']:.3f} · p {h['p']:.4f} "
              f"({'유의' if h['p'] <= 0.05 else '유의하지 않음'})")
    print(f"  제일 좋은 설정만 보고했다면        : "
          f"AUC {best['auc']:.3f} · p {best['p']:.4f} "
          f"({'유의' if best['p'] <= 0.05 else '유의하지 않음'})")
    print(f"     -> 그 설정: {best['label']}")
    print()
    print(f"  p<0.05를 주는 설정 : {n_sig}/{len(df)}개 ({n_sig/len(df)*100:.0f}%)")
    print(f"  p값 중앙값         : {p.median():.3f}")
    print()
    print("  상위 5개 설정:")
    for _, r in df.head(5).iterrows():
        print(f"    p {r['p']:.4f} · AUC {r['auc']:.3f} · {r['label']}")
    print("-" * 80)
    print("  해석: 데이터도 파이프라인도 똑같습니다. 바뀐 것은 연구자의 선택뿐인데,")
    print("        '무엇을 보고할지' 고르는 것만으로 결론이 달라집니다.")
    print("        이렇게 고른 p값은 더 이상 5% 유의수준을 의미하지 않습니다.")
    print()
    print("저장: experiment_multipletesting_results.csv")


if __name__ == "__main__":
    main()
