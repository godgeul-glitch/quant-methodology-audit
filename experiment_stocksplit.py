"""논문 실험: 모델이 '팩터 관계'를 배우는가, '종목별 습성'을 외우는가?

[배경]
지금까지의 교차검증은 **날짜로만** 나눕니다. 그래서 같은 종목이 학습 구간과
평가 구간에 모두 등장합니다. 이러면 모델이 "이익수익률이 높으면 좋다" 같은
일반적 관계 대신, **"이 종목은 대체로 평균을 이긴다"는 종목별 습성**을 외워도
평가에서 점수를 받습니다. 실전에서 새로운 종목에 적용하면 무용지물이죠.

무작위 유니버스 실험에서 42종목(AUC 중앙값 0.528)이 501종목(0.514)보다
체계적으로 높게 나온 것도 이걸로 설명될 수 있습니다. 종목이 적을수록 외우기
쉬우니까요.

[방법 - 학습을 고정하고 평가 종목만 바꾸는 통제 실험]
  유니버스를 무작위로 절반씩 S1, S2로 나눈 뒤
    조건 A(본 종목)  : S1으로 학습 -> S1으로 평가
    조건 B(안 본 종목): S1으로 학습 -> S2으로 평가
  학습 데이터가 완전히 동일하고 평가 날짜도 동일합니다. 다른 건 오직
  '평가 종목이 학습에 있었는가' 하나뿐입니다.

  A >> B 이면 -> 모델이 종목을 외우고 있었다는 뜻.
  A ≈ B 이면  -> 일반화되는 팩터 관계를 배웠다는 뜻.

실행: py experiment_stocksplit.py            (기본 20회 반복)
      py experiment_stocksplit.py --n 5
"""
import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import core


def build_panel(tickers, df_all, macro_prepared, workers=20):
    results = {}

    def one(tk):
        try:
            return tk, core.build_value_panel(df_all, None, macro_prepared, tk, include_fundamentals=False)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="무작위 절반 분할 반복 횟수")
    args = ap.parse_args()

    pool = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=core.SWING_YEARS)).strftime("%Y-%m-%d")

    print("가격·매크로 수집 중...")
    t0 = time.time()
    df_all = core.download_all_data(tuple(pool), start_date)
    macro_prepared = core.prepare_macro(core.download_macro_data(start_date))
    print(f"  완료 ({time.time() - t0:.0f}초)")

    print("전체 패널 생성 중 (한 번만; 이후 반복은 이 패널을 재사용)...")
    t0 = time.time()
    panel = build_panel(pool, df_all, macro_prepared)
    print(f"  완료 ({time.time() - t0:.0f}초) · {len(panel):,}행 · "
          f"{panel['Ticker'].nunique()}종목\n")

    tickers = sorted(panel["Ticker"].unique())
    rng = np.random.RandomState(42)
    rows = []

    for i in range(args.n):
        perm = rng.permutation(tickers)
        half = len(perm) // 2
        s1 = set(perm[:half])     # 학습 종목
        s2 = set(perm[half:])     # 한 번도 학습에 안 쓰인 종목

        seen = core.run_value_model(panel, horizon_override=core.SWING_HORIZON,
                                    require_fundamentals=False, train_tickers=s1, eval_tickers=s1)
        unseen = core.run_value_model(panel, horizon_override=core.SWING_HORIZON,
                                      require_fundamentals=False, train_tickers=s1, eval_tickers=s2)
        if (not seen or "error" in seen) or (not unseen or "error" in unseen):
            continue

        rows.append({
            "seen_auc": seen["auc"], "seen_t": seen.get("fm_tstat", np.nan),
            "seen_p": seen["pvalue"],
            "unseen_auc": unseen["auc"], "unseen_t": unseen.get("fm_tstat", np.nan),
            "unseen_p": unseen["pvalue"],
            "gap": seen["auc"] - unseen["auc"],
        })
        print(f"  {i+1}/{args.n}  본 종목 AUC {seen['auc']:.3f} (p {seen['pvalue']:.3f}) | "
              f"안 본 종목 AUC {unseen['auc']:.3f} (p {unseen['pvalue']:.3f}) | "
              f"차이 {rows[-1]['gap']:+.3f}", flush=True)

    if not rows:
        print("유효한 결과가 없습니다.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("experiment_stocksplit_results.csv", index=False, encoding="utf-8-sig")

    print()
    print("=" * 76)
    print(f"학습 종목 고정 · 평가 종목만 교체 ({len(df)}회 반복)")
    print("=" * 76)
    print(f"  학습에서 본 종목으로 평가   : AUC {df['seen_auc'].mean():.3f} "
          f"· p<0.05 비율 {(df['seen_p'] <= 0.05).mean()*100:.0f}%")
    print(f"  학습에 없던 종목으로 평가   : AUC {df['unseen_auc'].mean():.3f} "
          f"· p<0.05 비율 {(df['unseen_p'] <= 0.05).mean()*100:.0f}%")
    print(f"  차이(본 - 안 본)            : {df['gap'].mean():+.3f} "
          f"(표준편차 {df['gap'].std():.3f})")

    # 차이가 0인지 대응표본 t검정
    from scipy.stats import ttest_rel
    tt = ttest_rel(df["seen_auc"], df["unseen_auc"])
    print(f"  대응표본 t검정               : t={tt.statistic:+.2f}, p={tt.pvalue:.4f}")
    print("-" * 76)
    if tt.pvalue <= 0.05 and df["gap"].mean() > 0:
        print("  해석: 학습에서 본 종목일 때 성능이 유의하게 높음")
        print("        -> 모델이 일반적 팩터 관계가 아니라 '종목별 습성'을 외우고 있다.")
        print("        -> 날짜 기반 교차검증만으로는 이 과적합을 못 걸러낸다.")
    else:
        print("  해석: 본 종목/안 본 종목 간 차이가 유의하지 않음")
        print("        -> 종목 기억으로 성능이 부풀려졌다는 증거는 없다.")
    print()
    print("저장: experiment_stocksplit_results.csv")


if __name__ == "__main__":
    main()
