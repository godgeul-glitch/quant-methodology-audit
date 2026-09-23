"""논문 실험: 모델이 '팩터 관계'를 배우는가, '종목별 습성'을 외우는가?

[배경]
지금까지의 교차검증은 **날짜로만** 나눕니다. 그래서 같은 종목이 학습 구간과
평가 구간에 모두 등장합니다. 이러면 모델이 "이익수익률이 높으면 좋다" 같은
일반적 관계 대신, **"이 종목은 대체로 평균을 이긴다"는 종목별 습성**을 외워도
평가에서 점수를 받습니다. 실전에서 새로운 종목에 적용하면 무용지물이죠.

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

import numpy as np
import pandas as pd

import core
from stats_utils import paired_diff_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="무작위 절반 분할 반복 횟수")
    args = ap.parse_args()

    pool = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))

    print(f"가격·매크로 로드 ({core.SWING_START} ~ {core.SWING_END}, 스냅샷)...")
    t0 = time.time()
    df_all, macro_prepared = core.load_swing_data(pool)
    print(f"  완료 ({time.time() - t0:.0f}초)")

    print("전체 패널 생성 중 (한 번만; 이후 반복은 이 패널을 재사용)...")
    t0 = time.time()
    panel = core.build_swing_panel(pool, df_all, macro_prepared)
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

        # 같은 평가일에서 날짜별 AUC 차이를 대응 검정 (분할 1회 안에서의 차이)
        md, dt, _, dp, _ = paired_diff_test(seen["per_date_auc"], unseen["per_date_auc"],
                                            nw_lag=seen["nw_lag"], fixed_b=True)
        rows.append({
            "seen_auc": seen["fm_auc"], "seen_t": seen["fm_tstat"],
            "seen_p": seen["pvalue_two_sided"],
            "unseen_auc": unseen["fm_auc"], "unseen_t": unseen["fm_tstat"],
            "unseen_p": unseen["pvalue_two_sided"],
            "gap": md, "gap_t": dt, "gap_p": dp,
        })
        print(f"  {i+1}/{args.n}  본 종목 AUC {seen['fm_auc']:.4f} (p {seen['pvalue_two_sided']:.3f}) | "
              f"안 본 종목 AUC {unseen['fm_auc']:.4f} (p {unseen['pvalue_two_sided']:.3f}) | "
              f"차이 {md:+.4f} (대응 p {dp:.3f})", flush=True)

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

    # 차이가 0인지 대응표본 t검정 (분할 간 검정. 분할들은 같은 데이터를 공유하므로
    # 서로 독립이 아니다 - 보조 지표로만 쓰고, 분할별 대응 검정의 유의 비율을 함께 본다)
    from scipy.stats import ttest_rel
    tt = ttest_rel(df["seen_auc"], df["unseen_auc"])
    print(f"  대응표본 t검정(분할 간)       : t={tt.statistic:+.2f}, p={tt.pvalue:.4f}")
    print(f"  분할별 날짜 대응 검정 p<0.05  : {(df['gap_p'] <= 0.05).sum()}/{len(df)}회")
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
