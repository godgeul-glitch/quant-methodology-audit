"""논문 실험: 종목 선택(유니버스)이 유의성을 만들어내는가?

[배경]
ablation에서 '손으로 고른 42종목'만으로도 p=0.0188(유의)이 나왔는데,
같은 설정에서 501종목으로 늘리면 p=0.16(유의하지 않음)이 됩니다.
이 차이가 어디서 오는지에는 두 가지 가능성이 있습니다:

  (A) 소표본 자체의 문제 - 42종목이면 아무 종목을 뽑아도 우연히 유의한
      결과가 자주 나온다 (다중검정/소표본 문제)
  (B) 종목 선택의 문제 - 개발자가 관심 있는 종목(테크·성장주 편중)을
      손으로 고른 것 자체가 편향이다 (선택 편향)

[방법]
S&P500에서 무작위로 42종목을 N번 뽑아 매번 같은 파이프라인(전부 올바른
설정: Fama-MacBeth + Newey-West, 분기45/연간90 지연, 성장률 부호 수정)을
돌리고 p값 분포를 봅니다.

  - 무작위 draw 중 p<0.05 비율이 5%를 크게 넘으면  -> (A) 소표본 문제
  - 손으로 고른 명단이 그 분포의 꼬리에 있으면      -> (B) 선택 편향
  - 둘 다면 둘 다

실행: py experiment_universe.py           (기본 100회)
      py experiment_universe.py --n 30    (횟수 지정)
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import core

UNIVERSE_SIZE = 42   # 손으로 고른 명단과 같은 크기로 맞춤


def build_panel(tickers, df_all, macro_prepared, workers=20):
    """전부 올바른 설정으로 패널 생성 (원본 재무는 캐시 재사용)."""
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
    # 재현성: 날짜 우선, 같은 날짜 안에서는 티커 알파벳 순으로 고정
    return panel.sort_values("Ticker", kind="mergesort").sort_index(kind="mergesort")


def evaluate(tickers, df_all, macro_prepared):
    panel = build_panel(tickers, df_all, macro_prepared)
    if panel.empty:
        return None
    res = core.run_value_model(panel)
    if res is None or "error" in res:
        return None
    return {
        "auc": res["auc"],
        "t": res.get("fm_tstat", float("nan")),
        "p": res["pvalue"],
        "T": res.get("fm_n_dates", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="무작위 추출 횟수")
    args = ap.parse_args()

    pool = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))
    handpicked = sorted(set(core.ALL_TICKERS))
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=core.AUTO_YEARS)).strftime("%Y-%m-%d")

    print(f"전체 풀 {len(pool)}종목에서 {UNIVERSE_SIZE}종목씩 {args.n}회 무작위 추출\n")

    print("가격·매크로 수집 중...")
    t0 = time.time()
    df_all = core.download_all_data(tuple(pool), start_date)
    macro_prepared = core.prepare_macro(core.download_macro_data(start_date))
    print(f"  완료 ({time.time() - t0:.0f}초)")

    # 무작위 draw가 어떤 종목을 뽑든 즉시 쓰이도록 원본 재무를 미리 데워둔다
    print("재무 원본 캐시 예열 중 (이후 draw는 네트워크 호출 없음)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(lambda t: core.get_fundamental_history(t), pool))
    print(f"  완료 ({time.time() - t0:.0f}초)\n")

    # 기준: 손으로 고른 명단
    hp = evaluate(handpicked, df_all, macro_prepared)
    if hp:
        print(f"[손으로 고른 {len(handpicked)}종목] "
              f"AUC {hp['auc']:.3f} · t {hp['t']:+.2f} · p {hp['p']:.4f} · 날짜 {hp['T']}\n")

    rng = np.random.RandomState(42)
    rows = []
    t0 = time.time()
    for i in range(args.n):
        draw = sorted(rng.choice(pool, size=UNIVERSE_SIZE, replace=False).tolist())
        r = evaluate(draw, df_all, macro_prepared)
        if r is None:
            continue
        rows.append(r)
        done = len(rows)
        if done % 10 == 0 or done == 1:
            el = time.time() - t0
            eta = el / done * (args.n - done)
            print(f"  {done}/{args.n} 완료 · 경과 {el:.0f}초 · 남은 예상 {eta:.0f}초", flush=True)

    if not rows:
        print("유효한 결과가 없습니다.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("experiment_universe_results.csv", index=False, encoding="utf-8-sig")

    p = df["p"].dropna()
    auc = df["auc"].dropna()
    sig = (p <= 0.05).mean() * 100

    print()
    print("=" * 76)
    print(f"무작위 {UNIVERSE_SIZE}종목 추출 {len(df)}회 결과")
    print("=" * 76)
    print(f"  AUC   중앙값 {auc.median():.3f} · 5~95% 구간 [{auc.quantile(.05):.3f}, {auc.quantile(.95):.3f}]")
    print(f"  p값   중앙값 {p.median():.3f}")
    print(f"  p<0.05 비율  {sig:.1f}%   (참값이 5%여야 정상)")
    print(f"  p<0.10 비율  {(p <= 0.10).mean()*100:.1f}%")
    if hp:
        pct_auc = (auc < hp["auc"]).mean() * 100
        pct_p = (p > hp["p"]).mean() * 100
        print()
        print(f"  손으로 고른 명단의 위치:")
        print(f"    AUC {hp['auc']:.3f} -> 무작위 추출의 상위 {100 - pct_auc:.0f}% 지점")
        print(f"    p값 {hp['p']:.4f} -> 무작위 추출의 {pct_p:.0f}%보다 더 유의함")
    print("-" * 76)
    if sig > 15:
        print(f"  해석: 무작위로 뽑아도 {sig:.0f}%가 '유의'하게 나옴 -> 소표본 자체가 문제(A).")
        print("        종목 수가 적으면 아무 명단이나 유의한 결과를 만들 수 있다는 뜻.")
    elif hp and hp["p"] <= p.quantile(0.1):
        print("  해석: 무작위는 정상 수준인데 손으로 고른 명단만 유독 유의 -> 선택 편향(B).")
    else:
        print("  해석: 무작위 추출의 오탐률이 정상 범위이고, 손으로 고른 명단도 특별하지 않음.")
    print()
    print("저장: experiment_universe_results.csv")


if __name__ == "__main__":
    main()
