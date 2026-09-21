"""논문 실험: 생존 편향과 지수 편입일 룩어헤드의 기여도.

[두 가지 다른 문제]
  (가) 편입일 룩어헤드 - 현재 S&P500 구성종목 503개 중 79개(15.7%)는 최근 5년
       내에 편입됐습니다. 그런데 백테스트는 이들을 처음부터 포함시킵니다.
       "이 회사가 나중에 S&P500에 들어갈 것"을 미리 아는 셈입니다.
       지수 편입은 보통 좋은 성과의 결과이므로 체계적인 상방 편향입니다.

  (나) 생존 편향 - 그 기간에 지수에서 빠진 종목들이 아예 없습니다.
       성과가 나빠 탈락했거나, 인수·합병으로 사라진 기업들입니다.

[세 조건 비교]
  A. 현재 방식        : 현재 구성종목 전체를 전 기간에 사용 (가·나 둘 다 있음)
  B. 편입일 반영      : 각 종목을 '실제 편입일 이후'에만 사용 ((가) 제거)
  C. 시점별 구성종목  : 매 시점의 실제 구성종목만 사용, 탈락 종목도 복원
                        ((가)·(나) 둘 다 제거 - 단, 데이터 구할 수 있는 만큼만)

[한계 - 논문에 반드시 명시]
  C에서도 완전한 교정은 불가능합니다. 지수에서 빠진 종목의 상당수는 인수·합병
  되거나 사명이 바뀌어 yfinance에서 조회되지 않습니다(표본 점검 결과 약 60%).
  게다가 그 결측이 무작위가 아닙니다 - 인수된 기업은 대체로 프리미엄을 받고
  좋게 끝난 경우라, 빠진 쪽이 한 방향으로 치우쳐 있습니다.

실행: py experiment_survivorship.py
"""
import io
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import core

UA = {"User-Agent": "Mozilla/5.0 (academic research; contact via github)"}
HIST_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
            + requests.utils.quote("S&P 500 Historical Components & Changes (Updated).csv"))
HIST_CACHE = "sp500_historical_components.csv"


def load_historical_membership():
    """시점별 S&P500 구성종목. 한 번 받아서 로컬에 캐시."""
    if os.path.exists(HIST_CACHE):
        df = pd.read_csv(HIST_CACHE)
    else:
        r = requests.get(HIST_URL, headers=UA, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv(HIST_CACHE, index=False, encoding="utf-8")
        print(f"  시점별 구성종목 다운로드 -> {HIST_CACHE}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def membership_at(hist, when):
    """해당 시점에 유효한 구성종목 집합 (그 이전 마지막 스냅샷)."""
    idx = hist["date"].searchsorted(pd.Timestamp(when), side="right") - 1
    if idx < 0:
        return set()
    return {t.strip().replace(".", "-") for t in str(hist.loc[idx, "tickers"]).split(",")}


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


def apply_membership(panel, hist):
    """각 행(날짜, 종목)이 그 시점에 실제 지수 구성종목이었는지로 걸러냅니다."""
    dates = panel.index.unique()
    # 날짜별 구성종목 집합을 미리 만들어두고 매핑
    date_to_set = {d: membership_at(hist, d) for d in dates}
    keep = np.fromiter(
        (tk in date_to_set[d] for d, tk in zip(panel.index, panel["Ticker"])),
        dtype=bool, count=len(panel))
    return panel[keep]


def report(name, res):
    if not res or "error" in res:
        print(f"  {name:<34} 실패: {res.get('error') if res else '?'}")
        return None
    print(f"  {name:<34} AUC {res['auc']:.3f} · t {res.get('fm_tstat', float('nan')):+.2f} "
          f"· p {res['pvalue']:.4f} · 날짜 {res.get('fm_n_dates', 0)}"
          + ("  ***" if res["pvalue"] <= 0.05 else ""))
    return res


def main():
    start_ts = pd.Timestamp.today() - pd.DateOffset(years=core.SWING_YEARS)
    start_date = start_ts.strftime("%Y-%m-%d")

    print("시점별 구성종목 데이터 로드...")
    hist = load_historical_membership()
    print(f"  {hist['date'].min().date()} ~ {hist['date'].max().date()} · "
          f"스냅샷 {len(hist)}개")

    current = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))

    # 분석 기간에 한 번이라도 지수에 있었던 종목 전체 (= 탈락 종목 포함)
    window = hist[hist["date"] >= start_ts]
    ever = set()
    for s in window["tickers"]:
        ever |= {t.strip().replace(".", "-") for t in str(s).split(",")}
    ever |= set(current)
    removed = sorted(ever - set(current))
    print(f"  분석 기간 내 한 번이라도 편입된 종목: {len(ever)}개")
    print(f"  그중 현재는 빠진 종목: {len(removed)}개\n")

    print("가격·매크로 수집 중 (탈락 종목 포함)...")
    t0 = time.time()
    df_all = core.download_all_data(tuple(sorted(ever)), start_date)
    macro_prepared = core.prepare_macro(core.download_macro_data(start_date))
    print(f"  완료 ({time.time() - t0:.0f}초)")

    print("패널 생성 중 (전체 유니버스, 한 번만)...")
    t0 = time.time()
    panel_all = build_panel(sorted(ever), df_all, macro_prepared)
    got = set(panel_all["Ticker"].unique())
    got_removed = sorted(got & set(removed))
    print(f"  완료 ({time.time() - t0:.0f}초) · {len(panel_all):,}행 · {len(got)}종목")
    print(f"  탈락 종목 중 데이터 확보: {len(got_removed)}/{len(removed)}개 "
          f"({len(got_removed)/max(len(removed),1)*100:.0f}%)\n")

    print("=" * 84)
    print("조건별 결과")
    print("=" * 84)
    results = {}

    # A. 현재 방식 - 현재 구성종목을 전 기간에 사용
    pa = panel_all[panel_all["Ticker"].isin(current)]
    results["A"] = report("A. 현재 방식 (편입일 무시·탈락종목 없음)",
                          core.run_value_model(pa, horizon_override=core.SWING_HORIZON,
                                              require_fundamentals=False))

    # B. 편입일 반영 - 현재 구성종목만 쓰되 실제 편입 이후만
    pb = apply_membership(pa, hist)
    print(f"     (편입일 적용으로 {len(pa):,} -> {len(pb):,}행, "
          f"{(1 - len(pb)/len(pa))*100:.1f}% 제거)")
    results["B"] = report("B. 편입일 반영 (룩어헤드 제거)",
                          core.run_value_model(pb, horizon_override=core.SWING_HORIZON,
                                              require_fundamentals=False))

    # C. 시점별 구성종목 - 탈락 종목까지 복원
    pc = apply_membership(panel_all, hist)
    n_rm_rows = (pc["Ticker"].isin(got_removed)).sum()
    print(f"     (탈락 종목 복원으로 {n_rm_rows:,}행 추가, 총 {len(pc):,}행)")
    results["C"] = report("C. 시점별 구성종목 (생존편향 부분 교정)",
                          core.run_value_model(pc, horizon_override=core.SWING_HORIZON,
                                              require_fundamentals=False))

    print("-" * 84)
    a, b, c = results.get("A"), results.get("B"), results.get("C")
    if a and b:
        print(f"  편입일 룩어헤드의 기여도 : AUC {a['auc']:.3f} -> {b['auc']:.3f} "
              f"({b['auc'] - a['auc']:+.3f})")
    if b and c:
        print(f"  생존편향의 기여도(부분)  : AUC {b['auc']:.3f} -> {c['auc']:.3f} "
              f"({c['auc'] - b['auc']:+.3f})")
    print()
    print(f"  ⚠️ 한계: 탈락 종목 {len(removed)}개 중 {len(got_removed)}개만 복원됨. "
          f"나머지는 인수·합병·사명변경으로 조회 불가이며,")
    print("     그 결측은 무작위가 아니므로(인수는 대체로 좋은 결말) C도 완전한 교정이 아님.")

    out = pd.DataFrame([
        {"조건": k, "AUC": v["auc"], "t": v.get("fm_tstat"), "p": v["pvalue"],
         "날짜": v.get("fm_n_dates")}
        for k, v in results.items() if v
    ])
    out.to_csv("experiment_survivorship_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: experiment_survivorship_results.csv")


if __name__ == "__main__":
    main()
