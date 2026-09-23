"""논문 실험: 생존 편향과 지수 편입일 룩어헤드의 기여도.

[두 가지 다른 문제]
  (가) 편입일 룩어헤드 - 현재 구성종목 중 분석 기간 도중에 편입된 종목을
       백테스트가 처음부터 포함시킵니다. "이 회사가 나중에 S&P500에 들어갈 것"을
       미리 아는 셈입니다. 지수 편입은 보통 좋은 성과의 결과이므로 상방 편향입니다.

  (나) 생존 편향 - 그 기간에 지수에서 빠진 종목들이 아예 없습니다.
       성과가 나빠 탈락했거나, 인수·합병으로 사라진 기업들입니다.

[세 조건 비교]
  A. 현재 방식        : 현재 구성종목 전체를 전 기간에 사용 (가·나 둘 다 있음)
  B. 편입일 반영      : 각 종목을 '실제 편입 기간'에만 사용 ((가) 제거)
  C. 시점별 구성종목  : 매 시점의 실제 구성종목만 사용, 탈락 종목도 복원
                        ((가)·(나) 둘 다 제거 - 단, 가격을 구할 수 있는 만큼만)

[개정 내역 - 이전 버전의 두 가지 오류]
  1) '탈락 종목 100% 복원'은 집계 오류였습니다. 여러 티커를 한 번에 받으면 가격이
     없는 티커도 전 기간 NaN 행으로 딸려 오는데, 이 NaN 행을 '데이터 확보'로 셌습니다.
     이제는 '편입 기간 중 실제 가격이 있는 날'의 비율로 복원율을 셉니다.
  2) 미래수익률(라벨)을 구성종목 필터 *뒤에* 행 기준 shift로 만들어, 지수 탈락 직전
     21일이 표본에서 빠지고(탈락은 대개 급락 뒤에 옴) 재편입 종목은 라벨이 몇 달
     뒤를 가리켰습니다. 이제 전체 가격 이력에서 라벨을 먼저 만든 뒤 필터합니다.
     개정 전 방식의 결과도 'C(구 라벨)'로 함께 기록해 차이를 보입니다.

[비교 방식]
  세 조건은 같은 평가일을 쓰므로, 'p값이 유의/비유의로 갈렸는가'가 아니라 날짜별
  AUC 차이를 대응 검정(NW + fixed-b)해 조건 간 차이를 판단합니다.

[한계 - 논문에 반드시 명시]
  상장폐지(인수·합병·파산)된 종목은 yfinance에 가격이 없어 복원되지 않습니다.
  그 결측은 무작위가 아닙니다 - 인수는 대체로 프리미엄을 받는 좋은 결말이고,
  파산은 나쁜 결말이라 방향이 섞여 있어 잔여 편향의 부호도 단정할 수 없습니다.

실행: py experiment_survivorship.py
"""
import io
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time

import numpy as np
import pandas as pd
import requests

import core
from stats_utils import paired_diff_test

UA = {"User-Agent": "Mozilla/5.0 (academic research; contact via github)"}
HIST_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
            + requests.utils.quote("S&P 500 Historical Components & Changes (Updated).csv"))
HIST_CACHE = "sp500_historical_components.csv"
COVERED_MIN = 0.90   # 편입 기간 중 이 비율 이상 가격이 있으면 '복원'으로 본다


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


def apply_membership(panel, hist):
    """각 행(날짜, 종목)이 그 시점에 실제 지수 구성종목이었는지로 걸러냅니다."""
    dates = panel.index.unique()
    # 날짜별 구성종목 집합을 미리 만들어두고 매핑
    date_to_set = {d: membership_at(hist, d) for d in dates}
    keep = np.fromiter(
        (tk in date_to_set[d] for d, tk in zip(panel.index, panel["Ticker"])),
        dtype=bool, count=len(panel))
    return panel[keep]


def coverage_table(removed, prices, hist, calendar):
    """탈락 종목별로 '편입 기간 중 실제 가격이 있는 날'의 비율을 센다.

    분류:
      복원          : 편입 기간의 COVERED_MIN 이상에 가격이 있음
      부분 복원      : 일부만 있음
      재사용 의심    : 편입 기간엔 가격이 없는데 그 밖에는 있음
                      (옛 티커를 다른 회사가 쓰는 경우. 필터 후 행이 남지 않아 결과엔 무영향)
      가격 없음      : 상장폐지 등으로 가격이 전혀 없음
    """
    date_to_set = {d: membership_at(hist, d) for d in calendar}
    rows = []
    for tk in removed:
        mem_days = pd.DatetimeIndex([d for d in calendar if tk in date_to_set[d]])
        first, last, n_px = core.price_coverage(prices, tk)
        if n_px == 0:
            px_days = pd.DatetimeIndex([])
        else:
            px_days = prices[(tk, "Close")].dropna().index
        n_mem = len(mem_days)
        n_hit = len(mem_days.intersection(px_days))
        frac = n_hit / n_mem if n_mem else np.nan
        if n_px == 0:
            status = "가격 없음"
        elif n_hit == 0:
            status = "재사용 의심"
        elif frac >= COVERED_MIN:
            status = "복원"
        else:
            status = "부분 복원"
        rows.append({"ticker": tk, "편입일수": n_mem, "가격있는편입일수": n_hit,
                     "복원율": frac, "가격첫날": first, "가격마지막날": last,
                     "편입첫날": mem_days.min() if n_mem else pd.NaT,
                     "편입마지막날": mem_days.max() if n_mem else pd.NaT,
                     "분류": status})
    return pd.DataFrame(rows)


def run(name, panel):
    res = core.run_value_model(panel, horizon_override=core.SWING_HORIZON,
                               require_fundamentals=False)
    if not res or "error" in res:
        print(f"  {name:<40} 실패: {res.get('error') if res else '?'}")
        return None
    print(f"  {name:<40} {core.fmt_result(res)}", flush=True)
    return res


def main():
    start_ts, end_ts = pd.Timestamp(core.SWING_START), pd.Timestamp(core.SWING_END)

    print("시점별 구성종목 데이터 로드...")
    hist = load_historical_membership()
    print(f"  {hist['date'].min().date()} ~ {hist['date'].max().date()} · "
          f"스냅샷 {len(hist)}개")

    current = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))

    # 분석 기간에 한 번이라도 지수에 있었던 종목 전체 (= 탈락 종목 포함).
    # 기간 시작 시점에 유효했던 스냅샷(시작일 직전의 마지막 스냅샷)도 포함해야 한다.
    first_idx = max(0, hist["date"].searchsorted(start_ts, side="right") - 1)
    window = hist.iloc[first_idx:]
    window = window[window["date"] < end_ts]
    ever = set()
    for s in window["tickers"]:
        ever |= {t.strip().replace(".", "-") for t in str(s).split(",")}
    ever |= set(current)
    removed = sorted(ever - set(current))
    print(f"  분석 기간 내 한 번이라도 편입된 종목: {len(ever)}개")
    print(f"  그중 현재는 빠진 종목: {len(removed)}개\n")

    print("가격·매크로 로드 (탈락 종목 포함, 스냅샷)...")
    t0 = time.time()
    prices, macro_prepared = core.load_swing_data(ever)
    print(f"  완료 ({time.time() - t0:.0f}초)")

    calendar = macro_prepared.index[(macro_prepared.index >= start_ts)
                                    & (macro_prepared.index < end_ts)]
    cov = coverage_table(removed, prices, hist, calendar)
    cov.to_csv("experiment_survivorship_coverage.csv", index=False, encoding="utf-8-sig")
    counts = cov["분류"].value_counts()
    tot_mem = cov["편입일수"].sum()
    tot_hit = cov["가격있는편입일수"].sum()
    print("\n탈락 종목 가격 복원 현황 (편입 기간 기준)")
    for k in ["복원", "부분 복원", "재사용 의심", "가격 없음"]:
        print(f"  {k:<8}: {int(counts.get(k, 0)):>4}개")
    print(f"  종목-일 기준 복원율: {tot_hit:,}/{tot_mem:,} ({tot_hit / max(tot_mem, 1) * 100:.1f}%)\n")

    print("패널 생성 중 (전체 유니버스, 한 번만)...")
    t0 = time.time()
    panel_all = core.build_swing_panel(sorted(ever), prices, macro_prepared)
    # ⭐ 라벨은 필터 전에, 전체 가격 이력으로 만든다 (개정 내역 2)
    panel_all = core.add_forward_returns(panel_all, [core.SWING_HORIZON])
    print(f"  완료 ({time.time() - t0:.0f}초) · {len(panel_all):,}행 · "
          f"{panel_all['Ticker'].nunique()}종목\n")

    print("=" * 96)
    print("조건별 결과")
    print("=" * 96)
    results = {}
    pa = panel_all[panel_all["Ticker"].isin(current)]
    results["A"] = run("A. 현재 방식 (편입일 무시·탈락종목 없음)", pa)

    pb = apply_membership(pa, hist)
    print(f"     (편입일 적용으로 {len(pa):,} -> {len(pb):,}행, "
          f"{(1 - len(pb)/len(pa))*100:.1f}% 제거)")
    results["B"] = run("B. 편입일 반영 (룩어헤드 제거)", pb)

    pc = apply_membership(panel_all, hist)
    n_rm_rows = int(pc["Ticker"].isin(removed).sum())
    print(f"     (탈락 종목 복원으로 {n_rm_rows:,}행 추가, 총 {len(pc):,}행)")
    results["C"] = run("C. 시점별 구성종목 (생존편향 부분 교정)", pc)

    # 개정 전 라벨 방식(필터 후 행 기준 shift) - 수정의 영향을 보이기 위한 참고 행
    pc_old = pc.drop(columns=[f"Fwd_Return_{core.SWING_HORIZON}"])
    results["C_old"] = run("C(구 라벨: 필터 후 계산, 참고용)", pc_old)

    print("-" * 96)
    print("조건 간 차이 (날짜별 AUC 대응 검정, NW + fixed-b)")
    pairs = [("A", "B", "편입일 룩어헤드 (A-B)"),
             ("B", "C", "생존 편향 (B-C)"),
             ("A", "C", "합계 (A-C)"),
             ("C", "C_old", "라벨 수정의 영향 (C-C구)")]
    diff_rows = []
    for x, y, label in pairs:
        rx, ry = results.get(x), results.get(y)
        if not rx or not ry:
            continue
        md, tt, _, p2, Tc = paired_diff_test(rx["per_date_auc"], ry["per_date_auc"],
                                             nw_lag=rx["nw_lag"], fixed_b=True)
        diff_rows.append({"비교": label, "ΔAUC": md, "t": tt, "p양측": p2, "공통날짜": Tc})
        print(f"  {label:<24} ΔAUC {md:+.4f} · t {tt:+.2f} · 양측 p {p2:.3f} · 날짜 {Tc}")

    print()
    n_rest = int(counts.get("복원", 0)) + int(counts.get("부분 복원", 0))
    print(f"  ⚠️ 한계: 탈락 종목 {len(removed)}개 중 가격이 편입 기간에 있는 종목은 {n_rest}개뿐입니다.")
    print("     나머지는 상장폐지(인수·합병·파산) 등으로 가격이 없으며, 그 결측은 무작위가 아닙니다.")

    out = pd.DataFrame([
        {"조건": k, "AUC": v["fm_auc"], "t": v["fm_tstat"], "p단측": v["pvalue"],
         "p양측": v["pvalue_two_sided"], "날짜": v["fm_n_dates"], "종목수": v["n_tickers"]}
        for k, v in results.items() if v
    ])
    out.to_csv("experiment_survivorship_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diff_rows).to_csv("experiment_survivorship_diffs.csv", index=False,
                                   encoding="utf-8-sig")
    print("\n저장: experiment_survivorship_results.csv, experiment_survivorship_diffs.csv, "
          "experiment_survivorship_coverage.csv")


if __name__ == "__main__":
    main()
