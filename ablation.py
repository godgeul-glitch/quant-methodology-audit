"""논문 1부: 방법론 결함별 기여도 분해(ablation).

[무엇을 보이려는 연구인가]
"무료 공개 데이터로 만든 개인 투자자용 퀀트 파이프라인"에서, 흔히 저지르는
방법론 결함들이 각각 '없는 예측력'을 얼마나 만들어내는지 정량화합니다.
결함을 하나씩만 켜고 나머지는 올바른 설정으로 고정해(one-at-a-time ablation),
그 결함 단독의 기여도를 측정합니다.

[측정하는 결함]
  ① p값 계산      : 행(종목×날짜)을 독립 표본으로 셈 (iid 가정)
  ② 소표본 유니버스 : 종목 41개 (vs S&P500 확장)
  ③ 연간 룩어헤드   : 10-K에도 분기 기한(45일)을 적용 (실제 기한 60~90일)
  ④ 성장률 부호     : pct_change가 적자 기준에서 부호를 뒤집음

[측정하지 못하는 결함 - 논문에 명시할 것]
  ⑤ 생존 편향 : 현재 시점 S&P500 구성종목으로 과거를 학습. 상장폐지·지수
     제외 종목이 빠져 있습니다. 시점별 구성종목 데이터가 무료로는 사실상
     구할 수 없어 이 스크립트로는 켜고 끌 수 없고, 한계로 기술합니다.

실행: py ablation.py            (전체 유니버스, 오래 걸림)
      py ablation.py --quick    (종목 60개로 축소, 빠른 점검용)
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import core


# 결함을 하나씩만 켜는 설정들. 첫 줄이 '전부 올바른' 기준선입니다.
CONFIGS = [
    # (이름, universe, lag_quarterly, lag_annual, growth_formula, pvalue_method, eval_mode)
    ("기준선 (전부 수정됨)",       "large", 45, 90, "abs_base", "fama_macbeth", "overlap_nw"),
    ("① p값만 iid 가정",           "large", 45, 90, "abs_base", "iid",          "overlap_nw"),
    ("② 소표본 유니버스(41종목)",   "small", 45, 90, "abs_base", "fama_macbeth", "overlap_nw"),
    ("③ 연간 룩어헤드(연간도 45일)", "large", 45, 45, "abs_base", "fama_macbeth", "overlap_nw"),
    ("④ 성장률 부호 오류",         "large", 45, 90, "naive",    "fama_macbeth", "overlap_nw"),
    ("⑤ 평가일 솎아내기(구 방식)",   "large", 45, 90, "abs_base", "fama_macbeth", "nonoverlap_iid"),
    ("전부 순진하게 (원래 상태)",    "small", 45, 45, "naive",    "iid",          "nonoverlap_iid"),
]


def build_panel(tickers, df_all, macro_prepared, lag_q, lag_a, growth, workers=20):
    """주어진 설정으로 전 종목 패널을 만듭니다(원본 재무는 캐시 재사용).

    ⭐ [재현성] 결과를 종목별 dict에 담았다가 '정렬된 티커 순서'로 이어붙입니다.
    as_completed 순서(= 네트워크 응답 도착 순서)대로 붙이면 실행할 때마다 행
    순서가 달라지고, 같은 날짜 안의 행 순서가 바뀌면서 RandomForest의 부트스트랩
    표본이 달라져 AUC가 매번 미세하게 흔들립니다. 논문 수치는 재현되어야 하므로
    수집은 병렬로 하되 결합은 결정론적으로 합니다.
    """
    results = {}
    def one(tk):
        try:
            fh = core.get_fundamental_history(tk, lag_quarterly=lag_q,
                                              lag_annual=lag_a, growth_formula=growth)
            if fh.empty:
                return tk, None
            return tk, core.build_value_panel(df_all, fh, macro_prepared, tk)
        except Exception:
            return tk, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tk, p in ex.map(one, tickers):
            if p is not None and not p.empty:
                results[tk] = p

    ordered = [results[tk] for tk in sorted(results)]
    if not ordered:
        return pd.DataFrame()
    panel = pd.concat(ordered)
    # 날짜 우선, 같은 날짜 안에서는 티커 알파벳 순으로 고정
    return panel.sort_values("Ticker", kind="mergesort").sort_index(kind="mergesort")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="종목 수를 줄여 빠르게 점검")
    args = ap.parse_args()

    small = list(core.ALL_TICKERS)
    large = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))
    if args.quick:
        # 소표본 대비 효과를 보려면 두 유니버스 크기 차이는 유지해야 함
        small = small[:20]
        large = large[:60]
        print(f"[quick] 소표본 {len(small)}종목 / 확장 {len(large)}종목으로 축소 실행\n")

    start_date = (pd.Timestamp.today() - pd.DateOffset(years=core.AUTO_YEARS)).strftime("%Y-%m-%d")

    print("가격·매크로 데이터 수집 중...")
    t0 = time.time()
    df_all = core.download_all_data(tuple(large), start_date)
    macro_prepared = core.prepare_macro(core.download_macro_data(start_date))
    if df_all.empty:
        print("가격 데이터를 받지 못했습니다. 잠시 후 다시 시도하세요.")
        sys.exit(1)
    print(f"  완료 ({time.time() - t0:.0f}초)\n")

    rows = []
    for name, uni, lag_q, lag_a, growth, pmethod, emode in CONFIGS:
        tickers = small if uni == "small" else large
        t0 = time.time()
        print(f"[{name}] 종목 {len(tickers)}개 · 패널 생성 중...", flush=True)
        panel = build_panel(tickers, df_all, macro_prepared, lag_q, lag_a, growth)
        if panel.empty:
            rows.append((name, len(tickers), None, None, None, None, "패널 생성 실패"))
            print("   -> 패널 비어있음\n")
            continue

        res = core.run_value_model(panel, pvalue_method=pmethod, eval_mode=emode)
        if res is None or "error" in res:
            rows.append((name, len(tickers), None, None, None, None,
                         f"모델 실패: {res.get('error', '?') if res else '?'}"))
            print(f"   -> {res.get('error') if res else '실패'}\n")
            continue

        auc = res["auc"]
        pval = res["pvalue"]
        fm_T = res.get("fm_n_dates", 0)
        fm_t = res.get("fm_tstat", float("nan"))
        verdict = ("유의" if (np.isfinite(pval) and pval <= 0.05) else "유의하지 않음")
        rows.append((name, len(tickers), auc, fm_t, pval, fm_T, verdict))
        print(f"   -> AUC {auc:.3f} · t {fm_t if np.isfinite(fm_t) else float('nan'):.2f} "
              f"· p {pval:.4f} · 독립날짜 {fm_T}개 · {verdict}  ({time.time() - t0:.0f}초)\n")

    print("\n" + "=" * 92)
    print("결함별 기여도 (one-at-a-time ablation)")
    print("=" * 92)
    hdr = f"{'설정':<28}{'종목':>6}{'AUC':>8}{'t값':>9}{'p값':>10}{'독립날짜':>9}  판정"
    print(hdr)
    print("-" * 92)
    for name, n, auc, t, p, T, verdict in rows:
        if auc is None:
            print(f"{name:<28}{n:>6}{'-':>8}{'-':>9}{'-':>10}{'-':>9}  {verdict}")
            continue
        t_txt = f"{t:+.2f}" if (t is not None and np.isfinite(t)) else "-"
        p_txt = f"{p:.4f}" if (p is not None and np.isfinite(p)) else "-"
        print(f"{name:<28}{n:>6}{auc:>8.3f}{t_txt:>9}{p_txt:>10}{T:>9}  {verdict}")
    print("-" * 92)
    print("해석: 기준선 대비 특정 행에서 AUC가 높아지거나 p값이 작아졌다면,")
    print("      그 차이는 시장의 신호가 아니라 '그 결함이 만들어낸 착시'입니다.")
    print("주의: 생존 편향(현재 지수 구성종목으로 과거 학습)은 이 표로 측정 불가 — 한계로 기술.")

    out = pd.DataFrame(rows, columns=["설정", "종목수", "AUC", "t값", "p값", "독립날짜", "판정"])
    out.to_csv("ablation_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: ablation_results.csv")


if __name__ == "__main__":
    main()
