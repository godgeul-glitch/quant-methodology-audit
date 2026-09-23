"""논문 1부: 방법론 결함별 기여도 분해(ablation) — 스윙(가격 기반) 버전.

[무엇을 보이려는 연구인가]
"무료 공개 데이터로 만든 개인 투자자용 퀀트 파이프라인"에서, 흔히 저지르는
방법론 결함들이 각각 '없는 예측력'을 얼마나 만들어내는지 정량화합니다.
결함을 하나씩만 켜고 나머지는 올바른 설정으로 고정해(one-at-a-time ablation),
그 결함 단독의 기여도를 측정합니다.

[재무제표를 뺀 이유]
초기 실험에서 유의성이 재무 지표가 아니라 가격 변동성(1개월 구간)에서
나온다는 사실이 드러났습니다(4.4절 다중검정 실험). yfinance 무료
재무제표는 ~4년치밖에 없어 기간을 늘려 검정력을 보완할 수도 없었습니다.
그래서 재무제표를 아예 빼고, 가격만으로 계산되는 지표(모멘텀·변동성·
시장 대비 상대강도)와 더 짧은 예측 기간(1개월)으로 전환했습니다. 이제
기간 제약이 없어(SWING_YEARS) 검정력을 실질적으로 늘릴 수 있습니다.

[측정하는 결함]
  ① p값 계산       : 행(종목×날짜)을 독립 표본으로 셈 (iid 가정)
  ② 소표본 유니버스 : 손으로 고른 관심종목 42개 (vs S&P500 확장)
  ⑤ 평가일 솎아내기 : 라벨이 겹치지 않도록 evaluation date를 horizon 간격으로 솎음

[개정: 판정과 비교 방식]
  - 판정은 양측 p값 기준입니다(단측 p도 함께 기록).
  - 기준선과의 차이는 'p값이 유의/비유의로 갈렸는가'가 아니라, 같은 평가일의
    날짜별 AUC 차이를 대응 검정(NW + fixed-b)해서 판단합니다.
  - '검정만 구 방식' 행: 대역폭 h + t분포(개정 전 기본 검정)로 p값만 다시 계산.

[측정하지 못하는 결함 - 논문에 명시할 것]
  ⑧ 생존 편향·지수 편입일 룩어헤드 : experiment_survivorship.py에서 별도 측정.

실행: py ablation.py            (전체 유니버스, 오래 걸림)
      py ablation.py --quick    (종목 60개로 축소, 빠른 점검용)
"""
import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time

import numpy as np
import pandas as pd

from stats_utils import paired_diff_test

import core

# 결함을 하나씩만 켜는 설정들. 첫 줄이 '전부 올바른' 기준선입니다.
CONFIGS = [
    # (이름, universe, run_value_model 추가 인자)
    ("기준선 (⑧ 제외 전부 교정)",       "large", {}),
    ("검정만 구 방식 (NW lag=h, t분포)", "large", {"nw_lag_mult": 1, "fixed_b": False}),
    ("① p값만 iid 가정",                "large", {"pvalue_method": "iid"}),
    ("② 소표본 유니버스",               "small", {}),
    ("⑤ 평가일 솎아내기(구 방식)",       "large", {"eval_mode": "nonoverlap_iid"}),
    ("전부 순진하게 (원래 상태)",        "small", {"pvalue_method": "iid", "eval_mode": "nonoverlap_iid"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="종목 수를 줄여 빠르게 점검")
    args = ap.parse_args()

    small = sorted(set(core.ALL_TICKERS))
    large = sorted(set(core.VALUE_UNIVERSE_TICKERS) | set(core.ALL_TICKERS))
    if args.quick:
        # 소표본 대비 효과를 보려면 두 유니버스 크기 차이는 유지해야 함
        small = small[:20]
        large = large[:60]
        print(f"[quick] 소표본 {len(small)}종목 / 확장 {len(large)}종목으로 축소 실행\n")

    print(f"가격·매크로 데이터 로드 ({core.SWING_START} ~ {core.SWING_END}, 스냅샷)...")
    t0 = time.time()
    df_all, macro_prepared = core.load_swing_data(large)
    if df_all.empty:
        print("가격 데이터를 받지 못했습니다. 잠시 후 다시 시도하세요.")
        sys.exit(1)
    print(f"  완료 ({time.time() - t0:.0f}초)\n")

    panels = {}
    rows, base = [], None
    for name, uni, kw in CONFIGS:
        tickers = small if uni == "small" else large
        t0 = time.time()
        if uni not in panels:
            print(f"[{uni}] 종목 {len(tickers)}개 · 패널 생성 중...", flush=True)
            panels[uni] = core.build_swing_panel(tickers, df_all, macro_prepared)
        panel = panels[uni]
        n_tk = panel["Ticker"].nunique() if not panel.empty else 0
        res = (core.run_value_model(panel, horizon_override=core.SWING_HORIZON,
                                    require_fundamentals=False, **kw)
               if not panel.empty else {"error": "패널 비어있음"})
        if res is None or "error" in res:
            print(f"[{name}] -> {res.get('error') if res else '실패'}\n")
            rows.append({"설정": name, "종목수": n_tk, "판정": "실패"})
            continue
        if base is None:
            base = res
        # 기준선과의 대응 차이 검정 (공통 평가일)
        if res is base:
            d_auc = dt = dp = np.nan
        else:
            d_auc, dt, _, dp, _ = paired_diff_test(res["per_date_auc"], base["per_date_auc"],
                                                   nw_lag=base["nw_lag"], fixed_b=True)
        p2 = res["pvalue_two_sided"]
        verdict = "유의" if (np.isfinite(p2) and p2 <= 0.05) else "유의하지 않음"
        rows.append({"설정": name, "종목수": n_tk, "AUC": res["fm_auc"], "t값": res["fm_tstat"],
                     "p단측": res["pvalue"], "p양측": p2, "독립날짜": res["fm_n_dates"],
                     "ΔAUC(대 기준선)": d_auc, "차이 t": dt, "차이 p양측": dp, "판정": verdict})
        print(f"[{name}] {n_tk}종목 · {core.fmt_result(res)} · {verdict}"
              + (f"\n     기준선 대비 ΔAUC {d_auc:+.4f} (대응 t {dt:+.2f}, 양측 p {dp:.3f})"
                 if np.isfinite(d_auc) else "")
              + f"  ({time.time() - t0:.0f}초)\n", flush=True)

    print("주: ①은 AUC·날짜별 시계열이 기준선과 같고 p값만 다르므로 ΔAUC가 0입니다.")
    print("주: 생존 편향(현재 지수 구성종목으로 과거 학습)은 이 표로 측정 불가 — experiment_survivorship.py 참조.")
    out = pd.DataFrame(rows)
    out.to_csv("ablation_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: ablation_results.csv")


if __name__ == "__main__":
    main()
