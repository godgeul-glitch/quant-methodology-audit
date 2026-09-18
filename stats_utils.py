"""횡단면 예측력 검정 유틸.

app.py는 Streamlit 스크립트라 임포트만 해도 전체가 실행돼서 테스트가 불가능합니다.
논문의 핵심 통계량인 Fama-MacBeth 검정만 여기로 분리해 검증 가능하게 만듭니다.
이 파일을 직접 실행하면(`py stats_utils.py`) 자체 점검이 돌아갑니다.
"""
import numpy as np
from scipy.stats import norm, rankdata, t as t_dist
from sklearn.metrics import roc_auc_score


def fast_auc(y_true, scores):
    """AUC를 순위합(Mann-Whitney U)으로 직접 계산한다.

    왜 sklearn 대신 직접 구현하는가:
      본 연구는 '날짜별로 AUC를 따로 구한 뒤 시계열을 검정'하므로 AUC를
      수백~수십만 번 호출한다. sklearn의 roc_auc_score는 호출당 입력 검증
      오버헤드가 커서(120행 기준 약 880µs) 계산 자체보다 오버헤드가 지배적이다.
      AUC는 정의상 '양성 표본의 점수 순위합'으로 바로 구할 수 있다.

      AUC = (양성의 순위합 − n_pos(n_pos+1)/2) / (n_pos × n_neg)

    동점은 평균 순위로 처리하며(rankdata 기본값), 이는 roc_auc_score와 동일한
    처리다. 동점 처리는 중요하다 — 예컨대 정의 불가한 PEG를 동일한 최하값으로
    채우면 대량의 동점이 발생한다.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    pos = y > 0
    n_pos = int(pos.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def hanley_mcneil_pvalue(auc, n_pos, n_neg):
    """⚠️ 관측치가 서로 독립일 때만 유효한 기존 방식.

    횡단면 패널(같은 날짜의 여러 종목)에 그대로 쓰면 표준오차가 크게
    과소평가됩니다. 비교/설명 목적으로만 남겨둡니다.
    """
    if not np.isfinite(auc) or n_pos < 5 or n_neg < 5:
        return 1.0
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc ** 2 / (1.0 + auc)
    se = np.sqrt((auc * (1.0 - auc)
                  + (n_pos - 1.0) * (q1 - auc ** 2)
                  + (n_neg - 1.0) * (q2 - auc ** 2)) / (n_pos * n_neg))
    if se <= 0:
        return 1.0
    return float(1.0 - norm.cdf((auc - 0.5) / se))


def newey_west_se(x, lag):
    """겹치는 관측치로 생긴 자기상관을 보정한 평균의 표준오차 (Newey-West 1987).

    왜 필요한가:
      예측 기간이 h일이면, 오늘의 '미래 h일 수익률'과 내일의 그것은 h-1일치가
      겹칩니다. 그래서 날짜별 통계량이 서로 강하게 자기상관되고, 단순
      std/sqrt(T)는 표준오차를 과소평가합니다. 겹치는 관측치를 버리면(thinning)
      이 문제는 사라지지만 데이터의 1/h만 쓰게 되어 검정력이 무너집니다.
      Newey-West는 '전부 쓰되 분산을 보정'하는 표준 해법입니다.

    Bartlett 커널: w_j = 1 - j/(lag+1)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 2:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / T                      # gamma_0
    for j in range(1, min(int(lag), T - 1) + 1):
        w = 1.0 - j / (float(lag) + 1.0)
        gj = float(e[j:] @ e[:-j]) / T        # gamma_j
        s += 2.0 * w * gj
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    return float(np.sqrt(s / T))


def fama_macbeth_auc(y_true, scores, dates, min_rows_per_date=5, nw_lag=0):
    """날짜별 AUC를 구한 뒤 그 시계열을 t검정합니다.

    왜 이렇게 해야 하는가:
      같은 날짜의 종목들은 (1) 같은 시장 환경을 공유하고, (2) 라벨이
      '그날 중앙값 초과'라 정확히 절반이 1로 강제되어 서로 독립이 아닙니다.
      행 수를 표본 수로 쓰면 유의성이 크게 부풀려집니다. 날짜 안의 종속성을
      '날짜별 통계량 1개'로 압축하면 이 문제가 사라집니다.

    nw_lag:
      0이면 날짜별 통계량이 서로 독립이라고 보고 std/sqrt(T)를 씁니다
      (평가일을 예측기간 간격으로 솎아낸 경우에만 타당).
      0보다 크면 Newey-West로 자기상관을 보정합니다. 평가일을 솎아내지 않고
      전부 쓰는 경우 반드시 필요하며, 보통 lag = 예측기간(겹침 길이)으로 둡니다.

    Returns:
        (mean_auc, tstat, pvalue, n_dates)
        검정이 불가능하면 tstat/pvalue는 nan.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    d = np.asarray(dates)

    mask = ~np.isnan(s)
    y, s, d = y[mask], s[mask], d[mask]

    # ⭐ [성능] 날짜별로 나눌 때 매번 `d == dt` 마스크를 만들면 전체 행을 날짜
    # 수만큼 반복 훑게 되어 O(행수 × 날짜수)가 됩니다. 평가일 300개 × 15만 행
    # 규모에서 이것만으로 수천만 번의 비교가 발생합니다.
    # 한 번 정렬해 같은 날짜를 인접시킨 뒤 구간 슬라이스로 접근하면
    # O(행수 log 행수)로 끝납니다. 결과는 동일합니다.
    order = np.argsort(d, kind="mergesort")   # 안정 정렬: 같은 날짜 내 순서 보존
    d_sorted, y_sorted, s_sorted = d[order], y[order], s[order]
    # 날짜가 바뀌는 경계 위치
    if len(d_sorted) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    bounds = np.flatnonzero(d_sorted[1:] != d_sorted[:-1]) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(d_sorted)]))

    per_date = []
    for a, b in zip(starts, ends):            # 날짜 오름차순. NW는 순서가 중요.
        y_d = y_sorted[a:b]
        if (b - a) < min_rows_per_date:
            continue
        # 라벨이 한 종류뿐이면 AUC가 정의되지 않음 (min/max 비교가 unique보다 빠름)
        if y_d.min() == y_d.max():
            continue
        per_date.append(fast_auc(y_d, s_sorted[a:b]))

    T = len(per_date)
    if T == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean_auc = float(np.mean(per_date))
    if T < 2:
        return mean_auc, float("nan"), float("nan"), T

    if nw_lag and nw_lag > 0:
        se = newey_west_se(per_date, nw_lag)
    else:
        sd = float(np.std(per_date, ddof=1))
        se = sd / np.sqrt(T) if sd > 0 else float("nan")
    if not np.isfinite(se) or se <= 0:
        return mean_auc, float("nan"), float("nan"), T

    tstat = (mean_auc - 0.5) / se
    # 자유도는 관례대로 T-1을 씁니다. 겹침이 있으면 '유효 표본'은 T보다 작으므로
    # 이 검정도 여전히 다소 낙관적일 수 있다는 점은 논문에 명시할 것.
    pval = float(1.0 - t_dist.cdf(tstat, df=T - 1))
    return mean_auc, float(tstat), pval, T


def _make_panel(n_dates, n_stocks, signal_strength, seed, date_effect_sd=0.0):
    """날짜별로 '중앙값 초과' 라벨을 만드는 합성 패널.

    signal_strength : 모든 날짜에 공통으로 존재하는 '진짜' 예측력.
    date_effect_sd  : 날짜마다 신호 강도가 들쭉날쭉한 정도. 실제 시장이
                      이렇습니다 — 어떤 분기엔 밸류가 먹히고 어떤 분기엔
                      정반대로 움직입니다. 이 값이 크면, 평균적으로는 예측력이
                      0이어도 운 좋게 몇 날짜에서 잘 맞아 전체를 모아 구한
                      AUC가 0.5에서 크게 벗어날 수 있습니다.
    """
    rng = np.random.RandomState(seed)
    y, s, d = [], [], []
    for i in range(n_dates):
        score = rng.randn(n_stocks)
        # 그 날짜에만 해당하는 신호 강도 (평균은 signal_strength)
        beta_t = signal_strength + rng.randn() * date_effect_sd
        fwd = beta_t * score + rng.randn(n_stocks)
        target = (fwd > np.median(fwd)).astype(int)
        y.extend(target)
        s.extend(score)
        d.extend([i] * n_stocks)
    return np.array(y), np.array(s), np.array(d)


def _make_overlapping_panel(n_days, n_stocks, horizon, seed):
    """평가일을 매일 두고 h일 뒤 수익률을 라벨로 쓰는, 겹치는 구간 패널.

    실제 백테스트와 같은 구조입니다: t일의 라벨(t~t+h 수익률)과 t+1일의
    라벨(t+1~t+1+h)은 h-1일치가 겹칩니다. 진짜 예측력은 0으로 둡니다.
    """
    rng = np.random.RandomState(seed)
    # 종목별 일별 수익률 -> 누적가격
    daily = rng.randn(n_days + horizon, n_stocks)
    cum = np.vstack([np.zeros(n_stocks), np.cumsum(daily, axis=0)])
    # 점수는 종목 고유의 고정 특성(재무지표처럼 천천히 변함) + 약간의 노이즈
    base = rng.randn(n_stocks)
    y, s, d = [], [], []
    for t in range(n_days):
        fwd = cum[t + horizon] - cum[t]          # t~t+h 누적수익 (인접 t끼리 겹침)
        score = base + 0.1 * rng.randn(n_stocks)  # 진짜 예측력 없음
        target = (fwd > np.median(fwd)).astype(int)
        y.extend(target)
        s.extend(score)
        d.extend([t] * n_stocks)
    return np.array(y), np.array(s), np.array(d)


def _demo():
    # 1) [핵심] 평균적으로는 예측력이 0인데, 날짜가 3개뿐이라 그중 운 좋게
    #    잘 맞은 날짜가 섞이는 경우 = 이 앱이 실제로 처했던 상황.
    #    기존 방식은 행 수(900개)를 표본으로 세서 '유의함'이라고 착각하고,
    #    Fama-MacBeth는 독립 날짜가 3개뿐임을 알기에 속지 않아야 합니다.
    false_positives_old = 0
    false_positives_new = 0
    trials = 40
    for seed in range(trials):
        y, s, d = _make_panel(n_dates=3, n_stocks=300, signal_strength=0.0,
                              seed=seed, date_effect_sd=0.35)
        pooled_auc = roc_auc_score(y, s)
        n_pos = int(y.sum())
        p_old = hanley_mcneil_pvalue(pooled_auc, n_pos, len(y) - n_pos)
        _, _, p_new, T = fama_macbeth_auc(y, s, d)
        assert T == 3, f"독립 날짜 수가 3이어야 하는데 {T}"
        if p_old <= 0.05:
            false_positives_old += 1
        if np.isfinite(p_new) and p_new <= 0.05:
            false_positives_new += 1

    # 진짜 예측력은 0이므로 5% 유의수준에서 오탐은 5% 근처여야 정상.
    print(f"[신호 없음·날짜3개] 기존 방식 오탐 {false_positives_old}/{trials}, "
          f"Fama-MacBeth 오탐 {false_positives_new}/{trials}")
    assert false_positives_old > false_positives_new, (
        "기존 방식이 Fama-MacBeth보다 오탐이 많아야 하는데 그렇지 않음 "
        f"(기존 {false_positives_old}, 신규 {false_positives_new})")
    assert false_positives_new <= trials * 0.2, (
        f"Fama-MacBeth 오탐이 너무 많음: {false_positives_new}/{trials}")

    # 2) 진짜 신호가 있고 날짜도 충분하면 잡아내야 함.
    y, s, d = _make_panel(n_dates=40, n_stocks=200, signal_strength=0.5, seed=0)
    mean_auc, tstat, pval, T = fama_macbeth_auc(y, s, d)
    assert T == 40, f"독립 날짜 수가 40이어야 하는데 {T}"
    assert mean_auc > 0.6, f"신호가 있는데 AUC가 낮음: {mean_auc}"
    assert pval < 0.01, f"진짜 신호를 못 잡음: p={pval}"
    print(f"[신호 있음] 평균 AUC {mean_auc:.3f}, t={tstat:.2f}, p={pval:.5f}, 날짜 {T}개")

    # 3) 날짜가 1개뿐이면 시계열 분산 추정이 불가능하므로 검정 불가여야 함.
    y, s, d = _make_panel(n_dates=1, n_stocks=200, signal_strength=0.5, seed=1)
    mean_auc, tstat, pval, T = fama_macbeth_auc(y, s, d)
    assert T == 1 and np.isnan(tstat) and np.isnan(pval), "날짜 1개는 검정 불가여야 함"
    print(f"[날짜 1개] 검정 불가 처리 확인 (AUC 점추정 {mean_auc:.3f})")

    # 4) 결측 점수는 그 행만 빠지고 나머지는 정상 계산되어야 함.
    y, s, d = _make_panel(n_dates=10, n_stocks=100, signal_strength=0.5, seed=2)
    s_with_nan = s.copy()
    s_with_nan[::7] = np.nan
    _, _, p_nan, T_nan = fama_macbeth_auc(y, s_with_nan, d)
    assert T_nan == 10 and np.isfinite(p_nan), "결측이 섞여도 검정은 되어야 함"
    print(f"[결측 포함] 날짜 {T_nan}개로 정상 검정, p={p_nan:.5f}")

    # 5) [핵심] 겹치는 예측 구간(overlapping windows)에서 Newey-West가 필요한 이유.
    #    평가일을 매일 두고 h일 뒤 수익률을 라벨로 쓰면, 인접한 날짜의 통계량이
    #    h-1일치를 공유해 강하게 자기상관됩니다. 이때 std/sqrt(T)는 표준오차를
    #    과소평가해 오탐이 폭증하고, NW 보정은 이를 완화해야 합니다.
    H = 20
    fp_iid = fp_nw = 0
    trials = 40
    for seed in range(trials):
        y, s, d = _make_overlapping_panel(n_days=260, n_stocks=120, horizon=H, seed=seed)
        _, _, p_iid, T = fama_macbeth_auc(y, s, d, nw_lag=0)
        _, _, p_nw, _ = fama_macbeth_auc(y, s, d, nw_lag=H)
        if np.isfinite(p_iid) and p_iid <= 0.05:
            fp_iid += 1
        if np.isfinite(p_nw) and p_nw <= 0.05:
            fp_nw += 1
    print(f"[겹치는 구간·신호없음] 날짜 T={T} · "
          f"iid 오탐 {fp_iid}/{trials}, Newey-West 오탐 {fp_nw}/{trials}")
    assert fp_nw <= fp_iid, (
        f"NW 보정이 iid보다 오탐이 많으면 안 됨 (iid {fp_iid}, NW {fp_nw})")

    print("\n자체 점검 통과")


if __name__ == "__main__":
    _demo()
