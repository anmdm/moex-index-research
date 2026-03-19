"""
Pipeline v5: Find best index + optimize strategy.
Phase 1: Index search (max vol, fast reversion, zero drift) — on 10s bars
Phase 2: Strategy optimization — on 5s bars
  Constraints: tp_pct >= 0.3%, avg_hold >= 60s
"""
import sys, io, os, time, json, warnings
import numpy as np
import pandas as pd
import optuna
from numba import njit

warnings.filterwarnings('ignore')

PARQUET_DIR = 'moex_cache/ticks_parquet'
CAPITAL = 10_000.0
TOTAL_COST = 0.00015 + 0.0001
EXEC_DELAY_MS = 20
N_TRIALS_INDEX = 2000
N_TRIALS_STRATEGY = 1000

ASSET_NAMES = ['BR', 'NG', 'GD', 'SV', 'MX']
SMA_PERIODS = [60, 120, 240, 480]

from moex_full_pipeline import (
    build_relative_index, compute_metrics, basket_bt, mark_sessions
)


@njit(cache=True)
def compute_oscillation_ratio(idx, sma_period):
    """Compute oscillation vs trend at multiple scales.
    Returns: osc_vol, trend_vol, ratio, yearly_cv, macro_drift
    macro_drift = max absolute return of any yearly third (catches hidden trends)
    """
    n = len(idx)
    if n < sma_period + 100:
        return 0.0, 0.0, 0.0, 999.0, 999.0

    # Compute SMA
    sma = np.empty(n, dtype=np.float64)
    cumsum = 0.0
    for i in range(n):
        cumsum += idx[i]
        if i >= sma_period:
            cumsum -= idx[i - sma_period]
            sma[i] = cumsum / sma_period
        elif i >= sma_period - 1:
            sma[i] = cumsum / sma_period
        else:
            sma[i] = cumsum / (i + 1)

    start = sma_period

    # Oscillation: variance of (idx - sma) / sma
    osc_sum = 0.0; osc_sq = 0.0; osc_n = 0
    for i in range(start, n):
        if sma[i] > 1e-10:
            dev = (idx[i] - sma[i]) / sma[i]
            osc_sum += dev; osc_sq += dev * dev; osc_n += 1
    osc_mean = osc_sum / osc_n if osc_n > 0 else 0.0
    osc_var = osc_sq / osc_n - osc_mean * osc_mean if osc_n > 1 else 0.0
    osc_vol = np.sqrt(max(0, osc_var))

    # Trend: variance of SMA returns (bar-to-bar)
    tr_sum = 0.0; tr_sq = 0.0; tr_n = 0
    for i in range(start + 1, n):
        if sma[i-1] > 1e-10:
            ret = sma[i] / sma[i-1] - 1.0
            tr_sum += ret; tr_sq += ret * ret; tr_n += 1
    tr_mean = tr_sum / tr_n if tr_n > 0 else 0.0
    tr_var = tr_sq / tr_n - tr_mean * tr_mean if tr_n > 1 else 0.0
    trend_vol = np.sqrt(max(0, tr_var))

    ratio = osc_vol / trend_vol if trend_vol > 1e-15 else 999.0

    # Yearly uniformity + macro drift detection
    # Split into 6 equal chunks (~6 months each for 3 years)
    n_chunks = 6
    chunk_size = (n - start) // n_chunks
    if chunk_size > 100:
        chunk_means = np.empty(n_chunks, dtype=np.float64)
        chunk_rets = np.empty(n_chunks, dtype=np.float64)
        for c in range(n_chunks):
            cs = start + c * chunk_size
            ce = cs + chunk_size
            s = 0.0
            for i in range(cs, ce):
                s += idx[i]
            chunk_means[c] = s / chunk_size
            # Return of chunk: last vs first
            chunk_rets[c] = idx[ce - 1] / idx[cs] - 1.0 if idx[cs] > 1e-10 else 0.0

        cm_mean = 0.0
        for c in range(n_chunks):
            cm_mean += chunk_means[c]
        cm_mean /= n_chunks

        cm_var = 0.0
        for c in range(n_chunks):
            cm_var += (chunk_means[c] - cm_mean) ** 2
        cm_var /= n_chunks
        yearly_cv = np.sqrt(cm_var) / cm_mean if cm_mean > 1e-10 else 999.0

        # Macro drift: max absolute return of any chunk
        # If any 6-month period has >20% return, the index is trending
        macro_drift = 0.0
        for c in range(n_chunks):
            if abs(chunk_rets[c]) > macro_drift:
                macro_drift = abs(chunk_rets[c])
    else:
        yearly_cv = 999.0
        macro_drift = 999.0

    return osc_vol, trend_vol, ratio, yearly_cv, macro_drift


def score_index(hl, vol, mean_dev, max_down, max_up, trend,
                left_p5=0.0, left_p1=0.0, down_std=0.0, max_dd_idx=0.0,
                osc_vol=0.0, trend_vol=0.0, osc_ratio=0.0, yearly_cv=0.0,
                macro_drift=0.0):
    """Score for mean-reversion grid:
    - HIGH oscillation vol (moves around SMA)
    - HIGH osc/trend ratio (vol from oscillation, not trend)
    - FAST mean-reversion (low half-life, but tradeable >100s)
    - ZERO drift at ALL scales (micro + macro)
    - STABLE index level across years
    """
    hl = min(hl, 1000.0)
    if hl <= 0 or hl > 500:
        return 0.0
    if hl < 10:
        return 0.0

    # 1. Oscillation vol — what grid trades
    osc_score = osc_vol * 10000
    if osc_score < 0.1:
        return 0.0

    # 2. Osc/trend ratio
    if osc_ratio < 0.5:
        return 0.0
    ratio_score = min(osc_ratio, 20.0)

    # 3. Fast reversion
    hl_score = 100.0 / hl

    # 4. Micro drift penalty (3-year trend from compute_metrics)
    abs_trend = abs(trend)
    if abs_trend < 0.02:
        drift_score = 1.0
    elif abs_trend < 0.05:
        drift_score = 0.8
    elif abs_trend < 0.10:
        drift_score = 0.5
    elif abs_trend < 0.20:
        drift_score = 0.3
    else:
        drift_score = 0.1
    if trend > 0:
        drift_score *= 1.1

    # 5. MACRO drift penalty — catches hidden trends invisible on micro scale
    # macro_drift = max |return| of any 6-month chunk
    # If ANY half-year moves >15%, this is a trending index
    if macro_drift < 0.05:
        macro_score = 1.5   # excellent: no half-year moves >5%
    elif macro_drift < 0.10:
        macro_score = 1.0   # ok
    elif macro_drift < 0.15:
        macro_score = 0.5   # mediocre
    elif macro_drift < 0.25:
        macro_score = 0.15  # bad: some half-year has 15-25% move
    elif macro_drift < 0.50:
        macro_score = 0.03  # terrible: 25-50% move in half a year
    else:
        macro_score = 0.005 # catastrophic: >50% move

    # 6. Yearly stability of index level
    if yearly_cv < 0.03:
        stability = 1.5
    elif yearly_cv < 0.05:
        stability = 1.2
    elif yearly_cv < 0.10:
        stability = 1.0
    elif yearly_cv < 0.20:
        stability = 0.5
    else:
        stability = 0.2

    return osc_score * ratio_score * hl_score * drift_score * macro_score * stability


def load_resampled(instruments, resample_sec):
    """Memory-safe: resample each instrument first, then join."""
    dfs = {}
    for name in instruments:
        path = os.path.join(PARQUET_DIR, f'{name}_ticks.parquet')
        df = pd.read_parquet(path)
        print(f'  {name}: {len(df):,} ticks')
        resampled = df['price'].resample(f'{resample_sec}s').last()
        dfs[name] = resampled
        del df
    combined = pd.DataFrame(dfs).sort_index().ffill().dropna()
    print(f'  Aligned {resample_sec}s: {len(combined):,} bars')
    return combined


def run_phase1(data_10s):
    """Phase 1: Find best index."""
    prices = data_10s.values.astype(np.float64)
    n, m = prices.shape

    _ = build_relative_index(prices[:500], np.ones(m) / m, 60)
    _ = compute_metrics(build_relative_index(prices[:500], np.ones(m) / m, 60), 60)
    print("  Numba warmed up")

    all_results = []
    best_score = [0.0]

    def objective(trial):
        raw_weights = []
        for j in range(m):
            w = trial.suggest_float(f"w_{ASSET_NAMES[j]}", 0.0, 1.0, step=0.05)
            raw_weights.append(w)

        total = sum(raw_weights)
        if total < 0.01:
            return -999.0
        n_active = sum(1 for w in raw_weights if w >= 0.025)
        if n_active < 2:
            return -999.0

        weights = np.array(raw_weights, dtype=np.float64) / total

        # Max weight per asset <= 35%
        if max(weights) > 0.36:
            return -999.0

        sma_p = trial.suggest_categorical("sma", SMA_PERIODS)

        idx = build_relative_index(prices, weights, sma_p)
        hl, vol, md, maxd, maxu, trend, lp5, lp1, dstd, mdd = compute_metrics(idx, sma_p)
        osc_vol, trend_vol, osc_ratio, yearly_cv, macro_drift = compute_oscillation_ratio(idx, sma_p)
        sc = score_index(hl, vol, md, maxd, maxu, trend, lp5, lp1, dstd, mdd,
                         osc_vol, trend_vol, osc_ratio, yearly_cv, macro_drift)

        if sc > 0:
            all_results.append({
                "weights": weights.copy(), "sma": sma_p,
                "hl": hl, "vol": vol, "md": md, "trend": trend,
                "osc_vol": osc_vol, "osc_ratio": osc_ratio, "yearly_cv": yearly_cv,
                "macro_drift": macro_drift, "score": sc,
            })

        if sc > best_score[0]:
            best_score[0] = sc
            ws = " + ".join(f"{weights[j]:.2f}*{ASSET_NAMES[j]}" for j in range(m) if weights[j] > 0.01)
            print(f"  #{trial.number}: score={sc:.4f} SMA={sma_p} HL={hl:.0f} "
                  f"osc={osc_vol*10000:.1f}bps osc/tr={osc_ratio:.1f} "
                  f"macro={macro_drift*100:.1f}% yrCV={yearly_cv:.3f} trend={trend*100:+.1f}%")
            print(f"    {ws}")
        return sc

    print(f"\n  Optuna: {N_TRIALS_INDEX} trials")
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=300))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS_INDEX, show_progress_bar=True)

    all_results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  TOP 5:")
    for i, r in enumerate(all_results[:5]):
        ws = " + ".join(f"{r['weights'][j]:.2f}*{ASSET_NAMES[j]}" for j in range(len(ASSET_NAMES)) if r['weights'][j] > 0.01)
        print(f"  {i+1}. score={r['score']:.4f} SMA={r['sma']} HL={r['hl']:.0f} "
              f"osc={r['osc_vol']*10000:.1f}bps osc/tr={r['osc_ratio']:.1f} "
              f"macro={r['macro_drift']*100:.1f}% yrCV={r['yearly_cv']:.3f} trend={r['trend']*100:+.1f}%  {ws}")

    return all_results[:3]


def run_phase2(data_5s, top_indices):
    """Phase 2: Optimize strategy. Constraints: tp >= 0.3%, hold >= 60s, uniform yearly P&L."""
    prices = data_5s[ASSET_NAMES].values.astype(np.float64)
    ts_ms = (data_5s.index.astype(np.int64) // 10**6).values
    is_session = mark_sessions(data_5s.index)
    n = len(prices)
    print(f'\n  Bars: {n:,}, Session: {is_session.sum():,} ({is_session.mean()*100:.0f}%)')

    # Pre-compute year boundaries (for yearly P&L uniformity)
    ts_index = data_5s.index
    years = ts_index.year
    unique_years = sorted(years.unique())
    year_boundaries = []  # list of (start_idx, end_idx) per year
    for y in unique_years:
        mask = years == y
        indices = np.where(mask)[0]
        if len(indices) > 0:
            year_boundaries.append((indices[0], indices[-1]))
    print(f'  Years: {unique_years}, {len(year_boundaries)} segments')

    best_overall = None

    for idx_rank, idx_info in enumerate(top_indices):
        weights = idx_info['weights'].astype(np.float64)
        sma = idx_info['sma']
        sma_ticks = sma * 2

        ws = " + ".join(f"{weights[j]:.2f}*{ASSET_NAMES[j]}" for j in range(len(ASSET_NAMES)) if weights[j] > 0.01)
        print(f'\n  === Index #{idx_rank+1}: {ws} (SMA={sma}×10s = {sma_ticks}×5s) ===')

        idx_raw = build_relative_index(prices, weights, sma_ticks)
        idx_log = np.log(np.maximum(idx_raw, 1e-10)).astype(np.float64)

        _ = basket_bt(prices[:2000], weights, idx_log[:2000],
                      ts_ms[:2000], is_session[:2000],
                      5, 2.0, 1.0, 0.01, 3, 600, CAPITAL, TOTAL_COST, EXEC_DELAY_MS)

        best_score = [-999.0]

        def objective(trial, w=weights, il=idx_log):
            p = {
                "n_parts": trial.suggest_int("n_parts", 3, 20),
                "entry_z": trial.suggest_float("entry_z", 0.5, 4.0),
                "grid_step_z": trial.suggest_float("grid_step_z", 0.1, 3.0),
                "tp_pct": trial.suggest_float("tp_pct", 0.003, 0.10),  # MIN 0.3%
                "max_positions": trial.suggest_int("max_positions", 1, 10),
                "lookback_sec": trial.suggest_int("lookback_sec", 300, 36000),
            }
            p["max_positions"] = min(p["max_positions"], p["n_parts"])

            feq, mdd, nt, wt, eq, hms, nopen = basket_bt(
                prices, w, il, ts_ms, is_session,
                p["n_parts"], p["entry_z"], p["grid_step_z"],
                p["tp_pct"], p["max_positions"], p["lookback_sec"],
                CAPITAL, TOTAL_COST, EXEC_DELAY_MS)

            total_ret = feq / CAPITAL - 1
            if total_ret <= 0 or nt < 5:
                return -999.0

            # Constraint: avg hold >= 60 seconds
            avg_hold_s = (hms / nt / 1000) if nt > 0 else 0
            if avg_hold_s < 60:
                return -999.0

            # Yearly P&L uniformity
            yearly_rets = []
            for y_start, y_end in year_boundaries:
                eq_start = eq[y_start]
                eq_end = eq[y_end]
                if eq_start > 0:
                    yearly_rets.append(eq_end / eq_start - 1)

            # All years must be profitable
            if len(yearly_rets) >= 2:
                if any(r <= 0 for r in yearly_rets):
                    return -999.0  # reject if any year is negative

                # Uniformity: CV (coefficient of variation) = std/mean of yearly returns
                yr_mean = sum(yearly_rets) / len(yearly_rets)
                yr_var = sum((r - yr_mean)**2 for r in yearly_rets) / len(yearly_rets)
                yr_cv = (yr_var ** 0.5) / yr_mean if yr_mean > 0 else 999
                # CV=0 → perfectly uniform, CV=1 → std=mean (very uneven)
                # uniformity_score: CV<0.3 → 1.0, CV=0.5 → 0.7, CV=1.0 → 0.4, CV>2 → ~0
                uniformity = 1.0 / (1.0 + yr_cv * yr_cv)
            else:
                uniformity = 1.0

            wr = wt / nt if nt > 0 else 0
            dd = abs(mdd)
            dd_penalty = max(0, dd - 0.10) * 5
            reliability = min(1.0, nt / 30)
            trade_bonus = np.log(1 + nt) / np.log(1 + 50)

            score = total_ret * 100 * (0.5 + wr) * reliability * (1 - dd_penalty) * trade_bonus * uniformity

            if score > best_score[0]:
                best_score[0] = score
                yr_str = " ".join(f"{r*100:+.0f}%" for r in yearly_rets)
                print(f"    #{trial.number}: ret={total_ret*100:+.1f}% DD={mdd*100:.2f}% "
                      f"trades={nt}({wt}w) hold={avg_hold_s:.0f}s "
                      f"tp={p['tp_pct']*100:.2f}% ez={p['entry_z']:.2f} "
                      f"yrs=[{yr_str}] cv={yr_cv:.2f}")
            return score

        study = optuna.create_study(direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=200))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=N_TRIALS_STRATEGY, show_progress_bar=True)

        if study.best_value > -999:
            bp = study.best_params
            bp["max_positions"] = min(bp["max_positions"], bp["n_parts"])

            feq, mdd, nt, wt, equity, hms, nopen = basket_bt(
                prices, weights, idx_log, ts_ms, is_session,
                bp["n_parts"], bp["entry_z"], bp["grid_step_z"],
                bp["tp_pct"], bp["max_positions"], bp["lookback_sec"],
                CAPITAL, TOTAL_COST, EXEC_DELAY_MS)

            ret_pct = (feq / CAPITAL - 1) * 100
            avg_hold_s = (hms / nt / 1000) if nt > 0 else 0
            pnl_dd = abs(ret_pct / (mdd * 100)) if mdd != 0 else 0

            print(f'\n    RESULT: P&L={ret_pct:+.2f}% DD={mdd*100:.2f}% P&L/DD={pnl_dd:.1f} '
                  f'trades={nt}({wt}w) hold={avg_hold_s:.0f}s')

            result = {
                "index_rank": idx_rank + 1,
                "weights": {ASSET_NAMES[j]: round(float(weights[j]), 4) for j in range(len(ASSET_NAMES)) if weights[j] > 0.01},
                "sma": sma,
                "params": {k: round(v, 6) if isinstance(v, float) else v for k, v in bp.items()},
                "pnl_pct": round(ret_pct, 2),
                "maxdd_pct": round(mdd * 100, 2),
                "pnl_dd": round(pnl_dd, 2),
                "trades": nt, "wins": wt,
                "avg_hold_sec": round(avg_hold_s, 1),
                "optuna_score": round(study.best_value, 4),
            }

            if best_overall is None or study.best_value > best_overall['optuna_score']:
                best_overall = result

    return best_overall


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("  PIPELINE v5: Index + Strategy (tp>=0.3%, hold>=60s)")
    print("=" * 70)

    # Phase 1: 10s bars
    print(f'\nPhase 1: Loading 10s data...')
    data_10s = load_resampled(ASSET_NAMES, 10)
    top_indices = run_phase1(data_10s)
    del data_10s

    # Phase 2: 5s bars
    print(f'\nPhase 2: Loading 5s data...')
    data_5s = load_resampled(ASSET_NAMES, 5)
    best = run_phase2(data_5s, top_indices)

    if best:
        print(f'\n{"="*70}')
        print(f'  BEST OVERALL')
        print(f'{"="*70}')
        print(json.dumps(best, indent=2))

        with open('moex_v5_results.json', 'w') as f:
            json.dump(best, f, indent=2, default=str)
        print(f'\nSaved: moex_v5_results.json')

    print(f'\nTotal time: {(time.time()-t0)/60:.0f} min')
