"""
MOEX Backtest Engine — единый бэктестер для всех вариантов стратегий.

Правильная логика:
- free_cash tracking: позиция не открывается если нет денег
- При закрытии: invested + pnl возвращается в free_cash
- Комиссии вычитаются из pnl
- Execution delay (20ms)
- Session-only trading (MOEX Main 10:00-18:45, Evening 19:05-23:50 MSK)

Все варианты A-E используют общую базу, отличаются только TP логикой и sizing.
"""
import sys, io, os, time, json, warnings
import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings('ignore')

PARQUET_DIR = 'moex_cache/ticks_parquet'


# ════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════

def load_5s(asset_names_all):
    """Load tick data, resample to 5s, outer join + ffill."""
    dfs = {}
    for name in asset_names_all:
        path = os.path.join(PARQUET_DIR, f'{name}_ticks.parquet')
        df = pd.read_parquet(path)
        print(f'  {name}: {len(df):,} ticks')
        dfs[name] = df['price'].resample('5s').last()
        del df
    combined = pd.DataFrame(dfs).sort_index().ffill().dropna()
    print(f'  Aligned 5s: {len(combined):,} bars')
    return combined


def mark_sessions(ts_index):
    """MOEX sessions: Main 10:00-18:45, Evening 19:05-23:50 MSK (UTC+3)."""
    hour = ts_index.hour
    minute = ts_index.minute
    hm = (hour + 3) * 60 + minute
    hm = hm % 1440
    is_main = (hm >= 600) & (hm < 1125)
    is_evening = (hm >= 1145) & (hm < 1430)
    result = is_main | is_evening
    return result.values.astype(np.bool_) if hasattr(result, 'values') else result.astype(np.bool_)


# ════════════════════════════════════════════════════════════════════
#  INDEX & Z-SCORE (shared across all variants)
# ════════════════════════════════════════════════════════════════════

@njit(cache=True)
def build_index_and_z(close_matrix, weights, sma_period, lookback_bars):
    """Build relative index and Z-score.
    close_matrix: (n_bars, n_active_assets)
    Returns: z_scores, idx_log
    """
    n = close_matrix.shape[0]
    m = close_matrix.shape[1]

    # Relative index: Σ(w_i × price_i / SMA_i)
    idx_log = np.empty(n, dtype=np.float64)
    cumsum = np.zeros(m, dtype=np.float64)
    for i in range(n):
        val = 0.0
        for j in range(m):
            cumsum[j] += close_matrix[i, j]
            if i >= sma_period:
                cumsum[j] -= close_matrix[i - sma_period, j]
                sma_val = cumsum[j] / sma_period
            elif i >= sma_period - 1:
                sma_val = cumsum[j] / sma_period
            else:
                sma_val = cumsum[j] / (i + 1)
            if sma_val > 1e-10:
                val += weights[j] * (close_matrix[i, j] / sma_val)
        idx_log[i] = np.log(max(val, 1e-10))

    # Z-score with bar-based lookback
    z = np.zeros(n, dtype=np.float64)
    lb_start = 0; roll_sum = 0.0; roll_sq = 0.0; roll_cnt = 0
    for i in range(n):
        si = idx_log[i]
        roll_sum += si; roll_sq += si * si; roll_cnt += 1
        while lb_start < i and roll_cnt > lookback_bars:
            roll_sum -= idx_log[lb_start]
            roll_sq -= idx_log[lb_start] ** 2
            roll_cnt -= 1
            lb_start += 1
        if roll_cnt >= 10:
            mu = roll_sum / roll_cnt
            var = (roll_sq / roll_cnt - mu * mu) * roll_cnt / (roll_cnt - 1)
            if var > 1e-30:
                z[i] = (si - mu) / np.sqrt(var)
    return z, idx_log


# ════════════════════════════════════════════════════════════════════
#  CORE BACKTEST — shared position management
# ════════════════════════════════════════════════════════════════════
#
# All variants share:
# - free_cash tracking (no position if no money)
# - Execution delay
# - Session filter
# - Equity curve with unrealized P&L
# - Per-position entry prices for all assets
#
# Variants differ in:
# - TP condition (per-position vs group)
# - TP amount (fixed vs scaled)
# - Position sizing (fixed vs progressive)
# - Grid step (fixed vs vol-adaptive)


@njit(cache=True)
def backtest_variant(close_matrix, weights, z, idx_log, is_session,
                     variant,  # 0=A, 1=B, 2=C, 3=D, 4=E
                     n_parts, entry_z, grid_step_z, tp_pct, max_pos,
                     capital, total_cost, delay_bars,
                     # Variant-specific params:
                     tp_scale,      # B, E: TP = tp_pct * max(1, |Z| * tp_scale)
                     size_scale,    # C: size = base * (1 + size_scale * depth)
                     vol_lb_bars,   # E: rolling vol lookback in bars
                     ):
    """
    Unified backtest for all 5 variants.
    variant: 0=A(Base), 1=B(Dev-TP), 2=C(Progressive), 3=D(Group-TP), 4=E(Vol-adaptive)
    """
    n = close_matrix.shape[0]
    m = close_matrix.shape[1]
    base_size = capital / n_parts

    # ── Rolling vol for variant E ──
    vol_arr = np.zeros(n, dtype=np.float64)
    avg_vol = 1e-10
    if variant == 4:
        vl_start = 0; vl_sum = 0.0; vl_sq = 0.0; vl_cnt = 0
        g_vol_sum = 0.0; g_vol_n = 0
        for i in range(1, n):
            ret = idx_log[i] - idx_log[i - 1]
            vl_sum += ret; vl_sq += ret * ret; vl_cnt += 1
            while vl_start < i - 1 and vl_cnt > vol_lb_bars:
                r_old = idx_log[vl_start + 1] - idx_log[vl_start]
                vl_sum -= r_old; vl_sq -= r_old * r_old; vl_cnt -= 1; vl_start += 1
            if vl_cnt >= 10:
                vm = vl_sum / vl_cnt
                vv = vl_sq / vl_cnt - vm * vm
                if vv > 0:
                    vol_arr[i] = np.sqrt(vv)
                    g_vol_sum += vol_arr[i]; g_vol_n += 1
        avg_vol = g_vol_sum / g_vol_n if g_vol_n > 0 else 1e-10

    # ── State ──
    MAX_G = 30
    g_entry = np.zeros((MAX_G, m), dtype=np.float64)   # entry prices
    g_ez = np.zeros(MAX_G, dtype=np.float64)            # entry Z
    g_size = np.zeros(MAX_G, dtype=np.float64)          # invested $ per group
    g_tp = np.zeros(MAX_G, dtype=np.float64)            # TP threshold per group
    ng = 0                                               # number of open groups

    pending_entry = False
    pending_ez = 0.0
    pending_fill_bar = -1
    pending_close_all_bar = -1  # for D: bar to close all

    free_cash = capital
    realized_pnl = 0.0
    equity = np.empty(n, dtype=np.float64)
    n_trades = 0; win_trades = 0; hold_bars_total = np.int64(0)
    peak = capital; max_dd = 0.0
    pending_vol_ratio = 1.0

    for i in range(n):
        zi = z[i]
        sess = is_session[i]

        # Vol ratio for E
        vol_ratio = 1.0
        if variant == 4:
            vr = vol_arr[i] / avg_vol if avg_vol > 1e-15 and vol_arr[i] > 1e-15 else 1.0
            vol_ratio = max(0.3, min(vr, 3.0))

        # ════ CLOSE LOGIC ════

        if variant == 3:
            # D: Group-TP — close ALL when avg return >= tp
            if pending_close_all_bar >= 0 and i >= pending_close_all_bar and ng > 0:
                for g in range(ng):
                    pnl = 0.0
                    for k in range(m):
                        if g_entry[g, k] > 0:
                            pnl += weights[k] * (close_matrix[i, k] / g_entry[g, k] - 1.0) * g_size[g]
                            # Commission: entry + exit = 2 sides
                            pnl -= 2.0 * total_cost * g_size[g] * weights[k]
                    realized_pnl += pnl
                    free_cash += g_size[g] + pnl
                    n_trades += 1
                    if pnl > 0: win_trades += 1
                ng = 0
                pending_close_all_bar = -1

            # Check group TP signal (net of costs)
            if ng > 0 and sess and pending_close_all_bar < 0:
                total_inv = 0.0; total_pnl = 0.0
                for g in range(ng):
                    total_inv += g_size[g]
                    for k in range(m):
                        if g_entry[g, k] > 0:
                            total_pnl += weights[k] * (close_matrix[i, k] / g_entry[g, k] - 1.0) * g_size[g]
                            # Subtract estimated round-trip cost for TP check
                            total_pnl -= 2.0 * total_cost * g_size[g] * weights[k]
                if total_inv > 0 and total_pnl / total_inv >= tp_pct:
                    pending_close_all_bar = i + delay_bars

        else:
            # A, B, C, E: per-position TP
            g = 0
            while g < ng:
                # Check TP net of costs
                port_ret = 0.0
                for k in range(m):
                    if g_entry[g, k] > 0:
                        port_ret += weights[k] * (close_matrix[i, k] / g_entry[g, k] - 1.0)
                        port_ret -= 2.0 * total_cost * weights[k]  # round-trip cost as % of return
                if port_ret >= g_tp[g]:
                    # Close this group
                    pnl = 0.0
                    for k in range(m):
                        if g_entry[g, k] > 0:
                            pnl += weights[k] * (close_matrix[i, k] / g_entry[g, k] - 1.0) * g_size[g]
                            pnl -= 2.0 * total_cost * g_size[g] * weights[k]
                    realized_pnl += pnl
                    free_cash += g_size[g] + pnl
                    n_trades += 1
                    if pnl > 0: win_trades += 1
                    # Remove group (swap with last)
                    ng -= 1
                    if g < ng:
                        for k in range(m): g_entry[g, k] = g_entry[ng, k]
                        g_ez[g] = g_ez[ng]; g_size[g] = g_size[ng]; g_tp[g] = g_tp[ng]
                else:
                    g += 1

        # ════ ENTRY FILL ════
        if pending_entry and i >= pending_fill_bar:
            # Compute position size
            if variant == 2:  # C: Progressive
                depth = max(0.0, abs(pending_ez) - entry_z)
                pos_size = base_size * (1.0 + size_scale * depth)
            else:
                pos_size = base_size

            if ng < max_pos and free_cash >= pos_size:
                for k in range(m):
                    g_entry[ng, k] = close_matrix[i, k]
                g_ez[ng] = pending_ez
                g_size[ng] = pos_size
                free_cash -= pos_size

                # Compute TP for this group
                if variant == 0:      # A: fixed TP
                    g_tp[ng] = tp_pct
                elif variant == 1:    # B: Dev-TP
                    g_tp[ng] = tp_pct * max(1.0, abs(pending_ez) * tp_scale)
                elif variant == 2:    # C: Progressive + Dev-TP
                    g_tp[ng] = tp_pct * max(1.0, abs(pending_ez) * tp_scale)
                elif variant == 3:    # D: Group-TP (TP checked at group level, not here)
                    g_tp[ng] = 999.0  # never triggers per-position
                elif variant == 4:    # E: Vol-adaptive
                    depth_mult = max(1.0, abs(pending_ez) / entry_z)
                    g_tp[ng] = tp_pct * pending_vol_ratio * depth_mult

                ng += 1
            pending_entry = False

        # ════ ENTRY SIGNAL ════
        if sess and not pending_entry and (variant != 3 or pending_close_all_bar < 0):
            # Grid step (vol-adapted for E)
            eff_step = grid_step_z
            if variant == 4:
                eff_step = grid_step_z * vol_ratio

            # Size needed for cash check
            if variant == 2:
                depth_est = max(0.0, abs(zi) - entry_z)
                size_needed = base_size * (1.0 + size_scale * depth_est)
            else:
                size_needed = base_size

            if zi < -entry_z and ng < max_pos and free_cash >= size_needed:
                should = True
                if ng > 0:
                    # Find the lowest (most negative) entry Z among open groups
                    min_ez = g_ez[0]
                    for g in range(1, ng):
                        if g_ez[g] < min_ez:
                            min_ez = g_ez[g]
                    if zi > min_ez - eff_step:
                        should = False
                if should:
                    pending_entry = True
                    pending_ez = zi
                    pending_fill_bar = i + delay_bars
                    pending_vol_ratio = vol_ratio

        # ════ EQUITY ════
        unrealized = 0.0
        for g in range(ng):
            for k in range(m):
                if g_entry[g, k] > 0:
                    unrealized += weights[k] * (close_matrix[i, k] / g_entry[g, k] - 1.0) * g_size[g]
        eq_val = capital + realized_pnl + unrealized
        equity[i] = eq_val
        if eq_val > peak:
            peak = eq_val
        dd = (eq_val - peak) / peak
        if dd < max_dd:
            max_dd = dd

    final_eq = equity[-1] if n > 0 else capital
    return final_eq, max_dd, n_trades, win_trades, equity, ng


# ════════════════════════════════════════════════════════════════════
#  RUNNER
# ════════════════════════════════════════════════════════════════════

VARIANT_NAMES = {0: "A: Base (fixed TP)", 1: "B: Dev-TP", 2: "C: Progressive",
                 3: "D: Group-TP", 4: "E: Vol-adaptive"}

# Index config
ASSET_NAMES_ALL = ['BR', 'NG', 'GD', 'SV', 'MX']
WEIGHTS_FULL = np.array([0.36, 0.28, 0.0, 0.36, 0.0], dtype=np.float64)
ACTIVE_MASK = WEIGHTS_FULL > 0
ACTIVE_NAMES = [ASSET_NAMES_ALL[i] for i in range(len(ASSET_NAMES_ALL)) if ACTIVE_MASK[i]]
ACTIVE_WEIGHTS = WEIGHTS_FULL[ACTIVE_MASK]

SMA_PERIOD = 120      # 600s on 5s bars
LOOKBACK_BARS = 1573  # 7865s / 5s
CAPITAL = 10_000.0
TOTAL_COST = 0.00025  # commission + spread
DELAY_BARS = 4        # 20ms ≈ 4 bars at 5s? No, 20ms < 5s → 1 bar delay
# Actually 20ms is sub-bar on 5s. Use 1 bar delay (next bar execution).

# Best params per variant
PARAMS = {
    0: {"n_parts": 9, "entry_z": 3.997, "grid_step_z": 1.247, "tp_pct": 0.003,
        "max_pos": 9, "tp_scale": 0.0, "size_scale": 0.0, "vol_lb": 0},
    1: {"n_parts": 9, "entry_z": 3.242, "grid_step_z": 0.753, "tp_pct": 0.006,
        "max_pos": 9, "tp_scale": 0.336, "size_scale": 0.0, "vol_lb": 0},
    2: {"n_parts": 13, "entry_z": 2.100, "grid_step_z": 2.087, "tp_pct": 0.043,
        "max_pos": 10, "tp_scale": 0.254, "size_scale": 1.537, "vol_lb": 0},
    3: {"n_parts": 7, "entry_z": 3.296, "grid_step_z": 0.159, "tp_pct": 0.003,
        "max_pos": 9, "tp_scale": 0.0, "size_scale": 0.0, "vol_lb": 0},
    4: {"n_parts": 10, "entry_z": 2.969, "grid_step_z": 2.520, "tp_pct": 0.003,
        "max_pos": 10, "tp_scale": 0.0, "size_scale": 0.0, "vol_lb": 1966},
}


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    t0 = time.time()

    print("=" * 70)
    print("  MOEX BACKTEST ENGINE — All Variants")
    print("  Index: 36%BR + 28%NG + 36%SV")
    print("  Cash management: ON | Delay: 1 bar (5s)")
    print("=" * 70)

    print('\nLoading data...')
    data = load_5s(ASSET_NAMES_ALL)
    close_df = data[ACTIVE_NAMES]
    close_matrix = close_df.values.astype(np.float64)
    is_session = mark_sessions(data.index)
    n_bars = len(close_matrix)
    print(f'Active: {ACTIVE_NAMES}, Bars: {n_bars:,}, Session: {is_session.sum():,}')

    print('\nComputing index & Z-scores...')
    z, idx_log = build_index_and_z(close_matrix, ACTIVE_WEIGHTS, SMA_PERIOD, LOOKBACK_BARS)
    print(f'Z range: [{z.min():.1f}, {z.max():.1f}]')

    # Warmup numba
    print('Warming up numba...')
    cm2k = close_matrix[:2000]; z2k = z[:2000]; il2k = idx_log[:2000]; s2k = is_session[:2000]
    _ = backtest_variant(cm2k, ACTIVE_WEIGHTS, z2k, il2k, s2k,
                         0, 5, 2.0, 1.0, 0.01, 3, CAPITAL, TOTAL_COST, 1,
                         0.0, 0.0, 0)
    print('Done\n')

    results = {}
    for v in range(5):
        p = PARAMS[v]
        label = VARIANT_NAMES[v]
        print(f'{"="*70}')
        print(f'  {label}')
        print(f'{"="*70}')

        feq, mdd, nt, wt, equity, nopen = backtest_variant(
            close_matrix, ACTIVE_WEIGHTS, z, idx_log, is_session,
            v,
            p["n_parts"], p["entry_z"], p["grid_step_z"], p["tp_pct"], p["max_pos"],
            CAPITAL, TOTAL_COST, 1,  # 1 bar delay
            p["tp_scale"], p["size_scale"], p["vol_lb"])

        ret_pct = (feq / CAPITAL - 1) * 100
        dd_pct = mdd * 100
        pnl_dd = abs(ret_pct / dd_pct) if dd_pct != 0 else 0
        wr = wt / nt * 100 if nt > 0 else 0

        # Yearly
        eq_s = pd.Series(equity, index=data.index)
        yearly = {}
        for y in sorted(data.index.year.unique()):
            yr = eq_s[eq_s.index.year == y]
            if len(yr) > 1:
                yearly[y] = round((yr.iloc[-1] / yr.iloc[0] - 1) * 100, 1)

        # Daily returns for Sharpe/Sortino
        daily = eq_s.resample('D').last().pct_change().dropna()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 10 and daily.std() > 0 else 0
        neg = daily[daily < 0]
        sortino = daily.mean() / neg.std() * np.sqrt(252) if len(neg) > 5 and neg.std() > 0 else 0

        yr_str = " ".join(f"{yearly.get(y, 0):+.0f}%" for y in sorted(yearly.keys()))

        print(f'  P&L: {ret_pct:+.1f}%  MaxDD: {dd_pct:.1f}%  P&L/DD: {pnl_dd:.1f}')
        print(f'  Trades: {nt} ({wt}w, WR={wr:.0f}%)  Open: {nopen}')
        print(f'  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}')
        print(f'  Years: [{yr_str}]')

        results[VARIANT_NAMES[v]] = {
            "pnl_pct": round(ret_pct, 2), "maxdd_pct": round(dd_pct, 2),
            "pnl_dd": round(pnl_dd, 2), "trades": nt, "wins": wt,
            "wr_pct": round(wr, 1), "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2), "yearly": yearly,
        }

    # Comparison table
    print(f'\n{"="*70}')
    print(f'  COMPARISON')
    print(f'{"="*70}')
    print(f'  {"Variant":25s} {"P&L":>8s} {"MaxDD":>8s} {"P/DD":>6s} {"Trades":>7s} {"WR":>5s} {"Sharpe":>7s} {"Sortino":>8s}')
    for v in range(5):
        r = results[VARIANT_NAMES[v]]
        print(f'  {VARIANT_NAMES[v]:25s} {r["pnl_pct"]:+7.1f}% {r["maxdd_pct"]:7.1f}% {r["pnl_dd"]:6.1f} {r["trades"]:7d} {r["wr_pct"]:4.0f}% {r["sharpe"]:7.2f} {r["sortino"]:8.2f}')

    with open('moex_engine_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: moex_engine_results.json')
    print(f'Time: {(time.time()-t0)/60:.1f} min')
