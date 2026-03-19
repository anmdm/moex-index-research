"""
Optuna optimization of variant D (Group-TP) on NG (NatGas) SOLO.
Compare with index (36%BR+28%NG+36%NG) to see if index has advantage.
"""
import sys, io, os, time, json, warnings
import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from moex_backtest_engine import (
    load_5s, mark_sessions, build_index_and_z, backtest_variant,
    ASSET_NAMES_ALL, SMA_PERIOD, LOOKBACK_BARS, CAPITAL, TOTAL_COST
)

N_TRIALS = 1000


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("  OPTUNA — Variant D on NG (NatGas) SOLO")
    print("  Compare with index to check if index has advantage")
    print("=" * 70)

    print('\nLoading data...')
    data = load_5s(ASSET_NAMES_ALL)

    # NG solo: single asset
    sv_close = data[['NG']].values.astype(np.float64)
    sv_weights = np.array([1.0])
    is_session = mark_sessions(data.index)
    n_bars = len(sv_close)

    print(f'NG solo: {n_bars:,} bars')

    print('Computing Z-scores for NG solo...')
    z, idx_log = build_index_and_z(sv_close, sv_weights, SMA_PERIOD, LOOKBACK_BARS)

    # Warmup
    _ = backtest_variant(sv_close[:2000], sv_weights, z[:2000], idx_log[:2000],
                         is_session[:2000], 3, 5, 2.0, 1.0, 0.01, 3, CAPITAL, TOTAL_COST, 1,
                         0.0, 0.0, 0)
    print('Numba warmed up\n')

    # Year boundaries
    years = data.index.year
    unique_years = sorted(years.unique())
    year_bounds = []
    for y in unique_years:
        idx = np.where(years == y)[0]
        if len(idx) > 0:
            year_bounds.append((idx[0], idx[-1]))
    print(f'Years: {unique_years}\n')

    best_score = [-999.0]

    def objective(trial):
        p = {
            "n_parts": trial.suggest_int("n_parts", 3, 20),
            "entry_z": trial.suggest_float("entry_z", 0.5, 4.0),
            "grid_step_z": trial.suggest_float("grid_step_z", 0.05, 3.0),
            "tp_pct": trial.suggest_float("tp_pct", 0.001, 0.05),
            "max_pos": trial.suggest_int("max_pos", 1, 10),
        }
        p["max_pos"] = min(p["max_pos"], p["n_parts"])

        feq, mdd, nt, wt, equity, nopen = backtest_variant(
            sv_close, sv_weights, z, idx_log, is_session,
            3,  # variant D
            p["n_parts"], p["entry_z"], p["grid_step_z"],
            p["tp_pct"], p["max_pos"],
            CAPITAL, TOTAL_COST, 1,
            0.0, 0.0, 0)

        total_ret = feq / CAPITAL - 1
        if total_ret <= 0 or nt < 5:
            return -999.0

        yearly_rets = []
        for y_start, y_end in year_bounds:
            eq_s = equity[y_start]
            eq_e = equity[y_end]
            if eq_s > 0:
                yearly_rets.append(eq_e / eq_s - 1)

        if len(yearly_rets) >= 2:
            if any(r <= 0 for r in yearly_rets):
                return -999.0
            yr_mean = sum(yearly_rets) / len(yearly_rets)
            yr_var = sum((r - yr_mean) ** 2 for r in yearly_rets) / len(yearly_rets)
            yr_cv = (yr_var ** 0.5) / yr_mean if yr_mean > 0 else 999
            uniformity = 1.0 / (1.0 + yr_cv * yr_cv)
        else:
            uniformity = 1.0
            yr_cv = 0.0

        wr = wt / nt if nt > 0 else 0
        dd = abs(mdd)
        dd_penalty = max(0, dd - 0.10) * 5
        reliability = min(1.0, nt / 30)
        trade_bonus = np.log(1 + nt) / np.log(1 + 50)

        score = total_ret * 100 * (0.5 + wr) * reliability * (1 - dd_penalty) * trade_bonus * uniformity

        if score > best_score[0]:
            best_score[0] = score
            yr_str = " ".join(f"{r*100:+.0f}%" for r in yearly_rets)
            print(f"  #{trial.number}: ret={total_ret*100:+.1f}% DD={mdd*100:.1f}% "
                  f"trades={nt}({wt}w) tp={p['tp_pct']*100:.2f}% ez={p['entry_z']:.2f} "
                  f"np={p['n_parts']} mp={p['max_pos']} gs={p['grid_step_z']:.2f} "
                  f"yrs=[{yr_str}] cv={yr_cv:.2f}")
        return score

    print(f'Optuna: {N_TRIALS} trials')
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=200))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    bp = study.best_params
    bp["max_pos"] = min(bp["max_pos"], bp["n_parts"])

    feq, mdd, nt, wt, equity, nopen = backtest_variant(
        sv_close, sv_weights, z, idx_log, is_session,
        3, bp["n_parts"], bp["entry_z"], bp["grid_step_z"],
        bp["tp_pct"], bp["max_pos"],
        CAPITAL, TOTAL_COST, 1, 0.0, 0.0, 0)

    ret_pct = (feq / CAPITAL - 1) * 100
    dd_pct = mdd * 100

    eq_s = pd.Series(equity, index=data.index)
    yearly = {}
    for y in unique_years:
        yr = eq_s[eq_s.index.year == y]
        if len(yr) > 1:
            yearly[y] = round((yr.iloc[-1] / yr.iloc[0] - 1) * 100, 1)

    daily = eq_s.resample('D').last().pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 10 and daily.std() > 0 else 0
    neg = daily[daily < 0]
    sortino = daily.mean() / neg.std() * np.sqrt(252) if len(neg) > 5 and neg.std() > 0 else 0

    yr_str = " ".join(f"{yearly.get(y, 0):+.0f}%" for y in sorted(yearly.keys()))

    print(f'\n{"="*70}')
    print(f'  NG SOLO — Variant D (Group-TP)')
    print(f'{"="*70}')
    print(f'  P&L: {ret_pct:+.1f}%  MaxDD: {dd_pct:.1f}%  P&L/DD: {abs(ret_pct/dd_pct) if dd_pct != 0 else 0:.1f}')
    print(f'  Trades: {nt} ({wt}w, WR={wt/nt*100:.0f}%)')
    print(f'  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}')
    print(f'  Years: [{yr_str}]')
    print(f'  Params: {bp}')

    # Comparison
    print(f'\n  --- COMPARISON ---')
    print(f'  Index (36%BR+28%NG+36%NG): +259% DD=-12.5% P&L/DD=20.7 yrs=[+24/+20/+124/+9]')
    print(f'  NG solo:                    {ret_pct:+.1f}% DD={dd_pct:.1f}% P&L/DD={abs(ret_pct/dd_pct) if dd_pct != 0 else 0:.1f} yrs=[{yr_str}]')

    result = {
        "strategy": "D: Group-TP on NG solo",
        "pnl_pct": round(ret_pct, 2), "maxdd_pct": round(dd_pct, 2),
        "pnl_dd": round(abs(ret_pct / dd_pct) if dd_pct != 0 else 0, 2),
        "trades": nt, "wins": wt,
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
        "yearly": yearly, "params": bp,
    }
    with open('moex_ng_solo_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\nSaved: moex_ng_solo_results.json')
    print(f'Time: {(time.time()-t0)/60:.1f} min')
