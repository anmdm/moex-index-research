"""Compare Index+StrategyD with TRUR (Вечный Портфель) and TMON (Денежный Рынок)."""
import pandas as pd
import numpy as np
import json, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

def load_etf(path):
    df = pd.read_csv(path)
    df.columns = [c.strip('<>') for c in df.columns]
    df['date'] = pd.to_datetime(df['DATE'], format='%y%m%d')
    df = df.set_index('date').sort_index()
    return df['CLOSE'].astype(float)

def analyze(prices, name):
    total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    daily_ret = prices.pct_change().dropna()
    cum = (1 + daily_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    neg = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0

    yearly = {}
    for y in sorted(prices.index.year.unique()):
        yr = prices[prices.index.year == y]
        if len(yr) > 1:
            yearly[y] = round((yr.iloc[-1] / yr.iloc[0] - 1) * 100, 1)

    # Drawdown durations
    dd_periods = []
    in_dd = False
    dd_start = None
    for i in range(len(dd)):
        if dd.iloc[i] < -0.001:
            if not in_dd:
                in_dd = True
                dd_start = dd.index[i]
        else:
            if in_dd:
                dd_end = dd.index[i]
                dd_periods.append((dd_end - dd_start).days)
                in_dd = False
    if in_dd:
        dd_periods.append((dd.index[-1] - dd_start).days)

    dd_arr = np.array(dd_periods) if dd_periods else np.array([0])

    # Calmar ratio (annualized return / maxDD)
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    ann_ret = (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    calmar = abs(ann_ret * 100 / max_dd) if max_dd != 0 else 0

    # Max consecutive loss days
    loss_days = (daily_ret < 0).astype(int)
    max_consec_loss = 0
    cur = 0
    for v in loss_days:
        if v: cur += 1
        else: cur = 0
        max_consec_loss = max(max_consec_loss, cur)

    return {
        'name': name,
        'period': f'{prices.index[0].date()} -> {prices.index[-1].date()}',
        'total_ret': round(total_ret, 2),
        'ann_ret': round(ann_ret * 100, 2),
        'max_dd': round(max_dd, 2),
        'pnl_dd': round(abs(total_ret / max_dd) if max_dd != 0 else 0, 2),
        'calmar': round(calmar, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'yearly': yearly,
        'dd_count': len(dd_periods),
        'dd_median': round(np.median(dd_arr), 0),
        'dd_p75': round(np.percentile(dd_arr, 75), 0),
        'dd_p95': round(np.percentile(dd_arr, 95), 0),
        'dd_max': int(dd_arr.max()),
        'max_consec_loss_days': max_consec_loss,
        'daily_vol': round(daily_ret.std() * np.sqrt(252) * 100, 2),
    }

def print_result(r):
    print(f"  Period: {r['period']}")
    print(f"  Total P&L: {r['total_ret']:+.1f}%")
    print(f"  Annualized: {r['ann_ret']:+.1f}%/year")
    print(f"  MaxDD: {r['max_dd']:.1f}%")
    print(f"  P&L/DD: {r['pnl_dd']:.1f}")
    print(f"  Calmar: {r['calmar']:.2f}")
    print(f"  Sharpe: {r['sharpe']:.2f}")
    print(f"  Sortino: {r['sortino']:.2f}")
    print(f"  Annual Vol: {r['daily_vol']:.1f}%")
    yr_str = ' '.join(f"{v:+.0f}%" for v in r['yearly'].values())
    print(f"  Yearly: [{yr_str}]")
    print(f"  Drawdowns: {r['dd_count']} periods, median={r['dd_median']:.0f}d, P75={r['dd_p75']:.0f}d, P95={r['dd_p95']:.0f}d, max={r['dd_max']}d")
    print(f"  Max consecutive loss days: {r['max_consec_loss_days']}")


if __name__ == '__main__':
    trur = load_etf('moex_cache/etf/TRUR_230101_251201.txt')
    tmon = load_etf('moex_cache/etf/TMON_230101_251201.txt')

    with open('moex_d_optimized.json') as f:
        idx_res = json.load(f)

    print("=" * 70)
    print("  TRUR (Тинькофф Вечный Портфель)")
    print("=" * 70)
    r_trur = analyze(trur, 'TRUR')
    print_result(r_trur)

    print()
    print("=" * 70)
    print("  TMON (Тинькофф Денежный Рынок)")
    print("=" * 70)
    r_tmon = analyze(tmon, 'TMON')
    print_result(r_tmon)

    print()
    print("=" * 70)
    print("  INDEX (36%BR+28%NG+36%SV) + Strategy D (Group-TP)")
    print("=" * 70)
    # Annualize index (Mar 2023 - Mar 2026 = 3 years)
    idx_ann = ((1 + idx_res['pnl_pct']/100) ** (1/3) - 1) * 100
    print(f"  Period: 2023-03 -> 2026-03")
    print(f"  Total P&L: +{idx_res['pnl_pct']:.1f}%")
    print(f"  Annualized: {idx_ann:+.1f}%/year")
    print(f"  MaxDD: {idx_res['maxdd_pct']:.1f}%")
    print(f"  P&L/DD: {idx_res['pnl_dd']:.1f}")
    calmar_idx = abs(idx_ann / idx_res['maxdd_pct']) if idx_res['maxdd_pct'] != 0 else 0
    print(f"  Calmar: {calmar_idx:.2f}")
    print(f"  Sharpe: {idx_res['sharpe']:.2f}")
    print(f"  Sortino: {idx_res['sortino']:.2f}")
    yr_str = ' '.join(f"{v:+.0f}%" for v in idx_res['yearly'].values())
    print(f"  Yearly: [{yr_str}]")
    print(f"  Trades: {idx_res['trades']} ({idx_res['wins']}w, WR={idx_res['wins']/idx_res['trades']*100:.0f}%)")

    # ═══ COMPARISON TABLE ═══
    print()
    print("=" * 70)
    print("  СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("=" * 70)
    h = f"  {'Метрика':30s} {'TRUR':>12s} {'TMON':>12s} {'INDEX+D':>12s}"
    print(h)
    print("  " + "-" * 68)
    print(f"  {'Total P&L':30s} {r_trur['total_ret']:+11.1f}% {r_tmon['total_ret']:+11.1f}% +{idx_res['pnl_pct']:10.1f}%")
    print(f"  {'Annualized':30s} {r_trur['ann_ret']:+11.1f}% {r_tmon['ann_ret']:+11.1f}% {idx_ann:+11.1f}%")
    print(f"  {'MaxDD':30s} {r_trur['max_dd']:11.1f}% {r_tmon['max_dd']:11.1f}% {idx_res['maxdd_pct']:11.1f}%")
    print(f"  {'P&L/DD':30s} {r_trur['pnl_dd']:11.1f} {r_tmon['pnl_dd']:11.1f} {idx_res['pnl_dd']:11.1f}")
    print(f"  {'Calmar':30s} {r_trur['calmar']:11.2f} {r_tmon['calmar']:11.2f} {calmar_idx:11.2f}")
    print(f"  {'Sharpe':30s} {r_trur['sharpe']:11.2f} {r_tmon['sharpe']:11.2f} {idx_res['sharpe']:11.2f}")
    print(f"  {'Sortino':30s} {r_trur['sortino']:11.2f} {r_tmon['sortino']:11.2f} {idx_res['sortino']:11.2f}")
    print(f"  {'Annual Volatility':30s} {r_trur['daily_vol']:10.1f}% {r_tmon['daily_vol']:10.1f}% {'N/A':>11s}")
    print(f"  {'DD median duration (days)':30s} {r_trur['dd_median']:11.0f} {r_tmon['dd_median']:11.0f} {'N/A':>11s}")
    print(f"  {'DD P95 duration (days)':30s} {r_trur['dd_p95']:11.0f} {r_tmon['dd_p95']:11.0f} {'N/A':>11s}")
    print(f"  {'DD max duration (days)':30s} {r_trur['dd_max']:11d} {r_tmon['dd_max']:11d} {'N/A':>11s}")

    # Yearly comparison
    print()
    print("  Yearly returns:")
    all_years = sorted(set(list(r_trur['yearly'].keys()) + list(r_tmon['yearly'].keys()) + [int(y) for y in idx_res['yearly'].keys()]))
    for y in all_years:
        tr = r_trur['yearly'].get(y, None)
        tm = r_tmon['yearly'].get(y, None)
        ix = idx_res['yearly'].get(str(y), None)
        tr_s = f"{tr:+.1f}%" if tr is not None else "  -"
        tm_s = f"{tm:+.1f}%" if tm is not None else "  -"
        ix_s = f"{ix:+.1f}%" if ix is not None else "  -"
        print(f"    {y}: TRUR={tr_s:>8s}  TMON={tm_s:>8s}  INDEX={ix_s:>8s}")

    # Save
    comparison = {'TRUR': r_trur, 'TMON': r_tmon, 'INDEX_D': idx_res}
    with open('moex_etf_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nSaved: moex_etf_comparison.json")
