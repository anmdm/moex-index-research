"""
Full Backtrader validation with REAL orders (bt.buy/bt.sell).
Backtrader manages portfolio, cash, positions, commissions.
Data resampled to 30s (3.1M bars) to avoid timeout.

Strategy D: Group-TP Grid on 36%BR + 28%NG + 36%SV.
"""
import sys, io, os, time, json, warnings
import numpy as np
import pandas as pd
import backtrader as bt

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

WEIGHTS = {'BR': 0.36, 'NG': 0.28, 'SV': 0.36}
SMA_PERIOD = 20       # 600s / 30s = 20 bars
LOOKBACK = 262        # 7865s / 30s = 262 bars
CAPITAL = 10_000.0
COMMISSION = 0.00025

# Strategy D params
ENTRY_Z = 2.4379
GRID_STEP = 0.6020
TP_PCT = 0.003647
MAX_POS = 4
N_PARTS = 4


class GroupTPGridBT(bt.Strategy):
    """Group-TP grid — uses bt.buy()/bt.sell() for real order execution."""

    def __init__(self):
        self.names = ['BR', 'NG', 'SV']
        self.w = [0.36, 0.28, 0.36]

        # Pre-compute index and Z from data
        # (backtrader is too slow to compute rolling stats bar-by-bar in Python)
        self.idx_log = []
        self.cumsum = [0.0, 0.0, 0.0]

        # Group state
        self.groups = []  # [{entries: [p0,p1,p2], ez: float, sizes: [s0,s1,s2]}]
        self.bar_count = 0
        self.pending_entry = None  # (fill_bar, ez)
        self.pending_close = None  # fill_bar

        # Z state
        self.z_lb_start = 0
        self.z_sum = 0.0
        self.z_sq = 0.0
        self.z_cnt = 0

        # Manual tracking for comparison
        self.manual_trades = 0
        self.manual_wins = 0

    def next(self):
        self.bar_count += 1
        i = self.bar_count - 1

        # Current closes
        closes = [float(self.datas[k].close[0]) for k in range(3)]

        # Index = sum(w * price / SMA)
        iv = 0.0
        for j in range(3):
            self.cumsum[j] += closes[j]
            if i >= SMA_PERIOD:
                # Get price SMA_PERIOD bars ago
                self.cumsum[j] -= float(self.datas[j].close[-SMA_PERIOD])
                sma = self.cumsum[j] / SMA_PERIOD
            elif i >= SMA_PERIOD - 1:
                sma = self.cumsum[j] / SMA_PERIOD
            else:
                sma = self.cumsum[j] / (i + 1)
            if sma > 1e-10:
                iv += self.w[j] * (closes[j] / sma)

        li = np.log(max(iv, 1e-10))
        self.idx_log.append(li)

        # Z-score (rolling)
        si = li
        self.z_sum += si
        self.z_sq += si * si
        self.z_cnt += 1
        while self.z_lb_start < len(self.idx_log) - 1 and self.z_cnt > LOOKBACK:
            self.z_sum -= self.idx_log[self.z_lb_start]
            self.z_sq -= self.idx_log[self.z_lb_start] ** 2
            self.z_cnt -= 1
            self.z_lb_start += 1

        zi = 0.0
        if self.z_cnt >= 10:
            mu = self.z_sum / self.z_cnt
            var = (self.z_sq / self.z_cnt - mu * mu) * self.z_cnt / (self.z_cnt - 1)
            if var > 1e-30:
                zi = (si - mu) / np.sqrt(var)

        # Session filter (UTC+3 for MSK)
        dt = self.datas[0].datetime.datetime(0)
        hm = (dt.hour + 3) * 60 + dt.minute
        hm %= 1440
        sess = (600 <= hm < 1125) or (1145 <= hm < 1430)

        base_size = CAPITAL / N_PARTS

        # ═══ CLOSE FILL ═══
        if self.pending_close is not None and self.bar_count >= self.pending_close and self.groups:
            for g in self.groups:
                pnl = 0.0
                for k in range(3):
                    entry_px = g['entries'][k]
                    cur_px = closes[k]
                    qty = g['sizes'][k]
                    if qty > 0 and entry_px > 0:
                        # SELL via backtrader
                        self.sell(data=self.datas[k], size=qty)
                        pnl += self.w[k] * (cur_px / entry_px - 1) * base_size
                self.manual_trades += 1
                if pnl > 0:
                    self.manual_wins += 1
            self.groups.clear()
            self.pending_close = None

        # ═══ TP CHECK ═══
        if self.groups and sess and self.pending_close is None:
            total_inv = 0.0
            total_pnl = 0.0
            for g in self.groups:
                total_inv += base_size
                for k in range(3):
                    if g['entries'][k] > 0:
                        total_pnl += self.w[k] * (closes[k] / g['entries'][k] - 1) * base_size
                        total_pnl -= 2 * COMMISSION * base_size * self.w[k]
            if total_inv > 0 and total_pnl / total_inv >= TP_PCT:
                self.pending_close = self.bar_count + 1

        # ═══ ENTRY FILL ═══
        if self.pending_entry is not None:
            fill_bar, pend_ez = self.pending_entry
            if self.bar_count >= fill_bar:
                cash = self.broker.getcash()
                if len(self.groups) < MAX_POS and cash >= base_size:
                    g = {'entries': [], 'ez': pend_ez, 'sizes': []}
                    for k in range(3):
                        px = closes[k]
                        notional = base_size * self.w[k]
                        qty = round(notional / px, 4)
                        # BUY via backtrader
                        self.buy(data=self.datas[k], size=qty)
                        g['entries'].append(px)
                        g['sizes'].append(qty)
                    self.groups.append(g)
                self.pending_entry = None

        # ═══ ENTRY SIGNAL ═══
        if sess and self.pending_entry is None and self.pending_close is None:
            ng = len(self.groups)
            cash = self.broker.getcash()
            if zi < -ENTRY_Z and ng < MAX_POS and cash >= base_size:
                ok = True
                if ng > 0:
                    min_ez = min(g['ez'] for g in self.groups)
                    if zi > min_ez - GRID_STEP:
                        ok = False
                if ok:
                    self.pending_entry = (self.bar_count + 1, zi)


def main():
    print("=" * 70)
    print("  BACKTRADER FULL VALIDATION — Strategy D (Group-TP)")
    print("  Real bt.buy()/bt.sell() orders, BT manages cash/positions")
    print("  Data: 30s bars (resampled from 5s)")
    print("=" * 70)

    # Load data
    print('\nLoading...')
    PARQUET_DIR = 'moex_cache/ticks_parquet'
    dfs = {}
    for name in WEIGHTS:
        df = pd.read_parquet(os.path.join(PARQUET_DIR, f'{name}_ticks.parquet'))
        dfs[name] = df['price'].resample('30s').last()
        del df
    combined = pd.DataFrame(dfs).sort_index().ffill().dropna()
    print(f'  Bars (30s): {len(combined):,}')

    # Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)

    for name in WEIGHTS:
        df = combined[[name]].copy()
        df.columns = ['close']
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        df['volume'] = 1.0
        df['openinterest'] = 0
        feed = bt.feeds.PandasData(dataname=df, datetime=None,
            open='open', high='high', low='low', close='close',
            volume='volume', openinterest='openinterest')
        cerebro.adddata(feed, name=name)

    cerebro.addstrategy(GroupTPGridBT)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print(f'\nRunning backtrader...')
    t0 = time.time()
    results = cerebro.run()
    elapsed = time.time() - t0
    strat = results[0]

    # Results
    final_val = cerebro.broker.getvalue()
    ret_pct = (final_val / CAPITAL - 1) * 100
    dd = strat.analyzers.dd.get_analysis()
    max_dd = dd.max.drawdown if hasattr(dd.max, 'drawdown') else 0
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_val = sharpe.get('sharperatio', 0) or 0

    pnl_dd = abs(ret_pct / max_dd) if max_dd > 0 else 0
    wr = strat.manual_wins / strat.manual_trades * 100 if strat.manual_trades > 0 else 0

    print(f'\n{"="*70}')
    print(f'  BACKTRADER FULL VALIDATION RESULT')
    print(f'  (Real orders, BT manages cash/commissions/positions)')
    print(f'{"="*70}')
    print(f'  Final value: ${final_val:,.2f}')
    print(f'  P&L: {ret_pct:+.1f}%  MaxDD: -{max_dd:.1f}%  P&L/DD: {pnl_dd:.1f}')
    print(f'  Manual trades: {strat.manual_trades} ({strat.manual_wins}w, WR={wr:.0f}%)')
    print(f'  BT Sharpe: {sharpe_val:.2f}')
    print(f'  Bars: {strat.bar_count:,}  Time: {elapsed:.0f}s')

    print(f'\n  --- COMPARISON ---')
    print(f'  Python Fast BT:  +259.1% DD=-12.5% P&L/DD=20.7 trades=859')
    print(f'  C# standalone:   +259.1% DD=-12.5% (bitwise match)')
    print(f'  NautilusTrader:   +96.4% DD=-11.0% (30s bars, event-driven)')
    print(f'  Backtrader:      {ret_pct:+.1f}% DD=-{max_dd:.1f}% P&L/DD={pnl_dd:.1f} trades={strat.manual_trades}')

    result = {
        "engine": "Backtrader (real orders)",
        "timeframe": "30s",
        "pnl_pct": round(ret_pct, 2),
        "maxdd_pct": round(-max_dd, 2),
        "pnl_dd": round(pnl_dd, 2),
        "trades": strat.manual_trades,
        "wins": strat.manual_wins,
        "sharpe": round(sharpe_val, 2),
        "final_value": round(final_val, 2),
    }
    with open('moex_backtrader_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: moex_backtrader_results.json')


if __name__ == '__main__':
    main()
