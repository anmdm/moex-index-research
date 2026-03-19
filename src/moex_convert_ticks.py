"""
Convert tick CSV.GZ files to parquet.
Deduplication: records with same TRADE_TS + DIRECTION are partial fills
of one order → keep only last (highest-price for BUY, lowest for SELL,
or simply last by row order since files are already sorted by time).
Output: one parquet per instrument with columns [ts, price]
where ts = pandas Timestamp (UTC), price = last trade price per unique (ts, direction) group.
"""
import os, sys, io, gzip, time
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = 'moex_cache/ticks_2y'
OUT_DIR  = 'moex_cache/ticks_parquet'
os.makedirs(OUT_DIR, exist_ok=True)

INSTRUMENTS = ['Si', 'CR', 'BR', 'NG', 'GD', 'SV', 'MX']


def convert_instrument(prefix):
    in_dir = os.path.join(DATA_DIR, prefix)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.csv.gz'))
    print(f'{prefix}: {len(files)} files')

    chunks = []
    t0 = time.time()

    for i, fname in enumerate(files):
        fpath = os.path.join(in_dir, fname)
        try:
            with gzip.open(fpath, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, usecols=['TRADE_TS', 'DIRECTION', 'PRICE'])
        except Exception as e:
            print(f'  skip {fname}: {e}')
            continue

        if df.empty:
            continue

        # Parse timestamps (microsecond precision)
        df['ts'] = pd.to_datetime(df['TRADE_TS'], format='ISO8601', utc=True)

        # Deduplicate: same (ts, direction) = partial fills of one order → keep last
        df = df.drop_duplicates(subset=['ts', 'DIRECTION'], keep='last')

        # For index we only need last price per timestamp (regardless of direction)
        df = df.sort_values('ts').drop_duplicates(subset=['ts'], keep='last')

        chunks.append(df[['ts', 'PRICE']].rename(columns={'PRICE': 'price'}))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(files)} ({elapsed:.0f}s)')

    if not chunks:
        print(f'  {prefix}: no data!')
        return

    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values('ts').reset_index(drop=True)

    # Set ts as index
    result = result.set_index('ts')

    out_path = os.path.join(OUT_DIR, f'{prefix}_ticks.parquet')
    result.to_parquet(out_path)

    elapsed = time.time() - t0
    print(f'  {prefix}: {len(result):,} ticks, {result.index[0]} → {result.index[-1]}')
    print(f'  saved: {out_path}  ({os.path.getsize(out_path)/1024/1024:.1f} MB, {elapsed:.0f}s)')


if __name__ == '__main__':
    for prefix in INSTRUMENTS:
        convert_instrument(prefix)
    print('\nDone.')
