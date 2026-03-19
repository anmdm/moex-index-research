"""
Robust tick data downloader for T-Invest history-trades API.
- Auto-reconnect on connection loss with exponential backoff
- Session persistence (keep-alive)
- Configurable date range (default: 3 years back)
- Skips already cached files
- Handles rate limits, timeouts, connection drops gracefully
- Self-monitoring: tracks progress, detects stalls
"""
import sys, io, os, time, warnings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta, timezone

warnings.filterwarnings('ignore', message='Unverified HTTPS')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

TOKEN = os.environ.get("TINKOFF_TOKEN",
    "YOUR_TOKEN_HERE")
BASE_URL = 'https://invest-public-api.tbank.ru/history-trades'
DATA_DIR = 'moex_cache/ticks_2y'
os.makedirs(DATA_DIR, exist_ok=True)

PREFIXES = ['Si', 'CR', 'BR', 'NG', 'GD', 'SV', 'MX']

# === Date range: 3 years back (covers 2023-2026) ===
END_DATE = datetime.now() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=1095)  # ~3 years


def create_session():
    """Create a requests Session with retry strategy and keep-alive."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,       # 2, 4, 8, 16, 32 sec between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.verify = False
    return session


def get_all_futures_from_api():
    """Get futures list via gRPC SDK."""
    from t_tech.invest import Client, InstrumentStatus
    for attempt in range(5):
        try:
            with Client(TOKEN) as client:
                resp = client.instruments.futures(instrument_status=InstrumentStatus(2))
                return resp.instruments
        except Exception as e:
            print(f'  Futures API error (attempt {attempt+1}): {e}')
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Cannot load futures list after 5 attempts")


def build_front_month_schedule(all_futures, prefix, start_date, end_date):
    """For each trading date, determine which contract ticker to use."""
    contracts = []
    for f in all_futures:
        t = f.ticker
        if not t.startswith(prefix):
            continue
        suffix = t[len(prefix):]
        if len(suffix) > 3 or len(suffix) < 2:
            continue
        exp = f.expiration_date
        if not exp:
            continue
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        contracts.append({'ticker': t, 'exp': exp})

    contracts.sort(key=lambda c: c['exp'])

    schedule = {}
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            date_str = d.strftime('%Y-%m-%d')
            d_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            best = None
            for c in contracts:
                if c['exp'] >= d_utc:
                    best = c
                    break
            if best:
                schedule[date_str] = f"{best['ticker']}_SPBFUT"
        d += timedelta(days=1)

    return schedule


def download_day(session, ticker_cc, date_str, out_dir):
    """Download one day of trades with robust error handling.
    Returns: 'cached', 'downloaded', 'empty', or 'error'
    """
    out_file = os.path.join(out_dir, f'{ticker_cc}_{date_str}.csv.gz')

    # Check cache
    if os.path.exists(out_file) and os.path.getsize(out_file) > 100:
        return 'cached'

    url = f'{BASE_URL}/{date_str}?instrumentId={ticker_cc}'

    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            resp = session.get(url, timeout=120)

            if resp.status_code == 200 and len(resp.content) > 50:
                with open(out_file, 'wb') as f:
                    f.write(resp.content)
                return 'downloaded'

            elif resp.status_code == 404:
                return 'empty'

            elif resp.status_code == 429:
                wait = 65 + attempt * 10
                print(f'    429 rate limit, waiting {wait}s (attempt {attempt+1})...')
                time.sleep(wait)

            elif resp.status_code >= 500:
                wait = 10 * (attempt + 1)
                print(f'    Server error {resp.status_code}, waiting {wait}s...')
                time.sleep(wait)

            else:
                print(f'    Unexpected status {resp.status_code} for {ticker_cc} {date_str}')
                return 'error'

        except requests.exceptions.ConnectionError as e:
            wait = 15 * (attempt + 1)
            print(f'    Connection lost (attempt {attempt+1}/{max_attempts}), reconnecting in {wait}s...')
            time.sleep(wait)
            # Recreate session on connection errors
            try:
                session.close()
            except:
                pass
            session = create_session()

        except requests.exceptions.ReadTimeout:
            wait = 20 * (attempt + 1)
            print(f'    Read timeout (attempt {attempt+1}/{max_attempts}), retry in {wait}s...')
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            wait = 10 * (attempt + 1)
            print(f'    Request error: {type(e).__name__} (attempt {attempt+1}), retry in {wait}s...')
            time.sleep(wait)

    print(f'    FAILED after {max_attempts} attempts: {ticker_cc} {date_str}')
    return 'error'


def print_status(prefixes_done, current_prefix, current_i, current_total,
                 stats, t0, all_schedules):
    """Print comprehensive status."""
    elapsed = time.time() - t0
    total_expected = sum(len(s) for s in all_schedules.values())
    total_done = stats['cached'] + stats['downloaded'] + stats['empty'] + stats['error']
    pct = total_done / total_expected * 100 if total_expected > 0 else 0
    rate = (stats['downloaded']) / elapsed * 3600 if elapsed > 60 else 0

    print(f'\n  [{elapsed/60:.0f}min] Progress: {total_done}/{total_expected} ({pct:.0f}%) '
          f'| downloaded={stats["downloaded"]} cached={stats["cached"]} empty={stats["empty"]} err={stats["error"]} '
          f'| {rate:.0f}/hr')
    print(f'  Current: {current_prefix} {current_i}/{current_total}')
    done_list = ", ".join(prefixes_done) if prefixes_done else "none"
    print(f'  Done: {done_list}')


if __name__ == "__main__":
    t0 = time.time()

    print(f'Period: {START_DATE.date()} -> {END_DATE.date()} ({(END_DATE - START_DATE).days} days)')
    print(f'Instruments: {PREFIXES}')
    print()

    # Get futures list
    print('Loading futures list...')
    all_futures = get_all_futures_from_api()
    print(f'Total futures: {len(all_futures)}')

    # Build schedules
    schedules = {}
    for prefix in PREFIXES:
        schedule = build_front_month_schedule(all_futures, prefix, START_DATE, END_DATE)
        schedules[prefix] = schedule
        tickers_used = sorted(set(schedule.values()))
        n_contracts = len(tickers_used)
        print(f'  {prefix}: {len(schedule)} trading days, {n_contracts} contracts')

    total_files = sum(len(s) for s in schedules.values())
    print(f'\nTotal files: {total_files}')
    print()

    # Create robust session
    session = create_session()

    stats = {'downloaded': 0, 'cached': 0, 'empty': 0, 'error': 0}
    prefixes_done = []
    last_minute_start = time.time()
    req_in_minute = 0

    for prefix in PREFIXES:
        schedule = schedules[prefix]
        prefix_dir = os.path.join(DATA_DIR, prefix)
        os.makedirs(prefix_dir, exist_ok=True)

        dates = sorted(schedule.keys())
        print(f'\n=== {prefix} ({len(dates)} days) ===')

        for i, date_str in enumerate(dates):
            ticker_cc = schedule[date_str]

            # Smart rate limiting
            now = time.time()
            if now - last_minute_start >= 60:
                last_minute_start = now
                req_in_minute = 0
            if req_in_minute >= 26:  # conservative
                wait = 60 - (now - last_minute_start) + 2
                if wait > 0:
                    time.sleep(wait)
                last_minute_start = time.time()
                req_in_minute = 0

            result = download_day(session, ticker_cc, date_str, prefix_dir)
            stats[result] += 1

            if result != 'cached':
                req_in_minute += 1

            # Status every 100 files
            if (i + 1) % 100 == 0:
                print_status(prefixes_done, prefix, i+1, len(dates), stats, t0, schedules)

        prefixes_done.append(prefix)
        count = len([f for f in os.listdir(prefix_dir) if f.endswith('.csv.gz')])
        print(f'  {prefix} done: {count} files on disk')

    session.close()

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'ALL DONE in {elapsed/60:.0f} min')
    print(f'  Downloaded: {stats["downloaded"]}')
    print(f'  Cached: {stats["cached"]}')
    print(f'  Empty/404: {stats["empty"]}')
    print(f'  Errors: {stats["error"]}')

    # Show disk usage
    print(f'\nDisk usage:')
    total_size = 0
    for prefix in PREFIXES:
        prefix_dir = os.path.join(DATA_DIR, prefix)
        if os.path.exists(prefix_dir):
            files = [f for f in os.listdir(prefix_dir) if f.endswith('.csv.gz')]
            size = sum(os.path.getsize(os.path.join(prefix_dir, f)) for f in files)
            total_size += size
            print(f'  {prefix}: {len(files)} files, {size/1024/1024:.0f} MB')
    print(f'  Total: {total_size/1024/1024:.0f} MB')
