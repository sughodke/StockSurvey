"""One-shot local prep: fetch top-50 crypto daily OHLCV → pickle.

Run with:
    uv run python apps/factor/scripts/prep_crypto_universe.py

Writes `Output/crypto_universe_panel.pkl` — a dict
`{ticker: pandas.DataFrame(open, high, low, close, volume, adj_close)}`
that the Modal entrypoint reads as raw bytes (no project-venv deps in
the `uvx modal` local entrypoint).

Source: CryptoCompare public v2/histoday REST endpoint (paged in
2000-bar chunks to span >5y of history). No auth, rate-limited
~8000/hr free — fetching 50 names well below the cap.

**On `min_bars`**: align_tickers does a strict date intersection over
the loaded universe. With the default `min_bars=1100` you get 50
tickers but the common axis collapses to ARB's 2023-04-04 start
(~229 rebal blocks at rebal_days=5), which only fits ~2 walk-forward
windows. To hit the pre-registered 5-window target, prep with
`min_bars=2000` — that drops ~22 newer alts (TON, SHIB, APT, ARB,
OP, ICP, FLOW, GRT, etc.) and leaves ~28 names with a 2018-01-02
start, giving ~440 rebal blocks and 5 windows at the
146/73/73 (train/val/step) config.

Universe: hard-coded top-50 by 2026-vintage market cap (the
`s-and-p-500-companies/coin100.json` submodule is not present in the
worktree, so the canonical loader `ss_loaders.coin100()` is unavailable
here). The list below is a hand-curated top-50 snapshot suitable for
the venue-fit smallest test in `.research-venue-fit.md`. Stable assets
(USDT/USDC/DAI/BUSD/etc.) and wrapped assets (WBTC/WETH/STETH) are
excluded — they are not price-discovery instruments and would muddy
the cross-section.

The pickle is deterministic given the universe; re-run before each
Modal launch if the universe should be refreshed.
"""
from __future__ import annotations

import datetime as dt
import pickle
import time
from pathlib import Path

import pandas as pd
import requests

# The shipped `ss_loaders.load_cryptocompare` hit the legacy v1 endpoint
# whose response shape (`Data` as a list with a top-level `time` column)
# CryptoCompare has since changed (now `Data.Data` with `conversionType`
# fields). Re-implementing the fetch here against the current v2 API,
# with paging via `toTs` so we can pull >2000 bars (the per-request cap).
CC_HISTODAY_V2 = 'https://min-api.cryptocompare.com/data/v2/histoday'


def _fetch_histoday_paged(
    symbol: str, start_ts: int, end_ts: int,
    *, page_limit: int = 2000, sleep_between: float = 0.25,
) -> pd.DataFrame:
    """Page the v2/histoday endpoint backwards from `end_ts` until we've
    covered `start_ts` (or the API stops returning new bars).

    Returns a DataFrame indexed by datetime with columns
    open / high / low / close / volume / adj_close.
    """
    SEC_DAY = 86400
    frames: list[pd.DataFrame] = []
    cur_to = end_ts
    seen_earliest = None
    for _ in range(50):  # hard cap on paging iterations
        r = requests.get(CC_HISTODAY_V2, params={
            'fsym': symbol, 'tsym': 'USD',
            'limit': page_limit, 'toTs': cur_to,
        }, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get('Response') != 'Success':
            raise RuntimeError(
                f'CryptoCompare error for {symbol}: {payload.get("Message")}')
        data = payload.get('Data', {}).get('Data', [])
        if not data:
            break
        df = pd.DataFrame(data)
        df = df[df['close'] > 0]
        if df.empty:
            break
        frames.append(df)
        earliest = int(df['time'].min())
        if seen_earliest is not None and earliest >= seen_earliest:
            # No progress — bail.
            break
        seen_earliest = earliest
        if earliest <= start_ts:
            break
        # Step `toTs` back one day before our current earliest bar.
        cur_to = earliest - SEC_DAY
        time.sleep(sleep_between)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset='time', keep='first')
    out = out.sort_values('time').reset_index(drop=True)
    out.index = pd.to_datetime(out['time'], unit='s')
    out['volume'] = out['volumeto'] + out['volumefrom']
    out['adj_close'] = out['close']
    out = out[(out.index >= pd.to_datetime(start_ts, unit='s'))
              & (out.index <= pd.to_datetime(end_ts, unit='s'))]
    return out[['open', 'high', 'low', 'close', 'volume', 'adj_close']]


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / 'Output'
PICKLE_PATH = OUTPUT_DIR / 'crypto_universe_panel.pkl'


# Top-50 by 2026-vintage market cap, stable / wrapped excluded.
# Ordering is best-effort by mcap rank; the universe is what matters,
# not the order (walk-forward eval is symmetric in ticker order).
TOP50_CRYPTO: list[str] = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP',
    'ADA', 'DOGE', 'TRX', 'AVAX', 'DOT',
    'LINK', 'MATIC', 'TON', 'SHIB', 'LTC',
    'BCH', 'ATOM', 'UNI', 'XLM', 'ETC',
    'NEAR', 'XMR', 'ICP', 'APT', 'FIL',
    'HBAR', 'ARB', 'OP', 'VET', 'INJ',
    'MKR', 'AAVE', 'GRT', 'ALGO', 'RUNE',
    'SAND', 'AXS', 'EOS', 'FTM', 'XTZ',
    'EGLD', 'THETA', 'FLOW', 'CHZ', 'KAVA',
    'MANA', 'CRV', 'SNX', 'COMP', 'ZEC',
]


def main(
    *,
    start: str = '2018-01-01',
    end: str | None = None,
    # ~5.5 years of daily bars. Tighter than the "3 years post-filter"
    # mentioned in `.research-venue-fit.md` because align_tickers does a
    # strict date intersection — admitting any 2023-vintage alt (ARB, APT,
    # TON) collapses the common axis to that ticker's start and starves
    # the walk-forward (only 2 windows fit). At min_bars=2000 we drop
    # ~8 newer alts and retain 42 tickers with a 2018-or-earlier start,
    # giving ~405 rebal blocks at rebal_days=5 → 5 walk-forward windows
    # at the train/val/step=110/55/55 default (matches pre-reg).
    min_bars: int = 2000,
    sleep_between: float = 0.5,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    end_dt = (dt.datetime.fromisoformat(end) if end
              else dt.datetime.now(dt.timezone.utc).replace(
                  hour=0, minute=0, second=0, microsecond=0, tzinfo=None))
    start_dt = dt.datetime.fromisoformat(start)
    print(f'fetching {len(TOP50_CRYPTO)} crypto tickers from CryptoCompare')
    print(f'  range: {start_dt.date()} .. {end_dt.date()}')
    print(f'  min_bars filter: {min_bars}')

    panel: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, str]] = []
    t0 = time.perf_counter()
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    for i, sym in enumerate(TOP50_CRYPTO):
        try:
            df = _fetch_histoday_paged(sym, start_ts, end_ts,
                                       sleep_between=sleep_between)
        except Exception as e:
            skipped.append((sym, f'{type(e).__name__}: {e}'))
            print(f'  [{i+1:2d}/{len(TOP50_CRYPTO)}] {sym}: FAIL ({e})')
            time.sleep(sleep_between)
            continue
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < min_bars:
            skipped.append((sym, f'only {len(df)} bars (< {min_bars})'))
            print(f'  [{i+1:2d}/{len(TOP50_CRYPTO)}] {sym}: SHORT '
                  f'({len(df)} bars)')
            time.sleep(sleep_between)
            continue
        panel[sym] = df[['open', 'high', 'low', 'close',
                         'volume', 'adj_close']].copy()
        first = df.index[0].date()
        last = df.index[-1].date()
        print(f'  [{i+1:2d}/{len(TOP50_CRYPTO)}] {sym}: {len(df)} bars  '
              f'{first} .. {last}')
        time.sleep(sleep_between)

    wall = time.perf_counter() - t0
    print(f'\nfetched {len(panel)} usable tickers, '
          f'{len(skipped)} skipped, wall={wall:.1f}s')
    if skipped:
        print('  first 5 skipped:')
        for s, reason in skipped[:5]:
            print(f'    {s}: {reason}')

    if len(panel) < 8:
        raise RuntimeError(
            f'only {len(panel)} tickers usable — too few for IC walkforward')

    # Summary stats.
    bar_counts = [len(df) for df in panel.values()]
    print(f'  bar-count quantiles: min={min(bar_counts)}  '
          f'p25={sorted(bar_counts)[len(bar_counts)//4]}  '
          f'median={sorted(bar_counts)[len(bar_counts)//2]}  '
          f'max={max(bar_counts)}  total={sum(bar_counts)}')

    print(f'\npickling to {PICKLE_PATH} ...')
    payload = {
        'panel': panel,
        'universe_label': 'cryptocompare_top50_v1',
        'start': start,
        'end': end_dt.date().isoformat(),
        'min_bars_filter': min_bars,
        'skipped': skipped,
    }
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f'  {PICKLE_PATH.stat().st_size / 1024 / 1024:.2f} MB')


if __name__ == '__main__':
    main()
