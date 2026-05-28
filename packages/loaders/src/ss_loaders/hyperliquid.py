"""Hyperliquid public-API loaders for perp funding + candles.

Why Hyperliquid: Binance (`fapi.binance.com`) and Bybit (`api.bybit.com`)
both return HTTP 451 / 403 from this host (US-restricted). OKX's public
funding-rate-history endpoint exposes only the most recent ~3 months
of data. Hyperliquid's `info` endpoint exposes ~2.4 years of hourly
funding history across 230 perpetuals with NO auth or geo-restriction,
which is enough substrate for a 4-fold annual walk-forward eval.

Funding cadence on Hyperliquid is **hourly** (24/day), not 8h like
Binance/Bybit. Callers that want daily-summed funding should call
`load_hl_funding_panel(..., resample='1D')`.

Endpoint: POST https://api.hyperliquid.xyz/info
- `{"type": "meta"}` → list of perp instruments
- `{"type": "metaAndAssetCtxs"}` → current-snapshot meta + market context
  (incl. `dayNtlVlm`, `markPx`, `funding`, `openInterest`)
- `{"type": "fundingHistory", "coin": <name>, "startTime": ms,
   "endTime": ms}` → list of `{coin, fundingRate, premium, time}` for
  the window. Returns up to 500 records per call; paginate by
  advancing `startTime` to `max(time) + 1`.
- `{"type": "candleSnapshot",
   "req": {"coin": <name>, "interval": "1d"|"1h", "startTime": ms,
            "endTime": ms}}` → OHLCV candles. Returns up to 5000.

Disk cache at `.hl-cache/` (mirrors `ss_macro`'s on-disk cache
convention). Cache key = (endpoint, coin, interval, start_ms, end_ms);
files are pickled DataFrames or JSON. Force a refresh by passing
`refresh=True`.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path('.hl-cache')
HL_INFO_URL = 'https://api.hyperliquid.xyz/info'
HL_FUNDING_PAGE_LIMIT = 500
HL_CANDLE_PAGE_LIMIT = 5000
HL_REQUEST_SLEEP_S = 0.15  # conservative; HL allows ~1200 req/min but bursts trigger 429


def _post(payload: dict, timeout: float = 30.0, max_retries: int = 6) -> object:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            HL_INFO_URL,
            method='POST',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload).encode(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                # exponential backoff
                time.sleep(min(60.0, 1.0 * (2 ** attempt)))
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(min(30.0, 0.5 * (2 ** attempt)))
            continue
    raise last_err if last_err is not None else RuntimeError('post failed')


def _cache_path(cache_dir: Path, key_parts: tuple[str, ...], ext: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1('|'.join(key_parts).encode()).hexdigest()[:16]
    safe = '_'.join(p.replace('/', '_') for p in key_parts[:3])
    return cache_dir / f'{safe}_{h}.{ext}'


def load_hl_perp_universe(
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the current perp universe + market context as a DataFrame.

    Columns: `coin, dayNtlVlm, openInterest, markPx, funding, premium,
    oraclePx`. **Snapshot-only** — for point-in-time historical
    universe selection use `load_hl_perp_candles` and compute
    `close * volume` per day from candles.
    """
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cpath = _cache_path(cdir, ('hl_meta_ctxs',), 'json')
    if refresh or not cpath.exists():
        data = _post({'type': 'metaAndAssetCtxs'})
        cpath.write_text(json.dumps(data))
    else:
        data = json.loads(cpath.read_text())
    universe = data[0]['universe']
    ctxs = data[1]
    def _f(x):
        try:
            return float(x) if x is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    rows = []
    for i, u in enumerate(universe):
        c = ctxs[i]
        rows.append({
            'coin': u['name'],
            'dayNtlVlm': _f(c.get('dayNtlVlm')),
            'openInterest': _f(c.get('openInterest')),
            'markPx': _f(c.get('markPx')),
            'funding': _f(c.get('funding')),
            'premium': _f(c.get('premium')),
            'oraclePx': _f(c.get('oraclePx')),
        })
    return pd.DataFrame(rows)


def load_hl_funding_history(
    coin: str,
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    sleep_s: float = HL_REQUEST_SLEEP_S,
) -> pd.DataFrame:
    """Fetch full funding-rate history for one coin over [start, end] ms.

    Returns a DataFrame `[funding_time (UTC), funding_rate, premium]`.
    Paginates in 500-record batches. Cached as a per-(coin, span)
    parquet under `cache_dir/`.
    """
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cpath = _cache_path(
        cdir,
        ('hl_funding', coin, str(start_ts_ms), str(end_ts_ms)),
        'parquet',
    )
    if not refresh and cpath.exists():
        return pd.read_parquet(cpath)

    out: list[dict] = []
    cur = start_ts_ms
    while True:
        data = _post({
            'type': 'fundingHistory',
            'coin': coin,
            'startTime': cur,
            'endTime': end_ts_ms,
        })
        if not data:
            break
        out.extend(data)
        new_max = max(int(x['time']) for x in data)
        if new_max <= cur or len(data) < HL_FUNDING_PAGE_LIMIT:
            break
        cur = new_max + 1
        time.sleep(sleep_s)
    if not out:
        df = pd.DataFrame(columns=['funding_time', 'funding_rate', 'premium'])
    else:
        df = pd.DataFrame([
            {
                'funding_time': pd.Timestamp(int(x['time']), unit='ms', tz='UTC'),
                'funding_rate': float(x['fundingRate']),
                'premium': float(x['premium']),
            }
            for x in out
        ]).drop_duplicates(subset=['funding_time']).sort_values('funding_time')
    df.to_parquet(cpath, index=False)
    return df


def load_hl_perp_candles(
    coin: str,
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    interval: str = '1d',
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    sleep_s: float = HL_REQUEST_SLEEP_S,
) -> pd.DataFrame:
    """Fetch perp candles for one coin. Returns `[t (UTC), open, high,
    low, close, vol_base, vol_quote]` with `vol_quote ≈ close*vol_base`.

    HL's `candleSnapshot` returns up to 5000 records — large enough for
    daily candles over multi-year spans without paginating, but we
    paginate defensively for `1h`/finer intervals.
    """
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cpath = _cache_path(
        cdir,
        ('hl_candles', coin, interval, str(start_ts_ms), str(end_ts_ms)),
        'parquet',
    )
    if not refresh and cpath.exists():
        return pd.read_parquet(cpath)
    out: list[dict] = []
    cur = start_ts_ms
    while True:
        data = _post({
            'type': 'candleSnapshot',
            'req': {
                'coin': coin,
                'interval': interval,
                'startTime': cur,
                'endTime': end_ts_ms,
            },
        })
        if not data:
            break
        out.extend(data)
        new_max = max(int(x['t']) for x in data)
        if new_max <= cur or len(data) < HL_CANDLE_PAGE_LIMIT:
            break
        cur = new_max + 1
        time.sleep(sleep_s)
    if not out:
        df = pd.DataFrame(
            columns=['t', 'open', 'high', 'low', 'close', 'vol_base', 'vol_quote'])
    else:
        seen = set()
        rows = []
        for x in out:
            t = int(x['t'])
            if t in seen:
                continue
            seen.add(t)
            close = float(x['c'])
            vb = float(x['v'])
            rows.append({
                't': pd.Timestamp(t, unit='ms', tz='UTC'),
                'open': float(x['o']),
                'high': float(x['h']),
                'low': float(x['l']),
                'close': close,
                'vol_base': vb,
                'vol_quote': close * vb,
            })
        df = pd.DataFrame(rows).sort_values('t').reset_index(drop=True)
    df.to_parquet(cpath, index=False)
    return df


def load_hl_funding_panel(
    coins: list[str],
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    resample: str = '1D',
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build a `[date × coin]` panel of summed funding rates.

    With `resample='1D'`, the value at date `d` for coin `c` is the
    SUM of the (up to 24) hourly funding rates that paid on day `d`.
    This is the gross funding yield the long-spot/short-perp leg
    collects between successive midnights.
    """
    out = {}
    for coin in coins:
        df = load_hl_funding_history(
            coin, start_ts_ms, end_ts_ms, cache_dir=cache_dir, refresh=refresh
        )
        if df.empty:
            continue
        s = df.set_index('funding_time')['funding_rate']
        if resample is not None:
            s = s.resample(resample).sum()
        out[coin] = s
    if not out:
        return pd.DataFrame()
    panel = pd.DataFrame(out).sort_index()
    return panel


def load_hl_close_panel(
    coins: list[str],
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    interval: str = '1d',
    field: str = 'close',
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build a `[date × coin]` panel of one OHLCV field from perp candles."""
    out = {}
    for coin in coins:
        df = load_hl_perp_candles(
            coin, start_ts_ms, end_ts_ms, interval=interval,
            cache_dir=cache_dir, refresh=refresh,
        )
        if df.empty:
            continue
        s = df.set_index('t')[field]
        out[coin] = s
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


__all__ = [
    'load_hl_perp_universe',
    'load_hl_funding_history',
    'load_hl_perp_candles',
    'load_hl_funding_panel',
    'load_hl_close_panel',
]
