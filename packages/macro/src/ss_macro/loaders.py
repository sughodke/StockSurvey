"""FRED CSV loaders — no API key, on-disk cache, point-in-time discipline."""
from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd


DEFAULT_CACHE_DIR = Path('.macro-cache')

# Canonical six-feature regime stack. Each entry: (output_name,
# fred_series_id, transform). `transform`:
#   None       — use raw series value (already in the right unit)
#   'yoy_pct'  — year-over-year percent change from the raw level
DEFAULT_SERIES: list[tuple[str, str, str | None]] = [
    ('fed_funds',      'FEDFUNDS', None),  # %, monthly
    ('slope_10y_3m',   'T10Y3M',   None),  # %, daily
    ('credit_baa',     'BAA10Y',   None),  # %, daily
    ('m2_level',       'M2SL',     None),  # $B, monthly
    ('real_yield_10y', 'DFII10',   None),  # %, daily
    ('vix',            'VIXCLS',   None),  # index, daily
    ('gold_vix',       'GVZCLS',   None),  # CBOE Gold ETF VIX, daily 2008-06+
]
# Note: FRED's no-auth CSV endpoint does NOT expose the LBMA gold price
# series (`GOLDAMGBD228NLBM` / `PGOLDUSDM` both 404). Gold *price* level
# + YoY come from Stooq's `GLD.US` ETF instead — see
# `load_gold_features_from_stooq()` and the `stooq_data_dir` arg of
# `load_macro_panel()`. `gold_vix` (GVZCLS) is the gold-implied-vol
# regime signal; it loads cleanly from FRED but only starts 2008-06-03.


def fred_series_url(series_id: str) -> str:
    """FRED's CSV download endpoint for a series — no auth required."""
    return (
        f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    )


def load_fred_series(
    series_id: str,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Fetch one FRED series, return as a date-indexed pandas Series.

    First call downloads the CSV from FRED's public endpoint; subsequent
    calls read from `cache_dir/<series_id>.csv`. Set `refresh=True` to
    re-pull. Missing values in the source (FRED uses `.` for "no
    observation") are converted to NaN.
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    csv_path = cache / f'{series_id}.csv'

    if refresh or not csv_path.exists():
        url = fred_series_url(series_id)
        print(f'[ss_macro] downloading {url} → {csv_path}')
        urllib.request.urlretrieve(url, csv_path)
        size_kb = csv_path.stat().st_size / 1024
        print(f'[ss_macro] downloaded {size_kb:.1f} KB')

    df = pd.read_csv(csv_path, parse_dates=[0])
    df.columns = ['date', 'value']
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.set_index('date').sort_index()
    s = df['value'].rename(series_id)
    return s


def load_gold_features_from_stooq(
    stooq_data_dir: str | Path,
    *,
    ticker: str = 'GLD',
    yoy_periods: int = 252,
) -> pd.DataFrame:
    """Load gold price level + YoY % change from Stooq's GLD ETF.

    FRED's no-auth CSV endpoint doesn't expose the LBMA gold price
    series (`GOLDAMGBD228NLBM` and `PGOLDUSDM` both 404 on the
    `fredgraph.csv?id=` URL pattern), so we proxy gold price through
    the SPDR Gold Shares ETF (`GLD.US`) which has been in the Stooq
    archive since 2005-02-25 and tracks gold spot to within a
    management-fee drift.

    Returns a DataFrame indexed by trading date with columns:
      gold_level — close price (ETF units, ≈ 1/10 oz × spot − fees)
      gold_yoy   — `yoy_periods`-day pct change × 100, the regime-
                   signal form (rising = USD-debasement / risk-off)
    """
    from ss_loaders import load_stooq_matrix
    closes, _, _, _ = load_stooq_matrix(
        stooq_data_dir, min_history=60, tickers=[ticker],
        include_etfs=True)
    if ticker.upper() not in closes.columns:
        raise ValueError(
            f'gold ticker {ticker!r} not found in Stooq archive at '
            f'{stooq_data_dir!r}; expected `{ticker.lower()}.us.txt` '
            f'somewhere under daily/us/...')
    gold = closes[ticker.upper()].dropna().rename('gold_level')
    gold_yoy = (gold.pct_change(periods=yoy_periods) * 100.0).rename('gold_yoy')
    return pd.concat([gold, gold_yoy], axis=1)


def load_macro_panel(
    *,
    series: list[tuple[str, str, str | None]] | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    target_index: pd.DatetimeIndex | None = None,
    add_yoy_features: bool = True,
    stooq_data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build the canonical macro feature panel.

    Loads all series in `DEFAULT_SERIES` (or `series` if overridden),
    optionally derives `m2_yoy` from `m2_level` (and similar
    `*_yoy_chg` for any series with `add_yoy_features=True`), and
    aligns to `target_index` if provided.

    `target_index` is typically a daily trading-bar `DatetimeIndex`
    (e.g. from a Stooq panel). Macro releases are forward-filled
    onto trading bars to respect publishing-lag discipline — a
    macro reading dated `t` propagates to all trading bars `≥ t`
    until a newer reading arrives. **No look-ahead**: the value at
    trading bar `T` is the latest macro reading published `≤ T`.

    `stooq_data_dir` (optional): when provided, also loads gold
    price level + 252d YoY from `GLD.US` and adds `gold_level`
    and `gold_yoy` columns to the panel. Required because FRED's
    public CSV endpoint does not expose LBMA gold price series.
    """
    series = series if series is not None else DEFAULT_SERIES
    cols: dict[str, pd.Series] = {}
    for out_name, fred_id, _transform in series:
        s = load_fred_series(fred_id, cache_dir=cache_dir, refresh=refresh)
        cols[out_name] = s

    df = pd.concat(cols.values(), axis=1)
    df.columns = list(cols.keys())

    if add_yoy_features and 'm2_level' in df.columns:
        # M2 is monthly → take 12-month YoY % change.
        m2_monthly = df['m2_level'].dropna()
        m2_yoy = m2_monthly.pct_change(periods=12) * 100.0
        df['m2_yoy'] = m2_yoy.reindex(df.index)

    if stooq_data_dir is not None:
        gold_df = load_gold_features_from_stooq(stooq_data_dir)
        df = df.join(gold_df, how='outer')

    if target_index is not None:
        # Forward-fill macro readings onto trading-bar index. Because
        # we sort by source date and then ffill, no look-ahead.
        df = df.sort_index().reindex(
            df.index.union(target_index).sort_values()
        ).ffill().reindex(target_index)

    return df


__all__ = [
    'DEFAULT_CACHE_DIR',
    'DEFAULT_SERIES',
    'fred_series_url',
    'load_fred_series',
    'load_gold_features_from_stooq',
    'load_macro_panel',
]
