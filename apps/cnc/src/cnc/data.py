"""Panel construction for the crypto-and-carry backtest.

Builds three aligned `[date × coin]` panels:
- `funding_daily`: sum of hourly funding rates per UTC day (the gross
  yield the long-spot/short-perp leg collects between midnights).
- `vol_quote`: daily $-volume from perp candles (`close * vol_base`),
  for point-in-time universe ranking by trailing-30d $-volume.
- `close`: daily perp close price; used only for daily-return
  computation when we add price-tracking (not currently needed for the
  academic-clean carry approximation).

The universe-ranking panel is from the perp leg's volume (proxy for
basis-trade capacity at the venue). Returns coins with at least
`min_history_days` of jointly-observed funding + volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ss_loaders.hyperliquid import (
    load_hl_close_panel,
    load_hl_funding_panel,
    load_hl_perp_universe,
)


@dataclass
class CarryPanels:
    funding_daily: pd.DataFrame  # [date × coin], sum of hourly rates per day
    vol_quote: pd.DataFrame      # [date × coin], close*vol_base per day
    close: pd.DataFrame          # [date × coin], perp daily close
    coins: list[str]
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def _to_naive_utc_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a tz-aware datetime index to naive midnight-UTC dates."""
    if df.empty:
        return df
    idx = df.index
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    df = df.copy()
    df.index = pd.DatetimeIndex(idx).normalize()
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df


def build_panels(
    *,
    start_date: str = '2024-01-01',
    end_date: str | None = None,
    top_universe: int = 20,
    min_history_days: int = 180,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> CarryPanels:
    """Fetch HL panels for the top-N most-liquid current-snapshot perps.

    Universe selection here is the **current** top-N by `dayNtlVlm`,
    which is acceptable for a 2024-2026 walk-forward eval since
    Hyperliquid only launched mainnet 2023-05; the current top-N by
    volume is a reasonable approximation of the joint "had liquidity
    over the whole span" filter. Point-in-time ranking within the
    walk-forward (top-K-by-trailing-funding) happens downstream in
    `cnc.backtest`.
    """
    if end_date is None:
        end_date = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    universe_df = load_hl_perp_universe(cache_dir=cache_dir, refresh=refresh)
    universe_df = universe_df.sort_values('dayNtlVlm', ascending=False)
    candidate_coins = universe_df['coin'].head(top_universe).tolist()

    funding = load_hl_funding_panel(
        candidate_coins, start_ms, end_ms,
        resample='1D', cache_dir=cache_dir, refresh=refresh,
    )
    close = load_hl_close_panel(
        candidate_coins, start_ms, end_ms,
        interval='1d', field='close', cache_dir=cache_dir, refresh=refresh,
    )
    vol_b = load_hl_close_panel(
        candidate_coins, start_ms, end_ms,
        interval='1d', field='vol_base', cache_dir=cache_dir, refresh=refresh,
    )

    funding = _to_naive_utc_date_index(funding)
    close = _to_naive_utc_date_index(close)
    vol_b = _to_naive_utc_date_index(vol_b)

    # Intersect indices and columns.
    idx = funding.index.intersection(close.index).intersection(vol_b.index)
    cols = sorted(set(funding.columns) & set(close.columns) & set(vol_b.columns))
    funding = funding.loc[idx, cols]
    close = close.loc[idx, cols]
    vol_b = vol_b.loc[idx, cols]
    vol_quote = (close * vol_b).rename(columns=lambda c: c)

    # Per-coin history sufficiency on funding panel.
    keep = []
    for c in cols:
        nonna = funding[c].notna().sum()
        if nonna >= min_history_days:
            keep.append(c)
    funding = funding[keep]
    close = close[keep]
    vol_quote = vol_quote[keep]

    return CarryPanels(
        funding_daily=funding,
        vol_quote=vol_quote,
        close=close,
        coins=keep,
        start_date=funding.index.min(),
        end_date=funding.index.max(),
    )
