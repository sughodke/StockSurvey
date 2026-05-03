"""TickerData bundle + price loader shared across apps.

`TickerData` is the per-ticker container both apps/notebook (replay
trainer) and apps/factor (cross-sectional scorer) consume. The optional
`targets` and `target_grids` dicts are unused by the scorer (it only
reads `features` + `prices` + `dates` + `valid`); the replay trainer
populates both during SSL pretraining.

`load_prices` returns one ticker's adjusted-close series from one of
three sources (Stooq archive / Kaggle Nasdaq3347 / yfinance), centralised
here so feature builders don't reach into apps/notebook for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import (
    iter_stooq_ticker_files,
    load_price_matrix,
    read_stooq_file,
    stooq_ticker_from_path,
)


DEFAULT_STOOQ_DIR = './StooqData'


@dataclass
class TickerData:
    """One ticker's loaded prices, dates, features, ground-truth indicators,
    and the warm-up-aware valid mask.

    `target_grids` carries `(n_grid, n_dates)` arrays for parameter-
    conditioned targets; empty dict when no conditioning is in use. The
    1D anchor array in `targets` is what plotting/stats compare against.
    """
    name: str
    prices: np.ndarray
    dates: np.ndarray
    features: np.ndarray
    targets: dict[str, np.ndarray]
    valid: np.ndarray
    target_grids: dict[str, np.ndarray] = field(default_factory=dict)


def _find_stooq_path(stooq_dir: Path, ticker: str,
                     include_etfs: bool = True) -> Path | None:
    """Locate one ticker's `.txt` file inside the Stooq archive layout."""
    target = ticker.upper()
    for path in iter_stooq_ticker_files(stooq_dir, include_etfs=include_etfs):
        if stooq_ticker_from_path(path) == target:
            return path
    return None


def load_prices(
    ticker: str,
    *,
    stooq_dir: str | None = None,
    kaggle_dir: str | None = None,
    use_yahoo: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Return adjusted-close series for one ticker.

    Stooq path (default): walk the archive's file tree to find the
    matching ticker file, then `read_stooq_file` parses just that
    one CSV. Stooq close is already split-/dividend-adjusted, so no
    separate `adj_close` column is needed.

    Kaggle path (`kaggle_dir`): slice one column from the wide
    Nasdaq3347 close matrix. Note: that dataset has no adjustments
    or volume; `close` is raw.

    Yahoo path (`use_yahoo=True`): on-the-fly fetch via yfinance
    (`ss_loaders.load_yahoo`) — no on-disk archive needed. Returns
    `adj_close`. Use on Colab or any environment where the Stooq
    archive isn't present.
    """
    if use_yahoo:
        import datetime as _dt

        from ss_loaders import load_yahoo

        start_dt = (_dt.datetime.fromisoformat(start) if start
                    else _dt.datetime(1990, 1, 1))
        end_dt = (_dt.datetime.fromisoformat(end) if end
                  else _dt.datetime.now())
        df = load_yahoo(start_dt, end_dt, ticker)
        if df.empty or 'adj_close' not in df.columns:
            raise KeyError(
                f'{ticker} returned no usable rows from yfinance')
        return df['adj_close'].dropna().rename('adj_close')

    if kaggle_dir:
        end_date = end or '2099-12-31'
        prices, _, _ = load_price_matrix(
            kaggle_dir, min_history=1, start_date=start, end_date=end_date)
        if ticker not in prices.columns:
            raise KeyError(f'{ticker} not in {kaggle_dir}')
        return prices[ticker].dropna().rename('close')

    root = Path(stooq_dir or DEFAULT_STOOQ_DIR)
    if not root.exists():
        raise RuntimeError(
            f'Stooq archive not found at {root}. Pass --stooq-dir or '
            '--kaggle-dir.')
    path = _find_stooq_path(root, ticker)
    if path is None:
        raise KeyError(f'{ticker} not found in {root}')
    df = read_stooq_file(path)
    if df is None or df.empty:
        raise RuntimeError(f'failed to parse {path}')
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df['close'].dropna().rename('adj_close')
