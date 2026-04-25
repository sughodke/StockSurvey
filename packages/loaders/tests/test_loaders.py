"""Tests for ss_loaders: CSV matrix loader + symbol-list constants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import (
    MY_FAVES,
    NDX_CONSTITUENTS,
    load_price_matrix,
    load_stooq_matrix,
)


def _write_synthetic_csvs(tmp: Path, n_tickers: int = 5, n_days: int = 600) -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    for i in range(n_tickers):
        prices = np.cumsum(rng.standard_normal(n_days)) + 100
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices + rng.uniform(0.1, 0.5, n_days),
            'low': prices - rng.uniform(0.1, 0.5, n_days),
            'close': prices + rng.standard_normal(n_days) * 0.1,
        })
        df.to_csv(tmp / f'TICK{i}.csv', index=False)


def test_load_price_matrix_synthetic(tmp_path):
    _write_synthetic_csvs(tmp_path, n_tickers=5, n_days=600)
    prices, highs, lows = load_price_matrix(str(tmp_path), min_history=504)
    assert prices.shape == (600, 5)
    assert prices.shape == highs.shape == lows.shape
    assert list(prices.columns) == sorted(['TICK0', 'TICK1', 'TICK2', 'TICK3', 'TICK4'])
    # Index alignment
    assert prices.index.equals(highs.index)
    assert prices.index.equals(lows.index)
    # No NaN after ffill+dropna
    assert not prices.isna().any().any()


def test_load_price_matrix_drops_short_tickers(tmp_path):
    _write_synthetic_csvs(tmp_path, n_tickers=4, n_days=600)
    # Add one short ticker that should be dropped
    short = pd.DataFrame({
        'date': pd.bdate_range('2020-01-01', periods=100),
        'open': np.arange(100), 'high': np.arange(100), 'low': np.arange(100),
        'close': np.arange(100),
    })
    short.to_csv(tmp_path / 'SHORT.csv', index=False)
    prices, _, _ = load_price_matrix(str(tmp_path), min_history=504)
    assert 'SHORT' not in prices.columns
    assert len(prices.columns) == 4


def test_load_price_matrix_date_filter(tmp_path):
    _write_synthetic_csvs(tmp_path, n_tickers=3, n_days=600)
    prices, _, _ = load_price_matrix(
        str(tmp_path), min_history=504,
        start_date='2021-01-01', end_date='2021-12-31')
    assert prices.index.min() >= pd.Timestamp('2021-01-01')
    assert prices.index.max() <= pd.Timestamp('2021-12-31')


def test_symbol_constants_present():
    assert isinstance(NDX_CONSTITUENTS, list)
    assert 'AAPL' in NDX_CONSTITUENTS
    assert 'MSFT' in NDX_CONSTITUENTS
    assert isinstance(MY_FAVES, list)
    assert len(MY_FAVES) > 0


def _write_synthetic_stooq(
    tmp: Path, *, tickers: list[str], section: str = 'nasdaq stocks',
    n_days: int = 600, include_etfs: bool = False,
) -> None:
    """Stub the Stooq archive layout: daily/<country>/<section>/<bucket>/*.us.txt.

    Each ticker file uses the angle-bracketed header Stooq actually
    ships, so the loader's column rename and date-format parsing are
    exercised end-to-end.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    bucket = tmp / 'daily' / 'us' / section / '1'
    bucket.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        prices = np.cumsum(rng.standard_normal(n_days)) + 100
        df = pd.DataFrame({
            '<TICKER>': f'{ticker}.US',
            '<PER>': 'D',
            '<DATE>': dates.strftime('%Y%m%d').astype(int),
            '<TIME>': '000000',
            '<OPEN>': prices,
            '<HIGH>': prices + rng.uniform(0.1, 0.5, n_days),
            '<LOW>': prices - rng.uniform(0.1, 0.5, n_days),
            '<CLOSE>': prices + rng.standard_normal(n_days) * 0.1,
            '<VOL>': rng.integers(1_000_000, 10_000_000, n_days),
            '<OPENINT>': 0,
        })
        df.to_csv(bucket / f'{ticker.lower()}.us.txt', index=False)
    if include_etfs:
        etf_bucket = tmp / 'daily' / 'us' / 'nasdaq etfs' / '1'
        etf_bucket.mkdir(parents=True, exist_ok=True)
        prices = np.cumsum(rng.standard_normal(n_days)) + 100
        df = pd.DataFrame({
            '<TICKER>': 'SPY.US', '<PER>': 'D',
            '<DATE>': dates.strftime('%Y%m%d').astype(int),
            '<TIME>': '000000',
            '<OPEN>': prices, '<HIGH>': prices + 0.5, '<LOW>': prices - 0.5,
            '<CLOSE>': prices, '<VOL>': 50_000_000, '<OPENINT>': 0,
        })
        df.to_csv(etf_bucket / 'spy.us.txt', index=False)


def test_load_stooq_matrix_basic(tmp_path):
    """Loader pivots the Stooq long-form into 4 aligned wide DataFrames,
    including the volume panel that the Kaggle CSV loader can't provide."""
    _write_synthetic_stooq(tmp_path, tickers=['AAPL', 'MSFT', 'NVDA'])
    close, high, low, vol = load_stooq_matrix(str(tmp_path), min_history=504)
    assert close.shape == (600, 3)
    assert close.shape == high.shape == low.shape == vol.shape
    assert list(close.columns) == ['AAPL', 'MSFT', 'NVDA']
    assert close.index.equals(vol.index)
    # Volume is a real panel, not a constant placeholder
    assert (vol.values > 0).all()


def test_load_stooq_matrix_drops_short_history(tmp_path):
    """Same min_history semantic as the CSV loader: tickers with too few
    rows fall out before the panel is returned."""
    _write_synthetic_stooq(tmp_path, tickers=['AAPL', 'MSFT'], n_days=600)
    # Inject a too-short ticker into the same bucket
    bucket = tmp_path / 'daily' / 'us' / 'nasdaq stocks' / '1'
    short_dates = pd.bdate_range('2020-01-01', periods=100)
    pd.DataFrame({
        '<TICKER>': 'NEW.US', '<PER>': 'D',
        '<DATE>': short_dates.strftime('%Y%m%d').astype(int),
        '<TIME>': '000000',
        '<OPEN>': 1, '<HIGH>': 1, '<LOW>': 1, '<CLOSE>': 1,
        '<VOL>': 100, '<OPENINT>': 0,
    }).to_csv(bucket / 'new.us.txt', index=False)

    close, _, _, _ = load_stooq_matrix(str(tmp_path), min_history=504)
    assert 'NEW' not in close.columns
    assert set(close.columns) == {'AAPL', 'MSFT'}


def test_load_stooq_matrix_date_filter(tmp_path):
    _write_synthetic_stooq(tmp_path, tickers=['AAPL'], n_days=600)
    close, _, _, _ = load_stooq_matrix(
        str(tmp_path), min_history=100,
        start_date='2021-01-01', end_date='2021-12-31')
    assert close.index.min() >= pd.Timestamp('2021-01-01')
    assert close.index.max() <= pd.Timestamp('2021-12-31')


def test_load_stooq_matrix_excludes_etfs_by_default(tmp_path):
    """Stooq layout separates `<exchange> stocks` from `<exchange> etfs`;
    the regime app targets equities, so the loader skips ETFs unless
    asked. Confirms the section-name filter works both ways."""
    _write_synthetic_stooq(
        tmp_path, tickers=['AAPL'], n_days=600, include_etfs=True)
    close_default, _, _, _ = load_stooq_matrix(str(tmp_path), min_history=504)
    assert 'SPY' not in close_default.columns

    close_etfs, _, _, _ = load_stooq_matrix(
        str(tmp_path), min_history=504, include_etfs=True)
    assert 'SPY' in close_etfs.columns


def test_load_stooq_matrix_cache_round_trip(tmp_path):
    """Second call with a cache file present should load from disk
    without re-walking the archive (test by deleting source files
    after the first call and reading from cache)."""
    _write_synthetic_stooq(tmp_path, tickers=['AAPL', 'MSFT'])
    cache = tmp_path / 'cache.pkl'
    close1, _, _, _ = load_stooq_matrix(
        str(tmp_path), min_history=504, cache_path=str(cache))
    assert cache.exists()

    # Wipe the source files; cache must still satisfy the request.
    import shutil
    shutil.rmtree(tmp_path / 'daily')
    close2, _, _, _ = load_stooq_matrix(
        str(tmp_path), min_history=504, cache_path=str(cache))
    pd.testing.assert_frame_equal(close1, close2)


def test_load_stooq_matrix_raises_on_empty_dir(tmp_path):
    """No matching ticker files → loud failure rather than empty panel,
    so a typo in --data-dir doesn't silently train on nothing."""
    import pytest
    with pytest.raises(RuntimeError, match='no ticker files'):
        load_stooq_matrix(str(tmp_path))


def test_load_stooq_matrix_keeps_late_ipos(tmp_path):
    """A ticker that started trading partway through the date range
    (i.e., has leading NaN in the panel) must be kept — that's the
    survivorship-bias fix. The trainer's per-window filter handles
    the cumsum-leading-NaN issue downstream; the loader no longer
    drops these columns."""
    bucket = tmp_path / 'daily' / 'us' / 'nasdaq stocks' / '1'
    bucket.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # AAPL trades the whole window
    full_dates = pd.bdate_range('2020-01-01', periods=600)
    pd.DataFrame({
        '<TICKER>': 'AAPL.US', '<PER>': 'D',
        '<DATE>': full_dates.strftime('%Y%m%d').astype(int),
        '<TIME>': '000000',
        '<OPEN>': 100, '<HIGH>': 101, '<LOW>': 99, '<CLOSE>': 100,
        '<VOL>': 1_000_000, '<OPENINT>': 0,
    }).to_csv(bucket / 'aapl.us.txt', index=False)

    # NEW.US "IPOs" 200 bars in. Has leading NaN in the panel.
    late_dates = full_dates[200:]
    prices = np.cumsum(rng.standard_normal(len(late_dates))) + 100
    pd.DataFrame({
        '<TICKER>': 'NEW.US', '<PER>': 'D',
        '<DATE>': late_dates.strftime('%Y%m%d').astype(int),
        '<TIME>': '000000',
        '<OPEN>': prices, '<HIGH>': prices + 0.5, '<LOW>': prices - 0.5,
        '<CLOSE>': prices, '<VOL>': 500_000, '<OPENINT>': 0,
    }).to_csv(bucket / 'new.us.txt', index=False)

    closes, _, _, _ = load_stooq_matrix(str(tmp_path), min_history=252)
    # Both tickers survive: NEW has 400 bars in the window (>252 floor).
    assert set(closes.columns) == {'AAPL', 'NEW'}
    # NEW has leading NaN where it wasn't trading yet — that's the
    # point-in-time data we want to preserve.
    assert closes['NEW'].iloc[0:100].isna().all()
    assert closes['NEW'].iloc[300:].notna().all()


def test_load_stooq_matrix_keeps_delisted(tmp_path):
    """A ticker that delisted partway through the date range (trailing
    NaN) must be kept too — for the same survivorship-bias reason.
    `ffill(limit=5)` lets a 1-week halt extend the last value, then
    NaN takes over."""
    bucket = tmp_path / 'daily' / 'us' / 'nasdaq stocks' / '1'
    bucket.mkdir(parents=True, exist_ok=True)

    full_dates = pd.bdate_range('2020-01-01', periods=600)
    pd.DataFrame({
        '<TICKER>': 'SURV.US', '<PER>': 'D',
        '<DATE>': full_dates.strftime('%Y%m%d').astype(int),
        '<TIME>': '000000',
        '<OPEN>': 100, '<HIGH>': 101, '<LOW>': 99, '<CLOSE>': 100,
        '<VOL>': 1_000_000, '<OPENINT>': 0,
    }).to_csv(bucket / 'surv.us.txt', index=False)

    # DEAD delists after 300 bars
    early_dates = full_dates[:300]
    pd.DataFrame({
        '<TICKER>': 'DEAD.US', '<PER>': 'D',
        '<DATE>': early_dates.strftime('%Y%m%d').astype(int),
        '<TIME>': '000000',
        '<OPEN>': 50, '<HIGH>': 51, '<LOW>': 49, '<CLOSE>': 50,
        '<VOL>': 200_000, '<OPENINT>': 0,
    }).to_csv(bucket / 'dead.us.txt', index=False)

    closes, _, _, _ = load_stooq_matrix(str(tmp_path), min_history=252)
    assert 'DEAD' in closes.columns
    # First 300 bars valid, then ffill carries 5 more, then NaN.
    assert closes['DEAD'].iloc[:300].notna().all()
    assert closes['DEAD'].iloc[306:].isna().all()
