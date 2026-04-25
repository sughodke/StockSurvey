"""Wavelet scalogram that encodes all indicator behaviors of a stock in
a single image.

Each row = a timescale. Each column = a point in time.
Read horizontally to see how a specific timescale evolves.
Read vertically to see all timescales active at a moment.

Modernized from `apps/v1/src/v1/scripts/plot_scalogram.py`:

  * CWT comes from `ss_wavelets.causal_cwt` (Ricker, strictly causal,
    rolling z-norm). The legacy hand-rolled morlet/ricker convolution
    is gone. Causal output means the heatmap can be read as an EDA
    proxy for what the regime trainer actually sees.
  * RSI / EMA / MACD / rolling-std come from `ss_indicators` (JAX).
  * Single-ticker price data is fetched via the Stooq loader (split-/
    dividend-adjusted close, includes delistings). `iter_stooq_ticker_files`
    walks the archive once and `read_stooq_file` parses just the one
    ticker — no full-archive `load_stooq_matrix` scan. Pass
    `--kaggle-dir DIR` to slice one column out of the Kaggle Nasdaq3347
    wide matrix instead.
  * The main heatmap panel uses `ss_plotting.plot_scalogram_heatmap`;
    the composite layout (price strip + heatmap + RSI/MACD/BBands
    comparison strips) is rebuilt locally.

Usage:
    uv run python -m ss_notebook.scalogram TSLA
    uv run python -m ss_notebook.scalogram --save AAPL NVDA MSFT
    uv run python -m ss_notebook.scalogram --stooq-dir ./StooqData TSLA
    uv run python -m ss_notebook.scalogram --kaggle-dir ./Nasdaq3347 NVDA
    uv run python -m ss_notebook.scalogram --start 2016-06-01 --end 2019-02-01 TSLA
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_indicators import ema, macd, rolling_std, rsi
from ss_loaders import (
    iter_stooq_ticker_files,
    load_price_matrix,
    read_stooq_file,
    stooq_ticker_from_path,
)
from ss_plotting import plot_scalogram_heatmap
from ss_wavelets import causal_cwt

DEFAULT_STOOQ_DIR = './StooqData'


def _to_np(x) -> np.ndarray:
    return np.asarray(x)


def _scalogram_scales(min_scale: int = 2, max_scale: int = 200,
                      n: int = 120) -> np.ndarray:
    """Dense log-spaced integer scale grid for visual continuity.

    The regime trainer uses the sparse `ss_wavelets.ALL_SCALES` (13
    points). For a heatmap we want a denser grid — this matches the
    legacy v1 plot's 120-point log scan from 2 to 200 trading days.
    """
    raw = np.logspace(np.log10(min_scale), np.log10(max_scale), n).astype(int)
    return np.unique(raw)


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
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Return adjusted-close series for one ticker.

    Stooq path (default): walk the archive's file tree to find the
    matching ticker file, then `read_stooq_file` parses just that
    one CSV. Stooq close is already split-/dividend-adjusted, so no
    separate `adj_close` column is needed.

    Kaggle path (`--kaggle-dir`): slice one column from the wide
    Nasdaq3347 close matrix. Note: that dataset has no adjustments
    or volume; `close` is raw.
    """
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


def compute_scalogram_power(
    prices: np.ndarray,
    scales: np.ndarray,
    lookback: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """`(power, coeffs)` shaped `(n_scales, n_dates)`.

    `causal_cwt` expects `(T, N)` input — we add a singleton ticker
    axis and squeeze it back out.
    """
    px = prices.astype(np.float32).reshape(-1, 1)
    coeffs_3d = causal_cwt(px, list(map(int, scales)), lookback=lookback)
    coeffs = coeffs_3d[:, :, 0]
    return coeffs ** 2, coeffs


def compute_indicators(prices: np.ndarray) -> dict[str, np.ndarray]:
    rsi_7 = _to_np(rsi(prices, n=7))
    rsi_ma10 = _to_np(ema(rsi_7, span=10))
    macd_line, _, _ = macd(prices, fast=12, slow=26, signal=9)
    sigma_21 = _to_np(rolling_std(prices, window=21))
    return {
        'rsi': rsi_7,
        'rsi_ma10': rsi_ma10,
        'macd': _to_np(macd_line),
        'sigma_21': sigma_21,
    }


def plot_scalogram(
    ticker: str,
    prices: np.ndarray,
    dates: np.ndarray,
    *,
    lookback: int = 252,
) -> plt.Figure:
    """Composite figure: price + scalogram heatmap + 3 comparison strips."""
    scales = _scalogram_scales()
    power, coeffs = compute_scalogram_power(prices, scales, lookback=lookback)
    indicators = compute_indicators(prices)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'{ticker} — Wavelet Scalogram (causal Ricker)',
                 fontsize=14, fontweight='bold', y=0.98)

    ax_price = fig.add_axes([0.10, 0.87, 0.78, 0.08])
    ax_price.plot(dates, prices, color='black', linewidth=0.7)
    ax_price.set_ylabel('Price', fontsize=9)
    ax_price.set_title(f'Adj Close ({len(prices)} points)', fontsize=10)
    ax_price.tick_params(labelbottom=False)
    ax_price.set_xlim(dates[0], dates[-1])

    ax_sg = fig.add_axes([0.10, 0.42, 0.78, 0.42])
    plot_scalogram_heatmap(
        power, scales, dates,
        title='', annotate_indicators=True, ax=ax_sg)
    ax_sg.set_title('')
    ax_sg.tick_params(labelbottom=False)

    gs = fig.add_gridspec(3, 1, left=0.10, right=0.88,
                          bottom=0.04, top=0.38, hspace=0.35)

    ax_rsi = fig.add_subplot(gs[0])
    idx_7 = int(np.argmin(np.abs(scales - 7)))
    rsi_proxy = power[idx_7]
    rsi_proxy = rsi_proxy / (np.max(rsi_proxy) + 1e-9) * 100.0
    ax_rsi.plot(dates, indicators['rsi'], color='darkgoldenrod',
                linewidth=0.8, label='RSI(7)')
    ax_rsi.plot(dates, indicators['rsi_ma10'], color='steelblue',
                linewidth=0.8, alpha=0.7, label='RSI EMA(10)')
    ax_rsi2 = ax_rsi.twinx()
    ax_rsi2.fill_between(dates, rsi_proxy, alpha=0.15, color='red',
                         label='Scale=7 power')
    ax_rsi2.set_ylabel('Power', fontsize=8, color='red')
    ax_rsi2.tick_params(labelcolor='red', labelsize=7)
    ax_rsi.axhline(70, color='gray', linestyle=':', alpha=0.4)
    ax_rsi.axhline(30, color='gray', linestyle=':', alpha=0.4)
    ax_rsi.set_ylabel('RSI', fontsize=9)
    ax_rsi.set_title('Scale=7 power vs actual RSI(7)', fontsize=9)
    ax_rsi.legend(loc='upper left', fontsize=7)
    ax_rsi.tick_params(labelbottom=False)
    ax_rsi.set_xlim(dates[0], dates[-1])

    ax_macd = fig.add_subplot(gs[1])
    idx_12 = int(np.argmin(np.abs(scales - 12)))
    idx_26 = int(np.argmin(np.abs(scales - 26)))
    macd_proxy = coeffs[idx_12] - coeffs[idx_26]
    macd_norm = indicators['macd'] / (np.max(np.abs(indicators['macd'])) + 1e-9)
    proxy_norm = macd_proxy / (np.max(np.abs(macd_proxy)) + 1e-9)
    corr_macd = float(np.corrcoef(macd_norm, proxy_norm)[0, 1])
    ax_macd.plot(dates, macd_norm, color='darkgoldenrod',
                 linewidth=0.8, label='MACD(12,26)')
    ax_macd.plot(dates, proxy_norm, color='steelblue',
                 linewidth=0.8, alpha=0.7, label='Scale 12−26 diff')
    ax_macd.axhline(0, color='gray', linewidth=0.5)
    ax_macd.set_ylabel('Normalized', fontsize=9)
    ax_macd.set_title(
        f'Scale(12)−Scale(26) vs actual MACD  (r={corr_macd:.3f})',
        fontsize=9)
    ax_macd.legend(loc='upper left', fontsize=7)
    ax_macd.tick_params(labelbottom=False)
    ax_macd.set_xlim(dates[0], dates[-1])

    ax_vol = fig.add_subplot(gs[2])
    idx_21 = int(np.argmin(np.abs(scales - 21)))
    sg_vol = power[idx_21]
    sg_vol_norm = sg_vol / (np.max(sg_vol) + 1e-9)
    sigma_norm = indicators['sigma_21'] / (np.max(indicators['sigma_21']) + 1e-9)
    corr_vol = float(np.corrcoef(sigma_norm, sg_vol_norm)[0, 1])
    ax_vol.plot(dates, sigma_norm, color='darkgoldenrod',
                linewidth=0.8, label='Rolling σ(21)')
    ax_vol.plot(dates, sg_vol_norm, color='steelblue',
                linewidth=0.8, alpha=0.7, label='Scale=21 power')
    ax_vol.set_ylabel('Normalized', fontsize=9)
    ax_vol.set_title(
        f'Scale=21 power vs actual BBands width  (r={corr_vol:.3f})',
        fontsize=9)
    ax_vol.legend(loc='upper left', fontsize=7)
    ax_vol.set_xlim(dates[0], dates[-1])
    for label in ax_vol.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Wavelet scalogram with indicator-comparison strips.')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols')
    parser.add_argument('--stooq-dir', default=None,
                        help=f'Stooq archive root (contains daily/). '
                             f'Default: {DEFAULT_STOOQ_DIR}.')
    parser.add_argument('--kaggle-dir', default=None,
                        help='Slice one column from a Nasdaq3347-style CSV '
                             'matrix instead of using Stooq.')
    parser.add_argument('--start', default=None, help='YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='YYYY-MM-DD')
    parser.add_argument('--lookback', type=int, default=252,
                        help='Causal z-norm window for the CWT (default 252).')
    parser.add_argument('--save', action='store_true',
                        help='Save to Output/ instead of showing.')
    parser.add_argument('--output-dir', default='Output')
    args = parser.parse_args()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    for ticker in args.tickers:
        try:
            series = load_prices(
                ticker,
                stooq_dir=args.stooq_dir,
                kaggle_dir=args.kaggle_dir,
                start=args.start, end=args.end,
            )
        except (KeyError, RuntimeError) as exc:
            print(f'Skipping {ticker}: {exc}')
            continue

        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        fig = plot_scalogram(ticker, prices, dates, lookback=args.lookback)

        if args.save:
            fname = os.path.join(args.output_dir, f'{ticker}-scalogram.png')
            fig.savefig(fname, dpi=150)
            print(f'Saved {fname}')
            plt.close(fig)

    if not args.save:
        plt.show()


if __name__ == '__main__':
    main()
