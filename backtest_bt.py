"""
backtest_bt : portfolio backtesting using the bt library.

Produces a weight matrix [n_dates x n_tickers] where each cell is the
portfolio allocation for that stock on that date. Compares ranking
strategies (RSI, scalogram, regime) head-to-head with proper portfolio
metrics (Sharpe, drawdown, CAGR).

Data source: Kaggle NASDAQ Daily CSVs (svaningelgem/nasdaq-daily-stock-prices)

Usage:
    uv run python backtest_bt.py --data-dir ./Nasdaq3347 --top-n 20
    uv run python backtest_bt.py --data-dir ./Nasdaq3347 --rankers rsi scalogram regime
    uv run python backtest_bt.py --data-dir ./Nasdaq3347 --start 2016-01-01 --end 2025-01-01
"""

import argparse
import logging
import os
import warnings

import bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Numpy-vectorized core functions (operate on full ticker matrix at once)
# ---------------------------------------------------------------------------

def rsi_matrix(price_matrix, n=7):
    """Compute RSI for all tickers simultaneously.

    price_matrix: (n_dates, n_tickers)
    Returns: (n_dates, n_tickers) RSI values
    """
    deltas = np.diff(price_matrix, axis=0)  # (n_dates-1, n_tickers)
    up = np.where(deltas > 0, deltas, 0.0)
    down = np.where(deltas < 0, -deltas, 0.0)

    n_dates, n_tickers = price_matrix.shape
    avg_up = np.zeros(n_tickers)
    avg_down = np.zeros(n_tickers)

    rsi = np.full_like(price_matrix, 50.0)

    # Seed
    avg_up = up[:n].mean(axis=0)
    avg_down = down[:n].mean(axis=0)
    rs = avg_up / (avg_down + 1e-9)
    rsi[n] = 100.0 - 100.0 / (1.0 + rs)

    # Sequential EMA update (inherently sequential across time, but vectorized across tickers)
    for i in range(n + 1, n_dates):
        avg_up = (avg_up * (n - 1) + up[i - 1]) / n
        avg_down = (avg_down * (n - 1) + down[i - 1]) / n
        rs = avg_up / (avg_down + 1e-9)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def cwt_convolve(x_matrix, scale):
    """Apply Ricker wavelet at one scale to all tickers.

    x_matrix: (n_dates, n_tickers)
    Returns: (n_dates, n_tickers) coefficients
    """
    n_dates = x_matrix.shape[0]
    points = min(10 * scale, n_dates)
    half = points // 2
    t = np.arange(-half, half + 1) / scale
    wavelet = ((1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) / np.sqrt(scale))

    # FFT-based convolution (much faster than direct for large arrays)
    from scipy.signal import fftconvolve
    # fftconvolve handles 2D: convolve each column with the 1D wavelet
    w2d = wavelet[:, None]  # (kernel_len, 1) — broadcast over tickers
    result = fftconvolve(x_matrix, w2d, mode='same', axes=0)
    return result


def cwt_matrix(price_matrix, scales):
    """CWT for all tickers at all scales.

    price_matrix: (n_dates, n_tickers)
    Returns: (n_scales, n_dates, n_tickers) coefficients
    """
    # Normalize each ticker to zero-mean unit-variance
    mu = np.mean(price_matrix, axis=0, keepdims=True)
    std = np.std(price_matrix, axis=0, keepdims=True) + 1e-9
    x = (price_matrix - mu) / std

    coeffs = np.stack([cwt_convolve(x, s) for s in scales])
    return coeffs  # (n_scales, n_dates, n_tickers)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_price_matrix(data_dir, min_history=504, start_date=None, end_date=None):
    """Load per-ticker CSVs into a single price DataFrame.

    Returns DataFrame with DatetimeIndex, columns = tickers, values = close price.
    """
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
    frames = {}

    for fname in tqdm(csv_files, desc='Loading CSVs', unit='file'):
        ticker = fname.replace('.csv', '')
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
            series = df['close'].dropna()
            if len(series) >= min_history:
                frames[ticker] = series
        except Exception:
            continue

    prices = pd.DataFrame(frames)
    prices.sort_index(inplace=True)

    if start_date:
        prices = prices.loc[start_date:]
    if end_date:
        prices = prices.loc[:end_date]

    # Drop tickers with too many NaNs in the active period
    min_valid = int(len(prices) * 0.8)
    prices = prices.dropna(axis=1, thresh=min_valid)

    # Forward-fill small gaps, drop remaining NaN rows at start
    prices = prices.ffill().dropna()

    print(f'Price matrix: {prices.shape[0]} dates x {prices.shape[1]} tickers')
    print(f'Date range: {prices.index[0].date()} to {prices.index[-1].date()}')
    return prices


# ---------------------------------------------------------------------------
# Ranking strategies -> weight matrices
# ---------------------------------------------------------------------------

def select_top_n_matrix(scores_matrix, top_n, ascending=True):
    """Convert (n_dates, n_tickers) scores into equal-weight allocation matrix.

    At each date, allocate 1/top_n to the top_n tickers.
    """
    n_dates, n_tickers = scores_matrix.shape
    weights = np.zeros_like(scores_matrix)
    w = 1.0 / top_n

    for i in range(n_dates):
        row = scores_matrix[i]
        valid = ~np.isnan(row)
        if valid.sum() < top_n:
            continue
        if ascending:
            # lowest scores get selected
            ranked = np.argsort(np.where(valid, row, np.inf))
        else:
            # highest scores get selected
            ranked = np.argsort(np.where(valid, -row, np.inf))
        weights[i, ranked[:top_n]] = w

    return weights


def weights_rsi(prices, lookback=60, n_tail=5, top_n=20):
    """RSI-based: buy the most oversold stocks (lowest RSI).

    Computes RSI for ALL tickers in one pass, then selects top_n per date.
    """
    print('  Computing RSI matrix...')
    rsi = rsi_matrix(prices.values, n=7)  # (n_dates, n_tickers)

    # Rolling trailing mean of RSI
    n_dates, n_tickers = rsi.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)
    for i in range(lookback, n_dates):
        scores[i - lookback] = np.mean(rsi[i - n_tail + 1:i + 1], axis=0)

    # Mask tickers that had NaN prices in their lookback window
    price_arr = prices.values
    for i in range(lookback, n_dates):
        chunk = price_arr[i - lookback:i + 1]
        has_nan = np.any(np.isnan(chunk), axis=0)
        scores[i - lookback, has_nan] = np.nan

    weights = select_top_n_matrix(scores, top_n, ascending=True)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_scalogram(prices, lookback=120, n_tail=10, top_n=20):
    """Scalogram-based: multi-scale momentum + direction.

    Computes CWT for ALL tickers at once, then scores from trailing edge.
    """
    scales = [5, 7, 10, 12, 21, 26]
    print('  Computing CWT matrix...')
    coeffs = cwt_matrix(prices.values, scales)  # (n_scales, n_dates, n_tickers)
    power = coeffs ** 2

    n_dates, n_tickers = prices.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)

    for i in tqdm(range(lookback, n_dates), desc='Scalogram scores', unit='day'):
        # Momentum: mean trailing power across all scales
        trailing = power[:, i - n_tail + 1:i + 1, :]  # (n_scales, n_tail, n_tickers)
        momentum = np.mean(trailing, axis=(0, 1))  # (n_tickers,)

        # Direction: mean coefficient at shortest scale (scale=5)
        direction = np.mean(coeffs[0, i - n_tail + 1:i + 1, :], axis=0)

        # Coherence: correlation between shortest and longest scale power
        short_p = power[0, i - n_tail + 1:i + 1, :]   # (n_tail, n_tickers)
        long_p = power[-1, i - n_tail + 1:i + 1, :]

        short_m = short_p.mean(axis=0, keepdims=True)
        long_m = long_p.mean(axis=0, keepdims=True)
        cov = np.mean((short_p - short_m) * (long_p - long_m), axis=0)
        short_std = np.std(short_p, axis=0)
        long_std = np.std(long_p, axis=0)
        denom = short_std * long_std + 1e-9
        coherence = np.clip(cov / denom, 0, 1)

        scores[i - lookback] = direction - momentum * coherence

    # Mask NaN tickers
    price_arr = prices.values
    for i in range(lookback, n_dates):
        has_nan = np.any(np.isnan(price_arr[i - lookback:i + 1]), axis=0)
        scores[i - lookback, has_nan] = np.nan

    weights = select_top_n_matrix(scores, top_n, ascending=True)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_regime(prices, lookback=120, n_tail=20, top_n=20):
    """Regime detection: stocks with biggest power-spectrum shift."""
    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print('  Computing CWT matrix...')
    coeffs = cwt_matrix(prices.values, scales)
    power = coeffs ** 2

    n_dates, n_tickers = prices.shape
    n_scales = len(scales)
    scores = np.full((n_dates - lookback, n_tickers), np.nan)

    for i in tqdm(range(lookback, n_dates), desc='Regime scores', unit='day'):
        # Recent vs historical power distribution per scale
        recent = np.mean(power[:, i - n_tail + 1:i + 1, :], axis=1)    # (n_scales, n_tickers)
        historical = np.mean(power[:, i - lookback:i - n_tail + 1, :], axis=1)

        # Normalize to distributions per ticker
        recent_sum = recent.sum(axis=0, keepdims=True) + 1e-9
        hist_sum = historical.sum(axis=0, keepdims=True) + 1e-9
        rd = recent / recent_sum
        hd = historical / hist_sum

        # Symmetrized KL divergence per ticker
        kl = 0.5 * np.sum(rd * np.log((rd + 1e-9) / (hd + 1e-9)), axis=0)
        kl += 0.5 * np.sum(hd * np.log((hd + 1e-9) / (rd + 1e-9)), axis=0)

        scores[i - lookback] = kl

    # Mask NaN tickers
    price_arr = prices.values
    for i in range(lookback, n_dates):
        has_nan = np.any(np.isnan(price_arr[i - lookback:i + 1]), axis=0)
        scores[i - lookback, has_nan] = np.nan

    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_equal(prices, top_n=20):
    """Benchmark: equal-weight buy-and-hold of the top_n largest stocks
    (by last available price as a crude market-cap proxy).
    """
    last_prices = prices.iloc[-1].sort_values(ascending=False)
    selected = last_prices.index[:top_n].tolist()
    w = 1.0 / top_n
    row = {t: w if t in selected else 0.0 for t in prices.columns}
    return pd.DataFrame([row] * len(prices), index=prices.index)[prices.columns]


WEIGHT_BUILDERS = {
    'rsi': weights_rsi,
    'scalogram': weights_scalogram,
    'regime': weights_regime,
    'equal': weights_equal,
}


# ---------------------------------------------------------------------------
# bt strategy builder
# ---------------------------------------------------------------------------

def build_strategy(name, prices, weight_df, rebalance_days=5):
    """Build a bt.Strategy from a weight DataFrame.

    Subsamples weight_df to rebalance every N trading days.
    """
    # Subsample weights to rebalance frequency
    rebal_weights = weight_df.iloc[::rebalance_days]

    strategy = bt.Strategy(name, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])

    return bt.Backtest(strategy, prices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Portfolio backtest of ranking strategies using bt')
    parser.add_argument('--data-dir', required=True,
                        help='Path to Kaggle NASDAQ daily CSV directory')
    parser.add_argument('--rankers', nargs='+',
                        default=['rsi', 'scalogram', 'regime', 'equal'],
                        choices=list(WEIGHT_BUILDERS.keys()),
                        help='Strategies to compare')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of stocks to hold (default: 20)')
    parser.add_argument('--rebalance', type=int, default=5,
                        help='Rebalance every N trading days (default: 5)')
    parser.add_argument('--start', default='2016-01-01',
                        help='Backtest start date')
    parser.add_argument('--end', default='2025-01-01',
                        help='Backtest end date')
    parser.add_argument('--min-history', type=int, default=504,
                        help='Min trading days of history per ticker (default: 504)')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to Output/')
    args = parser.parse_args()

    # Load data
    prices = load_price_matrix(
        args.data_dir,
        min_history=args.min_history,
        start_date=args.start,
        end_date=args.end)

    # Build weight matrices and bt strategies
    backtests = []
    for name in args.rankers:
        print(f'\nComputing weights: {name}')
        builder = WEIGHT_BUILDERS[name]
        if name == 'equal':
            weight_df = builder(prices, top_n=args.top_n)
        else:
            weight_df = builder(prices, top_n=args.top_n)

        test = build_strategy(name, prices, weight_df, args.rebalance)
        backtests.append(test)

    # Run all backtests
    print('\nRunning backtests...')
    result = bt.run(*backtests)

    # Display results
    result.display()

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    result.plot(ax=axes[0])
    axes[0].set_title('Equity Curves')

    result.plot_weights(backtest=0, ax=axes[1])
    axes[1].set_title(f'Weight Allocation ({args.rankers[0]})')

    fig.tight_layout()

    if args.save:
        fname = 'Output/backtest-bt-comparison.png'
        fig.savefig(fname, dpi=150)
        print(f'\nSaved {fname}')

        # Also save the stats
        stats_fname = 'Output/backtest-bt-stats.txt'
        with open(stats_fname, 'w') as f:
            f.write(str(result.stats))
        print(f'Saved {stats_fname}')

        plt.close(fig)
    else:
        plt.show()
