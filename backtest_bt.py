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


def cwt_causal_convolve(x_matrix, scale):
    """Apply CAUSAL Ricker wavelet at one scale to all tickers.

    The kernel only looks backward in time: at index i, the convolution
    uses data from indices [i - kernel_len + 1, ..., i]. No future data.

    x_matrix: (n_dates, n_tickers)
    Returns: (n_dates, n_tickers) coefficients
    """
    from scipy.signal import fftconvolve

    n_dates = x_matrix.shape[0]
    points = min(10 * scale, n_dates)

    # Build one-sided (causal) Ricker wavelet: t in [-points, 0]
    t = np.arange(-points, 1) / scale
    wavelet = ((1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) / np.sqrt(scale))

    # fftconvolve with mode='full', then take the last n_dates values
    # This ensures output[i] only depends on input[:i+1]
    w2d = wavelet[:, None]  # (kernel_len, 1)
    full = fftconvolve(x_matrix, w2d, mode='full', axes=0)

    # full has length n_dates + kernel_len - 1; take the tail
    result = full[len(wavelet) - 1:len(wavelet) - 1 + n_dates]
    return result


def cwt_causal_matrix(price_matrix, scales, lookback):
    """Causal CWT for all tickers at all scales.

    Normalizes each ticker using a ROLLING window (lookback) mean/std,
    so normalization at time t only uses data up to t.

    price_matrix: (n_dates, n_tickers)
    Returns: (n_scales, n_dates, n_tickers) coefficients
    """
    n_dates, n_tickers = price_matrix.shape

    # Rolling mean and std (causal — only past data)
    df = pd.DataFrame(price_matrix)
    rolling_mu = df.rolling(lookback, min_periods=1).mean().values
    rolling_std = df.rolling(lookback, min_periods=1).std().values
    # First few rows have std=NaN from rolling; fill with 1 (no normalization)
    rolling_std = np.nan_to_num(rolling_std, nan=1.0, copy=True)
    rolling_std = np.where(rolling_std < 1e-9, 1e-9, rolling_std)

    x = (price_matrix - rolling_mu) / rolling_std

    coeffs = np.stack([cwt_causal_convolve(x, s) for s in scales])
    return coeffs  # (n_scales, n_dates, n_tickers)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_price_matrix(data_dir, min_history=504, start_date=None, end_date=None):
    """Load per-ticker CSVs into close, high, low DataFrames.

    Returns (prices, highs, lows) — all DataFrames with same shape/index.
    """
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
    close_frames = {}
    high_frames = {}
    low_frames = {}

    for fname in tqdm(csv_files, desc='Loading CSVs', unit='file'):
        ticker = fname.replace('.csv', '')
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
            if len(df['close'].dropna()) >= min_history:
                close_frames[ticker] = df['close']
                high_frames[ticker] = df['high']
                low_frames[ticker] = df['low']
        except Exception:
            continue

    prices = pd.DataFrame(close_frames)
    highs = pd.DataFrame(high_frames)
    lows = pd.DataFrame(low_frames)

    for df in [prices, highs, lows]:
        df.sort_index(inplace=True)

    if start_date:
        prices = prices.loc[start_date:]
        highs = highs.loc[start_date:]
        lows = lows.loc[start_date:]
    if end_date:
        prices = prices.loc[:end_date]
        highs = highs.loc[:end_date]
        lows = lows.loc[:end_date]

    # Keep same tickers across all three
    min_valid = int(len(prices) * 0.8)
    prices = prices.dropna(axis=1, thresh=min_valid)
    common = prices.columns
    highs = highs[common]
    lows = lows[common]

    prices = prices.ffill().dropna()
    highs = highs.ffill().dropna()
    lows = lows.ffill().dropna()

    # Align indices
    common_idx = prices.index.intersection(highs.index).intersection(lows.index)
    prices = prices.loc[common_idx]
    highs = highs.loc[common_idx]
    lows = lows.loc[common_idx]

    print(f'Price matrix: {prices.shape[0]} dates x {prices.shape[1]} tickers')
    print(f'Date range: {prices.index[0].date()} to {prices.index[-1].date()}')
    return prices, highs, lows


# ---------------------------------------------------------------------------
# Corwin-Schultz spread estimator
# ---------------------------------------------------------------------------

def corwin_schultz_spread(highs, lows, window=21):
    """Estimate bid-ask spread from OHLC using Corwin & Schultz (2012).

    Uses the ratio of 2-day vs 1-day high-low ranges to separate
    volatility (scales with sqrt(t)) from spread (constant).

    highs, lows: DataFrames (n_dates, n_tickers)
    window: rolling window for smoothing the estimate
    Returns: DataFrame of estimated spread as a fraction of price (0.01 = 1%)
    """
    # Log high-low ratio squared for single days
    log_hl = np.log(highs / lows)
    beta = log_hl ** 2

    # 2-day high-low: max of consecutive highs, min of consecutive lows
    high_2d = highs.rolling(2).max()
    low_2d = lows.rolling(2).min()
    log_hl_2d = np.log(high_2d / low_2d)
    gamma = log_hl_2d ** 2

    # Sum of single-day betas over 2 consecutive days
    beta_sum = beta + beta.shift(1)

    # Corwin-Schultz alpha
    # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    sqrt2 = np.sqrt(2)
    denom = 3 - 2 * sqrt2

    term1 = (np.sqrt(2) * np.sqrt(beta_sum) - np.sqrt(beta_sum)) / denom
    term2 = np.sqrt(gamma / denom)

    alpha = term1 - term2

    # Spread = 2(e^alpha - 1) / (1 + e^alpha)
    # Clamp alpha to avoid overflow and negative spreads
    alpha = alpha.clip(lower=0)
    exp_alpha = np.exp(alpha)
    spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)

    # Smooth with rolling mean
    spread = spread.rolling(window, min_periods=1).mean()

    # Clamp to reasonable range [0, 0.20] (0-20%)
    spread = spread.clip(lower=0, upper=0.20)

    return spread


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


def _apply_spread_mask(scores, spread_arr, lookback, max_spread):
    """NaN out scores for tickers with estimated spread > max_spread."""
    if spread_arr is None:
        return scores
    n_dates_scores = scores.shape[0]
    for i in range(n_dates_scores):
        spread_row = spread_arr[i + lookback]
        illiquid = spread_row > max_spread
        scores[i, illiquid] = np.nan
    return scores


def _apply_nan_mask(scores, price_arr, lookback):
    """NaN out scores for tickers with missing prices in lookback window."""
    n_dates = price_arr.shape[0]
    for i in range(lookback, n_dates):
        chunk = price_arr[i - lookback:i + 1]
        has_nan = np.any(np.isnan(chunk), axis=0)
        scores[i - lookback, has_nan] = np.nan
    return scores


def weights_rsi(prices, lookback=60, n_tail=5, top_n=20,
                spread_df=None, max_spread=0.02):
    """RSI-based: buy the most oversold stocks (lowest RSI)."""
    print('  Computing RSI matrix...')
    rsi = rsi_matrix(prices.values, n=7)

    n_dates, n_tickers = rsi.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)
    for i in range(lookback, n_dates):
        scores[i - lookback] = np.mean(rsi[i - n_tail + 1:i + 1], axis=0)

    scores = _apply_nan_mask(scores, prices.values, lookback)
    if spread_df is not None:
        scores = _apply_spread_mask(scores, spread_df.values, lookback, max_spread)

    weights = select_top_n_matrix(scores, top_n, ascending=True)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_scalogram(prices, lookback=120, n_tail=10, top_n=20,
                      spread_df=None, max_spread=0.02):
    """Scalogram-based: multi-scale momentum + direction."""
    scales = [5, 7, 10, 12, 21, 26]
    print('  Computing causal CWT matrix...')
    coeffs = cwt_causal_matrix(prices.values, scales, lookback)
    power = coeffs ** 2

    n_dates, n_tickers = prices.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)

    for i in tqdm(range(lookback, n_dates), desc='Scalogram scores', unit='day'):
        trailing = power[:, i - n_tail + 1:i + 1, :]
        momentum = np.mean(trailing, axis=(0, 1))
        direction = np.mean(coeffs[0, i - n_tail + 1:i + 1, :], axis=0)

        short_p = power[0, i - n_tail + 1:i + 1, :]
        long_p = power[-1, i - n_tail + 1:i + 1, :]
        short_m = short_p.mean(axis=0, keepdims=True)
        long_m = long_p.mean(axis=0, keepdims=True)
        cov = np.mean((short_p - short_m) * (long_p - long_m), axis=0)
        short_std = np.std(short_p, axis=0)
        long_std = np.std(long_p, axis=0)
        denom = short_std * long_std + 1e-9
        coherence = np.clip(cov / denom, 0, 1)

        scores[i - lookback] = direction - momentum * coherence

    scores = _apply_nan_mask(scores, prices.values, lookback)
    if spread_df is not None:
        scores = _apply_spread_mask(scores, spread_df.values, lookback, max_spread)

    weights = select_top_n_matrix(scores, top_n, ascending=True)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_regime(prices, lookback=120, n_tail=20, top_n=20,
                   spread_df=None, max_spread=0.02):
    """Regime detection: stocks with biggest power-spectrum shift."""
    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print('  Computing causal CWT matrix...')
    coeffs = cwt_causal_matrix(prices.values, scales, lookback)
    power = coeffs ** 2

    n_dates, n_tickers = prices.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)

    for i in tqdm(range(lookback, n_dates), desc='Regime scores', unit='day'):
        recent = np.mean(power[:, i - n_tail + 1:i + 1, :], axis=1)
        historical = np.mean(power[:, i - lookback:i - n_tail + 1, :], axis=1)

        recent_sum = recent.sum(axis=0, keepdims=True) + 1e-9
        hist_sum = historical.sum(axis=0, keepdims=True) + 1e-9
        rd = recent / recent_sum
        hd = historical / hist_sum

        kl = 0.5 * np.sum(rd * np.log((rd + 1e-9) / (hd + 1e-9)), axis=0)
        kl += 0.5 * np.sum(hd * np.log((hd + 1e-9) / (rd + 1e-9)), axis=0)
        scores[i - lookback] = kl

    scores = _apply_nan_mask(scores, prices.values, lookback)
    if spread_df is not None:
        scores = _apply_spread_mask(scores, spread_df.values, lookback, max_spread)

    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_equal(prices, top_n=20, spread_df=None, max_spread=0.02):
    """Benchmark: equal-weight buy-and-hold of the top_n largest stocks."""
    last_prices = prices.iloc[-1].sort_values(ascending=False)

    if spread_df is not None:
        last_spread = spread_df.iloc[-1]
        liquid = last_spread[last_spread <= max_spread].index
        last_prices = last_prices[last_prices.index.isin(liquid)]

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

def print_rebalance_events(weight_df, name, rebalance_days):
    """Print each rebalance event: date, holdings, and changes."""
    rebal_weights = weight_df.iloc[::rebalance_days]
    prev_holdings = set()

    for date, row in rebal_weights.iterrows():
        held = row[row > 0].sort_values(ascending=False)
        current = set(held.index)

        added = current - prev_holdings
        removed = prev_holdings - current

        tickers_str = ', '.join(f'{t} ({w:.0%})' for t, w in held.items())
        changes = []
        if added:
            changes.append(f'+{",".join(sorted(added))}')
        if removed:
            changes.append(f'-{",".join(sorted(removed))}')
        change_str = f'  [{" | ".join(changes)}]' if changes else ''

        print(f'  [{name}] {date.date()}  {tickers_str}{change_str}')
        prev_holdings = current


def make_commission_fn(spread_bps=10):
    """Create a commission function.

    Uses a flat spread cost in basis points. When per-ticker spread
    estimates are available, use make_spread_commission_fn instead.
    """
    frac = spread_bps / 10000.0

    def commission(q, p):
        return abs(q) * p * frac

    return commission


def build_strategy(name, prices, weight_df, rebalance_days=5, verbose=False,
                   commission_bps=10):
    """Build a bt.Strategy from a weight DataFrame.

    Subsamples weight_df to rebalance every N trading days.
    """
    rebal_weights = weight_df.iloc[::rebalance_days]

    if verbose:
        print_rebalance_events(weight_df, name, rebalance_days)

    strategy = bt.Strategy(name, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])

    return bt.Backtest(strategy, prices,
                       commissions=make_commission_fn(commission_bps))


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
    parser.add_argument('--start', default='2000-01-01',
                        help='Backtest start date')
    parser.add_argument('--end', default='2025-12-31',
                        help='Backtest end date')
    parser.add_argument('--min-history', type=int, default=504,
                        help='Min trading days of history per ticker (default: 504)')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to Output/')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each rebalance event')
    parser.add_argument('--commission-bps', type=int, default=10,
                        help='Transaction cost in basis points per trade (default: 10)')
    parser.add_argument('--max-spread', type=float, default=0.02,
                        help='Max Corwin-Schultz spread fraction to include (default: 0.02 = 2%%)')
    args = parser.parse_args()

    # Load data
    prices, highs, lows = load_price_matrix(
        args.data_dir,
        min_history=args.min_history,
        start_date=args.start,
        end_date=args.end)

    print('Computing Corwin-Schultz spread estimates...')
    spread_df = corwin_schultz_spread(highs, lows)
    liquid_pct = (spread_df.iloc[-1] <= args.max_spread).mean()
    print(f'Liquid tickers (spread <= {args.max_spread:.1%}): {liquid_pct:.1%} of universe')

    # Build weight matrices and bt strategies
    backtests = []
    for name in args.rankers:
        print(f'\nComputing weights: {name}')
        builder = WEIGHT_BUILDERS[name]
        if name == 'equal':
            weight_df = builder(prices, top_n=args.top_n,
                                spread_df=spread_df, max_spread=args.max_spread)
        else:
            weight_df = builder(prices, top_n=args.top_n,
                                spread_df=spread_df, max_spread=args.max_spread)

        test = build_strategy(name, prices, weight_df, args.rebalance,
                              verbose=args.verbose,
                              commission_bps=args.commission_bps)
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
