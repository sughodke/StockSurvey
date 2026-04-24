"""
backtest_ranking : walk-forward evaluation harness for ranking algorithms.

Compares different stock ranking strategies by measuring whether
top-ranked stocks actually outperform bottom-ranked stocks over
a forward horizon.

Supports two data sources:
  1. Kaggle NASDAQ Daily CSVs (svaningelgem/nasdaq-daily-stock-prices)
  2. Existing DataStore/ joblib cache

Usage:
    # Using Kaggle CSVs:
    uv run python backtest_ranking.py --data-dir ./NasdaqDaily --top-n 10

    # Using existing cache:
    uv run python backtest_ranking.py --use-cache --top-n 10

    # Compare rankers:
    uv run python backtest_ranking.py --data-dir ./NasdaqDaily --rankers rsi scalogram
"""

import argparse
import logging
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.basicConfig(level=logging.WARNING)


def ricker_cwt(x, scales):
    """Continuous wavelet transform with Ricker (Mexican hat) wavelet.

    Replaces scipy.signal.cwt which was removed in scipy 1.12+.
    """
    n = len(x)
    coeffs = np.zeros((len(scales), n))
    for i, s in enumerate(scales):
        # Ricker wavelet
        points = min(10 * s, n)
        t = np.arange(-points // 2, points // 2 + 1) / s
        wavelet = (1 - t ** 2) * np.exp(-t ** 2 / 2)
        wavelet /= np.sqrt(s)
        conv = np.convolve(x, wavelet, mode='same')
        coeffs[i] = conv[:n]
    return coeffs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kaggle_csvs(data_dir, min_history=252):
    """Load per-ticker CSVs from Kaggle dataset directory.

    Returns dict of {ticker: DataFrame} with standardized columns.
    Only includes tickers with at least min_history rows.
    """
    tickers = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for fname in tqdm(csv_files, desc='Loading CSVs', unit='file'):
        ticker = fname.replace('.csv', '')
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            if 'adj_close' not in df.columns:
                df['adj_close'] = df['close']
            df.sort_index(inplace=True)
            df.dropna(subset=['close'], inplace=True)
            if len(df) >= min_history:
                tickers[ticker] = df
        except Exception:
            continue

    return tickers


def load_cache(tickers_list=None, min_history=252):
    """Load from existing DataStore/ joblib cache.

    Returns dict of {ticker: DataFrame}.
    """
    from models.security import Security

    if tickers_list is None:
        store = os.path.join('./', 'DataStore')
        tickers_list = [f for f in os.listdir(store)
                        if not f.startswith('.') and not f.startswith('coin')]

    result = {}
    for ticker in tqdm(tickers_list, desc='Loading cache', unit='ticker'):
        s = Security.load(ticker, offline=True)
        if s is not None and s.daily is not None and len(s.daily) >= min_history:
            result[ticker] = s.daily

    return result


# ---------------------------------------------------------------------------
# Ranking strategies
# ---------------------------------------------------------------------------

def rank_rsi(prices_dict, date, lookback=60, n_tail=5):
    """Current approach: RSI(7) mean over last n_tail days.

    Lower RSI = more oversold = higher rank (lower score).
    """
    from util.indicators import relative_strength

    scores = {}
    for ticker, df in prices_dict.items():
        chunk = df.loc[:date].tail(lookback)
        if len(chunk) < lookback:
            continue
        rsi = relative_strength(chunk.adj_close.values, 7)
        scores[ticker] = np.mean(rsi[-n_tail:])

    return scores


def rank_scalogram(prices_dict, date, lookback=120, n_tail=10):
    """Multi-scale momentum + cross-scale coherence from wavelet scalogram.

    Reads power at indicator scales from trailing edge of scalogram.
    Combines momentum (how active) with coherence (how aligned across scales).
    Lower score = stronger buy signal.
    """
    indicator_scales = [5, 7, 10, 12, 21, 26]
    scores = {}

    for ticker, df in prices_dict.items():
        chunk = df.loc[:date].tail(lookback)
        if len(chunk) < lookback:
            continue

        prices = chunk.adj_close.values
        x = (prices - np.mean(prices)) / (np.std(prices) + 1e-9)

        scales = np.array(indicator_scales)
        coeffs = ricker_cwt(x, scales)
        power = coeffs ** 2

        # Momentum: mean trailing power across scales (high = active)
        trailing_power = power[:, -n_tail:]
        momentum = np.mean(trailing_power)

        # Direction: sign of wavelet coefficients at short scales
        # Negative = price below local mean = potential buy
        short_scale_direction = np.mean(coeffs[0, -n_tail:])  # scale=5

        # Coherence: correlation between short (scale=5) and long (scale=26)
        short_power = power[0, -n_tail:]
        long_power = power[-1, -n_tail:]
        if np.std(short_power) > 0 and np.std(long_power) > 0:
            coherence = np.corrcoef(short_power, long_power)[0, 1]
        else:
            coherence = 0

        # Combined score: oversold direction + high momentum + high coherence
        # Lower = better buy opportunity
        scores[ticker] = short_scale_direction - momentum * max(coherence, 0)

    return scores


def rank_regime_change(prices_dict, date, lookback=120, n_tail=20):
    """Regime detection: stocks where power distribution across scales
    is changing most vs historical norm.

    High divergence = regime shift = interesting.
    Returns negative divergence so lower = more change (consistent with other rankers).
    """
    scales = np.array([5, 7, 10, 12, 21, 26, 50, 90])
    scores = {}

    for ticker, df in prices_dict.items():
        chunk = df.loc[:date].tail(lookback)
        if len(chunk) < lookback:
            continue

        prices = chunk.adj_close.values
        x = (prices - np.mean(prices)) / (np.std(prices) + 1e-9)

        coeffs = ricker_cwt(x, scales)
        power = coeffs ** 2

        # Distribution of power across scales: recent vs historical
        recent = np.mean(power[:, -n_tail:], axis=1)
        historical = np.mean(power[:, :-n_tail], axis=1)

        # Normalize to distributions
        recent_dist = recent / (recent.sum() + 1e-9)
        hist_dist = historical / (historical.sum() + 1e-9)

        # KL divergence (symmetrized)
        kl = 0.5 * np.sum(recent_dist * np.log((recent_dist + 1e-9) / (hist_dist + 1e-9)))
        kl += 0.5 * np.sum(hist_dist * np.log((hist_dist + 1e-9) / (recent_dist + 1e-9)))

        scores[ticker] = -kl  # negative so lower = more divergence

    return scores


RANKERS = {
    'rsi': rank_rsi,
    'scalogram': rank_scalogram,
    'regime': rank_regime_change,
}


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def compute_forward_returns(prices_dict, date, horizon):
    """Compute forward returns for each ticker from date over horizon days."""
    returns = {}
    for ticker, df in prices_dict.items():
        future = df.loc[date:]
        if len(future) < horizon + 1:
            continue
        entry_price = future.iloc[0].adj_close
        exit_price = future.iloc[horizon].adj_close
        if entry_price > 0:
            returns[ticker] = (exit_price - entry_price) / entry_price
    return returns


def walk_forward(prices_dict, ranker_fn, rebalance_dates, horizon, top_n, bottom_n):
    """Run walk-forward backtest.

    At each rebalance date:
      1. Rank all tickers using ranker_fn
      2. Go long top_n, short bottom_n
      3. Measure forward returns over horizon

    Returns DataFrame of results per rebalance date.
    """
    results = []

    for date in tqdm(rebalance_dates, desc='Backtesting', unit='period'):
        scores = ranker_fn(prices_dict, date)
        if len(scores) < top_n + bottom_n:
            continue

        ranked = sorted(scores, key=scores.get)
        long_tickers = ranked[:top_n]
        short_tickers = ranked[-bottom_n:]

        fwd = compute_forward_returns(prices_dict, date, horizon)

        long_returns = [fwd[t] for t in long_tickers if t in fwd]
        short_returns = [fwd[t] for t in short_tickers if t in fwd]

        if not long_returns or not short_returns:
            continue

        long_mean = np.mean(long_returns)
        short_mean = np.mean(short_returns)

        results.append({
            'date': date,
            'long_return': long_mean,
            'short_return': short_mean,
            'spread': long_mean - short_mean,
            'long_tickers': long_tickers,
            'short_tickers': short_tickers,
            'n_scored': len(scores),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_results(all_results, horizon, top_n):
    """Plot comparison of ranking strategies."""
    n_rankers = len(all_results)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Ranking Strategy Backtest  (horizon={horizon}d, top/bottom {top_n})',
                 fontsize=13, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, n_rankers))

    # Cumulative spread
    ax = axes[0]
    for (name, df), color in zip(all_results.items(), colors):
        cum_spread = np.cumsum(df['spread'].values)
        ax.plot(df['date'].values, cum_spread,
                label=f'{name} (total: {cum_spread[-1]:.1%})',
                color=color, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Cumulative L/S spread')
    ax.set_title('Cumulative long-short spread')
    ax.legend(fontsize=9)

    # Per-period spread
    ax = axes[1]
    width = 0.8 / n_rankers
    for i, ((name, df), color) in enumerate(zip(all_results.items(), colors)):
        x = np.arange(len(df))
        ax.bar(x + i * width, df['spread'].values, width,
               label=name, color=color, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Period spread')
    ax.set_title('Per-period long minus short return')

    # Hit rate over time (rolling)
    ax = axes[2]
    for (name, df), color in zip(all_results.items(), colors):
        hits = (df['spread'] > 0).astype(float)
        rolling_hr = hits.rolling(min(10, len(hits)), min_periods=1).mean()
        ax.plot(df['date'].values, rolling_hr.values,
                label=f'{name} (overall: {hits.mean():.1%})',
                color=color, linewidth=1.5)
    ax.axhline(0.5, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Rolling hit rate')
    ax.set_title('Rolling hit rate (spread > 0)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    for label in axes[-1].get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')

    fig.tight_layout()
    return fig


def print_summary(all_results):
    """Print summary table of all strategies."""
    print(f'\n{"Strategy":<15} {"Periods":>8} {"Hit Rate":>10} {"Avg Spread":>12} '
          f'{"Total Spread":>14} {"Avg Long":>10} {"Avg Short":>11}')
    print('-' * 82)

    for name, df in all_results.items():
        n = len(df)
        hit_rate = (df['spread'] > 0).mean()
        avg_spread = df['spread'].mean()
        total_spread = df['spread'].sum()
        avg_long = df['long_return'].mean()
        avg_short = df['short_return'].mean()
        print(f'{name:<15} {n:>8} {hit_rate:>10.1%} {avg_spread:>12.3%} '
              f'{total_spread:>14.3%} {avg_long:>10.3%} {avg_short:>11.3%}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Walk-forward backtest of stock ranking strategies')
    parser.add_argument('--data-dir', default=None,
                        help='Path to Kaggle NASDAQ daily CSV directory')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use existing DataStore/ cache instead of CSVs')
    parser.add_argument('--rankers', nargs='+', default=['rsi', 'scalogram'],
                        choices=list(RANKERS.keys()),
                        help='Ranking strategies to compare')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Forward return horizon in trading days (default: 5)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of stocks in long/short baskets (default: 10)')
    parser.add_argument('--rebalance-freq', type=int, default=5,
                        help='Rebalance every N trading days (default: 5)')
    parser.add_argument('--start-date', default=None,
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None,
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--min-history', type=int, default=252,
                        help='Minimum trading days of history per ticker (default: 252)')
    parser.add_argument('--save', action='store_true',
                        help='Save plot to Output/')
    args = parser.parse_args()

    # Load data
    if args.use_cache:
        prices_dict = load_cache(min_history=args.min_history)
    elif args.data_dir:
        prices_dict = load_kaggle_csvs(args.data_dir, min_history=args.min_history)
    else:
        parser.error('Specify --data-dir or --use-cache')

    print(f'Loaded {len(prices_dict)} tickers')

    # Build rebalance schedule from common date range
    all_dates = None
    for df in prices_dict.values():
        idx = df.index
        if all_dates is None:
            all_dates = set(idx)
        else:
            all_dates &= set(idx)

    all_dates = sorted(all_dates)

    if args.start_date:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(args.start_date)]
    if args.end_date:
        all_dates = [d for d in all_dates if d <= pd.Timestamp(args.end_date)]

    # Need lookback warmup + forward horizon
    warmup = 120
    all_dates = all_dates[warmup:-args.horizon]
    rebalance_dates = all_dates[::args.rebalance_freq]

    print(f'Backtest: {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}, '
          f'{len(rebalance_dates)} rebalance periods')

    # Run each ranker
    all_results = {}
    for name in args.rankers:
        print(f'\nRunning: {name}')
        ranker_fn = RANKERS[name]
        results = walk_forward(
            prices_dict, ranker_fn, rebalance_dates,
            horizon=args.horizon, top_n=args.top_n, bottom_n=args.top_n)
        all_results[name] = results

    # Report
    print_summary(all_results)

    fig = plot_results(all_results, args.horizon, args.top_n)
    if args.save:
        fname = 'Output/backtest-ranking-comparison.png'
        fig.savefig(fname, dpi=150)
        print(f'\nSaved {fname}')
        plt.close(fig)
    else:
        plt.show()
