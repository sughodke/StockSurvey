"""backtest_ranking: walk-forward long/short evaluation of ranking algorithms.

Plain-numpy harness that measures whether top-ranked tickers actually
outperform bottom-ranked ones over a forward horizon. Lighter than the
bt-library backtest in `backtest_bt.py` — useful for fast signal
exploration before paying bt's overhead.

Usage:
    python -m regime.research.backtest_ranking --data-dir ./Nasdaq3347 --top-n 10
    python -m regime.research.backtest_ranking --data-dir ./Nasdaq3347 \\
        --rankers rsi scalogram
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ss_cli import add_save_args, add_universe_loader_args
from ss_indicators import rsi as rsi_indicator
from ss_indicators import symmetric_kl_divergence

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.basicConfig(level=logging.WARNING)


def _ricker_cwt_1d(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Symmetric (non-causal) Ricker CWT on a 1D series.

    Used here only for per-ticker rolling-window scoring where causality
    over the full series doesn't matter — each call is on a windowed
    chunk that ends at the rebalance date.
    """
    n = len(x)
    coeffs = np.zeros((len(scales), n))
    for i, s in enumerate(scales):
        points = min(10 * s, n)
        t = np.arange(-points // 2, points // 2 + 1) / s
        wavelet = (1 - t ** 2) * np.exp(-t ** 2 / 2) / np.sqrt(s)
        coeffs[i] = np.convolve(x, wavelet, mode='same')[:n]
    return coeffs


def load_kaggle_csvs(data_dir: str, min_history: int = 252) -> dict[str, pd.DataFrame]:
    """Load per-ticker CSVs into `{ticker: DataFrame}` for the ranking harness.

    The harness uses a per-ticker dict (not a wide matrix) because each
    rebalance date does ranker-specific lookups against arbitrary
    trailing windows. Standardizes columns to lowercase and ensures an
    `adj_close` column exists.
    """
    tickers: dict[str, pd.DataFrame] = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for fname in tqdm(csv_files, desc='Loading CSVs', unit='file'):
        ticker = fname.removesuffix('.csv')
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
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


def rank_rsi(prices_dict, date, lookback=60, n_tail=5):
    """Mean RSI(7) over the trailing `n_tail` days. Lower = more oversold."""
    scores = {}
    for ticker, df in prices_dict.items():
        chunk = df.loc[:date].tail(lookback)
        if len(chunk) < lookback:
            continue
        rsi_vals = np.asarray(rsi_indicator(chunk.adj_close.values, n=7))
        scores[ticker] = float(np.mean(rsi_vals[-n_tail:]))
    return scores


def rank_scalogram(prices_dict, date, lookback=120, n_tail=10):
    """Multi-scale momentum + cross-scale coherence from the wavelet scalogram.

    Lower score = stronger buy signal (downward direction with high
    coherent momentum).
    """
    indicator_scales = np.array([5, 7, 10, 12, 21, 26])
    scores = {}
    for ticker, df in prices_dict.items():
        chunk = df.loc[:date].tail(lookback)
        if len(chunk) < lookback:
            continue
        prices = chunk.adj_close.values
        x = (prices - np.mean(prices)) / (np.std(prices) + 1e-9)
        coeffs = _ricker_cwt_1d(x, indicator_scales)
        power = coeffs ** 2

        momentum = np.mean(power[:, -n_tail:])
        short_dir = np.mean(coeffs[0, -n_tail:])
        short_p = power[0, -n_tail:]
        long_p = power[-1, -n_tail:]
        if np.std(short_p) > 0 and np.std(long_p) > 0:
            coherence = np.corrcoef(short_p, long_p)[0, 1]
        else:
            coherence = 0.0
        scores[ticker] = float(short_dir - momentum * max(coherence, 0))
    return scores


def rank_regime_change(prices_dict, date, lookback=120, n_tail=20):
    """Stocks where the recent power distribution diverges most from history.

    Returns negative symmetric KL so lower = more divergence (matches
    the convention of the other rankers where lower = more interesting).
    """
    scales = np.array([5, 7, 10, 12, 21, 26, 50, 90])
    scores = {}
    for ticker, df in prices_dict.items():
        chunk = df.loc[:date].tail(lookback)
        if len(chunk) < lookback:
            continue
        prices = chunk.adj_close.values
        x = (prices - np.mean(prices)) / (np.std(prices) + 1e-9)
        coeffs = _ricker_cwt_1d(x, scales)
        power = coeffs ** 2
        recent = np.mean(power[:, -n_tail:], axis=1)
        historical = np.mean(power[:, :-n_tail], axis=1)
        log_w = np.zeros(len(scales))  # uniform scale weights
        kl = float(symmetric_kl_divergence(recent, historical, log_w))
        scores[ticker] = -kl
    return scores


RANKERS = {
    'rsi': rank_rsi,
    'scalogram': rank_scalogram,
    'regime': rank_regime_change,
}


def compute_forward_returns(prices_dict, date, horizon):
    """Forward returns from `date` over `horizon` trading days, per ticker."""
    returns = {}
    for ticker, df in prices_dict.items():
        future = df.loc[date:]
        if len(future) < horizon + 1:
            continue
        entry = future.iloc[0].adj_close
        exit_ = future.iloc[horizon].adj_close
        if entry > 0:
            returns[ticker] = (exit_ - entry) / entry
    return returns


def walk_forward(prices_dict, ranker_fn, rebalance_dates, horizon, top_n, bottom_n):
    """At each rebalance date: rank, long top_n / short bottom_n, measure spread."""
    results = []
    for date in tqdm(rebalance_dates, desc='Backtesting', unit='period'):
        scores = ranker_fn(prices_dict, date)
        if len(scores) < top_n + bottom_n:
            continue
        ranked = sorted(scores, key=scores.get)
        long_t = ranked[:top_n]
        short_t = ranked[-bottom_n:]
        fwd = compute_forward_returns(prices_dict, date, horizon)
        long_r = [fwd[t] for t in long_t if t in fwd]
        short_r = [fwd[t] for t in short_t if t in fwd]
        if not long_r or not short_r:
            continue
        results.append({
            'date': date,
            'long_return': np.mean(long_r),
            'short_return': np.mean(short_r),
            'spread': np.mean(long_r) - np.mean(short_r),
            'long_tickers': long_t,
            'short_tickers': short_t,
            'n_scored': len(scores),
        })
    return pd.DataFrame(results)


def plot_results(all_results, horizon, top_n):
    """Three-panel comparison: cumulative spread, per-period bars, rolling hit rate."""
    n_rankers = len(all_results)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Ranking Strategy Backtest  (horizon={horizon}d, top/bottom {top_n})',
                 fontsize=13, fontweight='bold')
    colors = plt.cm.Set2(np.linspace(0, 1, n_rankers))

    ax = axes[0]
    for (name, df), color in zip(all_results.items(), colors):
        cum = np.cumsum(df['spread'].values)
        ax.plot(df['date'].values, cum,
                label=f'{name} (total: {cum[-1]:.1%})',
                color=color, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Cumulative L/S spread')
    ax.set_title('Cumulative long-short spread')
    ax.legend(fontsize=9)

    ax = axes[1]
    width = 0.8 / n_rankers
    for i, ((name, df), color) in enumerate(zip(all_results.items(), colors)):
        x = np.arange(len(df))
        ax.bar(x + i * width, df['spread'].values, width,
               label=name, color=color, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Period spread')
    ax.set_title('Per-period long minus short return')

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
    """Tabular summary across rankers."""
    print(f'\n{"Strategy":<15} {"Periods":>8} {"Hit Rate":>10} {"Avg Spread":>12} '
          f'{"Total Spread":>14} {"Avg Long":>10} {"Avg Short":>11}')
    print('-' * 82)
    for name, df in all_results.items():
        n = len(df)
        hit = (df['spread'] > 0).mean()
        print(f'{name:<15} {n:>8} {hit:>10.1%} {df["spread"].mean():>12.3%} '
              f'{df["spread"].sum():>14.3%} {df["long_return"].mean():>10.3%} '
              f'{df["short_return"].mean():>11.3%}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Walk-forward backtest of stock ranking strategies')
    add_universe_loader_args(
        parser, data_dir_help='Path to Kaggle NASDAQ daily CSV directory.')
    parser.add_argument('--rankers', nargs='+', default=['rsi', 'scalogram'],
                        choices=list(RANKERS.keys()))
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--rebalance-freq', type=int, default=5)
    parser.add_argument('--min-history', type=int, default=252)
    add_save_args(parser)
    args = parser.parse_args(argv)

    prices_dict = load_kaggle_csvs(args.data_dir, min_history=args.min_history)
    print(f'Loaded {len(prices_dict)} tickers')

    all_dates: set | None = None
    for df in prices_dict.values():
        all_dates = set(df.index) if all_dates is None else (all_dates & set(df.index))
    all_dates = sorted(all_dates or [])
    if args.start:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(args.start)]
    if args.end:
        all_dates = [d for d in all_dates if d <= pd.Timestamp(args.end)]

    warmup = 120
    all_dates = all_dates[warmup:-args.horizon]
    rebalance_dates = all_dates[::args.rebalance_freq]
    print(f'Backtest: {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}, '
          f'{len(rebalance_dates)} rebalance periods')

    all_results: dict[str, pd.DataFrame] = {}
    for name in args.rankers:
        print(f'\nRunning: {name}')
        all_results[name] = walk_forward(
            prices_dict, RANKERS[name], rebalance_dates,
            horizon=args.horizon, top_n=args.top_n, bottom_n=args.top_n)

    print_summary(all_results)
    fig = plot_results(all_results, args.horizon, args.top_n)
    if args.save:
        fig.savefig('Output/backtest-ranking-comparison.png', dpi=150)
        print('\nSaved Output/backtest-ranking-comparison.png')
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
