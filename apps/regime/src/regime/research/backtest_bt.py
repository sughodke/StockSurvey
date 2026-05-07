"""backtest_bt: bt-library portfolio backtest of ranking strategies.

Compares four ranking algorithms head-to-head with proper portfolio
metrics (Sharpe, drawdown, CAGR) using the `bt` library:

  * rsi       — buy the most oversold names (lowest mean RSI(7) over n_tail)
  * scalogram — multi-scale momentum + cross-scale coherence
  * regime    — symmetric KL divergence between recent / historical CWT power
  * equal     — equal-weight buy-and-hold of the top-N largest stocks

Data source: Kaggle NASDAQ Daily CSVs (svaningelgem/nasdaq-daily-stock-prices)

Usage:
    python -m regime.research.backtest_bt --data-dir ./Nasdaq3347 --top-n 20
    python -m regime.research.backtest_bt --data-dir ./Nasdaq3347 \\
        --rankers rsi scalogram regime
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ss_indicators import corwin_schultz_spread, symmetric_kl_divergence
from ss_indicators import rsi as rsi_indicator
from ss_loaders import load_price_matrix
from ss_cli import add_save_args, add_universe_loader_args
from ss_portfolio import apply_nan_mask, apply_spread_mask, select_top_n_matrix
from ss_portfolio.bt_helpers import build_strategy
from ss_wavelets import causal_cwt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)


def weights_rsi(prices, lookback=60, n_tail=5, top_n=20,
                spread_df=None, max_spread=0.02):
    """Buy the most oversold names by mean RSI(7) over the trailing n_tail days."""
    print('  Computing RSI matrix...')
    rsi = np.asarray(rsi_indicator(prices.values, n=7))

    n_dates, n_tickers = rsi.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)
    for i in range(lookback, n_dates):
        scores[i - lookback] = np.mean(rsi[i - n_tail + 1:i + 1], axis=0)

    scores = apply_nan_mask(scores, prices.values, lookback)
    if spread_df is not None:
        scores = apply_spread_mask(scores, spread_df.values, lookback, max_spread)

    weights = select_top_n_matrix(scores, top_n, ascending=True)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_scalogram(prices, lookback=120, n_tail=10, top_n=20,
                      spread_df=None, max_spread=0.02):
    """Multi-scale momentum + direction from a causal CWT scalogram."""
    scales = [5, 7, 10, 12, 21, 26]
    print('  Computing causal CWT matrix...')
    coeffs = causal_cwt(prices.values, scales, lookback)
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
        denom = np.std(short_p, axis=0) * np.std(long_p, axis=0) + 1e-9
        coherence = np.clip(cov / denom, 0, 1)

        scores[i - lookback] = direction - momentum * coherence

    scores = apply_nan_mask(scores, prices.values, lookback)
    if spread_df is not None:
        scores = apply_spread_mask(scores, spread_df.values, lookback, max_spread)

    weights = select_top_n_matrix(scores, top_n, ascending=True)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_regime(prices, lookback=120, n_tail=20, top_n=20,
                   spread_df=None, max_spread=0.02):
    """Stocks with the biggest power-spectrum shift over the recent window."""
    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print('  Computing causal CWT matrix...')
    coeffs = causal_cwt(prices.values, scales, lookback)
    power = coeffs ** 2

    n_dates, n_tickers = prices.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)
    log_w = np.zeros(len(scales))  # uniform scale weights — softmax(0) = 1/n_scales

    for i in tqdm(range(lookback, n_dates), desc='Regime scores', unit='day'):
        recent = np.mean(power[:, i - n_tail + 1:i + 1, :], axis=1)
        historical = np.mean(power[:, i - lookback:i - n_tail + 1, :], axis=1)
        kl = symmetric_kl_divergence(recent, historical, log_w)
        scores[i - lookback] = np.asarray(kl)

    scores = apply_nan_mask(scores, prices.values, lookback)
    if spread_df is not None:
        scores = apply_spread_mask(scores, spread_df.values, lookback, max_spread)

    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def weights_equal(prices, top_n=20, spread_df=None, max_spread=0.02):
    """Equal-weight buy-and-hold of the top_n largest stocks (by last price)."""
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Portfolio backtest of ranking strategies using bt')
    add_universe_loader_args(
        parser,
        default_start='2000-01-01',
        default_end='2025-12-31',
        data_dir_help='Path to Kaggle NASDAQ daily CSV directory.')
    parser.add_argument('--rankers', nargs='+',
                        default=['rsi', 'scalogram', 'regime', 'equal'],
                        choices=list(WEIGHT_BUILDERS.keys()),
                        help='Strategies to compare')
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--rebalance', type=int, default=5,
                        help='Rebalance every N trading days')
    parser.add_argument('--min-history', type=int, default=504)
    add_save_args(parser)
    parser.add_argument('--verbose', action='store_true',
                        help='Print each rebalance event')
    parser.add_argument('--commission-bps', type=int, default=10)
    parser.add_argument('--max-spread', type=float, default=0.02,
                        help='Max Corwin-Schultz spread fraction to include')
    args = parser.parse_args(argv)

    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=args.min_history,
        start_date=args.start, end_date=args.end)

    print('Computing Corwin-Schultz spread estimates...')
    spread_df = corwin_schultz_spread(highs, lows)
    liquid_pct = (spread_df.iloc[-1] <= args.max_spread).mean()
    print(f'Liquid tickers (spread <= {args.max_spread:.1%}): {liquid_pct:.1%} of universe')

    backtests = []
    for name in args.rankers:
        print(f'\nComputing weights: {name}')
        builder = WEIGHT_BUILDERS[name]
        weight_df = builder(prices, top_n=args.top_n,
                            spread_df=spread_df, max_spread=args.max_spread)
        backtests.append(build_strategy(
            name, prices, weight_df,
            rebal_days=args.rebalance,
            verbose=args.verbose, commission_bps=args.commission_bps))

    print('\nRunning backtests...')
    result = bt.run(*backtests)
    result.display()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    result.plot(ax=axes[0])
    axes[0].set_title('Equity Curves')
    result.plot_weights(backtest=0, ax=axes[1])
    axes[1].set_title(f'Weight Allocation ({args.rankers[0]})')
    fig.tight_layout()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        png_path = os.path.join(args.output_dir, 'backtest-bt-comparison.png')
        stats_path = os.path.join(args.output_dir, 'backtest-bt-stats.txt')
        fig.savefig(png_path, dpi=150)
        print(f'\nSaved {png_path}')
        with open(stats_path, 'w') as f:
            f.write(str(result.stats))
        print(f'Saved {stats_path}')
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
