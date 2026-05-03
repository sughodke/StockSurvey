"""Head-to-head bt backtest: regime baseline vs sector-excess regime.

Runs two strategies on the same universe + dates:
  - regime:        weights_regime (per-stock CWT divergence; baseline)
  - excess-regime: weights_excess_regime (stock - sector divergence)

Prints side-by-side bt stats and saves an equity-comparison PNG. The
question: does subtracting the sector-wide regime shift improve
risk-adjusted returns by isolating the idiosyncratic component?

Intentionally simpler than `regime.research.backtest_bt`:
  - Single ticker universe (Phase-2 21 names, hardcoded for this script)
  - No spread filtering (the universe is already liquid)
  - No Optuna sweep (compares the two scoring families at one config)

Hyperparameters (lookback, n_tail, top_n, divergence) are passed in
from the CLI / `run()` arg list.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_indicators import get_divergence
from ss_loaders import load_stooq_matrix
from ss_portfolio import apply_nan_mask, select_top_n_matrix
from ss_wavelets import causal_cwt, precompute_windows

from relational.scoring import weights_excess_regime

warnings.filterwarnings('ignore')


# Phase-2 universe (mirrors the Stooq subset baked into apps/notebook/data/).
PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'CSCO',
    'JPM', 'BAC', 'GE', 'BA', 'XOM', 'KO', 'WMT', 'JNJ', 'UNH', 'T', 'DIS',
    'TSLA',
)


def _weights_regime_baseline(
    prices: pd.DataFrame, *,
    lookback: int, n_tail: int, top_n: int,
    scales: list[int], divergence: str = 'kl',
) -> pd.DataFrame:
    """Inline copy of `regime.trainer.weights_regime` to avoid a
    cross-app dep on `regime` (which would pull in jax/optax/alpaca-py
    just to compare). Numerically identical to the regime trainer's
    baseline."""
    coeffs = causal_cwt(prices.values, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(power, lookback, n_tail)
    div_fn = get_divergence(divergence)
    scale_log_weights = np.zeros(len(scales), dtype=np.float32)
    scores = np.array(div_fn(recent, historical, scale_log_weights),
                      copy=True)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)


def _make_commission_fn(bps: float):
    frac = bps / 10000.0

    def commission(q, p):
        return abs(q) * p * frac

    return commission


def _build_strategy(name: str, prices: pd.DataFrame,
                    weight_df: pd.DataFrame, *,
                    rebal_days: int, commission_bps: float):
    rebal_weights = weight_df.iloc[::rebal_days]
    strategy = bt.Strategy(name, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])
    return bt.Backtest(strategy, prices,
                       commissions=_make_commission_fn(commission_bps),
                       integer_positions=False)


def run(
    *, data_dir: str,
    top_n: int = 10,
    lookback: int = 120,
    n_tail: int = 20,
    divergence: str = 'kl',
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> None:
    """Programmatic entrypoint — called from `relational.cli`."""
    print(f'Loading Stooq prices from {data_dir} ...')
    prices_full, _highs, _lows, _volume = load_stooq_matrix(
        data_dir, min_history=lookback + n_tail + 10,
        start_date=start, end_date=end)
    # load_stooq_matrix returns the whole archive; subset to the
    # Phase-2 universe so the sector aggregates have known constituents.
    available = [t for t in PHASE2_TICKERS if t in prices_full.columns]
    missing = [t for t in PHASE2_TICKERS if t not in prices_full.columns]
    if missing:
        print(f'  WARN: tickers missing from Stooq archive (insufficient '
              f'history?): {missing}')
    prices = prices_full[available].copy()
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'({list(prices.columns)})')

    # Same scales as the regime trainer's default (excludes the very
    # short and very long ends to keep the divergence signal stable).
    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, divergence={divergence}')

    print('\n[1/2] Computing baseline regime weights...')
    w_baseline = _weights_regime_baseline(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence)

    print('[2/2] Computing sector-excess regime weights...')
    w_excess = weights_excess_regime(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence, sector_mode='equal')

    print('\nRunning bt backtests...')
    bt_baseline = _build_strategy(
        'regime', prices, w_baseline,
        rebal_days=rebal_days, commission_bps=commission_bps)
    bt_excess = _build_strategy(
        'excess-regime', prices, w_excess,
        rebal_days=rebal_days, commission_bps=commission_bps)
    result = bt.run(bt_baseline, bt_excess)
    result.display()

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    ax.set_title(f'Sector-excess regime vs baseline regime — Phase-2 '
                 f'({start} → {end}, top-{top_n}, rebal={rebal_days}d, '
                 f'div={divergence})')
    fig.tight_layout()
    fig_path = out / 'relational-sector-excess-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-sector-excess-stats.txt'
    with open(stats_path, 'w') as f:
        f.write(str(result.stats))
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    # Allow direct invocation: python -m relational.research.backtest_sector_excess
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--divergence', default='kl')
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
