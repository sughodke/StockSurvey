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
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio import weights_regime as _weights_regime_baseline

from relational.scoring import weights_excess_regime

warnings.filterwarnings('ignore')


# Phase-2 universe (mirrors the Stooq subset baked into apps/notebook/data/).
PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'CSCO',
    'JPM', 'BAC', 'GE', 'BA', 'XOM', 'KO', 'WMT', 'JNJ', 'UNH', 'T', 'DIS',
    'TSLA',
)


# `_weights_regime_baseline` is re-exported from
# `ss_portfolio.weights_regime` (the canonical home post-refactor).
# Local alias keeps the call sites below stable; no inline copy needed.


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
    # `tickers=` filters at the file-walk level so we read 21 files
    # instead of 12K. Tickers absent from the archive are silently
    # dropped — Phase-2 names are all liquid US large-caps, so missing
    # entries indicate a data-dir mismatch rather than a delisting we'd
    # want to warn about.
    prices, _highs, _lows, _volume = load_stooq_matrix(
        data_dir, min_history=lookback + n_tail + 10,
        start_date=start, end_date=end,
        tickers=list(PHASE2_TICKERS))
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
