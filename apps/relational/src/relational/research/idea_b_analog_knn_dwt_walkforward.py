"""Walk-forward eval of the analog-DWT-L1 result.

The full-period bt backtest in `idea_b_analog_knn_dwt.py` showed
DWT-L1 winning on Daily Sharpe (1.11 vs 1.07) over 2013-2025. That's
in-sample on the 12-year horizon. This script splits the same backtest
into the canonical Phase-2 train (2013-2020) and val (2021-2025)
windows used by `build_canonical_checkpoints.py` and reports
per-window stats so we can tell whether the compressed-fingerprint
edge persists out-of-sample or was a regime-specific artifact of the
2013-2020 period.

Note on terminology: there are no learned parameters in analog-kNN —
the scorer is deterministic given hyperparameters. "Walk-forward" here
is the colloquial usage (a fixed train/val split with no parameter
fitting), not the rolling-window protocol used by `apps/factor`. The
kNN's existing causality guard (`s + h < t`) already restricts each
query's candidate pool to past dates with completed forward-return
horizons, so the train/val split is purely a reporting partition over
the same equity curve.

Output: `Output/relational-idea-b-analog-knn-dwt-walkforward-{stats.txt,
equity.png}` with a vertical split-date marker on the equity plot.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_features import Compression
from ss_loaders import load_stooq_matrix
from ss_portfolio.bt_helpers import build_strategy

from relational.analog_knn import weights_regime_analog
from relational.sectors import PHASE2_TICKERS

warnings.filterwarnings('ignore')


# Mirrors PHASE2_TRAIN/VAL constants in build_canonical_checkpoints.py.
TRAIN_START = '2013-01-29'
TRAIN_END = '2020-12-31'
VAL_START = '2021-01-01'
VAL_END = '2025-12-11'

TRADING_DAYS_PER_YEAR = 252


def segment_stats(
    equity: pd.Series, start: str, end: str,
) -> dict[str, float]:
    """Annualised Sharpe / Sortino / MaxDD / CAGR / total return on a
    date-bounded slice of an equity curve. The slice is `[start, end]`
    inclusive at both ends. Public so the Modal multi-arm entrypoint
    can reuse the same segmentation logic."""
    seg = equity.loc[start:end].astype(float)
    if len(seg) < 2:
        return {'n_bars': 0}
    rets = seg.pct_change().dropna().to_numpy()
    n_bars = len(rets)
    n_years = n_bars / TRADING_DAYS_PER_YEAR

    total_return = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    mu = float(rets.mean())
    sd = float(rets.std(ddof=1))
    sharpe = (mu / sd) * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else 0.0

    downside = rets[rets < 0]
    downside_sd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = ((mu / downside_sd) * np.sqrt(TRADING_DAYS_PER_YEAR)
               if downside_sd > 0 else 0.0)

    cummax = np.maximum.accumulate(seg.to_numpy())
    drawdown = seg.to_numpy() / cummax - 1.0
    max_dd = float(drawdown.min())

    return {
        'n_bars': n_bars,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
    }


def run(
    *, data_dir: str,
    top_n: int = 10,
    lookback: int = 120,
    fp_window: int = 21,
    k_neighbors: int = 50,
    forward_horizon: int = 20,
    min_sep_days: int = 21,
    pool_mode: str = 'cross_ticker',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _highs, _lows, _vol = load_stooq_matrix(
        data_dir, min_history=lookback + 30,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=list(PHASE2_TICKERS))
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, top_n={top_n}, '
          f'fp_window={fp_window}, k={k_neighbors}, h={forward_horizon}')
    print(f'  train: {TRAIN_START} → {TRAIN_END}')
    print(f'  val:   {VAL_START} → {VAL_END}')

    arms = [
        ('analog',        None),
        ('analog-dwt-L1', Compression(kind='dwt', levels=1, wavelet='haar',
                                      pad_mode='periodization')),
    ]

    weights_by_arm: dict[str, pd.DataFrame] = {}
    for name, comp in arms:
        print(f'\n[{name}] computing weights (single full-period sweep) ...')
        weights_by_arm[name] = weights_regime_analog(
            prices, lookback=lookback, top_n=top_n,
            scales=scales, fp_window=fp_window,
            k_neighbors=k_neighbors, forward_horizon=forward_horizon,
            min_sep_days=min_sep_days, pool_mode=pool_mode,
            compression=comp)

    print('\nRunning bt backtests over the full period ...')
    strategies = [
        build_strategy(name, prices, w,
                       rebal_days=rebal_days, commission_bps=commission_bps)
        for name, w in weights_by_arm.items()
    ]
    result = bt.run(*strategies)

    # Slice each arm's equity curve into train and val windows, compute
    # segment stats independently. `result.prices` is a wide DataFrame
    # of equity curves keyed by strategy name.
    eq_panel = result.prices
    rows: list[dict] = []
    for name, _comp in arms:
        equity = eq_panel[name]
        for window_name, w_start, w_end in (('full', TRAIN_START, VAL_END),
                                            ('train', TRAIN_START, TRAIN_END),
                                            ('val', VAL_START, VAL_END)):
            stats = segment_stats(equity, w_start, w_end)
            stats.update({'arm': name, 'window': window_name,
                          'start': w_start, 'end': w_end})
            rows.append(stats)
    summary = pd.DataFrame(rows)[
        ['arm', 'window', 'start', 'end', 'n_bars',
         'total_return', 'cagr', 'sharpe', 'sortino', 'max_dd']
    ]

    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print('\n=== Segmented walk-forward stats ===')
    print(summary.to_string(index=False))

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    stats_path = out / 'relational-idea-b-analog-knn-dwt-walkforward-stats.txt'
    with open(stats_path, 'w') as f:
        f.write(summary.to_string(index=False))
        f.write('\n\n--- delta (dwt-L1 minus baseline) ---\n')
        for window in ('full', 'train', 'val'):
            base = summary[(summary.arm == 'analog') &
                           (summary.window == window)].iloc[0]
            comp = summary[(summary.arm == 'analog-dwt-L1') &
                           (summary.window == window)].iloc[0]
            f.write(
                f'  {window:6s}  '
                f'Δsharpe={comp.sharpe - base.sharpe:+.4f}  '
                f'Δcagr={comp.cagr - base.cagr:+.4f}  '
                f'Δret={comp.total_return - base.total_return:+.4f}  '
                f'Δmaxdd={comp.max_dd - base.max_dd:+.4f}\n')
    print(f'\nSaved {stats_path}')

    fig, ax = plt.subplots(figsize=(13, 7))
    eq_panel.plot(ax=ax, lw=1.5)
    split_date = pd.Timestamp(VAL_START)
    ax.axvline(split_date, color='k', ls='--', alpha=0.5,
               label=f'train/val split ({VAL_START})')
    ax.set_title(
        f'Idea B walk-forward — analog k-NN baseline vs DWT-L1 — Phase-2 '
        f'(top-{top_n}, rebal={rebal_days}d, k={k_neighbors}, '
        f'h={forward_horizon}, w={fp_window})')
    ax.legend()
    fig.tight_layout()
    fig_path = out / 'relational-idea-b-analog-knn-dwt-walkforward-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Saved {fig_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-neighbors', type=int, default=50)
    p.add_argument('--forward-horizon', type=int, default=20)
    p.add_argument('--min-sep-days', type=int, default=21)
    p.add_argument('--pool-mode', default='cross_ticker',
                   choices=['cross_ticker', 'per_ticker'])
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
