"""Compare 4 sizing variants atop the 3 scoreboard-relevant scorers.

Strategies tested: `baseline` (`weights_regime`), `empirical` (idea A),
`farthest` (idea C). For each, four sizing variants:

  * `equal`   — original 1/N within top-N basket (the scoreboard
    baseline; reproduces the prior bt result).
  * `rp`      — risk-parity reweighting via `sizing.risk_parity_weights`
    (1/σ_i, sum still ≈ 1).
  * `vt`      — equal-weight basket scaled by `target_vol / σ_p` to hit
    a 15% annualized portfolio-vol target (`sizing.vol_target_weights`).
  * `rp+vt`   — risk-parity inside the basket *plus* vol-target overlay.

Output: bt-stats leaderboard ordered by daily Sharpe, plus a saved
equity-curve PNG and stats text file. Same Phase-2 universe and date
window as the relational scoreboard for direct comparability.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio import weights_regime as _baseline_weights_regime

from relational.empirical_sectors import weights_excess_regime_empirical
from relational.farthest import weights_regime_farthest
from relational.sizing import risk_parity_weights, vol_target_weights

warnings.filterwarnings('ignore')


PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'CSCO',
    'JPM', 'BAC', 'GE', 'BA', 'XOM', 'KO', 'WMT', 'JNJ', 'UNH', 'T', 'DIS',
    'TSLA',
)


def _make_commission_fn(bps: float):
    frac = bps / 10000.0

    def commission(q, p):
        return abs(q) * p * frac

    return commission


def _build_strategy(
    name: str, prices: pd.DataFrame, weights: pd.DataFrame,
    *, rebal_days: int, commission_bps: float,
) -> bt.Backtest:
    rebal_weights = weights.iloc[::rebal_days]
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
    fp_window: int = 21,
    k_clusters: int = 11,
    target_vol: float = 0.15,
    sizing_vol_window: int = 60,
    max_leverage: float = 2.0,
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=lookback + n_tail + 10,
        start_date=start, end_date=end,
        tickers=list(PHASE2_TICKERS))
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, target_vol={target_vol}, '
          f'sizing_vol_window={sizing_vol_window}, max_leverage={max_leverage}')

    print('\n[scoring] computing per-strategy equal-weight baskets...')
    eq_baseline = _baseline_weights_regime(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n, scales=scales)
    eq_empirical = weights_excess_regime_empirical(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, k_clusters=k_clusters, fp_window=fp_window)
    eq_farthest = weights_regime_farthest(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window)

    base_strategies = {
        'baseline': eq_baseline,
        'empirical': eq_empirical,
        'farthest': eq_farthest,
    }

    print('\n[sizing] applying overlays...')
    backtests: list[bt.Backtest] = []
    for strat_name, eq in base_strategies.items():
        rp = risk_parity_weights(eq, prices, vol_window=sizing_vol_window)
        vt = vol_target_weights(
            eq, prices, target_vol=target_vol,
            vol_window=sizing_vol_window, max_leverage=max_leverage)
        rp_vt = vol_target_weights(
            rp, prices, target_vol=target_vol,
            vol_window=sizing_vol_window, max_leverage=max_leverage)
        for variant_name, w in [
            ('equal', eq), ('rp', rp), ('vt', vt), ('rp+vt', rp_vt),
        ]:
            label = f'{strat_name}|{variant_name}'
            print(f'  building {label}: '
                  f'mean_gross={w.iloc[::rebal_days].sum(axis=1).mean():.3f}')
            backtests.append(_build_strategy(
                label, prices, w,
                rebal_days=rebal_days, commission_bps=commission_bps))

    print('\nRunning bt backtests (12 strategies)...')
    result = bt.run(*backtests)
    result.display()

    # Order by daily_sharpe.
    stats = result.stats
    sharpe_row = stats.loc['daily_sharpe'].astype(float)
    order = sharpe_row.sort_values(ascending=False).index.tolist()

    print('\n' + '=' * 100)
    print('Sizing-overlay leaderboard — sorted by daily Sharpe')
    print('=' * 100)
    headline_rows = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                     'daily_vol', 'total_return']
    leaderboard = stats.loc[headline_rows, order].T
    with pd.option_context(
        'display.float_format', lambda x: f'{x:.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print(leaderboard.to_string())

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    result.plot(ax=ax)
    ax.set_title(
        f'Sizing overlays on scoreboard winners — Phase-2 '
        f'({start} → {end}, top-{top_n}, target_vol={target_vol}, '
        f'rebal={rebal_days}d, commission={commission_bps}bps)')
    fig.tight_layout()
    fig_path = out / 'relational-sizing-overlays-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-sizing-overlays-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Sizing-overlay leaderboard — sorted by daily Sharpe\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        f.write('Full bt stats:\n')
        f.write(str(result.stats) + '\n')
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--target-vol', type=float, default=0.15)
    p.add_argument('--sizing-vol-window', type=int, default=60)
    p.add_argument('--max-leverage', type=float, default=2.0)
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
