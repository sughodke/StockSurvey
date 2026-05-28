"""Walk-forward eval driver for the crypto-and-carry arc.

Test design (locked pre-eval; recorded as `pre_registered_bar` in NPZ):

  Hypothesis: A constant-weight long-spot / short-perp basket over the
  top-K most-funded HL perps, rebalanced every `rebal_days`, earns
  positive net Sharpe after realistic round-trip leg friction.

  Verdict bar:
    confirmed-OOS:  net Sharpe ≥ +1.0 AND deflated-t > +2.0 AND pos-quarter ≥ 0.80
    partial-OOS:    net Sharpe ≥ +0.5 AND deflated-t > +1.0
    confirmed-null: net Sharpe < +0.3
    reversed-OOS:   net Sharpe < 0
    diagnostic:     anything else in between

  Friction (applied per leg, charged on weight delta):
    - perp open/close: 10 bps per leg
    - spot rebal:       10 bps per leg
    - slippage buffer:   5 bps per leg
    → 15 bps per leg × 2 legs = 30 bps round-trip basis re-establish

  Universe: HL top-20 perps by current-snapshot $-volume; aligned to
  the joint funding-history coverage 2024-01-01 → today.

  Walk-forward: 4 non-overlapping calendar years 2024, 2025, 2026YTD,
  plus a pooled "full" arm. (HL only launched 2023-05; 2021/2022/2023
  folds from the original brief are not available on this venue.)

  Hyperparameter robustness arms (16 cells):
    - K ∈ {3, 5, 10}
    - rebal_days ∈ {1, 3, 7}
    - sign ∈ {'positive', 'both'}
    - trailing_window ∈ {7, 30}

  Pre-reg cell: K=5, rebal_days=1, sign='positive', trailing_window=30.

NPZ output keys (in `Output/cnc-walkforward.npz`):
  - oos_block_returns : pre-reg cell daily net stream over full span
  - pre_registered_bar : str
  - periods_per_year   : 365
  - universe_label     : str
  - cell_<i>_returns   : per-cell streams
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cnc.backtest import block_sharpe, max_drawdown, pos_quarter_fraction, run_carry
from cnc.data import build_panels


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'

PRE_REGISTERED_BAR = (
    'confirmed-OOS: net Sharpe>=+1.0 AND deflated-t>+2.0 AND pos-quarter>=0.80 | '
    'partial-OOS: net Sharpe>=+0.5 AND deflated-t>+1.0 | '
    'confirmed-null: net Sharpe<+0.3 | reversed-OOS: net Sharpe<0'
)

GRID = list(itertools.product(
    [3, 5, 10],          # top_k
    [1, 3, 7],           # rebal_days
    ['positive', 'both'],  # sign
    [7, 30],             # trailing_window
))

PRE_REG = dict(top_k=5, rebal_days=1, sign='positive', trailing_window=30)


def fold_slice(returns: pd.Series, year: int) -> pd.Series:
    return returns[(returns.index.year == year)]


def deflated_t(sharpe_ann: float, n_obs: int, n_trials: int,
               sharpe_std_ann: float = 0.40,
               periods_per_year: int = 365) -> float:
    """Deflated Sharpe t-stat (Lopez de Prado 2014, simplified).

    Mirrors the conservative form used in `apps/docs/scripts/compute_dsr.py`:
    expected_max_sharpe combines structural cross-trial dispersion with
    null estimation noise floor `1/sqrt(n_obs-1)`. We then compute
    `(sharpe - E[max_sharpe]) * sqrt(n_obs / periods_per_year)`.
    """
    if n_obs <= 2 or n_trials <= 0:
        return 0.0
    # Estimation-noise floor (annualized).
    noise = 1.0 / np.sqrt(max(1, n_obs - 1)) * np.sqrt(periods_per_year)
    sigma = float(np.sqrt(sharpe_std_ann ** 2 + noise ** 2))
    # E[max of n_trials draws from N(0, sigma^2)] ≈ sigma * (
    #   (1-gamma)*invPhi(1-1/n) + gamma*invPhi(1-1/(n*e)) )
    if n_trials == 1:
        e_max = 0.0
    else:
        from math import e
        from scipy.stats import norm
        gamma = 0.5772156649
        e_max = sigma * ((1 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
                         + gamma * norm.ppf(1.0 - 1.0 / (n_trials * e)))
    # t-stat: (sharpe - e_max) * sqrt(n_obs / periods_per_year)
    return float((sharpe_ann - e_max) * np.sqrt(n_obs / periods_per_year))


def assign_verdict(sharpe_ann: float, dsr_t: float, pos_q: float) -> str:
    if sharpe_ann >= 1.0 and dsr_t > 2.0 and pos_q >= 0.80:
        return 'confirmed-OOS'
    if sharpe_ann >= 0.5 and dsr_t > 1.0:
        return 'partial-OOS'
    if sharpe_ann < 0.0:
        return 'reversed-OOS'
    if sharpe_ann < 0.3:
        return 'confirmed-null'
    return 'diagnostic'


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    t0 = time.time()
    print('[cnc] building panels (Hyperliquid 2024-01-01 → today)')
    panels = build_panels(
        start_date='2024-01-01',
        end_date=None,
        top_universe=20,
        min_history_days=180,
    )
    print(f'[cnc] panels: {len(panels.funding_daily)} days × '
          f'{len(panels.coins)} coins '
          f'({panels.start_date.date()} → {panels.end_date.date()})')
    print(f'[cnc] coins: {panels.coins}')

    # Pre-reg cell.
    pre = run_carry(panels.funding_daily, **PRE_REG,
                    rebal_friction_bps_per_leg=15.0)
    pre_returns = pre.daily_return
    pre_sharpe = block_sharpe(pre_returns, periods_per_year=365)

    # Per-fold (calendar years).
    years_available = sorted(set(pre_returns.index.year))
    fold_results = {}
    for y in years_available:
        s = fold_slice(pre_returns, y)
        if len(s) < 30:
            continue
        fold_results[y] = dict(
            n_days=len(s),
            sharpe_ann=block_sharpe(s, periods_per_year=365),
            total_return=float(s.sum()),
            max_dd=max_drawdown(s),
        )

    # Grid sweep.
    print(f'[cnc] running {len(GRID)} grid cells')
    grid_rows = []
    cell_streams: dict[str, np.ndarray] = {}
    for i, (k, rd, sg, tw) in enumerate(GRID):
        r = run_carry(
            panels.funding_daily,
            top_k=k, rebal_days=rd, sign=sg, trailing_window=tw,
            rebal_friction_bps_per_leg=15.0,
        )
        sr = block_sharpe(r.daily_return, periods_per_year=365)
        mdd = max_drawdown(r.daily_return)
        pq = pos_quarter_fraction(r.daily_return)
        grid_rows.append(dict(
            cell=i, top_k=k, rebal_days=rd, sign=sg, trailing_window=tw,
            sharpe_ann=sr, max_dd=mdd, pos_quarter=pq,
            total_return=float(r.daily_return.sum()),
            gross_sharpe_ann=block_sharpe(r.gross_return, periods_per_year=365),
            friction_total=float(r.friction_cost.sum()),
        ))
        cell_streams[f'cell_{i:02d}_returns'] = r.daily_return.values
        print(f'  cell {i:>2d}: k={k} rd={rd} sg={sg:<8} tw={tw:>2} '
              f'net_Sh={sr:+.3f} pos_q={pq:.2f} mdd={mdd*100:+.2f}%')

    # DSR on pre-reg cell.
    n_obs = int(pre_returns.dropna().shape[0])
    dsr_t = deflated_t(pre_sharpe, n_obs, n_trials=len(GRID),
                       sharpe_std_ann=0.40, periods_per_year=365)
    pre_pq = pos_quarter_fraction(pre_returns)
    pre_mdd = max_drawdown(pre_returns)
    verdict = assign_verdict(pre_sharpe, dsr_t, pre_pq)

    # Best grid cell (by net Sharpe).
    best = max(grid_rows, key=lambda r: r['sharpe_ann'])

    # Save NPZ + JSON.
    npz_path = OUTPUT / 'cnc-walkforward.npz'
    np.savez(
        npz_path,
        oos_block_returns=pre_returns.values,
        oos_block_dates=pre_returns.index.astype('int64').values,
        pre_registered_bar=np.array([PRE_REGISTERED_BAR]),
        periods_per_year=np.array([365]),
        universe_label=np.array(['hyperliquid-perp-top20']),
        n_trials=np.array([len(GRID)]),
        **cell_streams,
    )
    print(f'[cnc] wrote NPZ → {npz_path}')

    summary = dict(
        venue='hyperliquid',
        funding_cadence_per_day=24,
        universe_label='hyperliquid-perp-top20',
        coins=panels.coins,
        n_coins=len(panels.coins),
        date_start=str(panels.start_date.date()),
        date_end=str(panels.end_date.date()),
        n_days=int(len(panels.funding_daily)),
        pre_reg_cell=PRE_REG,
        pre_reg_sharpe_ann=pre_sharpe,
        pre_reg_pos_quarter=pre_pq,
        pre_reg_max_dd=pre_mdd,
        pre_reg_total_return=float(pre_returns.sum()),
        deflated_t=dsr_t,
        n_trials=len(GRID),
        n_obs=n_obs,
        verdict=verdict,
        fold_results=fold_results,
        grid=grid_rows,
        best_cell=best,
        pre_registered_bar=PRE_REGISTERED_BAR,
        wall_seconds=time.time() - t0,
    )
    json_path = OUTPUT / 'cnc-walkforward.json'
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'[cnc] wrote JSON → {json_path}')

    print()
    print('=' * 70)
    print(f'PRE-REG CELL: K={PRE_REG["top_k"]} rebal={PRE_REG["rebal_days"]}d '
          f'sign={PRE_REG["sign"]} trail={PRE_REG["trailing_window"]}d')
    print(f'  net Sharpe (ann)      : {pre_sharpe:+.3f}')
    print(f'  deflated-t            : {dsr_t:+.3f}')
    print(f'  pos-quarter fraction  : {pre_pq:.2f}')
    print(f'  max drawdown          : {pre_mdd*100:+.2f}%')
    print(f'  total return          : {pre_returns.sum()*100:+.2f}%')
    print(f'  VERDICT               : {verdict}')
    print()
    print('Per-fold (pre-reg cell, calendar years):')
    for y, fr in fold_results.items():
        print(f'  {y}: Sh={fr["sharpe_ann"]:+.3f}  '
              f'tot={fr["total_return"]*100:+.2f}%  '
              f'mdd={fr["max_dd"]*100:+.2f}%  n={fr["n_days"]}')
    print()
    print(f'Best grid cell: {best}')
    print(f'Wall: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
