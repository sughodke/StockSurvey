"""Short-vol position evaluator — turn a per-(date, ticker) score matrix
plus an IV anchor into a vol-points P&L stream.

Two independent observations from the IV-anchored diagnostic motivate
this module:
  * Universe-wide forward realized vol < IV in 66% of (date, ticker)
    cases on Phase-2 → a generic short-vol bias, no scorer required.
  * Some scorers (idea C, r1_ot) point at the negative t-stat side —
    their top-N picks have forward / IV expansion further below 1
    than the rest. Reading the diagnostic table, that's a short-vol
    setup, but no `weights_*` builder ever materialized it.

Public surface:
  * `short_vol_pnl_panel(iv, forward)` — per-(date, ticker) cycle P&L
    `iv - forward`. Both inputs annualized fraction (0.30 = 30%).
  * `evaluate_short_vol(scores, iv, forward, prices, ...)` — picks
    top-N by score per rebalance, returns vol-point summary stats.
  * `evaluate_universe_short_vol(iv, forward, prices, ...)` — equal-
    weight every active ticker per rebalance: the trivial vrp baseline.

Output is *vol-point P&L*, not dollar-P&L. To approximate dollar P&L
of an at-the-money straddle, multiply by `vega × notional`. For
cross-scorer comparison, vol points are sufficient and unit-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def short_vol_pnl_panel(
    iv: np.ndarray, forward: np.ndarray,
) -> np.ndarray:
    """Per-(date, ticker) cycle P&L of a short-vol position opened at
    signal time t and closed at t+vol_window: `iv[t] - forward[t+w]`.
    Positive when realized vol came in below IV (vrp captured)."""
    return iv - forward


def evaluate_short_vol(
    scores: np.ndarray,
    iv_ann: np.ndarray,
    forward_ann: np.ndarray,
    prices: pd.DataFrame,
    *,
    lookback: int,
    top_n: int,
    rebal_days: int,
    vol_window: int,
    descending: bool = True,
) -> dict:
    """Evaluate short-vol P&L of taking the top-N by `scores` per
    rebalance. With `descending=True` the highest scores are selected;
    flip to `False` if the scorer's negative tail is the short-vol side.
    """
    n_dates = prices.shape[0]
    n_eval = scores.shape[0]
    rebal_eval_idx = np.arange(0, n_eval, rebal_days)

    cycle_pnls: list[float] = []
    cycle_dates: list[pd.Timestamp] = []
    n_picks_per_cycle: list[int] = []
    raw_pnl_pairs: list[float] = []  # for win-rate / per-pair stats

    for e in rebal_eval_idx:
        t = e + lookback
        if t + vol_window >= n_dates:
            break
        score_row = scores[e]
        anchor = iv_ann[t]
        forward = forward_ann[t + vol_window]
        ok = (np.isfinite(score_row) & np.isfinite(anchor)
              & np.isfinite(forward) & (anchor > 0))
        if ok.sum() < top_n:
            continue
        active = np.where(ok)[0]
        ordered = np.argsort(score_row[active])
        if descending:
            ordered = ordered[::-1]
        picks = active[ordered[:top_n]]
        per_pair = anchor[picks] - forward[picks]   # vol-points P&L
        cycle_pnls.append(float(per_pair.mean()))
        cycle_dates.append(prices.index[t])
        n_picks_per_cycle.append(int(len(picks)))
        raw_pnl_pairs.extend(per_pair.tolist())

    arr = np.asarray(cycle_pnls, dtype=np.float64)
    pairs = np.asarray(raw_pnl_pairs, dtype=np.float64)
    n_cycles = len(arr)
    if n_cycles == 0:
        return {
            'n_cycles': 0, 'mean_cycle_pnl': float('nan'),
            'std_cycle_pnl': float('nan'), 'sharpe': float('nan'),
            'win_rate_per_pair': float('nan'),
            'win_rate_per_cycle': float('nan'),
            'cum_pnl': float('nan'), 'max_dd': float('nan'),
            'ann_factor': float('nan'),
        }

    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(drawdowns.min())
    # Annualize assuming `rebal_days` is the cycle length: ~252/r cycles/yr.
    ann_factor = np.sqrt(252.0 / rebal_days)
    sharpe = (
        float(arr.mean() / arr.std() * ann_factor)
        if arr.std() > 0 else float('nan'))
    return {
        'n_cycles': n_cycles,
        'mean_cycle_pnl': float(arr.mean()),
        'std_cycle_pnl': float(arr.std()),
        'sharpe': sharpe,
        'win_rate_per_pair': float((pairs > 0).mean()),
        'win_rate_per_cycle': float((arr > 0).mean()),
        'cum_pnl': float(cumulative[-1]),
        'max_dd': max_dd,
        'ann_factor': ann_factor,
    }


def evaluate_universe_short_vol(
    iv_ann: np.ndarray,
    forward_ann: np.ndarray,
    prices: pd.DataFrame,
    *,
    lookback: int,
    rebal_days: int,
    vol_window: int,
) -> dict:
    """Equal-weight every active ticker every rebalance — the trivial
    "sell vol on everything" baseline that captures the universe-wide
    vrp without any scorer. Useful reference: any scorer that doesn't
    beat this is adding noise."""
    n_dates, n_tickers = prices.shape
    rebal_eval_idx = np.arange(
        0, n_dates - lookback - vol_window, rebal_days)

    cycle_pnls: list[float] = []
    cycle_dates: list[pd.Timestamp] = []
    raw_pnl_pairs: list[float] = []

    for e in rebal_eval_idx:
        t = e + lookback
        if t + vol_window >= n_dates:
            break
        anchor = iv_ann[t]
        forward = forward_ann[t + vol_window]
        ok = np.isfinite(anchor) & np.isfinite(forward) & (anchor > 0)
        if ok.sum() < 2:
            continue
        per_pair = anchor[ok] - forward[ok]
        cycle_pnls.append(float(per_pair.mean()))
        cycle_dates.append(prices.index[t])
        raw_pnl_pairs.extend(per_pair.tolist())

    arr = np.asarray(cycle_pnls, dtype=np.float64)
    pairs = np.asarray(raw_pnl_pairs, dtype=np.float64)
    if len(arr) == 0:
        return {'n_cycles': 0, 'mean_cycle_pnl': float('nan')}
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = float((cumulative - running_max).min())
    ann_factor = np.sqrt(252.0 / rebal_days)
    sharpe = (
        float(arr.mean() / arr.std() * ann_factor)
        if arr.std() > 0 else float('nan'))
    return {
        'n_cycles': len(arr),
        'mean_cycle_pnl': float(arr.mean()),
        'std_cycle_pnl': float(arr.std()),
        'sharpe': sharpe,
        'win_rate_per_pair': float((pairs > 0).mean()),
        'win_rate_per_cycle': float((arr > 0).mean()),
        'cum_pnl': float(cumulative[-1]),
        'max_dd': max_dd,
        'ann_factor': ann_factor,
    }
