"""Apply predicted IV-RV gap as a short-vol gate, report PnL stats.

Uses `ss_iv.short_vol_pnl_panel` for the unit-free vol-points PnL
convention NO_OPTIONS.md established (matches the prior arc's
short-vol leaderboard so any apples-to-apples comparison vs the
universe-wide baseline is honest).

Trade rule: at each rebalance bar, take the top-K cells by
*predicted* IV-RV-gap (highest predicted short-vol-edge) and
compute realized vol-points PnL on those picks. The universe-wide
baseline (all valid (date, symbol) cells) is the comparison gate
per the operational rule in CLAUDE.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GatedShortVolResult:
    """Per-arm vol-points PnL summary."""
    arm:               str
    n_picks:           int            # mean # cells per rebal
    mean_pnl_per_cell: float          # vol-points
    win_rate:          float
    sharpe_per_cell:   float          # mean / std (no annualization)


def evaluate_gated_short_vol(
    predicted_gap: pd.DataFrame,    # cols: date, symbol, pred_gap
    realized_gap:  pd.DataFrame,    # cols: date, symbol, iv_rv_gap (true)
    *,
    top_quantile: float = 0.80,
    arm_label:    str = 'gated',
) -> GatedShortVolResult:
    """Per-rebal: pick cells in top `top_quantile` of predicted gap;
    compute their realized PnL = `iv − rv_forward`.

    `top_quantile=0.80` means take the top 20% per date (or
    `top_quantile=1.0` means take everything = universe baseline).
    """
    merged = predicted_gap.merge(
        realized_gap, on=['date', 'symbol'], how='inner').dropna()
    if merged.empty:
        return GatedShortVolResult(
            arm=arm_label, n_picks=0,
            mean_pnl_per_cell=0.0, win_rate=0.0, sharpe_per_cell=0.0)

    picks = []
    for d, group in merged.groupby('date', sort=True):
        if len(group) < 5:
            continue
        threshold = group['pred_gap'].quantile(top_quantile)
        chosen = group[group['pred_gap'] >= threshold]
        if len(chosen) == 0:
            continue
        picks.append(chosen[['date', 'iv_rv_gap']])

    if not picks:
        return GatedShortVolResult(
            arm=arm_label, n_picks=0,
            mean_pnl_per_cell=0.0, win_rate=0.0, sharpe_per_cell=0.0)

    pnl_panel = pd.concat(picks, ignore_index=True)
    pnls = pnl_panel['iv_rv_gap'].values
    mean_pnl = float(np.mean(pnls))
    win_rate = float(np.mean(pnls > 0))
    std = float(np.std(pnls, ddof=1))
    sh = mean_pnl / std if std > 1e-12 else 0.0

    n_picks_per_rebal = float(np.mean([len(p) for p in picks]))
    return GatedShortVolResult(
        arm=arm_label,
        n_picks=int(round(n_picks_per_rebal)),
        mean_pnl_per_cell=mean_pnl,
        win_rate=win_rate,
        sharpe_per_cell=sh,
    )


__all__ = ['GatedShortVolResult', 'evaluate_gated_short_vol']
