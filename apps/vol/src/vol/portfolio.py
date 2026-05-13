"""Per-rebal portfolio aggregator + costs-in-loop for short-vol gating.

v1 replacement for `vol.backtest.evaluate_gated_short_vol`. The v0
metric (`sharpe_per_cell = mean(iv_rv_gap) / std(iv_rv_gap)` pooled
across all `(date, symbol)` picks) is a "weak metric" per
[`findings/vol-surface-v0.md`](../../apps/docs/docs/findings/vol-surface-v0.md):
it discards temporal structure (treats every cell as an independent
observation) and ignores friction. This module produces the honest
deployment metric: time-series annualized Sharpe of a top-K basket
rebal'd every 20 trading days, net of explicit options friction.

Trade rule:
  - At each rebal date t (every `rebal_days=20` bars), pick the top-K
    cells by *predicted* `iv_rv_gap`.
  - Each pick produces vol-points PnL = `iv_rv_gap` realized over the
    next 20 days (already pre-computed in `forward_iv_rv_gap`, so
    we just look it up).
  - Per-rebal portfolio PnL = mean realized `iv_rv_gap` across picks
    (equal vega-weighting, matches `ss_iv.short_vol_pnl_panel` convention).
  - Friction = `friction_bps / 10_000 * turnover` per rebal. Conservative
    upper bound: turnover = 1.0 (assume 100% basket churn per rebal,
    i.e. previous picks fully exited, new picks fully entered).
  - Net per-rebal PnL series → annualized Sharpe with
    `sqrt(252 / rebal_days) ≈ sqrt(12.6) ≈ 3.55`.

Why turnover = 1.0 is the right conservative cut for v1: the predicted
top-K varies materially window-to-window (val_pred is OLS-driven by
features that swing on each rebal), and we're shorting vol exposure
which is generally a closed-out trade at expiry anyway. A more
sophisticated v2 could track per-name overlap across rebals and only
charge friction on `|Δposition|`; the simpler version here is the
honest upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REBAL_DAYS_DEFAULT = 20
ANNUALIZATION_DAYS = 252


@dataclass(frozen=True)
class PortfolioShortVolResult:
    """Per-arm portfolio summary."""
    arm:              str
    n_rebals:         int            # number of 20d rebals in this val window
    n_picks_per_rebal: int           # K (constant across rebals)
    mean_pnl_per_rebal_vol_points: float  # mean over rebals (vol pts)
    std_pnl_per_rebal_vol_points: float
    annualized_sharpe: float          # gross of friction
    annualized_sharpe_net: float      # net of `friction_bps`
    friction_bps_roundtrip: float
    win_rate_per_rebal: float
    per_rebal_pnl_vol_points: list[float]
    per_rebal_dates: list[str]


def _build_rebal_dates(val_dates: pd.DatetimeIndex,
                       rebal_days: int) -> pd.DatetimeIndex:
    """Pick val dates at `rebal_days` intervals starting from the first.

    Returns the actual dates from `val_dates` at indices [0, rebal_days,
    2*rebal_days, ...]. Stops before val_dates end.
    """
    sorted_dates = val_dates.sort_values()
    n = len(sorted_dates)
    indices = list(range(0, n, rebal_days))
    return sorted_dates[indices]


def evaluate_portfolio_short_vol(
    predicted_gap: pd.DataFrame,   # cols: date, symbol, pred_gap
    realized_gap:  pd.DataFrame,   # cols: date, symbol, iv_rv_gap
    *,
    top_k: int = 100,
    friction_bps_roundtrip: float = 100.0,
    rebal_days: int = REBAL_DAYS_DEFAULT,
    arm_label: str = 'gated',
) -> PortfolioShortVolResult:
    """Compute per-rebal portfolio Sharpe of a short-vol top-K basket.

    At each rebal date, pick the top-K cells by predicted gap; realized
    PnL per rebal is the mean of realized `iv_rv_gap` across picks.
    Friction is `friction_bps / 10_000` subtracted per rebal (100%
    turnover assumed). Time-series Sharpe is annualized with
    `sqrt(252 / rebal_days)`.

    A `top_k=0` setting means "use the universe-baseline" — keep all
    cells per rebal, ignore predictions; PnL is the mean iv_rv_gap of
    every valid cell on that rebal date.
    """
    merged = predicted_gap.merge(
        realized_gap, on=['date', 'symbol'], how='inner'
    ).dropna()
    if merged.empty:
        return PortfolioShortVolResult(
            arm=arm_label, n_rebals=0, n_picks_per_rebal=top_k,
            mean_pnl_per_rebal_vol_points=0.0,
            std_pnl_per_rebal_vol_points=0.0,
            annualized_sharpe=0.0, annualized_sharpe_net=0.0,
            friction_bps_roundtrip=friction_bps_roundtrip,
            win_rate_per_rebal=0.0,
            per_rebal_pnl_vol_points=[],
            per_rebal_dates=[],
        )

    val_dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    rebal_dates = _build_rebal_dates(val_dates, rebal_days)

    per_rebal_pnl: list[float] = []
    rebal_dates_kept: list[str] = []
    for rd in rebal_dates:
        day_group = merged[merged['date'] == rd]
        if len(day_group) < max(top_k, 5):
            continue
        if top_k <= 0:
            picks = day_group
        else:
            picks = day_group.nlargest(top_k, 'pred_gap')
        if len(picks) == 0:
            continue
        per_rebal_pnl.append(float(picks['iv_rv_gap'].mean()))
        rebal_dates_kept.append(str(rd.date()))

    if not per_rebal_pnl:
        return PortfolioShortVolResult(
            arm=arm_label, n_rebals=0, n_picks_per_rebal=top_k,
            mean_pnl_per_rebal_vol_points=0.0,
            std_pnl_per_rebal_vol_points=0.0,
            annualized_sharpe=0.0, annualized_sharpe_net=0.0,
            friction_bps_roundtrip=friction_bps_roundtrip,
            win_rate_per_rebal=0.0,
            per_rebal_pnl_vol_points=[],
            per_rebal_dates=[],
        )

    pnls = np.asarray(per_rebal_pnl, dtype=float)
    friction_per_rebal = friction_bps_roundtrip / 10_000.0  # vol points
    pnls_net = pnls - friction_per_rebal
    mean_p = float(pnls.mean())
    std_p = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    ann_factor = float(np.sqrt(ANNUALIZATION_DAYS / rebal_days))
    sh_gross = (mean_p / std_p * ann_factor) if std_p > 1e-12 else 0.0
    mean_p_net = float(pnls_net.mean())
    sh_net = (mean_p_net / std_p * ann_factor) if std_p > 1e-12 else 0.0

    return PortfolioShortVolResult(
        arm=arm_label,
        n_rebals=len(pnls),
        n_picks_per_rebal=top_k,
        mean_pnl_per_rebal_vol_points=mean_p,
        std_pnl_per_rebal_vol_points=std_p,
        annualized_sharpe=sh_gross,
        annualized_sharpe_net=sh_net,
        friction_bps_roundtrip=friction_bps_roundtrip,
        win_rate_per_rebal=float((pnls > 0).mean()),
        per_rebal_pnl_vol_points=pnls.tolist(),
        per_rebal_dates=rebal_dates_kept,
    )


__all__ = ['PortfolioShortVolResult', 'evaluate_portfolio_short_vol']
