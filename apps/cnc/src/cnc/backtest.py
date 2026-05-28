"""Cash-and-carry walk-forward backtest.

Model (academic-clean approximation):
- Per coin `c`, per day `d`, the per-unit-notional carry PnL is the
  signed funding rate summed across the day's funding intervals
  (panel value at `funding_daily.loc[d, c]`):
    * sign = +1  → long-spot, short-perp (collects positive funding)
    * sign = −1  → long-perp, short-spot (collects |negative| funding)
- Portfolio per-day PnL = sum_c weight[d, c] * sign[d, c] * funding[d, c].
  Weights are 1/K_active spread across K active legs picked at the
  previous rebal.
- On rebal days, friction = `rebal_friction_bps_per_leg` × 2
  (spot + perp legs) × |Δweight|, summed across coins. This charges
  full round-trip on first-entry and on full exit; partial rebalances
  scale linearly with weight change.

Walk-forward:
- Per-fold: trailing-`trailing_window` mean funding rank at each rebal
  picks the top-K coins.
- Sign treatment: `'positive'` → only enter coins with positive
  trailing funding (long-spot/short-perp). `'both'` → also enter the
  most-negative-funding coins on the inverse side (long-perp/short-spot).
- Rebal cadence = `rebal_days` (in business days of the panel).
- Friction model encoded into return stream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CarryResult:
    daily_return: pd.Series         # net carry stream (after friction)
    gross_return: pd.Series         # before friction
    friction_cost: pd.Series        # per-day cost contribution
    weights: pd.DataFrame           # [date × coin]
    signs: pd.DataFrame             # [date × coin] in {-1, 0, +1}
    config: dict


def _rank_top_k(scores: pd.Series, k: int, sign: str) -> tuple[list[str], dict[str, int]]:
    """Return (coins_to_hold, sign_per_coin) per `sign` policy."""
    s = scores.dropna()
    if s.empty:
        return [], {}
    if sign == 'positive':
        pos = s[s > 0].sort_values(ascending=False).head(k)
        return list(pos.index), {c: +1 for c in pos.index}
    elif sign == 'both':
        # split capital: top-K/2 positive (long-spot/short-perp),
        # bottom-K/2 negative (long-perp/short-spot).
        k_each = max(1, k // 2)
        pos = s[s > 0].sort_values(ascending=False).head(k_each)
        neg = s[s < 0].sort_values(ascending=True).head(k_each)
        signs = {c: +1 for c in pos.index} | {c: -1 for c in neg.index}
        return list(pos.index) + list(neg.index), signs
    else:
        raise ValueError(f'sign must be positive|both, got {sign!r}')


def run_carry(
    funding_daily: pd.DataFrame,
    *,
    top_k: int = 5,
    rebal_days: int = 1,
    trailing_window: int = 30,
    sign: str = 'positive',
    rebal_friction_bps_per_leg: float = 15.0,  # 10 bps fee + 5 bps slippage
    n_legs: int = 2,                            # spot + perp
) -> CarryResult:
    """Run the carry backtest over the full panel.

    `rebal_friction_bps_per_leg` × `n_legs` is charged on |Δweight|
    on rebal days. Default 15 bps × 2 legs = 30 bps per unit of
    weight turnover (single-sided basis re-establish).
    """
    fp = funding_daily.copy().sort_index()
    coins = list(fp.columns)
    dates = fp.index
    n = len(dates)

    weights = pd.DataFrame(0.0, index=dates, columns=coins)
    signs = pd.DataFrame(0.0, index=dates, columns=coins)

    cur_w = pd.Series(0.0, index=coins)
    cur_s = pd.Series(0.0, index=coins)
    friction_cost = pd.Series(0.0, index=dates)

    # Trailing mean of funding (rank score). Use shift(1) to ensure
    # PIT — the rank decision at date d uses funding up through d−1.
    rank_score_panel = fp.rolling(trailing_window, min_periods=max(7, trailing_window // 2)).mean().shift(1)

    fric_per_leg = rebal_friction_bps_per_leg * 1e-4
    fric_round_trip = n_legs * fric_per_leg  # per unit of new notional

    last_rebal_idx = -10**9
    for i, d in enumerate(dates):
        if i - last_rebal_idx >= rebal_days:
            scores = rank_score_panel.loc[d]
            chosen, sign_map = _rank_top_k(scores, top_k, sign)
            new_w = pd.Series(0.0, index=coins)
            new_s = pd.Series(0.0, index=coins)
            if chosen:
                w_per = 1.0 / len(chosen)
                for c in chosen:
                    new_w[c] = w_per
                    new_s[c] = sign_map[c]
            # Friction = round-trip cost × notional change. We model
            # entry/exit/re-direction symmetrically: any change in
            # (sign * weight) costs fric_round_trip × |Δ(sign*w)|.
            delta = (new_s * new_w) - (cur_s * cur_w)
            friction_cost.iloc[i] = fric_round_trip * float(delta.abs().sum())
            cur_w = new_w
            cur_s = new_s
            last_rebal_idx = i
        weights.iloc[i] = cur_w.values
        signs.iloc[i] = cur_s.values

    # PnL = signed-weighted funding payment received that day.
    # Carry payment at day d uses positions in force AT THAT DAY.
    gross = (weights * signs * fp.fillna(0.0)).sum(axis=1)
    net = gross - friction_cost
    return CarryResult(
        daily_return=net,
        gross_return=gross,
        friction_cost=friction_cost,
        weights=weights,
        signs=signs,
        config=dict(
            top_k=top_k,
            rebal_days=rebal_days,
            trailing_window=trailing_window,
            sign=sign,
            rebal_friction_bps_per_leg=rebal_friction_bps_per_leg,
            n_legs=n_legs,
        ),
    )


def block_sharpe(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Annualized Sharpe of a daily return stream (numpy fallback)."""
    r = returns.dropna().values
    if len(r) < 5 or np.nanstd(r) == 0:
        return 0.0
    return float(np.nanmean(r) / np.nanstd(r) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def pos_quarter_fraction(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    q = returns.resample('Q').sum()
    if len(q) == 0:
        return 0.0
    return float((q > 0).mean())
