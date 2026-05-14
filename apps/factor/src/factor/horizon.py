"""Eval primitives for the endogenous-horizon scoring head.

The training loss (`objectives.horizon_mixture_loss`) is a per-bar
state-conditional surrogate for "deploy at horizon argmax(π_t) and hold
flat between rebalances". This module supplies the deployment-side
counterpart used at val time: simulate the actual irregular rebal
cadence on the daily axis, accumulate per-day portfolio PnL net of
turnover costs, and report annualized Sharpe.

The metric is `daily-PnL Sharpe × sqrt(252)`. No rebal-cadence
gymnastics — daily granularity is uniform, costs are debited on the day
they happen, and a single number is comparable to any baseline whose
PnL is also reported daily. This is *not* `block_sharpe`: that uses
fixed-cadence block returns and annualizes by `sqrt(252 / rebal_days)`,
which doesn't make sense when the cadence is endogenous.

All functions are pure numpy — no autograd, no Tensor. Called from the
walk-forward trainer once per window after training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TRADING_DAYS: int = 252


def _softmax_weights(
    scores: np.ndarray, mask: np.ndarray, temperature: float = 1.0,
) -> np.ndarray:
    """Per-bar softmax weights over the liquid universe.

    Mirrors `objectives.block_sharpe`'s constructor (long-only,
    sum-to-one over the liquid set). Masked entries get pushed to
    `-inf` via `+ log(mask + eps)` before the row-max subtraction so
    they receive zero weight. Bars with no liquid tickers return all
    zeros (handled by the caller as "no rebal action").
    """
    s = scores / temperature + np.log(mask + 1e-12)
    s = s - s.max(axis=1, keepdims=True)
    exp_s = np.exp(s) * mask
    denom = exp_s.sum(axis=1, keepdims=True)
    out = np.where(denom > 0, exp_s / np.maximum(denom, 1e-12), 0.0)
    return out


@dataclass(frozen=True)
class IrregularRunResult:
    """One simulated deployment of an endogenous-horizon policy.

    Fields
    ------
    daily_pnl       : `(n_val_days,)` log-PnL stream after costs.
    rebal_log       : list of `(rebal_bar_idx, daily_idx, horizon_chosen)`
                      tuples covering every actual rebal event.
    sharpe          : annualized Sharpe of `daily_pnl` (mean / std *
                      sqrt(252)). 0.0 when daily_pnl is degenerate.
    mean_holding_days : average horizon actually deployed across rebals.
    n_rebals        : count of rebal events.
    avg_turnover    : mean one-sided L1(Δw) across rebals (initial entry
                      from cash counts as full leverage).
    """
    daily_pnl:         np.ndarray
    rebal_log:         list[tuple[int, int, int]]
    sharpe:            float
    mean_holding_days: float
    n_rebals:          int
    avg_turnover:      float


def simulate_irregular_daily_pnl(
    scores: np.ndarray,
    pi: np.ndarray,
    mask: np.ndarray,
    daily_log_ret: np.ndarray,
    rebal_idx: np.ndarray,
    horizons: tuple[int, ...],
    *,
    daily_start: int,
    daily_end: int,
    commission_bps: float = 10.0,
    temperature: float = 1.0,
    horizon_picker: str = 'argmax',
    rng: np.random.Generator | None = None,
) -> IrregularRunResult:
    """Simulate a greedy endogenous-horizon deployment over a val window.

    Inputs
    ------
    scores       : `(n_bars, N)` head scores at every fine-grid rebal bar.
    pi           : `(n_bars, K)` horizon distributions at every fine bar.
    mask         : `(n_bars, N)` per-bar liquidity mask at fine bars.
    daily_log_ret: `(D, N)` daily log returns on the full daily axis.
    rebal_idx    : `(n_bars,)` daily-axis positions of the fine rebal bars.
    horizons     : `(K,)` tuple of horizon lengths (in daily bars). Must
                   match `pi.shape[1]`.
    daily_start  : daily-axis index where the val window starts. The
                   simulator picks the first fine-grid bar `>= daily_start`
                   as its initial rebal.
    daily_end    : exclusive daily-axis index where the val window ends.
                   No rebals or PnL accrual past this point. Holding
                   periods that would extend past `daily_end` are
                   truncated to it (the trailing partial block still
                   contributes daily PnL within the window).
    commission_bps: bps cost on one-sided turnover (initial entry from
                   cash = full leverage). Same convention as
                   `objectives.block_sharpe`.
    temperature  : softmax temperature for `scores → w_t`. 1.0 default.
    horizon_picker: `'argmax'` (default, greedy deployment) or
                   `'sample'` (Monte Carlo: draw `h_t ~ π_t` per rebal).
                   The training loss approximates the marginal over
                   π — at eval, argmax is the deterministic operating
                   policy and `sample` is a diagnostic for the
                   distribution.
    rng          : seed source for `sample` mode. Required when
                   `horizon_picker='sample'`; ignored otherwise.

    Returns IrregularRunResult — see docstring above for fields.

    Algorithm:
      1. Find first fine-grid bar `b0` with `rebal_idx[b0] >= daily_start`.
      2. While current daily position `d < daily_end`:
         a. Find the fine bar `b` with `rebal_idx[b] == d` (or first
            bar strictly past `d` if `d` is between fine bars — should
            not happen if `daily_start` is a fine bar; treat as
            error otherwise).
         b. Compute `w_t = softmax(scores[b])` over `mask[b]`. Pay
            turnover cost on day `d` (`commission_frac × |Δw|`).
         c. Pick `h_chosen = horizons[argmax(pi[b])]` (or sample).
         d. For each daily bar `d, d+1, ..., min(d+h_chosen, daily_end)-1`:
            daily_pnl[day] += w_t · daily_log_ret[day]. Cost is debited
            on day `d` only.
         e. Advance `d <- d + h_chosen`. Find next fine-grid bar
            `b' = first b' with rebal_idx[b'] >= d`. If `d >= daily_end`,
            stop.
      3. Compute Sharpe on the populated daily_pnl array.
    """
    if pi.shape[1] != len(horizons):
        raise ValueError(
            f'pi has {pi.shape[1]} horizons but `horizons` has {len(horizons)}')
    if horizon_picker not in ('argmax', 'sample'):
        raise ValueError(
            f'horizon_picker={horizon_picker!r} not in {{argmax, sample}}')
    if horizon_picker == 'sample' and rng is None:
        raise ValueError("horizon_picker='sample' requires rng")
    if daily_end <= daily_start:
        raise ValueError(
            f'daily_end={daily_end} must be > daily_start={daily_start}')

    n_bars, N = scores.shape
    D = daily_log_ret.shape[0]
    if daily_end > D:
        raise ValueError(
            f'daily_end={daily_end} exceeds daily_log_ret length {D}')
    commission_frac = commission_bps / 1e4
    horizons_arr = np.asarray(horizons, dtype=np.int64)

    n_val_days = daily_end - daily_start
    daily_pnl = np.zeros(n_val_days, dtype=np.float64)

    # Fine-grid rebal bars whose daily position is inside [daily_start,
    # daily_end). Anything outside that range can't be a rebal action
    # for this val window.
    bar_pool = np.where(
        (rebal_idx >= daily_start) & (rebal_idx < daily_end))[0]
    if len(bar_pool) == 0:
        return IrregularRunResult(
            daily_pnl=daily_pnl, rebal_log=[],
            sharpe=0.0, mean_holding_days=0.0,
            n_rebals=0, avg_turnover=0.0)

    rebal_log: list[tuple[int, int, int]] = []
    turnover_acc = 0.0
    prev_w = np.zeros(N, dtype=np.float64)
    d = int(rebal_idx[bar_pool[0]])
    while d < daily_end:
        # Find the fine-grid bar at this daily position (or first one
        # past). Greedy step (b) above.
        b_candidates = np.where(rebal_idx[bar_pool] >= d)[0]
        if len(b_candidates) == 0:
            # No more rebal opportunities — hold prev_w until daily_end
            # (consistent with "operator does nothing", which matches
            # what deployment would do absent a new score+π emission).
            if np.any(prev_w):
                tail_days = np.arange(d, daily_end) - daily_start
                tail_ret = daily_log_ret[d:daily_end] @ prev_w
                daily_pnl[tail_days] += tail_ret
            break
        b = int(bar_pool[b_candidates[0]])
        # If the next fine bar is past d, advance d to it — we don't have
        # scores on intermediate days. This skips the gap; the previous
        # position simply runs longer (already accounted for: prev_w is
        # still in effect because we didn't rebal yet).
        if rebal_idx[b] > d:
            # Apply PnL of prev_w over the gap [d, rebal_idx[b]) — but
            # if prev_w is zero (first iteration), this is a no-op.
            gap_end = min(int(rebal_idx[b]), daily_end)
            if gap_end > d and np.any(prev_w):
                gap_days = np.arange(d, gap_end) - daily_start
                seg_ret = daily_log_ret[d:gap_end] @ prev_w
                daily_pnl[gap_days] += seg_ret
            d = int(rebal_idx[b])
            if d >= daily_end:
                break

        # Compute target weights at this rebal.
        w = _softmax_weights(scores[b:b + 1], mask[b:b + 1], temperature)[0]
        # Turnover cost (one-sided) — initial entry from cash counts the
        # full leverage. `prev_w` is zeros for the first rebal so this
        # collapses to `commission_frac * sum(|w|)`.
        delta = w - prev_w
        one_sided_l1 = 0.5 * np.abs(delta).sum() if np.any(prev_w) else (
            np.abs(w).sum())
        cost = commission_frac * one_sided_l1
        # Pay cost on day d.
        day_idx = d - daily_start
        if 0 <= day_idx < n_val_days:
            daily_pnl[day_idx] -= cost
        turnover_acc += one_sided_l1

        # Pick horizon.
        if horizon_picker == 'argmax':
            k = int(np.argmax(pi[b]))
        else:
            k = int(rng.choice(len(horizons_arr), p=pi[b]))
        h_chosen = int(horizons_arr[k])
        rebal_log.append((b, d, h_chosen))

        # Accumulate PnL over [d, min(d+h_chosen, daily_end)) at weight w.
        seg_end = min(d + h_chosen, daily_end)
        if seg_end > d:
            seg_days = np.arange(d, seg_end) - daily_start
            seg_ret = daily_log_ret[d:seg_end] @ w
            daily_pnl[seg_days] += seg_ret
        prev_w = w
        d = seg_end

    # Sharpe — degenerate streams (all zero or near-zero std) report 0.
    if daily_pnl.size < 2:
        sharpe = 0.0
    else:
        mu = float(daily_pnl.mean())
        sd = float(daily_pnl.std())
        sharpe = (mu / sd * (TRADING_DAYS ** 0.5)) if sd > 1e-12 else 0.0

    n_rebals = len(rebal_log)
    mean_h = (sum(h for _, _, h in rebal_log) / n_rebals) if n_rebals else 0.0
    avg_to = (turnover_acc / n_rebals) if n_rebals else 0.0

    return IrregularRunResult(
        daily_pnl=daily_pnl, rebal_log=rebal_log,
        sharpe=sharpe, mean_holding_days=mean_h,
        n_rebals=n_rebals, avg_turnover=avg_to,
    )


def simulate_fixed_horizon_daily_pnl(
    scores: np.ndarray,
    mask: np.ndarray,
    daily_log_ret: np.ndarray,
    rebal_idx: np.ndarray,
    horizon: int,
    *,
    daily_start: int,
    daily_end: int,
    commission_bps: float = 10.0,
    temperature: float = 1.0,
) -> IrregularRunResult:
    """Baseline: deploy at fixed horizon `h` regardless of state.

    Mirrors `simulate_irregular_daily_pnl` but ignores `pi` — every
    rebal advances by `horizon` days. Used to construct the
    "best-fixed-h" and "fixed h_min/h_max" baselines for the null-
    rejection table.

    Returns an `IrregularRunResult` with `mean_holding_days == horizon`
    for every rebal (by construction).
    """
    n_bars, N = scores.shape
    D = daily_log_ret.shape[0]
    if daily_end > D:
        raise ValueError(
            f'daily_end={daily_end} exceeds daily_log_ret length {D}')
    commission_frac = commission_bps / 1e4

    n_val_days = daily_end - daily_start
    daily_pnl = np.zeros(n_val_days, dtype=np.float64)

    bar_pool = np.where(
        (rebal_idx >= daily_start) & (rebal_idx < daily_end))[0]
    if len(bar_pool) == 0:
        return IrregularRunResult(
            daily_pnl=daily_pnl, rebal_log=[],
            sharpe=0.0, mean_holding_days=0.0,
            n_rebals=0, avg_turnover=0.0)

    rebal_log: list[tuple[int, int, int]] = []
    turnover_acc = 0.0
    prev_w = np.zeros(N, dtype=np.float64)
    d = int(rebal_idx[bar_pool[0]])
    while d < daily_end:
        b_candidates = np.where(rebal_idx[bar_pool] >= d)[0]
        if len(b_candidates) == 0:
            if np.any(prev_w):
                tail_days = np.arange(d, daily_end) - daily_start
                tail_ret = daily_log_ret[d:daily_end] @ prev_w
                daily_pnl[tail_days] += tail_ret
            break
        b = int(bar_pool[b_candidates[0]])
        if rebal_idx[b] > d:
            gap_end = min(int(rebal_idx[b]), daily_end)
            if gap_end > d and np.any(prev_w):
                gap_days = np.arange(d, gap_end) - daily_start
                seg_ret = daily_log_ret[d:gap_end] @ prev_w
                daily_pnl[gap_days] += seg_ret
            d = int(rebal_idx[b])
            if d >= daily_end:
                break

        w = _softmax_weights(scores[b:b + 1], mask[b:b + 1], temperature)[0]
        delta = w - prev_w
        one_sided_l1 = 0.5 * np.abs(delta).sum() if np.any(prev_w) else (
            np.abs(w).sum())
        cost = commission_frac * one_sided_l1
        day_idx = d - daily_start
        if 0 <= day_idx < n_val_days:
            daily_pnl[day_idx] -= cost
        turnover_acc += one_sided_l1
        rebal_log.append((b, d, horizon))

        seg_end = min(d + horizon, daily_end)
        if seg_end > d:
            seg_days = np.arange(d, seg_end) - daily_start
            seg_ret = daily_log_ret[d:seg_end] @ w
            daily_pnl[seg_days] += seg_ret
        prev_w = w
        d = seg_end

    if daily_pnl.size < 2:
        sharpe = 0.0
    else:
        mu = float(daily_pnl.mean())
        sd = float(daily_pnl.std())
        sharpe = (mu / sd * (TRADING_DAYS ** 0.5)) if sd > 1e-12 else 0.0
    n_rebals = len(rebal_log)
    mean_h = float(horizon) if n_rebals else 0.0
    avg_to = (turnover_acc / n_rebals) if n_rebals else 0.0
    return IrregularRunResult(
        daily_pnl=daily_pnl, rebal_log=rebal_log,
        sharpe=sharpe, mean_holding_days=mean_h,
        n_rebals=n_rebals, avg_turnover=avg_to,
    )


__all__ = [
    'IrregularRunResult',
    'simulate_irregular_daily_pnl',
    'simulate_fixed_horizon_daily_pnl',
    'TRADING_DAYS',
]
