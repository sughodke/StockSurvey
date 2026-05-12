"""Baselines for the meta-allocator walk-forward.

Three baselines per Phase 1 spec:

- **Passive EW** — equal-weight rebal every `rebal_days`, 10bps
  commission on L1 turnover. Matches the
  [passive-EW benchmark](../../docs/findings/passive-ew-benchmark.md)
  convention so alpha is apples-to-apples.
- **TrailingBestGreedy** — at each rebal, pick the menu action
  whose trailing-`T_trail` realized Sharpe was highest, deploy at
  full gross. Represents the "no theory, just pick the recent
  winner" naive ensemble.
- **NaiveUniform** — at each rebal, mix uniformly across all
  actions in the menu at gross 1.0. Represents the "no
  meta-allocator, blend everything" do-nothing.

Each baseline outputs a daily simple-return series over the val
window. The driver compares each to passive EW per window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfr.menu import ActionMenu


def _daily_simple_returns(prices: pd.DataFrame) -> np.ndarray:
    """Per-bar simple returns. `(T, N)`. NaN for first row + any
    ticker leading-NaN. Treated as zero in portfolio integration so
    inactive tickers contribute zero return."""
    ret = prices.pct_change(fill_method=None).values
    return np.where(np.isnan(ret), 0.0, ret)


def _drift_weights_one_bar(
    w: np.ndarray, ret_t: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Move weights one bar forward by passive drift.

    Returns `(new_w, portfolio_simple_return)`. The portfolio return
    is computed as `Σ w_i * ret_i` (linear approximation, accurate
    for daily-scale returns). New weights are rescaled so they sum
    to the same gross (1 + port_ret) * old_gross / new_gross =
    (gross_i_pre * (1+ret_i)) / Σ — but for our purposes we don't
    need to re-leverage; we just let the gross drift.
    """
    gross_old = float(w.sum())
    gross_new_per_name = w * (1.0 + ret_t)
    port_ret = float(gross_new_per_name.sum()) - gross_old
    return gross_new_per_name, port_ret


def _portfolio_simulate(
    prices: pd.DataFrame,
    target_weights_per_rebal: np.ndarray,
    rebal_indices: np.ndarray,
    *, commission_bps: float = 10.0,
) -> np.ndarray:
    """Simulate a portfolio with discrete rebalancing.

    Parameters
    ----------
    prices : `(T, N)` prices panel.
    target_weights_per_rebal : `(n_rebal, N)` — target weights to
        snap to at each rebal index.
    rebal_indices : `(n_rebal,)` — int positions into `prices.index`
        where the rebal happens. The rebal at `t` uses weights
        decided at *close of t-1* (no same-bar peek): we apply the
        new weights to `prices` at index `t`, so weights actually
        take effect for the bar after they are decided.

        For Phase 1 cleanness we use the convention that weights
        in `target_weights_per_rebal[k]` are *decided* using info
        through bar `rebal_indices[k] - 1` and deployed starting at
        bar `rebal_indices[k]`. The driver assigns `rebal_indices`
        so the first rebal is at val_start + 1 (one-bar lag).

    Returns
    -------
    `(T,)` daily simple returns. Bars before the first rebal carry
    zero (cash) by convention.
    """
    T, N = prices.shape
    simple_ret = _daily_simple_returns(prices)
    out = np.zeros(T, dtype=np.float64)
    w = np.zeros(N, dtype=np.float64)
    fee = commission_bps / 10_000.0
    rebal_set = set(int(i) for i in rebal_indices)
    rebal_lookup = {int(rebal_indices[k]): k for k in range(len(rebal_indices))}
    for t in range(T):
        if t in rebal_set:
            target = target_weights_per_rebal[rebal_lookup[t]]
            target = np.where(np.isnan(target), 0.0, target)
            turnover = float(np.abs(target - w).sum())
            cost = fee * turnover
            w = target.copy()
            # Commission paid as a one-shot return drag at the rebal
            # bar itself.
            out[t] -= cost
        # Drift one bar.
        if t < T:
            ret_t = simple_ret[t]
            gross_old = float(w.sum())
            new_w_per_name = w * (1.0 + np.where(np.isnan(ret_t), 0.0, ret_t))
            gross_new = float(new_w_per_name.sum())
            out[t] += gross_new - gross_old
            w = new_w_per_name
    return out


@dataclass
class PassiveEW:
    """Equal-weight passive baseline (uniform over liquid universe).

    Identical to `apps/relational/scripts/equal_weight_benchmark.py`
    `ew_rebal20` arm: target = 1/N over liquid names, reset every
    `rebal_days`, commission on L1 turnover.
    """
    rebal_days: int = 20
    commission_bps: float = 10.0
    min_lookback: int = 21

    def daily_returns(self, prices: pd.DataFrame) -> np.ndarray:
        from cfr.menu import EqualWeightMode
        ew_weights = EqualWeightMode(min_lookback=self.min_lookback).precompute(prices)
        T = len(prices)
        rebal_indices = np.arange(self.min_lookback, T, self.rebal_days, dtype=np.int64)
        if len(rebal_indices) == 0:
            return np.zeros(T)
        target_per_rebal = ew_weights[rebal_indices]
        return _portfolio_simulate(
            prices, target_per_rebal, rebal_indices,
            commission_bps=self.commission_bps,
        )


@dataclass
class TrailingBestGreedy:
    """Pick the menu action with highest trailing Sharpe per rebal.

    At each rebal `t`, compute trailing `T_trail` daily-return Sharpe
    for each action (as if it had been deployed continuously over
    that window) and pick the argmax. Deploy at the picked action's
    pre-computed weights (which already encode its mode's gross).

    Restricts to non-cash actions when at least one beats trailing
    Sharpe of zero; otherwise falls back to cash.
    """
    rebal_days: int = 20
    commission_bps: float = 10.0
    trail_days: int = 63
    min_lookback: int = 21

    def daily_returns(
        self,
        prices: pd.DataFrame,
        menu: ActionMenu,
        action_weights_panel: np.ndarray,
    ) -> np.ndarray:
        """`action_weights_panel` is `(T, n_actions, N)` precomputed."""
        T, A, N = action_weights_panel.shape
        simple = _daily_simple_returns(prices)   # (T, N)
        # Per-bar per-action portfolio simple return as if deployed:
        # ret_action_t = Σ_n w_a(t-1) * ret_n_t. Use weights at t-1
        # to avoid same-bar peek. We use lag-1 weights for trailing
        # Sharpe estimation.
        per_action_ret = np.zeros((T, A), dtype=np.float64)
        for a in range(A):
            w_lag = np.concatenate([
                np.zeros((1, N)), action_weights_panel[:-1, a, :]
            ], axis=0)
            per_action_ret[:, a] = np.einsum('tn,tn->t', w_lag, simple)
        rebal_indices = np.arange(
            max(self.min_lookback, self.trail_days), T, self.rebal_days,
            dtype=np.int64,
        )
        target_per_rebal = np.zeros((len(rebal_indices), N), dtype=np.float64)
        for k, t in enumerate(rebal_indices):
            lo = max(0, int(t) - self.trail_days)
            trail = per_action_ret[lo:int(t)]
            with np.errstate(invalid='ignore'):
                sd = trail.std(axis=0, ddof=1)
                mu = trail.mean(axis=0)
                sh = np.where(sd > 1e-12, mu / sd, 0.0)
            # Prefer non-cash actions when any is positive
            best = int(np.argmax(sh))
            if sh[best] <= 0:
                # Cash by default (action 0 by menu convention)
                best = 0
            target_per_rebal[k] = action_weights_panel[int(t), best]
        return _portfolio_simulate(
            prices, target_per_rebal, rebal_indices,
            commission_bps=self.commission_bps,
        )


@dataclass
class NaiveUniform:
    """Uniform mix across all actions at every rebal."""
    rebal_days: int = 20
    commission_bps: float = 10.0
    min_lookback: int = 21

    def daily_returns(
        self,
        prices: pd.DataFrame,
        action_weights_panel: np.ndarray,
    ) -> np.ndarray:
        T, A, N = action_weights_panel.shape
        rebal_indices = np.arange(self.min_lookback, T, self.rebal_days, dtype=np.int64)
        if len(rebal_indices) == 0:
            return np.zeros(T)
        target = action_weights_panel[rebal_indices].mean(axis=1)  # (n_rebal, N)
        return _portfolio_simulate(
            prices, target, rebal_indices,
            commission_bps=self.commission_bps,
        )


def evaluate_baseline(daily_ret: np.ndarray) -> dict:
    """Annualized Sharpe + sortino + max DD on a daily simple-return series."""
    from ss_portfolio.metrics import (
        annualized_sharpe, cagr, max_drawdown, sortino,
    )
    return {
        'sharpe':   float(annualized_sharpe(daily_ret)),
        'sortino':  float(sortino(daily_ret)),
        'cagr':     float(cagr(daily_ret)),
        'maxdd':    float(max_drawdown(daily_ret)),
        'n_days':   int(len(daily_ret)),
    }


__all__ = [
    'PassiveEW', 'TrailingBestGreedy', 'NaiveUniform',
    'evaluate_baseline',
]
