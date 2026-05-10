"""Per-pair PnL + aggregator across pairs.

Trade construction: when `position[t] = +1` (long-spread), hold $0.5
of A and short $0.5 of B (gross exposure $1, net exposure $0). The
spread payoff over `[t, t+1]` is roughly `(r_a − β · r_b) / (1+β)`
in simple-return terms — for v1 we use the cleaner
`(log_p_a[t+1] − log_p_a[t]) − β · (log_p_b[t+1] − log_p_b[t])`
times `position`, which approximates the PnL of $1-gross dollar-
neutral position scaled appropriately. This is the standard
academic spec; the magnitudes are interpretable as Sharpe-on-
spread-trade.

Costs: each open or close pays `commission_bps × 2` (one per leg)
on top of slippage. v1 uses 10 bps × 2 = 20 bps per state
transition. Aggregation across pairs is equal-weight `1/N`
allocation per pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ss_portfolio.metrics import (
    annualized_sharpe, cagr, max_drawdown, sortino,
)

from pairs.cointegration import EngleGrangerResult, engle_granger_test
from pairs.spread import SpreadStats, compute_spread, spread_stats, zscore
from pairs.predictor import trade_signals


@dataclass(frozen=True)
class PairBacktestResult:
    """Per-pair PnL + diagnostic stats over a single val window."""
    a:                str
    b:                str
    n_val_bars:       int
    sharpe:           float
    sortino:          float
    cagr_pct:         float
    max_drawdown_pct: float
    n_trades:         int        # # of position transitions
    avg_holding_bars: float
    pct_in_trade:     float
    train_half_life:  float
    val_daily_ret:    np.ndarray  # for aggregation


def backtest_pair(
    log_p_a_train: np.ndarray, log_p_b_train: np.ndarray,
    log_p_a_val:   np.ndarray, log_p_b_val:   np.ndarray,
    val_dates: pd.DatetimeIndex,
    *,
    a_name: str, b_name: str,
    hedge_beta: float, intercept: float,
    entry: float = 2.0, exit_z: float = 0.5, stop: float = float('inf'),
    commission_bps: float = 10.0,
) -> PairBacktestResult:
    """Per-pair val-window PnL given a hedge ratio fit on train."""
    spread_train = compute_spread(
        log_p_a_train, log_p_b_train, hedge_beta, intercept)
    stats = spread_stats(spread_train)

    spread_val = compute_spread(
        log_p_a_val, log_p_b_val, hedge_beta, intercept)
    z_val = zscore(spread_val, stats)
    pos_val = trade_signals(z_val, entry=entry, exit_z=exit_z, stop=stop)

    # Lag position by 1 — decision uses z[t-1] but acts on bar t.
    pos_lag = np.concatenate([[0], pos_val[:-1]])

    # Spread per-bar log return: Δs_t = (log_p_a_t − log_p_a_{t-1})
    #                                  − β · (log_p_b_t − log_p_b_{t-1})
    d_log_a = np.diff(log_p_a_val, prepend=log_p_a_val[0])
    d_log_b = np.diff(log_p_b_val, prepend=log_p_b_val[0])
    spread_ret = d_log_a - hedge_beta * d_log_b

    # PnL = position × spread return; convert to per-dollar-gross-
    # exposure terms by dividing by (1 + |β|) so the PnL series is
    # comparable across pairs with different hedge ratios.
    leverage_normalizer = 1.0 + abs(hedge_beta)
    pnl = pos_lag * spread_ret / leverage_normalizer

    # Commission on state transitions. Each leg pays bps; pair has
    # 2 legs so total cost is 2 × bps × |Δposition|. Δposition can
    # be 1 (open or close), 2 (flip), per state machine — but
    # |Δposition_t| × commission still gives the right per-leg cost
    # if we interpret it as one-leg turnover.
    pos_change = np.abs(np.diff(pos_lag, prepend=0))
    cost = (commission_bps / 1e4) * 2.0 * pos_change
    pnl_net = pnl - cost

    daily = pd.Series(pnl_net, index=val_dates)

    transitions = int(np.sum(np.abs(np.diff(pos_lag)) > 0))
    in_trade = pos_lag != 0
    pct_in_trade = float(np.mean(in_trade))
    n_open_periods = int(np.sum(np.diff(in_trade.astype(int), prepend=0) > 0))
    avg_hold = (
        float(np.sum(in_trade)) / max(n_open_periods, 1)
        if n_open_periods > 0 else 0.0)

    return PairBacktestResult(
        a=a_name, b=b_name,
        n_val_bars=int(len(daily)),
        sharpe=float(annualized_sharpe(daily)),
        sortino=float(sortino(daily)),
        cagr_pct=float(cagr(daily) * 100.0),
        max_drawdown_pct=float(max_drawdown(daily) * 100.0),
        n_trades=transitions,
        avg_holding_bars=avg_hold,
        pct_in_trade=pct_in_trade,
        train_half_life=stats.half_life,
        val_daily_ret=daily.values,
    )


def aggregate_pair_pnl(
    pair_results: list[PairBacktestResult],
    val_dates: pd.DatetimeIndex,
) -> dict:
    """Equal-weight `1/N` aggregation of per-pair daily PnL."""
    if not pair_results:
        return {
            'n_pairs': 0, 'sharpe': 0.0, 'sortino': 0.0,
            'cagr_pct': 0.0, 'max_drawdown_pct': 0.0,
            'mean_pair_sharpe': 0.0, 'pos_pair_sharpe_frac': 0.0,
            'agg_daily_ret': np.zeros(len(val_dates)),
        }
    n = len(pair_results)
    stack = np.stack([r.val_daily_ret for r in pair_results], axis=0)  # (P, T)
    agg = np.mean(stack, axis=0)
    agg_series = pd.Series(agg, index=val_dates)
    pair_sharpes = np.array([r.sharpe for r in pair_results])
    return {
        'n_pairs': n,
        'sharpe':           float(annualized_sharpe(agg_series)),
        'sortino':          float(sortino(agg_series)),
        'cagr_pct':         float(cagr(agg_series) * 100.0),
        'max_drawdown_pct': float(max_drawdown(agg_series) * 100.0),
        'mean_pair_sharpe': float(np.mean(pair_sharpes)),
        'pos_pair_sharpe_frac': float(np.mean(pair_sharpes > 0)),
        'agg_daily_ret':    agg,
    }


__all__ = [
    'PairBacktestResult',
    'aggregate_pair_pnl',
    'backtest_pair',
]
