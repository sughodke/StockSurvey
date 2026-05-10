"""Apply a gate to the EW return series and report metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ss_portfolio.metrics import (
    annualized_sharpe, cagr, max_drawdown, sortino,
)


@dataclass(frozen=True)
class GatedBacktestResult:
    """Per-arm summary including both the unconditional EW baseline and
    the gated-EW arm. `transition_count` is the number of times the
    gate flipped state (used for diagnosing turnover-driven costs)."""
    arm:               str
    n_days:            int
    sharpe:            float
    sortino:           float
    cagr_pct:          float
    max_drawdown_pct:  float
    avg_exposure:      float
    transition_count:  int
    daily_ret:         np.ndarray


def gated_returns(
    ew_ret: np.ndarray, gate: np.ndarray,
    flat_yield: float = 0.0,
) -> np.ndarray:
    """Apply gate to EW return series.

    `gate[t] ∈ [0, 1]` is the *target* exposure for the next bar.
    Cash sleeve earns `flat_yield` per bar (default 0 — conservative;
    real T-bill yield could lift this slightly but we want to keep
    comparison clean against the unconditional EW arm which also
    earns 0 in cash).
    """
    if len(ew_ret) != len(gate):
        raise ValueError(f'ew_ret len {len(ew_ret)} != gate len {len(gate)}')
    return gate * ew_ret + (1.0 - gate) * flat_yield


def evaluate_gated_arm(
    ew_ret: np.ndarray, gate: np.ndarray, dates: pd.DatetimeIndex,
    *, arm_label: str,
) -> GatedBacktestResult:
    daily = pd.Series(gated_returns(ew_ret, gate), index=dates)
    transitions = int(np.sum(np.abs(np.diff(gate)) > 0.5))
    return GatedBacktestResult(
        arm=arm_label,
        n_days=int(len(daily)),
        sharpe=float(annualized_sharpe(daily)),
        sortino=float(sortino(daily)),
        cagr_pct=float(cagr(daily) * 100.0),
        max_drawdown_pct=float(max_drawdown(daily) * 100.0),
        avg_exposure=float(np.mean(gate)),
        transition_count=transitions,
        daily_ret=daily.values,
    )


__all__ = [
    'GatedBacktestResult', 'evaluate_gated_arm', 'gated_returns',
]
