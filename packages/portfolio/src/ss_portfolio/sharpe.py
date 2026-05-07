"""Block-level Sharpe with transaction costs (numpy).

Used as a forward-only metric for Optuna trials and offline diagnostics.
The historical raison d'être (autograd through this for `optimize_adam`)
is gone: `ss_indicators` is numpy now and there is no end-to-end gradient
path through the regime trainer's loss anymore. The `apps/factor` tinygrad
trainer carries its own `block_sharpe` mirror at `factor.objectives` for
the cases that genuinely want gradients.
"""

from __future__ import annotations

import numpy as np


TRADING_DAYS: int = 252


def block_sharpe_with_costs(
    rebal_scores: np.ndarray,
    log_temperature: float | np.ndarray,
    block_log_ret: np.ndarray,
    rebal_mask: np.ndarray,
    rebal_days: int,
    commission_frac: float,
) -> np.ndarray:
    """Annualized portfolio Sharpe at block (rebalance) granularity.

    A soft top-N is implemented as a temperature-scaled softmax of the
    regime score; small temperature approaches a hard argmax over the
    liquid universe.

    Costs
    -----
    Initial entry from cash incurs full one-sided turnover (sum of
    weights, which is 1) since there is no offsetting outflow. Subsequent
    rebalances pay 0.5 * L1(delta_w) per period — the factor of 0.5
    converts the bidirectional L1 to a one-sided cost (each unit moving
    out of A and into B is counted on both sides of |dw|).

    Block returns are converted to a daily-equivalent annualized Sharpe
    via sqrt(TRADING_DAYS / rebal_days), exact under iid block returns.
    """
    rebal_scores = np.asarray(rebal_scores, dtype=np.float64)
    block_log_ret = np.asarray(block_log_ret, dtype=np.float64)
    rebal_mask = np.asarray(rebal_mask, dtype=np.float64)
    log_temperature = float(np.asarray(log_temperature))

    temp = float(np.exp(log_temperature))
    s = rebal_scores / temp + np.log(rebal_mask + 1e-12)
    s = s - s.max(axis=1, keepdims=True)
    exp_s = np.exp(s) * rebal_mask
    w = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)

    port_block_ret = (w * block_log_ret).sum(axis=1)

    init_cost = np.abs(w[0]).sum()
    diff_cost = 0.5 * np.abs(w[1:] - w[:-1]).sum(axis=1)
    costs = commission_frac * np.concatenate([init_cost[None], diff_cost])
    port_block_ret = port_block_ret - costs

    mean = port_block_ret.mean()
    std = port_block_ret.std() + 1e-9
    return np.asarray(mean / std * np.sqrt(TRADING_DAYS / rebal_days))
