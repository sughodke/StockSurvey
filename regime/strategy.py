"""Differentiable regime scoring and portfolio Sharpe.

These two functions form the JAX core of the optimizer. They operate on
small block-level tensors (n_blocks ~= n_valid / rebal_days, ~150 in a
typical 13-year run) so the trace and gradient fit comfortably on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


TRADING_DAYS: int = 252


def regime_scores(
    recent: jax.Array,
    historical: jax.Array,
    scale_log_weights: jax.Array,
) -> jax.Array:
    """Symmetric KL divergence between weighted recent vs historical power.

    Parameters
    ----------
    recent, historical : (n_scales, n_blocks, n_tickers)
        Pre-computed windowed power means, output of `precompute_windows`.
    scale_log_weights : (n_scales,)
        Pre-softmax weights over CWT scales (a learned parameter).

    Returns
    -------
    (n_blocks, n_tickers) — larger value = bigger regime shift.
    """
    sw = jax.nn.softmax(scale_log_weights)
    rw = sw[:, None, None] * recent
    hw = sw[:, None, None] * historical

    eps = 1e-9
    rd = rw / (rw.sum(axis=0, keepdims=True) + eps)
    hd = hw / (hw.sum(axis=0, keepdims=True) + eps)

    kl = 0.5 * jnp.sum(rd * jnp.log((rd + eps) / (hd + eps)), axis=0)
    kl += 0.5 * jnp.sum(hd * jnp.log((hd + eps) / (rd + eps)), axis=0)
    return kl


def block_sharpe_with_costs(
    rebal_scores: jax.Array,
    log_temperature: jax.Array,
    block_log_ret: jax.Array,
    rebal_mask: jax.Array,
    rebal_days: int,
    commission_frac: float,
) -> jax.Array:
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
    out of A and into B is counted on both sides of |Δw|).

    Block returns are converted to a daily-equivalent annualized Sharpe
    via sqrt(TRADING_DAYS / rebal_days), exact under iid block returns.
    """
    temp = jnp.exp(log_temperature)
    s = rebal_scores / temp + jnp.log(rebal_mask + 1e-12)
    s = s - s.max(axis=1, keepdims=True)
    exp_s = jnp.exp(s) * rebal_mask
    w = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)

    port_block_ret = (w * block_log_ret).sum(axis=1)

    init_cost = jnp.abs(w[0]).sum()
    diff_cost = 0.5 * jnp.abs(w[1:] - w[:-1]).sum(axis=1)
    costs = commission_frac * jnp.concatenate([init_cost[None], diff_cost])
    port_block_ret = port_block_ret - costs

    mean = port_block_ret.mean()
    std = port_block_ret.std() + 1e-9
    return mean / std * jnp.sqrt(TRADING_DAYS / rebal_days)
