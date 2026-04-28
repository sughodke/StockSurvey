"""Differentiable training + eval objectives for the scoring head.

`pearson_rank_ic` is the training signal — per-rebalance Pearson
correlation of the head's score vector with forward log-returns,
masked to the liquid universe at that bar, then averaged across bars.
Pearson on raw scores is the Grinold "information coefficient": a
*per-decision* signal, dense and well-conditioned, vs Sharpe which is
one number per backtest.

`block_sharpe` re-exports `ss_portfolio.block_sharpe_with_costs` for
eval — Sharpe is what we ultimately care about, but it is too noisy a
single-number objective to optimize against directly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ss_portfolio import block_sharpe_with_costs as block_sharpe


def pearson_rank_ic(
    scores: jax.Array, fwd_returns: jax.Array, mask: jax.Array,
) -> jax.Array:
    """Mean over rebalance bars of Pearson(scores[bar], fwd_returns[bar]).

    All inputs shape `(n_bars, n_tickers)`. `mask` is 1.0 for liquid
    tickers at that bar, 0.0 otherwise. Bars with fewer than 2 valid
    tickers contribute 0 to the mean (correlation undefined).

    Inputs are sanitized: NaN/Inf values are replaced with 0 before any
    arithmetic. Callers are still expected to mask them out, but the
    guard avoids gradient contamination if a caller forgets — under
    autograd, `0 * NaN = NaN` and the head's update goes NaN.
    """
    scores = jnp.where(jnp.isfinite(scores), scores, 0.0)
    fwd_returns = jnp.where(jnp.isfinite(fwd_returns), fwd_returns, 0.0)
    counts = mask.sum(axis=1)
    safe_counts = jnp.maximum(counts, 1.0)
    s_mean = (scores * mask).sum(axis=1) / safe_counts
    r_mean = (fwd_returns * mask).sum(axis=1) / safe_counts
    s_dev = (scores - s_mean[:, None]) * mask
    r_dev = (fwd_returns - r_mean[:, None]) * mask
    cov = (s_dev * r_dev).sum(axis=1)
    s_var = (s_dev ** 2).sum(axis=1)
    r_var = (r_dev ** 2).sum(axis=1)
    denom = jnp.sqrt(jnp.maximum(s_var * r_var, 1e-18))
    per_bar_ic = cov / denom
    bar_valid = (counts >= 2).astype(scores.dtype)
    return (per_bar_ic * bar_valid).sum() / jnp.maximum(bar_valid.sum(), 1.0)


__all__ = ['block_sharpe', 'pearson_rank_ic']
