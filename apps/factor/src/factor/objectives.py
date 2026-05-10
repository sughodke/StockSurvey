"""Differentiable training + eval objectives for the scoring head.

`pearson_rank_ic` is the training signal — per-rebalance Pearson
correlation of the head's score vector with forward log-returns,
masked to the liquid universe at that bar, then averaged across bars.
Pearson on raw scores is the Grinold "information coefficient": a
*per-decision* signal, dense and well-conditioned, vs Sharpe which is
one number per backtest.

`block_sharpe` is the eval-only annualized Sharpe at rebalance
granularity, with one-sided turnover costs at `commission_frac`.
Block returns are converted to a daily-equivalent annualized Sharpe
via `sqrt(TRADING_DAYS / rebal_days)` (exact under iid block returns,
matching the JAX `ss_portfolio.block_sharpe_with_costs` definition).
"""
from __future__ import annotations

from tinygrad.tensor import Tensor


TRADING_DAYS: int = 252


def _isfinite(x: Tensor) -> Tensor:
    inf = float('inf')
    return (x == x) & (x < inf) & (x > -inf)


def pearson_rank_ic(
    scores: Tensor, fwd_returns: Tensor, mask: Tensor,
) -> Tensor:
    """Mean over rebalance bars of Pearson(scores[bar], fwd_returns[bar]).

    All inputs shape `(n_bars, n_tickers)`. `mask` is 1.0 for liquid
    tickers at that bar, 0.0 otherwise. Bars with fewer than 2 valid
    tickers contribute 0 to the mean (correlation undefined).

    Inputs are sanitized: NaN/Inf values are replaced with 0 before any
    arithmetic. Callers are still expected to mask them out, but the
    guard avoids gradient contamination if a caller forgets — under
    autograd, `0 * NaN = NaN` and the head's update goes NaN.

    Assumes `fwd_returns` is the cumulative log-return over the same
    rebalance horizon used to subsample `scores` and `mask`; period
    selection happens upstream in `precompute_inputs`.
    """
    scores = _isfinite(scores).where(scores, 0.0)
    fwd_returns = _isfinite(fwd_returns).where(fwd_returns, 0.0)
    counts = mask.sum(axis=1)
    safe_counts = counts.maximum(1.0)
    s_mean = (scores * mask).sum(axis=1) / safe_counts
    r_mean = (fwd_returns * mask).sum(axis=1) / safe_counts
    s_dev = (scores - s_mean.reshape(-1, 1)) * mask
    r_dev = (fwd_returns - r_mean.reshape(-1, 1)) * mask
    cov = (s_dev * r_dev).sum(axis=1)
    s_var = (s_dev * s_dev).sum(axis=1)
    r_var = (r_dev * r_dev).sum(axis=1)
    denom = (s_var * r_var).maximum(1e-18).sqrt()
    per_bar_ic = cov / denom
    bar_valid = (counts >= 2).cast(scores.dtype)
    return (per_bar_ic * bar_valid).sum() / bar_valid.sum().maximum(1.0)


def masked_mse(
    scores: Tensor, targets: Tensor, mask: Tensor,
) -> Tensor:
    """Mean squared error averaged over `(bar, ticker)` cells with `mask=1`.

    All inputs shape `(n_bars, n_tickers)`. `mask` is 1.0 for cells the
    auxiliary loss should see, 0.0 otherwise. NaN/Inf values are
    sanitized to 0 before the diff so a missed mask upstream cannot
    NaN-poison the gradient — same defensive pattern as
    `pearson_rank_ic`.

    Used as the auxiliary loss in the multi-task path: the aux head
    predicts cross-sectionally winsorized + z-scored forward returns
    (see `factor.data.forward_robust_z`). Mean-over-cells rather than
    mean-over-bars-then-mean keeps gradient magnitude proportional to
    actual sample count, which matters for early bars with fewer valid
    tickers.
    """
    scores = _isfinite(scores).where(scores, 0.0)
    targets = _isfinite(targets).where(targets, 0.0)
    sq = (scores - targets) ** 2 * mask
    n_valid = mask.sum().maximum(1.0)
    return sq.sum() / n_valid


def block_sharpe(
    rebal_scores: Tensor,
    log_temperature: Tensor,
    block_log_ret: Tensor,
    rebal_mask: Tensor,
    rebal_days: int,
    commission_frac: float,
) -> Tensor:
    """Annualized portfolio Sharpe at block (rebalance) granularity.

    Mirrors the contract of `ss_portfolio.block_sharpe_with_costs`
    (which still runs JAX in the regime app). Soft top-N is implemented
    as a temperature-scaled softmax of the regime score; small
    temperature approaches a hard argmax over the liquid universe.

    Costs: initial entry from cash incurs full one-sided turnover
    (sum of weights = 1). Subsequent rebalances pay `0.5 * L1(delta_w)`
    per period — the factor of 0.5 converts bidirectional L1 to a
    one-sided cost.

    Block returns are converted to daily-equivalent annualized Sharpe
    via `sqrt(TRADING_DAYS / rebal_days)`.
    """
    temp = log_temperature.exp()
    # `+ log(mask + eps)` drives masked entries to -inf in the softmax,
    # so they get zero weight regardless of score. Subtract row-max for
    # numerical stability before exp.
    s = rebal_scores / temp + (rebal_mask + 1e-12).log()
    s = s - s.max(axis=1, keepdim=True)
    exp_s = s.exp() * rebal_mask
    w = exp_s / (exp_s.sum(axis=1, keepdim=True) + 1e-12)

    port_block_ret = (w * block_log_ret).sum(axis=1)

    init_cost = w[0].abs().sum()
    diff_cost = 0.5 * (w[1:] - w[:-1]).abs().sum(axis=1)
    costs = commission_frac * init_cost.reshape(1).cat(diff_cost, dim=0)
    port_block_ret = port_block_ret - costs

    mean = port_block_ret.mean()
    std = port_block_ret.std() + 1e-9
    return mean / std * Tensor((TRADING_DAYS / rebal_days) ** 0.5)


__all__ = ['TRADING_DAYS', 'block_sharpe', 'masked_mse', 'pearson_rank_ic']
