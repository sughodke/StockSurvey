"""Market-neutral long/short per-block net return stream (pure numpy).

Mirrors `factor.objectives.block_port_returns_long_short_np` exactly (z-score
→ clip → demean → L1-normalize constructor, validated against the tinygrad
`block_sharpe_long_short`), but lives here so `lie` needs no `factor`/tinygrad
dependency. Used to turn a cross-sectional score matrix into a deployable
dollar-neutral book for the Deflated-Sharpe harness.
"""

from __future__ import annotations

import numpy as np


def long_short_net_returns(
    scores: np.ndarray,
    block_log_ret: np.ndarray,
    mask: np.ndarray,
    commission_frac: float,
    leverage: float = 1.0,
    clip_sigma: float = 3.0,
) -> np.ndarray:
    """Per-block net (post-cost) return stream of the market-neutral book.

    `scores`, `block_log_ret`, `mask` are `(n_block, n_name)`. Weights per
    block are dollar-neutral (sum w = 0) with gross `leverage` (sum |w| =
    leverage), so market beta cancels. Commission is charged on L1 turnover
    between consecutive blocks (full leverage on the initial entry).
    """
    scores = np.asarray(scores, dtype=np.float64)
    blr = np.asarray(block_log_ret, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    counts = mask.sum(axis=1, keepdims=True)
    safe_counts = np.maximum(counts, 1.0)
    s_mean = (scores * mask).sum(axis=1, keepdims=True) / safe_counts
    s_dev = (scores - s_mean) * mask
    s_var = (s_dev * s_dev).sum(axis=1, keepdims=True) / safe_counts
    s_std = np.sqrt(s_var + 1e-12)
    z = s_dev / s_std
    z = np.clip(z, -clip_sigma, clip_sigma) * mask
    z_mean = z.sum(axis=1, keepdims=True) / safe_counts
    z = (z - z_mean) * mask
    l1 = np.abs(z).sum(axis=1, keepdims=True)
    safe_l1 = np.maximum(l1, 1e-12)
    w = leverage * z / safe_l1
    w = w * (l1 > 1e-9).astype(np.float64)
    port = (w * blr).sum(axis=1)
    init_cost = np.abs(w[0]).sum()
    diff_cost = np.abs(w[1:] - w[:-1]).sum(axis=1)
    costs = commission_frac * np.concatenate([[init_cost], diff_cost])
    return port - costs
