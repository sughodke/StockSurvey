"""Fibonacci retracement levels.

Kept as a plain numpy routine — used by legacy v1 plotting, not in any
hot loop, so JAX-ifying buys nothing.
"""

from __future__ import annotations

import numpy as np

FIB_LEVELS: list[float] = [0.0, 14.6, 23.6, 38.2, 50.0, 61.8, 100.0]


def fibonacci_retracement(
    prices: np.ndarray,
    n: int = 90,
) -> tuple[int, int, list[float]]:
    """Return `(t1, t2, [fib_prices])` over the trailing `n` samples.

    `t1`, `t2` are absolute indices into `prices` (start and end of the
    trend window); `fib_prices` are the price levels at each ratio in
    `FIB_LEVELS`.
    """
    p = prices[-n:]
    min_idx, max_idx = int(np.argmin(p)), int(np.argmax(p))
    t1, t2 = min(min_idx, max_idx), max(min_idx, max_idx)
    p1, p2 = float(p[t1]), float(p[t2])
    diff = p2 - p1
    levels = [p1 + diff * lvl / 100.0 for lvl in FIB_LEVELS]
    offset = max(len(prices) - n, 0)
    return offset + t1, offset + t2, levels
