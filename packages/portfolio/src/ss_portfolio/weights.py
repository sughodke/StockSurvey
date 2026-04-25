"""Portfolio weight construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def softmax_weights(
    scores: np.ndarray,
    mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Temperature-scaled softmax of `scores`, masked to liquid names.

    Inputs are 1D arrays of equal length. Masked positions get weight 0
    (regardless of their score). Output sums to 1.
    """
    s = scores / temperature + np.log(mask + 1e-12)
    s = s - s.max()
    exp_s = np.exp(s) * mask
    return exp_s / (exp_s.sum() + 1e-12)


def apply_position_cap(weights: pd.Series, max_position: float) -> pd.Series:
    """Water-fill weights to `max_position` while preserving total mass.

    Names above the cap are pinned at the cap; the remaining
    `(1 - n_capped * cap)` mass is redistributed proportionally to the
    free names' original weights. Repeats until no free name exceeds
    the cap. If `n_names * max_position < 1`, the result is uniform at
    `1 / n_names` (cap is binding for everyone).
    """
    if not 0 < max_position <= 1:
        raise ValueError(f'max_position must be in (0, 1], got {max_position}')
    n = len(weights)
    if n * max_position < 1.0 - 1e-9:
        return pd.Series(1.0 / n, index=weights.index)

    base = weights / weights.sum() if weights.sum() > 0 else weights
    w = base.copy()
    fixed: set[str] = set()
    for _ in range(n + 1):
        over = w[(w > max_position + 1e-12) & (~w.index.isin(fixed))].index
        if not len(over):
            break
        fixed.update(over)
        w.loc[list(fixed)] = max_position
        free = w.index.difference(list(fixed))
        free_total = max(1.0 - max_position * len(fixed), 0.0)
        free_orig_sum = base.loc[free].sum()
        if free_orig_sum > 0:
            w.loc[free] = base.loc[free] * (free_total / free_orig_sum)
        else:
            w.loc[free] = 0.0
    return w
