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


def select_top_n_matrix(
    scores_matrix: np.ndarray,
    top_n: int,
    ascending: bool = True,
) -> np.ndarray:
    """Convert `(n_dates, n_tickers)` scores into an equal-weight allocation.

    At each date allocate `1 / top_n` to the `top_n` best names. NaN
    scores are excluded; if fewer than `top_n` valid scores exist the
    row is left at zero (no allocation that day).

    `ascending=True` selects the lowest scores (e.g. lowest RSI = most
    oversold); `ascending=False` selects the highest (e.g. highest
    regime divergence).
    """
    n_dates, n_tickers = scores_matrix.shape
    weights = np.zeros_like(scores_matrix)
    w = 1.0 / top_n

    for i in range(n_dates):
        row = scores_matrix[i]
        valid = ~np.isnan(row)
        if valid.sum() < top_n:
            continue
        if ascending:
            ranked = np.argsort(np.where(valid, row, np.inf))
        else:
            ranked = np.argsort(np.where(valid, -row, np.inf))
        weights[i, ranked[:top_n]] = w
    return weights


def apply_position_cap(weights: pd.Series, max_position: float) -> pd.Series:
    """Water-fill weights to `max_position` while preserving total mass.

    Names above the cap are pinned at the cap; the remaining
    `(1 - n_capped * cap)` mass is redistributed proportionally to the
    free names' original weights. Repeats until no free name exceeds
    the cap.

    Degenerate small-universe handling: if the number of *nonzero*
    input weights times `max_position` is less than the input total
    mass, the cap relaxes — distribute uniformly across the nonzero
    names only, **scaled to `min(input_sum, 1.0)`**. This preserves
    intentional sub-100% gross exposure (CB-4): a caller that hands in
    weights summing to 0.5 (e.g. half the universe spread-gated out)
    receives an output that sums to 0.5, not 1.0. Zero-weight names
    stay at zero (e.g. spread-gated illiquid names must not get
    re-introduced by the cap step). All-zero input short-circuits to
    all-zero output.
    """
    if not 0 < max_position <= 1:
        raise ValueError(f'max_position must be in (0, 1], got {max_position}')
    nonzero_mask = weights > 0
    n_nonzero = int(nonzero_mask.sum())
    if n_nonzero == 0:
        return weights.copy()
    if n_nonzero * max_position < 1.0 - 1e-9:
        out = pd.Series(0.0, index=weights.index, name=weights.name)
        target_total = min(float(weights.sum()), 1.0)
        out.loc[nonzero_mask] = target_total / n_nonzero
        return out

    base = weights / weights.sum() if weights.sum() > 0 else weights
    w = base.copy()
    fixed: set[str] = set()
    for _ in range(len(weights) + 1):
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
