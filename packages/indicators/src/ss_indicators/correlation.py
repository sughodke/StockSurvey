"""Rolling-window Pearson correlation between two 1-D series — pure numpy.

Used as the cheap deterministic-indicator analogue of the regime
trainer's `coherence` term (see `apps/regime/src/regime/trainer.py`'s
`weights_scalogram`): coherence there is the trailing-window Pearson
correlation between shortest-scale and longest-scale CWT power. Without
the CWT we substitute realized-vol over short and long windows and
correlate those instead; that's what `apps/factor` consumes.
"""

from __future__ import annotations

import numpy as np

EPS: float = 1e-12


def rolling_pearson_corr(
    x: np.ndarray, y: np.ndarray, window: int,
) -> np.ndarray:
    """Trailing-`window` Pearson correlation between two aligned 1-D series.

    Uses cumulative sums for O(T) total cost regardless of `window`.
    Output is 1-D with the same length as the inputs; positions before
    the first bar with a fully populated window of finite (x, y) pairs
    are NaN. If the trailing window contains any NaN in either series,
    that bar's output is NaN (strict — no expanding-window fallback,
    unlike `sma` / `rolling_std`).

    Numerical-stability rules mirror `divergence.cosine_divergence`:
    the variance of either series is floored at `EPS` before the sqrt;
    if either floor is hit the correlation is set to `0.0` (degenerate
    constant series — correlation undefined). Output is clipped to
    `[-1, 1]` to absorb float drift.

    Promotes to float64 internally because `cumsum` of squared price-
    scale values suffers catastrophic cancellation at float32 (same
    reason `rolling_std` does it).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f'rolling_pearson_corr requires 1-D inputs, got x.ndim={x.ndim} '
            f'y.ndim={y.ndim}')
    if x.shape != y.shape:
        raise ValueError(
            f'rolling_pearson_corr x and y must have the same shape, got '
            f'{x.shape} vs {y.shape}')
    if window < 2:
        raise ValueError(
            f'rolling_pearson_corr window must be >= 2, got {window}')

    T = len(x)
    out = np.full(T, np.nan, dtype=np.float64)
    if T < window:
        return out

    finite = np.isfinite(x) & np.isfinite(y)
    x_clean = np.where(finite, x, 0.0)
    y_clean = np.where(finite, y, 0.0)

    # Prepend a 0 so window sum [lo, hi) becomes cs[hi] - cs[lo] without
    # special-casing lo=0.
    cs_x = np.concatenate([[0.0], np.cumsum(x_clean)])
    cs_y = np.concatenate([[0.0], np.cumsum(y_clean)])
    cs_xx = np.concatenate([[0.0], np.cumsum(x_clean * x_clean)])
    cs_yy = np.concatenate([[0.0], np.cumsum(y_clean * y_clean)])
    cs_xy = np.concatenate([[0.0], np.cumsum(x_clean * y_clean)])
    cs_n = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])

    # Window covering bars [lo, t] inclusive (length `window`) for every
    # bar t where a full window fits.
    t_idx = np.arange(window - 1, T)
    lo = t_idx - window + 1
    hi = t_idx + 1

    n_in_window = cs_n[hi] - cs_n[lo]
    full_window = n_in_window == window

    sx = cs_x[hi] - cs_x[lo]
    sy = cs_y[hi] - cs_y[lo]
    sxx = cs_xx[hi] - cs_xx[lo]
    syy = cs_yy[hi] - cs_yy[lo]
    sxy = cs_xy[hi] - cs_xy[lo]

    inv_w = 1.0 / window
    mu_x = sx * inv_w
    mu_y = sy * inv_w
    var_x = sxx * inv_w - mu_x * mu_x
    var_y = syy * inv_w - mu_y * mu_y
    cov = sxy * inv_w - mu_x * mu_y

    var_x_safe = np.maximum(var_x, EPS)
    var_y_safe = np.maximum(var_y, EPS)
    corr = cov / np.sqrt(var_x_safe * var_y_safe)
    # Constant series in either window -> correlation undefined; pick 0.
    corr = np.where((var_x < EPS) | (var_y < EPS), 0.0, corr)
    # Clamp float drift to [-1, 1].
    corr = np.clip(corr, -1.0, 1.0)
    # Window straddling NaN(s) in either series -> NaN per the docstring.
    corr = np.where(full_window, corr, np.nan)

    out[t_idx] = corr
    return out
