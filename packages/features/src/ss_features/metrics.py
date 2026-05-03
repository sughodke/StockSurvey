"""Regression-eval metrics shared across apps.

`fit_stats` returns R²/RMSE/max-|Δ| for a prediction-vs-truth
comparison. Lifted from `replay.metrics` so any consumer (replay,
factor, relational) can compute eval stats without depending on
replay (which would drag in tinygrad + scipy + matplotlib for a
single math helper).
"""

from __future__ import annotations

import numpy as np


def fit_stats(snap: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """`max_abs`, `rmse`, and `r2` of `snap` vs `gt` over finite indices.

    R² uses 1 − SS_res/SS_tot; if `gt` is constant we return 1.0 when
    SS_res == 0 else NaN.
    """
    diff = snap - gt
    finite = np.isfinite(diff) & np.isfinite(gt)
    if not finite.any():
        return {'max_abs': float('nan'), 'rmse': float('nan'),
                'r2': float('nan')}
    d = diff[finite]
    g = gt[finite]
    ss_res = float(np.sum(d ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else float('nan')
    else:
        r2 = 1.0 - ss_res / ss_tot
    return {
        'max_abs': float(np.max(np.abs(d))),
        'rmse': float(np.sqrt(ss_res / len(d))),
        'r2': r2,
    }
