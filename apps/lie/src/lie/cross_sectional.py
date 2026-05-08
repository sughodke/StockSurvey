"""Cross-sectional information-coefficient summary helpers.

The v3 hypothesis is cross-sectional: given the universe-state coordinate
plus per-ticker shape features, can a kNN over historical (date, ticker)
pairs rank today's tickers by forward excess return?

The natural metric is per-date Spearman rank correlation between
predicted excess returns and realized excess returns, averaged across
test dates. With 21 tickers per day the per-date IC is noisy; the
mean-IC across ~1000+ test dates is what carries statistical weight.

We also return a t-statistic of the mean IC, the standard "is this
better than zero" gut-check for a cross-sectional signal:

    t = mean(IC) / (std(IC) / sqrt(n_dates))

|t| > 2 is the conventional threshold; > 3 is publishable. Anything
positive and beating the AR(1) baseline is at least directionally
informative.
"""

from __future__ import annotations

import numpy as np


def cross_sectional_ic_summary(
    preds: np.ndarray,
    targets: np.ndarray,
    date_idx: np.ndarray,
    method: str = 'spearman',
    min_per_date: int = 5,
) -> dict[str, float | int | list[float]]:
    """Compute per-date IC + summary stats.

    Parameters
    ----------
    preds, targets :
        `(M,)` aligned arrays of predictions and realized targets.
    date_idx :
        `(M,)` integer date indices. Pairs sharing the same `date_idx`
        are treated as one cross-section.
    method :
        `'spearman'` (default) or `'pearson'`.
    min_per_date :
        Drop any date with fewer than this many finite (pred, target)
        pairs. Default 5.

    Returns
    -------
    dict with keys:
        `mean_ic`       -- mean of per-date ICs
        `std_ic`        -- sample std of per-date ICs
        `t_stat`        -- mean / (std / sqrt(n))
        `n_dates`       -- number of dates that contributed
        `frac_positive` -- fraction of dates with IC > 0
        `ic_p25`, `ic_p50`, `ic_p75` -- IC distribution quantiles
        `ic_series`     -- per-date IC list (for plotting)
    """
    if preds.shape != targets.shape or preds.shape != date_idx.shape:
        raise ValueError('preds / targets / date_idx must be same shape')
    if method not in ('spearman', 'pearson'):
        raise ValueError(f"method must be 'spearman' or 'pearson'; got {method!r}")

    finite = np.isfinite(preds) & np.isfinite(targets)
    p = preds[finite]
    t = targets[finite]
    d = date_idx[finite]

    if len(p) == 0:
        return _empty_summary()

    # Sort by date so we can find run boundaries cheaply.
    order = np.argsort(d, kind='stable')
    p = p[order]; t = t[order]; d = d[order]

    ics: list[float] = []
    starts = np.where(np.diff(d, prepend=d[0] - 1) != 0)[0]
    starts = np.append(starts, len(d))

    for i in range(len(starts) - 1):
        s, e = int(starts[i]), int(starts[i + 1])
        if e - s < min_per_date:
            continue
        ic = _rankcorr(p[s:e], t[s:e]) if method == 'spearman' \
            else _pearson(p[s:e], t[s:e])
        if np.isfinite(ic):
            ics.append(float(ic))

    if not ics:
        return _empty_summary()

    arr = np.asarray(ics)
    n = len(arr)
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {
        'mean_ic': float(arr.mean()),
        'std_ic': std,
        't_stat': float(arr.mean() / (std / np.sqrt(n))) if std > 0 else float('nan'),
        'n_dates': n,
        'frac_positive': float((arr > 0).mean()),
        'ic_p25': float(np.percentile(arr, 25)),
        'ic_p50': float(np.percentile(arr, 50)),
        'ic_p75': float(np.percentile(arr, 75)),
        'ic_series': ics,
    }


def _empty_summary() -> dict[str, float | int | list[float]]:
    return {
        'mean_ic': float('nan'), 'std_ic': float('nan'), 't_stat': float('nan'),
        'n_dates': 0, 'frac_positive': float('nan'),
        'ic_p25': float('nan'), 'ic_p50': float('nan'), 'ic_p75': float('nan'),
        'ic_series': [],
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean(); ym = y - y.mean()
    denom = float(np.sqrt(np.sum(xm * xm) * np.sum(ym * ym)))
    return float(np.sum(xm * ym) / denom) if denom > 0 else float('nan')


def _rankcorr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation via average-rank Pearson. Avoids the scipy
    dep on the hot path; ties are handled via `argsort.argsort` ranking."""
    rx = _avg_rank(x)
    ry = _avg_rank(y)
    return _pearson(rx, ry)


def _avg_rank(a: np.ndarray) -> np.ndarray:
    """1-based average rank with ties broken by mean (matches scipy default).
    Vectorized via sort + run-length tie group averaging."""
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    order = np.argsort(a, kind='stable')
    ranks = np.empty(n, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0  # 1-based mean of [i+1 .. j]
        ranks[order[i:j]] = avg
        i = j
    return ranks


__all__ = ['cross_sectional_ic_summary']
