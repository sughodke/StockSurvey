"""Effective rank of the correlation spectrum -- the Lie-group symmetry signal.

A correlation matrix `C` of `N` assets has eigenvalues `lambda_1, ..., lambda_N`
with `sum(lambda_i) = N`. Define the participation ratio / spectral effective
rank:

    p_i = lambda_i / sum(lambda_j)
    H   = -sum_i p_i log p_i           (Shannon entropy of the spectrum)
    erank(C) = exp(H)

`erank` is in `[1, N]`:

* `erank == N`  -> uniform spectrum (`p_i = 1/N` for all i): the correlation
  structure is "as full-rank as possible", no direction dominates. The
  symmetry group acting trivially on returns is the full O(N).

* `erank -> 1`  -> all spectral mass on one mode: a single eigenvector
  explains everything. Markets are moving as a unit. The effective symmetry
  group has collapsed to the 1-D subgroup spanned by that mode.

The Lie-group framing identifies the *crisis* with the *symmetry breaking*:
historical drawdowns are preceded by `erank` falling -- the system is
selecting a low-dimensional subgroup before prices price it in. This is the
operationalization of the "all correlations going to 1" precondition.

We expose two surfaces:

* `effective_rank(corr)` -- the scalar from a precomputed correlation matrix.
* `trailing_effective_rank(prices, lookback)` -- the convenience wrapper that
  computes the rolling correlation and feeds it through `effective_rank`.

Plus `gross_exposure_modulator` -- the canonical mapping from a single
`erank` reading to a [floor, 1] gross-exposure scalar that `inference` applies
on top of HRP weights when the checkpoint opts in. The floor exists because
zero-erank readings during data anomalies (e.g. a freshly-listed name with
identical returns to its index) shouldn't kill exposure; treat the modulator
as a soft de-risker, not a kill switch.
"""

from __future__ import annotations

import numpy as np

from lie.correlation_network import trailing_correlation


def effective_rank(corr: np.ndarray) -> float:
    """Participation-ratio effective rank of a correlation matrix.

    `exp(H(lambda_normalized))`, where `lambda` are the (clipped to
    non-negative) eigenvalues of `corr`. NaN cells -> NaN result; an
    all-zero / degenerate matrix -> NaN.

    Equivalent intuition for the wary: this is also the number of equally-
    weighted modes that would carry the same Shannon entropy as the true
    spectrum. It's a *softer* version of matrix rank: smooth, differentiable
    in the entries of `corr`, and continuous as eigenvalues cross zero."""
    if not np.all(np.isfinite(corr)):
        # masked cells from `trailing_correlation` -> propagate, don't paper over
        return float('nan')
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.maximum(eigvals, 0.0)
    s = float(eigvals.sum())
    if s <= 0:
        return float('nan')
    p = eigvals / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def trailing_effective_rank(prices: np.ndarray, lookback: int) -> float:
    """Effective rank of the trailing-window correlation matrix.

    Convenience wrapper: builds the correlation, drops any name with a NaN
    return inside the window, and returns the scalar `erank` of the surviving
    sub-block. Returns NaN if fewer than 2 names have full-window data."""
    corr = trailing_correlation(prices, lookback=lookback)
    valid = ~np.isnan(corr).any(axis=1)
    if int(valid.sum()) < 2:
        return float('nan')
    sub = corr[np.ix_(valid, valid)]
    return effective_rank(sub)


def gross_exposure_modulator(
    eff_rank: float,
    n_assets: int,
    floor: float = 0.25,
) -> float:
    """Map effective rank to a gross-exposure scalar in `[floor, 1.0]`.

    Ratio = `eff_rank / n_assets`, clipped into `[floor, 1.0]`. A market in
    full diversity (`erank ~ N`) leaves exposure at 1.0; a market collapsed
    to a single mode (`erank -> 1`) is throttled toward `floor`.

    `floor` exists because (a) the participation-ratio interpretation gets
    noisy at very low erank when a few illiquid names dominate the spectrum
    and (b) a hard zero-out is the wrong shape for a continuous regime
    signal. Treat this as risk-off lean, not a kill switch -- the live
    pipeline already has a kill switch as one of its four risk rails."""
    if not np.isfinite(eff_rank):
        return 1.0
    ratio = eff_rank / max(int(n_assets), 1)
    return float(min(max(ratio, floor), 1.0))


__all__ = [
    'effective_rank',
    'trailing_effective_rank',
    'gross_exposure_modulator',
]
