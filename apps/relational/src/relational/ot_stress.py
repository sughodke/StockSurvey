"""R1 — Sinkhorn optimal-transport distance from today's fingerprint
cloud to a fixed "calm regime" reference cloud, with per-ticker
attribution.

The reference cloud is built once from the first `ref_window_days`
post-lookback dates and is *not* rolled — rolling a "calm" reference
would let it drift into the high-vol regime and lose its grounding.
Per evaluation date, we compute the cost matrix between today's
finite-fingerprint tickers and the reference cloud, run Sinkhorn-Knopp
to get a soft transport plan `P`, and report each ticker's expected
transport cost `sum_j P[i, j] * M[i, j]` as its stress score. High
score = "fingerprint hard to transport to anything in the calm
reference," i.e. unusual relative to the historical low-stress regime.

Pure-numpy Sinkhorn — no `pot` dep. Reference cloud is downsampled to
`max_ref` points to keep per-rebalance compute cheap (the cost matrix
is n_tickers × max_ref).

Eval dates inside the reference window are emitted as NaN: scoring a
date against a reference that includes itself is degenerate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


def _sinkhorn_knopp(
    M: np.ndarray, a: np.ndarray, b: np.ndarray,
    *, eps: float, n_iter: int, tol: float,
) -> np.ndarray:
    """Sinkhorn-Knopp on cost matrix `M` (shape `(n_a, n_b)`) with
    marginals `a`, `b`. Returns transport plan `P`."""
    K = np.exp(-M / eps)
    u = np.ones_like(a, dtype=np.float64)
    v = np.ones_like(b, dtype=np.float64)
    for _ in range(n_iter):
        u_prev = u
        v = b / (K.T @ u + 1e-30)
        u = a / (K @ v + 1e-30)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]


def ot_stress_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    fp_window: int = 21,
    ref_window_days: int = 252,
    max_ref: int = 500,
    eps: float = 1.0,
    n_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
    cache_dir=None,
) -> np.ndarray:
    """Per-(date, ticker) OT stress score relative to a calm reference.

    Returns `(n_eval, n_tickers)` matching the scorer shape contract.
    The first `ref_window_days` rows are NaN (those dates *built* the
    reference). With `eps=1.0` and unit-normed fingerprints (so squared
    L2 ∈ [0, 4]), the kernel `exp(-M/eps)` is numerically tame.
    """
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)
    fps_eval = fps[lookback:]
    n_eval, n_tickers, fp_dim = fps_eval.shape

    # Build the calm reference: pool all (ticker, date) finite
    # fingerprints from the first ref_window_days post-lookback. We
    # treat them as a flat point cloud regardless of which ticker each
    # came from; the OT distance is between distributions, not panels.
    ref_block = fps_eval[:ref_window_days].reshape(-1, fp_dim)
    finite = np.isfinite(ref_block).all(axis=1)
    ref_cloud = ref_block[finite]
    if len(ref_cloud) < 2:
        raise RuntimeError(
            f'OT reference cloud is empty/sparse '
            f'(n_ref={len(ref_cloud)}); '
            f'increase ref_window_days or check fingerprint NaN rate.')
    if len(ref_cloud) > max_ref:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ref_cloud), size=max_ref, replace=False)
        ref_cloud = ref_cloud[idx]
    n_ref = len(ref_cloud)

    scores = np.full((n_eval, n_tickers), np.nan, dtype=np.float32)
    b = np.full(n_ref, 1.0 / n_ref)
    for t in range(ref_window_days, n_eval):
        row = fps_eval[t]
        finite_t = np.isfinite(row).all(axis=1)
        n_t = int(finite_t.sum())
        if n_t < 2:
            continue
        valid = np.where(finite_t)[0]
        src = row[valid]
        a = np.full(n_t, 1.0 / n_t)
        # Squared-L2 cost; unit-norm fingerprints → cost in [0, 4].
        diff = src[:, None, :] - ref_cloud[None, :, :]
        M = (diff * diff).sum(axis=-1)
        P = _sinkhorn_knopp(M, a, b, eps=eps, n_iter=n_iter, tol=tol)
        per_ticker_cost = (P * M).sum(axis=1)
        scores[t, valid] = per_ticker_cost.astype(np.float32)
    return scores
