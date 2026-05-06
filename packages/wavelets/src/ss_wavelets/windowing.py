"""Recent vs historical windowed power means for regime scoring."""

from __future__ import annotations

import numpy as np


def precompute_windows(
    power: np.ndarray,
    lookback: int,
    n_tail: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (recent, historical) windowed power means for every valid date.

    Each output has shape `(n_scales, n_valid, n_tickers)` where
    `n_valid = n_dates - lookback`. For valid index i (date t = lookback + i):

        recent[..., i, ...]     = mean of power over (t - n_tail + 1, t]
        historical[..., i, ...] = mean of power over [i, t - n_tail + 1)

    The historical window has length `lookback - n_tail + 1`, the recent
    has length `n_tail`; together they tile the `lookback + 1` days
    ending at t. Computed via cumsum so cost is
    `O(n_scales * n_dates * n_tickers)` independent of `lookback`.

    Per-ticker normalization
    ------------------------
    Each ticker's power is divided by its mean over (scales, time) before
    cumsum, to keep the float32 cumsum well-conditioned across ~3000
    dates (raw CWT power on high-priced names exceeds 1e11 and overflows
    float32 cumsum). This *does* use full-history information, but the
    KL divergence in `ss_indicators.symmetric_kl_divergence` is
    invariant to a per-ticker uniform rescaling of power, so no future
    information actually leaks into training scores. If the divergence
    is ever swapped for a non-scale-invariant one, this normalizer must
    be made causal.
    """
    n_scales, n_dates, n_tickers = power.shape
    n_valid = n_dates - lookback
    n_hist = lookback - n_tail

    # TODO(review #10): per-ticker mean uses ALL TIME (axis 0=scales,
    # axis 1=dates). Safe today only because every active divergence
    # (KL/JS/cosine/L2) is invariant to a per-ticker uniform rescaling
    # — see docstring above. Add an `assert divergence in {...}` at the
    # call sites or refactor to a causal rolling mean before exposing
    # this to a non-scale-invariant downstream op.
    pm = power.mean(axis=(0, 1), keepdims=True)
    power = power / np.maximum(pm, 1e-12)

    cs = np.cumsum(power.astype(np.float64), axis=1)
    cs = np.concatenate(
        [np.zeros((n_scales, 1, n_tickers), dtype=np.float64), cs],
        axis=1,
    )

    recent = (cs[:, lookback + 1:, :]
              - cs[:, lookback - n_tail + 1: n_dates - n_tail + 1, :]) / n_tail
    historical = (cs[:, n_hist + 1: n_valid + n_hist + 1, :]
                  - cs[:, :n_valid, :]) / (n_hist + 1)
    return recent.astype(np.float32), historical.astype(np.float32)
