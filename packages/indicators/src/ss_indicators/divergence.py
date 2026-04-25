"""Regime-shift divergence between two CWT power distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def symmetric_kl_divergence(
    recent: jax.Array,
    historical: jax.Array,
    scale_log_weights: jax.Array,
) -> jax.Array:
    """Symmetric KL between weighted recent and historical power.

    Parameters
    ----------
    recent, historical : (n_scales, ..., n_tickers)
        Pre-computed windowed power means, e.g. output of
        `ss_wavelets.precompute_windows`.
    scale_log_weights : (n_scales,)
        Pre-softmax weights over CWT scales (typically a learned param).

    Returns
    -------
    Array with the scale axis collapsed: `(..., n_tickers)`. Larger
    values = bigger regime shift.
    """
    sw = jax.nn.softmax(scale_log_weights)
    extra = (None,) * (recent.ndim - 1)
    sw_b = sw[(slice(None),) + extra]
    rw = sw_b * recent
    hw = sw_b * historical

    eps = 1e-9
    rd = rw / (rw.sum(axis=0, keepdims=True) + eps)
    hd = hw / (hw.sum(axis=0, keepdims=True) + eps)

    kl = 0.5 * jnp.sum(rd * jnp.log((rd + eps) / (hd + eps)), axis=0)
    kl += 0.5 * jnp.sum(hd * jnp.log((hd + eps) / (rd + eps)), axis=0)
    return kl
