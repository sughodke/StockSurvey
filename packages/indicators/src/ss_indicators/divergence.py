"""Regime-shift divergences between two CWT power distributions.

Each function takes weighted recent vs historical power tensors and a
learnable `scale_log_weights` vector, returns a per-(block, ticker)
score where larger = bigger regime shift. All are differentiable so
they're drop-in for the JAX trainer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS: float = 1e-9


def _normalize(
    recent: jax.Array,
    historical: jax.Array,
    scale_log_weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply scale softmax weights and renormalize both tensors to sum to 1
    along the scale axis. Common preamble for all divergences."""
    sw = jax.nn.softmax(scale_log_weights)
    extra = (None,) * (recent.ndim - 1)
    sw_b = sw[(slice(None),) + extra]
    rw = sw_b * recent
    hw = sw_b * historical
    rd = rw / (rw.sum(axis=0, keepdims=True) + EPS)
    hd = hw / (hw.sum(axis=0, keepdims=True) + EPS)
    return rd, hd


def symmetric_kl_divergence(
    recent: jax.Array,
    historical: jax.Array,
    scale_log_weights: jax.Array,
) -> jax.Array:
    """Symmetric KL: 0.5 * (KL(rd||hd) + KL(hd||rd))."""
    rd, hd = _normalize(recent, historical, scale_log_weights)
    kl = 0.5 * jnp.sum(rd * jnp.log((rd + EPS) / (hd + EPS)), axis=0)
    kl += 0.5 * jnp.sum(hd * jnp.log((hd + EPS) / (rd + EPS)), axis=0)
    return kl


def js_divergence(
    recent: jax.Array,
    historical: jax.Array,
    scale_log_weights: jax.Array,
) -> jax.Array:
    """Jensen-Shannon divergence: 0.5 * (KL(rd||m) + KL(hd||m)) where m=(rd+hd)/2."""
    rd, hd = _normalize(recent, historical, scale_log_weights)
    m = 0.5 * (rd + hd)
    js = 0.5 * jnp.sum(rd * jnp.log((rd + EPS) / (m + EPS)), axis=0)
    js += 0.5 * jnp.sum(hd * jnp.log((hd + EPS) / (m + EPS)), axis=0)
    return js


def cosine_divergence(
    recent: jax.Array,
    historical: jax.Array,
    scale_log_weights: jax.Array,
) -> jax.Array:
    """1 - cosine(rd, hd) along the scale axis. Range [0, 2], 0 = identical.

    Floors the squared norms before sqrt to keep the gradient finite when
    a distribution collapses to ~zero (otherwise d/dx sqrt(x) at x=0 is
    inf and Adam state turns NaN within a few steps).
    """
    rd, hd = _normalize(recent, historical, scale_log_weights)
    dot = jnp.sum(rd * hd, axis=0)
    norm_r = jnp.sqrt(jnp.maximum(jnp.sum(rd ** 2, axis=0), 1e-12))
    norm_h = jnp.sqrt(jnp.maximum(jnp.sum(hd ** 2, axis=0), 1e-12))
    return 1.0 - dot / (norm_r * norm_h + EPS)


def l2_divergence(
    recent: jax.Array,
    historical: jax.Array,
    scale_log_weights: jax.Array,
) -> jax.Array:
    """Euclidean distance between rd and hd along the scale axis.

    Floors the sum-of-squares before sqrt for the same gradient-stability
    reason as `cosine_divergence`.
    """
    rd, hd = _normalize(recent, historical, scale_log_weights)
    return jnp.sqrt(jnp.maximum(jnp.sum((rd - hd) ** 2, axis=0), 1e-12))


DIVERGENCES: dict[str, callable] = {
    'kl': symmetric_kl_divergence,
    'js': js_divergence,
    'cosine': cosine_divergence,
    'l2': l2_divergence,
}


def get_divergence(name: str):
    """Look up a divergence by short name; raises KeyError if unknown."""
    try:
        return DIVERGENCES[name]
    except KeyError as e:
        raise KeyError(
            f'unknown divergence {name!r}; available: {sorted(DIVERGENCES)}'
        ) from e
