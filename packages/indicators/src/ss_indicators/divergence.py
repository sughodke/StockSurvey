"""Regime-shift divergences between two CWT power distributions — numpy.

Each function takes weighted recent vs historical power tensors and a
`scale_log_weights` vector, returns a per-(block, ticker) score where
larger = bigger regime shift. Pure-numpy port — gradients no longer
flow through these (autograd-driven `optimize_adam.py` is parked); the
default Optuna+vectorbt regime trainer just consumes the scores.
"""

from __future__ import annotations

import numpy as np

EPS: float = 1e-9


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis 0."""
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def _normalize(
    recent: np.ndarray,
    historical: np.ndarray,
    scale_log_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply scale softmax weights and renormalize both tensors to sum to 1
    along the scale axis. Common preamble for all divergences."""
    sw = _softmax(np.asarray(scale_log_weights))
    extra = (None,) * (recent.ndim - 1)
    sw_b = sw[(slice(None),) + extra]
    rw = sw_b * recent
    hw = sw_b * historical
    rd = rw / (rw.sum(axis=0, keepdims=True) + EPS)
    hd = hw / (hw.sum(axis=0, keepdims=True) + EPS)
    return rd, hd


def symmetric_kl_divergence(
    recent: np.ndarray,
    historical: np.ndarray,
    scale_log_weights: np.ndarray,
) -> np.ndarray:
    """Symmetric KL: 0.5 * (KL(rd||hd) + KL(hd||rd))."""
    rd, hd = _normalize(recent, historical, scale_log_weights)
    kl = 0.5 * np.sum(rd * np.log((rd + EPS) / (hd + EPS)), axis=0)
    kl += 0.5 * np.sum(hd * np.log((hd + EPS) / (rd + EPS)), axis=0)
    return kl


def js_divergence(
    recent: np.ndarray,
    historical: np.ndarray,
    scale_log_weights: np.ndarray,
) -> np.ndarray:
    """Jensen-Shannon divergence: 0.5 * (KL(rd||m) + KL(hd||m)) where m=(rd+hd)/2."""
    rd, hd = _normalize(recent, historical, scale_log_weights)
    m = 0.5 * (rd + hd)
    js = 0.5 * np.sum(rd * np.log((rd + EPS) / (m + EPS)), axis=0)
    js += 0.5 * np.sum(hd * np.log((hd + EPS) / (m + EPS)), axis=0)
    return js


def cosine_divergence(
    recent: np.ndarray,
    historical: np.ndarray,
    scale_log_weights: np.ndarray,
) -> np.ndarray:
    """1 - cosine(rd, hd) along the scale axis. Range [0, 2], 0 = identical.

    Floors the squared norms before sqrt to keep the result finite when a
    distribution collapses to ~zero.
    """
    rd, hd = _normalize(recent, historical, scale_log_weights)
    dot = np.sum(rd * hd, axis=0)
    norm_r = np.sqrt(np.maximum(np.sum(rd ** 2, axis=0), 1e-12))
    norm_h = np.sqrt(np.maximum(np.sum(hd ** 2, axis=0), 1e-12))
    return 1.0 - dot / (norm_r * norm_h + EPS)


def l2_divergence(
    recent: np.ndarray,
    historical: np.ndarray,
    scale_log_weights: np.ndarray,
) -> np.ndarray:
    """Euclidean distance between rd and hd along the scale axis.

    Floors the sum-of-squares before sqrt for the same numerical-stability
    reason as `cosine_divergence`.
    """
    rd, hd = _normalize(recent, historical, scale_log_weights)
    return np.sqrt(np.maximum(np.sum((rd - hd) ** 2, axis=0), 1e-12))


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
