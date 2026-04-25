"""Moving-average primitives: SMA, EMA, rolling_std.

All accept a `(T, ...)` array and operate on the leading time axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sma(x: jax.Array, window: int) -> jax.Array:
    """Causal simple moving average of length `window` over axis 0.

    Uses a cumsum trick: O(T) regardless of window size. For positions
    `t < window` the average is taken over the first `t + 1` samples
    (i.e. expanding window during warm-up), matching pandas
    `rolling(window, min_periods=1).mean()`.
    """
    x = jnp.asarray(x)
    T = x.shape[0]
    cs = jnp.concatenate([jnp.zeros((1,) + x.shape[1:], dtype=x.dtype),
                          jnp.cumsum(x, axis=0)], axis=0)
    idx = jnp.arange(T)
    lo = jnp.maximum(0, idx - window + 1)
    counts = (idx - lo + 1).astype(x.dtype)
    extra_dims = (None,) * (x.ndim - 1)
    return (cs[idx + 1] - cs[lo]) / counts[(slice(None),) + extra_dims]


def rolling_std(x: jax.Array, window: int) -> jax.Array:
    """Causal rolling sample-std of length `window` over axis 0.

    Uses cumsum of x and x**2; expanding window during warm-up. Matches
    `pandas.rolling(window, min_periods=1).std(ddof=0)` (population std).
    """
    x = jnp.asarray(x)
    T = x.shape[0]
    cs = jnp.concatenate([jnp.zeros((1,) + x.shape[1:], dtype=x.dtype),
                          jnp.cumsum(x, axis=0)], axis=0)
    cs2 = jnp.concatenate([jnp.zeros((1,) + x.shape[1:], dtype=x.dtype),
                           jnp.cumsum(x ** 2, axis=0)], axis=0)
    idx = jnp.arange(T)
    lo = jnp.maximum(0, idx - window + 1)
    counts = (idx - lo + 1).astype(x.dtype)
    extra_dims = (None,) * (x.ndim - 1)
    cnt = counts[(slice(None),) + extra_dims]
    mu = (cs[idx + 1] - cs[lo]) / cnt
    mu2 = (cs2[idx + 1] - cs2[lo]) / cnt
    return jnp.sqrt(jnp.maximum(mu2 - mu ** 2, 0.0))


def ema(x: jax.Array, span: int) -> jax.Array:
    """Exponential moving average with smoothing factor 2/(span+1).

    Time-recurrent so implemented via `jax.lax.scan` along axis 0. The
    first sample is used as the seed (matches pandas
    `ewm(span=..., adjust=False)`).
    """
    x = jnp.asarray(x)
    alpha = jnp.asarray(2.0 / (span + 1), dtype=x.dtype)

    def step(carry, xi):
        new = alpha * xi + (1.0 - alpha) * carry
        return new, new

    _, out = jax.lax.scan(step, x[0], x[1:])
    return jnp.concatenate([x[:1], out], axis=0)
