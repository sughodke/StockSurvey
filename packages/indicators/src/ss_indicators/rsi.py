"""Wilder relative strength index (RSI)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rsi(prices: jax.Array, n: int = 7) -> jax.Array:
    """Wilder RSI of period `n` over axis 0.

    Output range [0, 100]. Positions before index `n` are filled with the
    neutral value 50. Vectorized over all trailing axes (e.g. `(T, N)`
    input gives `(T, N)` output).
    """
    prices = jnp.asarray(prices)
    deltas = jnp.diff(prices, axis=0)
    up = jnp.where(deltas > 0, deltas, 0.0)
    down = jnp.where(deltas < 0, -deltas, 0.0)

    avg_up_seed = up[:n].mean(axis=0)
    avg_down_seed = down[:n].mean(axis=0)
    rs_seed = avg_up_seed / (avg_down_seed + 1e-9)
    rsi_seed = 100.0 - 100.0 / (1.0 + rs_seed)

    def step(carry, x):
        avg_up, avg_down = carry
        u, d = x
        avg_up = (avg_up * (n - 1) + u) / n
        avg_down = (avg_down * (n - 1) + d) / n
        rs = avg_up / (avg_down + 1e-9)
        rsi_val = 100.0 - 100.0 / (1.0 + rs)
        return (avg_up, avg_down), rsi_val

    _, rsi_tail = jax.lax.scan(
        step, (avg_up_seed, avg_down_seed),
        (up[n:], down[n:]))

    head = jnp.full((n,) + prices.shape[1:], 50.0, dtype=prices.dtype)
    return jnp.concatenate([head, rsi_seed[None], rsi_tail], axis=0)
