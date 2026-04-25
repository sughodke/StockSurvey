"""Bollinger Bands."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ss_indicators.moving_average import rolling_std, sma


def bbands(
    prices: jax.Array,
    window: int = 21,
    nsd: float = 2.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (middle, upper, lower) bands over axis 0.

    middle = SMA(window); upper = middle + nsd * rolling_std(window);
    lower = middle - nsd * rolling_std(window). Expanding window during
    warm-up (matches pandas `rolling(window, min_periods=1)`).
    """
    prices = jnp.asarray(prices)
    middle = sma(prices, window)
    sd = rolling_std(prices, window)
    return middle, middle + nsd * sd, middle - nsd * sd
