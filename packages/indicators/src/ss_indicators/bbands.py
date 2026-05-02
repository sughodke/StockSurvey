"""Bollinger Bands — pure numpy."""

from __future__ import annotations

import numpy as np

from ss_indicators.moving_average import rolling_std, sma


def bbands(
    prices: np.ndarray,
    window: int = 21,
    nsd: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (middle, upper, lower) bands over axis 0.

    middle = SMA(window); upper = middle + nsd * rolling_std(window);
    lower = middle - nsd * rolling_std(window). Expanding window during
    warm-up (matches pandas `rolling(window, min_periods=1)`).
    """
    prices = np.asarray(prices)
    middle = sma(prices, window)
    sd = rolling_std(prices, window)
    return middle, middle + nsd * sd, middle - nsd * sd
