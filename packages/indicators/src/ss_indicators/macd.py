"""MACD (Moving Average Convergence/Divergence) — pure numpy."""

from __future__ import annotations

import numpy as np

from ss_indicators.moving_average import ema


def macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (macd_line, signal_line, histogram) over axis 0.

    macd_line  = ema(fast) - ema(slow)
    signal     = ema(macd_line, signal)
    histogram  = macd_line - signal_line
    """
    prices = np.asarray(prices)
    macd_line = ema(prices, fast) - ema(prices, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line
