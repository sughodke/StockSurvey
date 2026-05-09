"""MACD (Moving Average Convergence/Divergence) — pure numpy."""

from __future__ import annotations

import numpy as np

from ss_indicators.moving_average import ema


# Textbook MACD is `(fast=12, slow=26, signal=9)`. The slow/fast and
# signal/fast ratios encode the canonical timescale relationship; any
# parameter sweep over `fast` alone should hold those ratios fixed
# rather than re-deriving them from a simpler (and wrong) rule like
# `slow = 2 * fast`. Centralizing them here keeps the SSL trainer
# (apps/replay), the indicator-feature scorer (apps/factor), and the
# eval helpers (apps/replay/eval) using one consistent definition;
# prior to 2026-05-09 each had its own `slow = 2 * fast` re-derivation,
# which collided with the canonical anchor at fast=12 and contaminated
# multi-head training (slow=24 grid cell vs slow=26 anchor at the same
# FiLM cond).
CANONICAL_MACD_FAST: int = 12
CANONICAL_MACD_SLOW: int = 26
CANONICAL_MACD_SIGNAL: int = 9
CANONICAL_SLOW_RATIO: float = CANONICAL_MACD_SLOW / CANONICAL_MACD_FAST    # 26/12
CANONICAL_SIGNAL_RATIO: float = CANONICAL_MACD_SIGNAL / CANONICAL_MACD_FAST  # 9/12


def macd_periods_from_fast(fast: int) -> tuple[int, int, int]:
    """Return `(fast, slow, signal)` scaling the canonical (12, 26, 9)
    triple by the requested `fast` period.

    Holds the textbook timescale ratios (slow ≈ 2.167*fast,
    signal ≈ 0.75*fast) so every MACD-grid cell is a simple
    re-parameterization of the same canonical operator. At
    `fast=CANONICAL_MACD_FAST=12` this returns `(12, 26, 9)` exactly,
    so a grid that includes 12 collapses cleanly onto the canonical
    anchor (no slow=24 vs slow=26 mismatch).

    `signal` is floored at 2 since `ema(.., signal=1)` is the identity
    and yields a zero histogram.
    """
    fast = int(fast)
    slow = int(round(fast * CANONICAL_SLOW_RATIO))
    sig = max(2, int(round(fast * CANONICAL_SIGNAL_RATIO)))
    return fast, slow, sig


def macd(
    prices: np.ndarray,
    fast: int = CANONICAL_MACD_FAST,
    slow: int = CANONICAL_MACD_SLOW,
    signal: int = CANONICAL_MACD_SIGNAL,
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


def macd_from_fast(
    prices: np.ndarray, fast: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-knob MACD over `fast`, holding canonical slow/signal ratios.

    Convenience wrapper used by FiLM-conditioned MACD heads where the
    grid sweeps `fast` only. Equivalent to
    `macd(prices, *macd_periods_from_fast(fast))` — exposed as a
    separate function so callers can't accidentally mismatch the
    triple by passing a custom `slow` while expecting canonical
    proportions.
    """
    f, s, sig = macd_periods_from_fast(fast)
    return macd(prices, fast=f, slow=s, signal=sig)


def macd_log(
    prices: np.ndarray,
    fast: int = CANONICAL_MACD_FAST,
    slow: int = CANONICAL_MACD_SLOW,
    signal: int = CANONICAL_MACD_SIGNAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-price MACD: `EMA(log(p), fast) - EMA(log(p), slow)`.

    Canonical MACD is in dollar units, which is fine on a single
    ticker but pathological as a cross-sectional target across
    price-disparate names: a $1 stock and a $1000 stock have MACD
    magnitudes that differ by a factor of ~1000. Pooling such tickers
    and standardizing globally then makes predictions on the
    low-priced ticker hundreds of times too large.

    The log-price variant is **scale-invariant** — `MACD(log(c*p)) =
    MACD(log(p))` for any positive constant `c`, so a $1 stock and a
    $1000 stock with the same percentage trend produce the same
    log-MACD. Typical magnitude is ~0.001-0.1 across all tickers,
    matching the rest of the SSL bundle (which already operates on
    log-returns and cumulative-log-return Gaussian smoothing).

    This is the recommended target series for SSL-style multi-ticker
    pooled training; reserve `macd()` (raw) for single-ticker
    indicators-as-features use cases where the dollar-unit
    interpretation is desired.
    """
    return macd(np.log(np.asarray(prices)), fast=fast, slow=slow,
                signal=signal)


def macd_log_from_fast(
    prices: np.ndarray, fast: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-price MACD with canonical `(fast, slow, signal)` ratios.

    The scale-invariant analog of `macd_from_fast`. See `macd_log`
    for why the log-price formulation is the canonical
    cross-sectional target.
    """
    f, s, sig = macd_periods_from_fast(fast)
    return macd_log(prices, fast=f, slow=s, signal=sig)
