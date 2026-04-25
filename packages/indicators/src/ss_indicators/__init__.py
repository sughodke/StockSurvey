"""ss_indicators: JAX technical indicators (matrix form).

All indicators operate on a `(T, ...)` time-leading array where the first
axis is the date axis. Single-ticker callers pass `(T,)` and get `(T,)`
back; multi-ticker callers pass `(T, N)` and get `(T, N)` back. Outputs
are `jax.numpy` arrays — convert with `np.asarray(...)` at the boundary
if you need numpy.

Indicators
----------
  * `rsi`                       — Wilder relative strength index
  * `macd`                      — MACD line + signal + histogram
  * `bbands`                    — Bollinger middle/upper/lower
  * `sma`, `ema`, `rolling_std` — moving-average primitives
  * `corwin_schultz_spread`     — bid-ask spread proxy from H/L
  * `symmetric_kl_divergence`   — regime score from CWT power distributions
  * `fibonacci_retracement`     — support/resistance levels (legacy plotting)
"""

from ss_indicators.bbands import bbands
from ss_indicators.divergence import (
    DIVERGENCES,
    cosine_divergence,
    get_divergence,
    js_divergence,
    l2_divergence,
    symmetric_kl_divergence,
)
from ss_indicators.fibonacci import FIB_LEVELS, fibonacci_retracement
from ss_indicators.macd import macd
from ss_indicators.moving_average import ema, rolling_std, sma
from ss_indicators.rsi import rsi
from ss_indicators.spread import corwin_schultz_spread

__all__ = [
    'DIVERGENCES',
    'FIB_LEVELS',
    'bbands',
    'corwin_schultz_spread',
    'cosine_divergence',
    'ema',
    'fibonacci_retracement',
    'get_divergence',
    'js_divergence',
    'l2_divergence',
    'macd',
    'rolling_std',
    'rsi',
    'sma',
    'symmetric_kl_divergence',
]
