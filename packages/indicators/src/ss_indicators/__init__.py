"""ss_indicators: technical indicators (numpy matrix form).

All indicators operate on a `(T, ...)` time-leading array where the first
axis is the date axis. Single-ticker callers pass `(T,)` and get `(T,)`
back; multi-ticker callers pass `(T, N)` and get `(T, N)` back.

Pure numpy after the JAX migration — there's no autograd path through
these anymore. The legacy JAX-Adam regime trainer
(`apps/regime/src/regime/research/optimize_adam.py`) is parked because
it depended on differentiable indicators; the default Optuna+vectorbt
trainer just consumes the scores and is unaffected.

Indicators
----------
  * `rsi`, `rsi_strided`         — Wilder relative strength index (matrix
                                   form + per-bar stride-w grid for FiLM)
  * `cci`, `cci_strided`         — Commodity Channel Index (close-only)
  * `macd`                       — MACD line + signal + histogram
  * `bbands`                     — Bollinger middle/upper/lower
  * `sma`, `ema`, `rolling_std`  — moving-average primitives
  * `corwin_schultz_spread`      — bid-ask spread proxy from H/L (pandas)
  * `symmetric_kl_divergence`    — regime score from CWT power
                                   distributions (KL/JS/cosine/L2)
  * `fibonacci_retracement`      — support/resistance levels (legacy plot)
"""

from ss_indicators.bbands import bbands
from ss_indicators.cci import cci, cci_strided
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
from ss_indicators.rsi import rsi, rsi_strided
from ss_indicators.spread import corwin_schultz_spread

__all__ = [
    'DIVERGENCES',
    'FIB_LEVELS',
    'bbands',
    'cci',
    'cci_strided',
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
    'rsi_strided',
    'sma',
    'symmetric_kl_divergence',
]
