"""Corwin-Schultz bid-ask spread estimator (price-based liquidity proxy).

Corwin & Schultz (2012) separate the volatility component of high-low
ranges (which scales with sqrt(t)) from the spread component (which is
constant), using the ratio of 2-day vs 1-day high-low ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def corwin_schultz_spread(
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """Estimate per-(date, ticker) bid-ask spread from OHLC ranges.

    Returns a fraction of price (0.01 = 1%), smoothed over `window` days
    and clipped to [0, 0.20]. Pandas-based — used as an offline liquidity
    filter, not in any autograd path. If you need a differentiable
    spread, build one on top of `ss_indicators.sma` + JAX rolling.
    """
    log_hl = np.log(highs / lows)
    beta = log_hl ** 2
    beta_sum = beta + beta.shift(1)

    high_2d = highs.rolling(2).max()
    low_2d = lows.rolling(2).min()
    gamma = np.log(high_2d / low_2d) ** 2

    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2) * np.sqrt(beta_sum) - np.sqrt(beta_sum)) / denom \
        - np.sqrt(gamma / denom)
    alpha = alpha.clip(lower=0)

    exp_alpha = np.exp(alpha)
    spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)
    spread = spread.rolling(window, min_periods=1).mean()
    return spread.clip(lower=0, upper=0.20)
