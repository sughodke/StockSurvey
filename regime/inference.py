"""Inference forward pass: prices in, target weights out.

This is the train-time forward pass with the optimization scaffolding
removed. It loads a `Checkpoint`, builds the same CWT + recent/historical
windows + softmax pipeline, and returns the soft top-N weights for the
*latest* date in the supplied OHLC frames.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from regime.cwt import causal_cwt, precompute_windows
from regime.data import corwin_schultz_spread
from regime.persist import Checkpoint
from regime.strategy import regime_scores


def target_weights(
    prices: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    checkpoint: Checkpoint,
) -> pd.Series:
    """Compute the strategy's target portfolio weights for the latest bar.

    Parameters
    ----------
    prices, highs, lows :
        Wide DataFrames indexed by date, columns are tickers. Must share
        an index and column set, and contain at least `lookback + 1` rows.
        The most recent row is the rebalance date.
    checkpoint :
        Trained model produced by `persist.load_checkpoint`.

    Returns
    -------
    pd.Series
        Target portfolio weights indexed by ticker, summing to 1.
        Tickers absent from `prices.columns` (e.g. delisted) are dropped.
    """
    if not (prices.columns.equals(highs.columns) and prices.columns.equals(lows.columns)):
        raise ValueError('prices/highs/lows must share columns')
    if not (prices.index.equals(highs.index) and prices.index.equals(lows.index)):
        raise ValueError('prices/highs/lows must share index')
    if len(prices) < checkpoint.lookback + 1:
        raise ValueError(
            f'need at least {checkpoint.lookback + 1} bars, got {len(prices)}')

    prices_np = prices.values.astype(np.float64)
    coeffs = causal_cwt(prices_np, checkpoint.scales, checkpoint.lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(power, checkpoint.lookback, checkpoint.n_tail)

    spread_df = corwin_schultz_spread(highs, lows)
    liquid = (spread_df.values <= checkpoint.max_spread).astype(np.float32)
    liquid_last = liquid[-1]

    params = checkpoint.jax_params()
    recent_last = jnp.asarray(recent[:, -1:, :])
    historical_last = jnp.asarray(historical[:, -1:, :])
    scores = np.asarray(regime_scores(
        recent_last, historical_last, params['scale_log_weights']))[0]

    temp = float(np.exp(checkpoint.log_temperature))
    s = scores / temp + np.log(liquid_last + 1e-12)
    s = s - s.max()
    exp_s = np.exp(s) * liquid_last
    weights = exp_s / (exp_s.sum() + 1e-12)
    return pd.Series(weights, index=prices.columns, name=prices.index[-1])


def apply_position_cap(weights: pd.Series, max_position: float) -> pd.Series:
    """Water-fill weights to `max_position` while preserving total mass.

    Names above the cap are pinned at the cap; the remaining
    (1 - n_capped * cap) mass is redistributed proportionally to the
    free names' original weights. Repeats until no free name exceeds
    the cap. If `n_names * max_position < 1`, the result is uniform at
    `1/n_names` (cap is binding for everyone).
    """
    if not 0 < max_position <= 1:
        raise ValueError(f'max_position must be in (0, 1], got {max_position}')
    n = len(weights)
    if n * max_position < 1.0 - 1e-9:
        return pd.Series(1.0 / n, index=weights.index)

    base = weights / weights.sum() if weights.sum() > 0 else weights
    w = base.copy()
    fixed: set[str] = set()
    for _ in range(n + 1):
        over = w[(w > max_position + 1e-12) & (~w.index.isin(fixed))].index
        if not len(over):
            break
        fixed.update(over)
        w.loc[list(fixed)] = max_position
        free = w.index.difference(list(fixed))
        free_total = max(1.0 - max_position * len(fixed), 0.0)
        free_orig_sum = base.loc[free].sum()
        if free_orig_sum > 0:
            w.loc[free] = base.loc[free] * (free_total / free_orig_sum)
        else:
            w.loc[free] = 0.0
    return w
