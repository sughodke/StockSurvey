"""Inference forward pass: prices in, target weights out.

Loads a `Checkpoint` and returns target weights for the *latest* date
in the supplied OHLC frames. Two code paths, dispatched on
`checkpoint.mode`:

  * **adam**   — soft top-N via temperature-scaled softmax of the
    symmetric-KL score with learned per-scale weights.
  * **optuna** — hard top-N equal-weight basket using the chosen
    divergence (`kl`/`js`/`cosine`/`l2`) with uniform per-scale
    weighting. Mirrors `ss_portfolio.select_top_n_matrix` semantics.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from regime.persist import Checkpoint
from ss_indicators import corwin_schultz_spread, get_divergence
from ss_indicators import symmetric_kl_divergence as regime_scores
from ss_wavelets import causal_cwt, precompute_windows


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
        Trained model produced by `persist.load_checkpoint`. Mode is
        read from `checkpoint.mode` to choose adam vs optuna semantics.

    Returns
    -------
    pd.Series
        Target portfolio weights indexed by ticker, summing to 1.
        Tickers absent from `prices.columns` (e.g. delisted) are dropped.
    """
    _validate_inputs(prices, highs, lows, checkpoint)

    # Score the latest bar — both modes need the divergence values for
    # the most recent date, computed from a full causal CWT pass.
    scores, liquid_last = _score_latest_bar(prices, highs, lows, checkpoint)

    if checkpoint.mode == 'optuna':
        weights = _hard_top_n(scores, liquid_last, checkpoint.top_n)
    else:
        weights = _soft_top_n(scores, liquid_last, checkpoint.log_temperature)

    return pd.Series(weights, index=prices.columns, name=prices.index[-1])


def _validate_inputs(prices, highs, lows, checkpoint):
    if not (prices.columns.equals(highs.columns) and prices.columns.equals(lows.columns)):
        raise ValueError('prices/highs/lows must share columns')
    if not (prices.index.equals(highs.index) and prices.index.equals(lows.index)):
        raise ValueError('prices/highs/lows must share index')
    if len(prices) < checkpoint.lookback + 1:
        raise ValueError(
            f'need at least {checkpoint.lookback + 1} bars, got {len(prices)}')


def _score_latest_bar(prices, highs, lows, checkpoint):
    """Compute the chosen divergence at the latest bar, plus the
    liquidity mask for that bar. Returns `(scores, liquid_last)`,
    both 1-D arrays over tickers.
    """
    prices_np = prices.values.astype(np.float64)
    coeffs = causal_cwt(prices_np, checkpoint.scales, checkpoint.lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(
        power, checkpoint.lookback, checkpoint.n_tail)

    spread_df = corwin_schultz_spread(highs, lows)
    liquid_last = (spread_df.values[-1] <= checkpoint.max_spread).astype(np.float32)

    # Pick the divergence by checkpoint mode. Adam mode uses the trained
    # symmetric-KL with learned per-scale weights. Optuna mode uses the
    # checkpoint-recorded divergence with uniform per-scale weights.
    if checkpoint.mode == 'optuna':
        div_fn = get_divergence(checkpoint.divergence or 'kl')
        scale_log_weights = jnp.zeros(len(checkpoint.scales), dtype=jnp.float32)
    else:
        div_fn = regime_scores
        scale_log_weights = checkpoint.jax_params()['scale_log_weights']

    recent_last = jnp.asarray(recent[:, -1:, :])
    historical_last = jnp.asarray(historical[:, -1:, :])
    scores = np.asarray(div_fn(
        recent_last, historical_last, scale_log_weights))[0]
    return scores, liquid_last


def _soft_top_n(scores: np.ndarray, mask: np.ndarray, log_temperature: float) -> np.ndarray:
    """Adam-mode allocation: temperature-scaled softmax × liquidity mask."""
    temp = float(np.exp(log_temperature))
    s = scores / temp + np.log(mask + 1e-12)
    s = s - s.max()
    exp_s = np.exp(s) * mask
    return exp_s / (exp_s.sum() + 1e-12)


def _hard_top_n(scores: np.ndarray, mask: np.ndarray, top_n: int | None) -> np.ndarray:
    """Optuna-mode allocation: pick the `top_n` highest-divergence
    liquid names, allocate `1/top_n` each. If fewer than `top_n` liquid
    names exist, equal-weight whatever's left.
    """
    if top_n is None or top_n < 1:
        raise ValueError(f'optuna checkpoint missing or invalid top_n: {top_n!r}')

    masked = np.where(mask >= 0.5, scores, np.nan)
    valid = ~np.isnan(masked)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros_like(scores, dtype=np.float64)

    weights = np.zeros_like(scores, dtype=np.float64)
    if n_valid <= top_n:
        weights[valid] = 1.0 / n_valid
    else:
        # Highest divergence wins — descending sort, take top_n indices.
        ranked = np.argsort(np.where(valid, -masked, np.inf))
        weights[ranked[:top_n]] = 1.0 / top_n
    return weights
