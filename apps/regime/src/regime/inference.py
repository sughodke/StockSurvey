"""Inference forward pass: prices in, target weights out.

Loads a `Checkpoint` and returns target weights for the *latest* date
in the supplied OHLC frames. The dispatch is two-dimensional:

  * `checkpoint.mode` ∈ {'adam', 'optuna'} — how the model was trained.
  * `checkpoint.strategy` ∈ {'regime', 'scalogram'} — which weight
    builder produced the score.

Combinations:

  * **adam + regime**     — soft top-N via temperature-scaled softmax
    of the symmetric-KL score with learned per-scale weights.
  * **optuna + regime**   — hard top-N equal-weight basket using the
    checkpoint-recorded divergence (`kl`/`js`/`cosine`/`l2`) with
    uniform per-scale weighting.
  * **optuna + scalogram** — hard top-N equal-weight basket ranked
    ascending by `direction − momentum × coherence`.

There's no adam + scalogram path today — scalogram has no continuous
parameters to gradient-descend over.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from regime.persist import Checkpoint
from regime.trainer import _log_returns
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

    # Branch on strategy first — scalogram has its own scoring math,
    # different ranking direction (ascending), and only exists in
    # optuna (hard top-N) form today.
    if checkpoint.strategy == 'scalogram':
        scores, liquid_last = _score_latest_bar_scalogram(
            prices, highs, lows, checkpoint)
        weights = _hard_top_n(
            scores, liquid_last, checkpoint.top_n, ascending=True)
        return pd.Series(weights, index=prices.columns, name=prices.index[-1])

    # Regime strategy — same score code path for both adam and optuna,
    # only the allocation rule differs.
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
    coeffs = causal_cwt(
        _log_returns(prices_np), checkpoint.scales, checkpoint.lookback)
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


def _score_latest_bar_scalogram(prices, highs, lows, checkpoint):
    """Compute the scalogram score for the latest bar across all tickers.

    Mirrors the math in `regime.trainer.weights_scalogram` but only
    needs the trailing-`n_tail` window ending at the last date, so we
    avoid the full cumsum sweep.
    """
    prices_np = prices.values.astype(np.float64)
    coeffs = causal_cwt(
        _log_returns(prices_np), checkpoint.scales, checkpoint.lookback)
    power = (coeffs ** 2).astype(np.float32)

    # Slice the trailing n_tail bars ending at the last index.
    tail = slice(-checkpoint.n_tail, None)
    momentum = power[:, tail, :].mean(axis=(0, 1))
    direction = coeffs[0, tail, :].mean(axis=0)

    short_p = power[0, tail, :]
    long_p = power[-1, tail, :]
    e_s = short_p.mean(axis=0)
    e_l = long_p.mean(axis=0)
    cov = (short_p * long_p).mean(axis=0) - e_s * e_l
    var_s = np.maximum((short_p ** 2).mean(axis=0) - e_s ** 2, 1e-12)
    var_l = np.maximum((long_p ** 2).mean(axis=0) - e_l ** 2, 1e-12)
    coherence = np.clip(cov / (np.sqrt(var_s * var_l) + 1e-9), 0.0, 1.0)

    scores = (direction - momentum * coherence).astype(np.float32)

    spread_df = corwin_schultz_spread(highs, lows)
    liquid_last = (spread_df.values[-1] <= checkpoint.max_spread).astype(np.float32)
    return scores, liquid_last


def _soft_top_n(scores: np.ndarray, mask: np.ndarray, log_temperature: float) -> np.ndarray:
    """Adam-mode allocation: temperature-scaled softmax × liquidity mask."""
    temp = float(np.exp(log_temperature))
    s = scores / temp + np.log(mask + 1e-12)
    s = s - s.max()
    exp_s = np.exp(s) * mask
    return exp_s / (exp_s.sum() + 1e-12)


def _hard_top_n(
    scores: np.ndarray,
    mask: np.ndarray,
    top_n: int | None,
    *,
    ascending: bool = False,
) -> np.ndarray:
    """Optuna-mode allocation: pick `top_n` liquid names, allocate
    `1/top_n` each. `ascending=False` (default) keeps the largest
    scores (regime: highest divergence). `ascending=True` keeps the
    smallest scores (scalogram: most negative direction−momentum×
    coherence). If fewer than `top_n` liquid names exist, equal-
    weight whatever's left.
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
        # Sort: place invalid at the back. For descending (default),
        # we sort by -score; for ascending, by +score.
        keys = np.where(valid, masked if ascending else -masked, np.inf)
        ranked = np.argsort(keys)
        weights[ranked[:top_n]] = 1.0 / top_n
    return weights
