"""Inference forward pass: prices in, target weights out.

Loads a `Checkpoint` and returns target weights for the *latest* date
in the supplied OHLC frames. Dispatch is on `checkpoint.strategy`:

  * **regime**   — hard top-N equal-weight basket using the checkpoint-
    recorded divergence (`kl`/`js`/`cosine`/`l2`) with uniform per-scale
    weighting.
  * **scalogram** — hard top-N equal-weight basket ranked ascending by
    `direction − momentum × coherence`.
  * **rsi**      — hard top-N equal-weight basket ranked ascending by
    trailing-`n_tail` mean Wilder RSI(`rsi_n`); lowest score = most
    oversold. No CWT.

All three are Optuna-search outputs; the legacy "adam" mode (gradient-
descended scale weights + soft top-N) was removed when the autograd
path was severed by the ss_indicators numpy migration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.persist import Checkpoint
from regime.trainer import _log_returns
from ss_indicators import corwin_schultz_spread, get_divergence, rsi
from ss_wavelets import KERNEL_HALF_EXTENT, causal_cwt, precompute_windows


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

    if checkpoint.strategy == 'scalogram':
        scores, liquid_last = _score_latest_bar_scalogram(
            prices, highs, lows, checkpoint)
        weights = _hard_top_n(
            scores, liquid_last, checkpoint.top_n, ascending=True)
    elif checkpoint.strategy == 'rsi':
        scores, liquid_last = _score_latest_bar_rsi(
            prices, highs, lows, checkpoint)
        weights = _hard_top_n(
            scores, liquid_last, checkpoint.top_n, ascending=True)
    else:
        scores, liquid_last = _score_latest_bar(prices, highs, lows, checkpoint)
        weights = _hard_top_n(scores, liquid_last, checkpoint.top_n)

    return pd.Series(weights, index=prices.columns, name=prices.index[-1])


def _validate_inputs(prices, highs, lows, checkpoint):
    if not (prices.columns.equals(highs.columns) and prices.columns.equals(lows.columns)):
        raise ValueError('prices/highs/lows must share columns')
    if not (prices.index.equals(highs.index) and prices.index.equals(lows.index)):
        raise ValueError('prices/highs/lows must share index')
    # CWT-using strategies need KERNEL_HALF_EXTENT*max_scale + lookback
    # bars for the latest-bar wavelet to have full kernel support
    # (otherwise it's silently zero-padded). RSI checkpoints have empty
    # scales and only need lookback+1.
    max_scale = max(checkpoint.scales) if checkpoint.scales else 0
    min_bars = KERNEL_HALF_EXTENT * max_scale + checkpoint.lookback + 1
    if len(prices) < min_bars:
        raise ValueError(
            f'need at least {min_bars} bars (KERNEL_HALF_EXTENT*max_scale + '
            f'lookback + 1 = {KERNEL_HALF_EXTENT}*{max_scale} + '
            f'{checkpoint.lookback} + 1), got {len(prices)}')


def _score_latest_bar(prices, highs, lows, checkpoint):
    """Compute the chosen divergence at the latest bar, plus the
    liquidity mask for that bar. Returns `(scores, liquid_last)`,
    both 1-D arrays over tickers.
    """
    prices_np = prices.values.astype(np.float64)
    # Score with the SAME CWT input the trainer used. The empirical
    # finding (see comment in regime.trainer above `_log_returns`) is
    # that raw close beats log-returns for cross-sectional ranking, so
    # `checkpoint.use_log_returns` defaults to False on new training
    # runs. Older checkpoints predating this field also default to
    # False thanks to the dataclass default — that matches their actual
    # train-time behavior (raw close was the only option then).
    cwt_input = (_log_returns(prices_np)
                 if checkpoint.use_log_returns else prices_np)
    coeffs = causal_cwt(cwt_input, checkpoint.scales, checkpoint.lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(
        power, checkpoint.lookback, checkpoint.n_tail)

    spread_df = corwin_schultz_spread(highs, lows)
    liquid_last = (spread_df.values[-1] <= checkpoint.max_spread).astype(np.float32)

    div_fn = get_divergence(checkpoint.divergence or 'kl')
    scale_log_weights = np.zeros(len(checkpoint.scales), dtype=np.float32)
    recent_last = recent[:, -1:, :]
    historical_last = historical[:, -1:, :]
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
    cwt_input = (_log_returns(prices_np)
                 if checkpoint.use_log_returns else prices_np)
    coeffs = causal_cwt(cwt_input, checkpoint.scales, checkpoint.lookback)
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


def _score_latest_bar_rsi(prices, highs, lows, checkpoint):
    """Compute the trailing-`n_tail` mean Wilder RSI(`rsi_n`) for the
    latest bar across all tickers, plus the liquidity mask.

    Mirrors `regime.trainer.weights_rsi` but only needs the trailing
    window ending at the last index, so we slice instead of computing
    a full cumsum sweep over history.
    """
    if checkpoint.rsi_n is None:
        raise ValueError(
            'rsi checkpoint missing rsi_n; checkpoint may be from a '
            'pre-rsi training run')
    rsi_arr = np.asarray(rsi(prices.values, n=checkpoint.rsi_n))
    tail = slice(-checkpoint.n_tail, None)
    scores = rsi_arr[tail].mean(axis=0).astype(np.float32)

    spread_df = corwin_schultz_spread(highs, lows)
    liquid_last = (spread_df.values[-1] <= checkpoint.max_spread).astype(np.float32)
    return scores, liquid_last


def _hard_top_n(
    scores: np.ndarray,
    mask: np.ndarray,
    top_n: int | None,
    *,
    ascending: bool = False,
) -> np.ndarray:
    """Pick `top_n` liquid names, allocate `1/top_n` each.

    `ascending=False` (default) keeps the largest scores (regime: highest
    divergence). `ascending=True` keeps the smallest scores (scalogram:
    most negative direction−momentum×coherence). If fewer than `top_n`
    liquid names exist, equal-weight whatever's left.
    """
    if top_n is None or top_n < 1:
        raise ValueError(f'checkpoint missing or invalid top_n: {top_n!r}')

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
