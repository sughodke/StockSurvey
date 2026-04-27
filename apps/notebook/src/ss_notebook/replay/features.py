"""CWT-slice feature builders + `TickerData` bundle for one ticker.

Per-bar features are a stack of channels lag-windowed over the trailing
`K = window_cols` bars. Channels always include the CWT (signed coeffs,
power) per scale; optionally also include the rolling z-norm stats
(mu, std) and a raw daily-return channel. Each addition is one CLI flag
and one extra entry in the channel stack, so the CNN reshape from
`(n, K * C)` → `(n, K, C)` works uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ss_indicators import macd, rsi
from ss_notebook.scalogram import _to_np, load_prices
from ss_wavelets import causal_cwt


TARGET_NAMES = ('price', 'rsi', 'macd')


def compute_scalogram(
    prices: np.ndarray,
    scales: list[int],
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(coeffs, power)` of shape `(n_scales, n_dates)` from the
    causal CWT of raw close. `causal_cwt` expects a 2-D `(T, N)` matrix,
    so we add and squeeze a singleton ticker axis."""
    px = prices.astype(np.float32).reshape(-1, 1)
    coeffs_3d = causal_cwt(px, list(map(int, scales)), lookback=lookback)
    coeffs = coeffs_3d[:, :, 0]
    return coeffs, (coeffs ** 2).astype(np.float32)


def rolling_zscore_stats(
    prices: np.ndarray, lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean + std over the trailing `lookback` bars.

    Mirrors the rolling z-norm inside `ss_wavelets.causal_cwt` so the
    decoder can be handed back the level information that the
    z-normalization strips out before the Ricker convolution.
    """
    n = len(prices)
    cs = np.cumsum(np.concatenate([[0.0], prices.astype(np.float64)]))
    cs2 = np.cumsum(
        np.concatenate([[0.0], (prices.astype(np.float64) ** 2)]))
    idx = np.arange(n)
    lo = np.maximum(0, idx - lookback + 1)
    counts = idx - lo + 1
    mu = (cs[idx + 1] - cs[lo]) / counts
    mu2 = (cs2[idx + 1] - cs2[lo]) / counts
    std = np.sqrt(np.maximum(mu2 - mu ** 2, 1e-4))
    return mu, std


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Per-bar log returns padded with NaN at index 0 so the array
    aligns with the price series. The leading NaN is gated out by the
    warm-up filter in `build_features_and_targets`."""
    log_p = np.log(prices.astype(np.float64))
    return np.concatenate([[np.nan], np.diff(log_p)])


def channels_per_lag(
    n_scales: int, *, include_zscore_stats: bool, include_returns: bool,
) -> int:
    """Per-lag channel count used by the CNN reshape `(n, K, C)`."""
    return (2 * n_scales
            + (2 if include_zscore_stats else 0)
            + (1 if include_returns else 0))


def build_lagged_features(
    channels_cn: np.ndarray, window_cols: int,
) -> np.ndarray:
    """Stack the trailing `window_cols` columns of `channels_cn`
    (shape `(C, n_dates)`) per date into a single feature row.

    Returns shape `(n_dates, window_cols * C)`. The first
    `window_cols - 1` rows contain NaN — the trailing window doesn't
    fit yet. Lag-0 (current bar) lives in the leading `C` columns;
    lag-`(K-1)` in the trailing block.
    """
    if window_cols < 1:
        raise ValueError(f'window_cols must be >= 1, got {window_cols}')
    C, n_dates = channels_cn.shape
    feats = np.full((n_dates, window_cols, C), np.nan, dtype=np.float64)
    for k in range(window_cols):
        feats[k:, k] = channels_cn[:, :n_dates - k].T
    return feats.reshape(n_dates, window_cols * C)


def build_features_and_targets(
    prices: np.ndarray, *,
    scales: list[int], lookback: int, window_cols: int,
    include_zscore_stats: bool, include_returns: bool, decoder: str,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Returns `(features, gt-by-target, valid-mask)` for one price series.

    `valid` is the AND of: warm-up complete (max of CWT lookback and
    window-cols), feature row finite, and every target finite at that
    bar.

    `decoder` is accepted for backwards compatibility but no longer
    gates which channels can be combined — every channel is now lag-
    windowed, so the CNN reshape works regardless of which optional
    channels are active.
    """
    del decoder  # all channel combinations are CNN-compatible
    coeffs, power = compute_scalogram(prices, scales, lookback=lookback)
    channels: list[np.ndarray] = [
        coeffs.astype(np.float64),
        power.astype(np.float64),
    ]
    if include_zscore_stats:
        mu, std = rolling_zscore_stats(prices, lookback=lookback)
        channels.append(mu[None, :])
        channels.append(std[None, :])
    if include_returns:
        channels.append(log_returns(prices)[None, :])
    channels_cn = np.vstack(channels)
    features = build_lagged_features(channels_cn, window_cols)

    rsi_gt = _to_np(rsi(prices, n=rsi_n)).astype(np.float64)
    macd_line, _, _ = macd(prices, fast=macd_fast, slow=macd_slow,
                          signal=macd_signal)
    macd_gt = _to_np(macd_line).astype(np.float64)
    price_gt = prices.astype(np.float64)
    gt = {'price': price_gt, 'rsi': rsi_gt, 'macd': macd_gt}

    valid = np.zeros(len(prices), dtype=bool)
    valid[max(lookback, window_cols - 1):] = True
    valid &= np.isfinite(features).all(axis=1)
    for arr in gt.values():
        valid &= np.isfinite(arr)
    return features, gt, valid


@dataclass
class TickerData:
    """One ticker's loaded prices, dates, features, ground-truth indicators,
    and the warm-up-aware valid mask. Constructed by `load_ticker`."""
    name: str
    prices: np.ndarray
    dates: np.ndarray
    features: np.ndarray
    targets: dict[str, np.ndarray]
    valid: np.ndarray


def load_ticker(
    name: str, *,
    stooq_dir: str | None, kaggle_dir: str | None,
    use_yahoo: bool = False,
    start: str | None, end: str | None,
    scales: list[int], lookback: int, window_cols: int,
    include_zscore_stats: bool, include_returns: bool, decoder: str,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
) -> TickerData:
    """Load one ticker and pre-compute features + targets + valid mask."""
    series = load_prices(
        name, stooq_dir=stooq_dir, kaggle_dir=kaggle_dir,
        use_yahoo=use_yahoo,
        start=start, end=end)
    prices = series.values.astype(np.float64)
    dates = np.asarray(series.index)
    features, targets, valid = build_features_and_targets(
        prices, scales=scales, lookback=lookback, window_cols=window_cols,
        include_zscore_stats=include_zscore_stats,
        include_returns=include_returns, decoder=decoder,
        rsi_n=rsi_n, macd_fast=macd_fast, macd_slow=macd_slow,
        macd_signal=macd_signal)
    return TickerData(name, prices, dates, features, targets, valid)
