"""CWT-slice feature builders + `TickerData` bundle for one ticker.

The lagged-window features feeding the decoder come from three places:

  * `compute_scalogram` → causal CWT coeffs and power per scale.
  * `build_lagged_features` → trailing K columns stacked per date.
  * `rolling_zscore_stats` (optional) → causal μ, σ that `causal_cwt`
    strips out before convolution. Restoring them lets the decoder
    recover price-level info the bandpass filter discards.
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


def build_lagged_features(
    coeffs: np.ndarray, power: np.ndarray, window_cols: int,
) -> np.ndarray:
    """Stack the trailing `window_cols` columns of (coeffs, power)
    per date into a single feature row.

    Returns shape `(n_dates, 2 * n_scales * window_cols)`. The first
    `window_cols - 1` rows contain NaN — at those bars the trailing
    window doesn't fit yet. Lag-0 (current bar) lives in the leading
    `2 * n_scales` columns; lag-1 in the next block; ...; lag-(K-1)
    in the trailing block.
    """
    if window_cols < 1:
        raise ValueError(f'window_cols must be >= 1, got {window_cols}')
    n_scales, n_dates = coeffs.shape
    F = 2 * n_scales
    stacked = np.concatenate([coeffs, power], axis=0).astype(np.float64)
    feats = np.full((n_dates, window_cols, F), np.nan, dtype=np.float64)
    for k in range(window_cols):
        feats[k:, k] = stacked[:, :n_dates - k].T
    return feats.reshape(n_dates, window_cols * F)


def build_features_and_targets(
    prices: np.ndarray, *,
    scales: list[int], lookback: int, window_cols: int,
    include_zscore_stats: bool, decoder: str,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Returns `(features, gt-by-target, valid-mask)` for one price series.

    `valid` is the AND of: warm-up complete (max of CWT lookback and
    window-cols), feature row finite, and every target finite at that
    bar.
    """
    coeffs, power = compute_scalogram(prices, scales, lookback=lookback)
    features = build_lagged_features(coeffs, power, window_cols)
    if include_zscore_stats:
        if decoder == 'cnn':
            raise ValueError(
                '--include-zscore-stats is incompatible with --decoder cnn '
                '(stats are not lag-windowed and would break the CNN '
                'reshape). Drop one of the two flags.')
        mu, std = rolling_zscore_stats(prices, lookback=lookback)
        features = np.column_stack([features, mu, std])

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
    include_zscore_stats: bool, decoder: str,
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
        include_zscore_stats=include_zscore_stats, decoder=decoder,
        rsi_n=rsi_n, macd_fast=macd_fast, macd_slow=macd_slow,
        macd_signal=macd_signal)
    return TickerData(name, prices, dates, features, targets, valid)
