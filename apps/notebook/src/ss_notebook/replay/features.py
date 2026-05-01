"""CWT-slice feature builders + `TickerData` bundle for one ticker.

Per-bar features are a stack of channels lag-windowed over the trailing
`K = window_cols` bars. Channels always include the CWT (signed coeffs,
power) per scale; optionally also include the rolling z-norm stats
(mu, std) and a raw daily-return channel. Each addition is one CLI flag
and one extra entry in the channel stack, so the CNN reshape from
`(n, K * C)` → `(n, K, C)` works uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ss_indicators import macd, rsi
from ss_notebook.scalogram import _to_np, load_prices
from ss_wavelets import causal_cwt


TARGET_NAMES = ('price', 'rsi', 'macd', 'vol')


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


def log_return_signs(prices: np.ndarray) -> np.ndarray:
    """Per-bar sign of log return ∈ {-1, 0, +1} aligned to `prices`.

    Diagnostic alternative input channel to `log_returns`: gives the
    model directional anchor without leaking magnitude. Per the
    attention plot finding (2026-05-01) all four supervised heads
    learned to live ~entirely on the raw `return` channel because that
    channel was the lazy shortcut to indicator reconstruction; sign-
    only forces the model to derive magnitude from the wavelets while
    keeping a direction crutch.
    """
    return np.sign(log_returns(prices))


def rsi_strided(prices: np.ndarray, n: int, w: int = 1) -> np.ndarray:
    """Wilder RSI(n) computed over stride-`w` price changes.

    At every bar `t` uses `Δ_i = price[i] - price[i-w]` instead of the
    standard 1-bar `Δ_i = price[i] - price[i-1]`. Wilder-smoothes the
    gains and losses over `n` strided observations. Output is a 1-D
    numpy array aligned with `prices`; positions before the warmup
    (`w + n - 1` bars) are NaN.

    `w=1` reduces to the canonical daily RSI(n) (matches
    `ss_indicators.rsi` for indices ≥ n). `w>1` is the rolling
    weekly/biweekly/monthly view evaluated at every bar — equals the
    discretely-resampled RSI(n) on the resampled-bar boundaries and
    smoothly interpolates off-boundary, giving dense supervision.
    """
    if w < 1:
        raise ValueError(f'rsi_strided w must be >= 1, got {w}')
    if n < 2:
        raise ValueError(f'rsi_strided n must be >= 2, got {n}')
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    out = np.full(T, np.nan, dtype=np.float64)
    if T < w + n:
        return out
    deltas = np.empty(T, dtype=np.float64)
    deltas[:w] = 0.0
    deltas[w:] = prices[w:] - prices[:-w]
    up = np.where(deltas > 0, deltas, 0.0)
    down = np.where(deltas < 0, -deltas, 0.0)
    # Seed Wilder smoothing on the first n strided deltas (bars w..w+n-1).
    avg_up = up[w:w + n].mean()
    avg_down = down[w:w + n].mean()
    rs = avg_up / (avg_down + 1e-9)
    out[w + n - 1] = 100.0 - 100.0 / (1.0 + rs)
    for t in range(w + n, T):
        avg_up = (avg_up * (n - 1) + up[t]) / n
        avg_down = (avg_down * (n - 1) + down[t]) / n
        rs = avg_up / (avg_down + 1e-9)
        out[t] = 100.0 - 100.0 / (1.0 + rs)
    return out


def realized_vol(prices: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling std of log returns over the trailing `window` bars.

    Output is NaN until the window is full. Used as a backbone-pretraining
    target — it's what the rolling-z-normed scalogram is best positioned
    to recover (squared-coefficient structure survives the z-norm) and
    a known cross-sectional return predictor in its own right.
    """
    if window < 2:
        raise ValueError(f'realized_vol window must be >= 2, got {window}')
    rets = log_returns(prices)  # (n,) with rets[0] = NaN
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    rets_clean = np.where(np.isnan(rets), 0.0, rets)
    cs = np.cumsum(np.concatenate([[0.0], rets_clean]))
    cs2 = np.cumsum(np.concatenate([[0.0], rets_clean ** 2]))
    # First valid return is index 1; first full window ends at index `window`.
    for i in range(window, n):
        s = cs[i + 1] - cs[i + 1 - window]
        s2 = cs2[i + 1] - cs2[i + 1 - window]
        m = s / window
        v = max(s2 / window - m * m, 0.0)
        out[i] = np.sqrt(v)
    return out


def channels_per_lag(
    n_scales: int, *, include_zscore_stats: bool, include_returns: bool,
    include_return_sign: bool = False,
) -> int:
    """Per-lag channel count used by the CNN reshape `(n, K, C)`.

    `include_returns` and `include_return_sign` are mutually exclusive
    — they occupy the same channel slot but with different content
    (raw log return vs sign-of-return). At most one may be True.
    """
    if include_returns and include_return_sign:
        raise ValueError('include_returns and include_return_sign are '
                         'mutually exclusive — they share a channel slot.')
    return (2 * n_scales
            + (2 if include_zscore_stats else 0)
            + (1 if include_returns else 0)
            + (1 if include_return_sign else 0))


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
    vol_window: int = 20,
    rsi_n_grid: tuple[int, ...] = (),
    rsi_w_grid: tuple[int, ...] = (),
    include_return_sign: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray,
           dict[str, np.ndarray]]:
    """Returns `(features, gt-by-target, valid-mask, target-grids)` for
    one price series.

    `valid` is the AND of: warm-up complete (max of CWT lookback and
    window-cols), feature row finite, and every target finite at that
    bar.

    `target_grids` carries the parameter-conditioning auxiliary arrays
    for conditioned targets:
      - `rsi_n_grid` only — `target_grids['rsi']` is shape
        `(n_n, n_dates)` holding RSI(n) for every n in the grid.
        Conditioning width p_dim=1.
      - `rsi_n_grid` + `rsi_w_grid` (both non-empty) —
        `target_grids['rsi']` is shape `(n_w * n_n, n_dates)`
        holding RSI on stride-w price changes for every (w, n) pair,
        flattened with row index `w_idx * n_n + n_idx`. Conditioning
        width p_dim=2 = `(n / max_n, w / max_w)`.
    Empty dict when no grid is provided. The 1-D anchor target in
    `gt['rsi']` (computed at `rsi_n`, w=1 implicitly) is what plotting/
    stats compare against, so the head must be evaluated at the anchor
    `(w=1, n=rsi_n)` during prediction.

    `decoder` is accepted for backwards compatibility but no longer
    gates which channels can be combined — every channel is now lag-
    windowed, so the CNN reshape works regardless of which optional
    channels are active.
    """
    del decoder  # all channel combinations are CNN-compatible
    if include_returns and include_return_sign:
        raise ValueError('include_returns and include_return_sign are '
                         'mutually exclusive — they share a channel slot.')
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
    elif include_return_sign:
        channels.append(log_return_signs(prices)[None, :])
    channels_cn = np.vstack(channels)
    features = build_lagged_features(channels_cn, window_cols)

    rsi_gt = _to_np(rsi(prices, n=rsi_n)).astype(np.float64)
    macd_line, _, _ = macd(prices, fast=macd_fast, slow=macd_slow,
                          signal=macd_signal)
    macd_gt = _to_np(macd_line).astype(np.float64)
    price_gt = prices.astype(np.float64)
    vol_gt = realized_vol(prices, window=vol_window)
    gt = {'price': price_gt, 'rsi': rsi_gt, 'macd': macd_gt, 'vol': vol_gt}

    target_grids: dict[str, np.ndarray] = {}
    if rsi_n_grid:
        if rsi_w_grid:
            # (n_w * n_n, n_dates) flattened. Row layout: outer = w,
            # inner = n, so row `w_idx * n_n + n_idx` holds the strided
            # RSI for (w_grid[w_idx], n_grid[n_idx]). RSI on stride-w
            # price changes; w=1 gives the canonical daily RSI(n).
            n_n = len(rsi_n_grid)
            grid_rows = []
            for w in rsi_w_grid:
                for n in rsi_n_grid:
                    grid_rows.append(rsi_strided(prices, n=int(n), w=int(w)))
            target_grids['rsi'] = np.stack(grid_rows, axis=0)
            assert target_grids['rsi'].shape[0] == len(rsi_w_grid) * n_n
        else:
            # (n_n, n_dates) — single-axis (n) conditioning, p_dim=1.
            target_grids['rsi'] = np.stack(
                [_to_np(rsi(prices, n=int(n))).astype(np.float64)
                 for n in rsi_n_grid],
                axis=0)

    valid = np.zeros(len(prices), dtype=bool)
    valid[max(lookback, window_cols - 1):] = True
    valid &= np.isfinite(features).all(axis=1)
    for arr in gt.values():
        valid &= np.isfinite(arr)
    # When grid mode is active, every grid row must also be finite at a bar
    # for the augmented training row to be usable.
    for grid_arr in target_grids.values():
        valid &= np.isfinite(grid_arr).all(axis=0)
    return features, gt, valid, target_grids


@dataclass
class TickerData:
    """One ticker's loaded prices, dates, features, ground-truth indicators,
    and the warm-up-aware valid mask. Constructed by `load_ticker`.

    `target_grids` carries `(n_grid, n_dates)` arrays for parameter-
    conditioned targets; empty dict when no conditioning is in use. The
    1D anchor array in `targets` is what plotting/stats compare against.
    """
    name: str
    prices: np.ndarray
    dates: np.ndarray
    features: np.ndarray
    targets: dict[str, np.ndarray]
    valid: np.ndarray
    target_grids: dict[str, np.ndarray] = field(default_factory=dict)


def load_ticker(
    name: str, *,
    stooq_dir: str | None, kaggle_dir: str | None,
    use_yahoo: bool = False,
    start: str | None, end: str | None,
    scales: list[int], lookback: int, window_cols: int,
    include_zscore_stats: bool, include_returns: bool, decoder: str,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
    vol_window: int = 20,
    rsi_n_grid: tuple[int, ...] = (),
    rsi_w_grid: tuple[int, ...] = (),
    include_return_sign: bool = False,
) -> TickerData:
    """Load one ticker and pre-compute features + targets + valid mask."""
    series = load_prices(
        name, stooq_dir=stooq_dir, kaggle_dir=kaggle_dir,
        use_yahoo=use_yahoo,
        start=start, end=end)
    prices = series.values.astype(np.float64)
    dates = np.asarray(series.index)
    features, targets, valid, target_grids = build_features_and_targets(
        prices, scales=scales, lookback=lookback, window_cols=window_cols,
        include_zscore_stats=include_zscore_stats,
        include_returns=include_returns, decoder=decoder,
        rsi_n=rsi_n, macd_fast=macd_fast, macd_slow=macd_slow,
        macd_signal=macd_signal,
        vol_window=vol_window,
        include_return_sign=include_return_sign,
        rsi_n_grid=rsi_n_grid, rsi_w_grid=rsi_w_grid)
    return TickerData(name, prices, dates, features, targets, valid,
                      target_grids=target_grids)
