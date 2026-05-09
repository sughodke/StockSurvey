"""CWT-slice feature builders shared between apps/notebook (replay
trainer) and apps/factor (cross-sectional scorer).

Per-bar features are a stack of 7 channels per scale, lag-windowed
over the trailing `K = window_cols` bars:

  1. `|c|`            Morlet amplitude (envelope; vol / extrema)
  2. `|c|^2`          Morlet power (squared-magnitude convenience)
  3. `cos(arg(c))`    Morlet phase x (unit-circle representation)
  4. `sin(arg(c))`    Morlet phase y
  5. `g`              Gaussian (scaling-function) coeff — lowpass /
                      trend companion that recovers the DC content
                      the bandpass Morlet structurally cannot carry.
                      Computed on cumulative log-returns so growth
                      stays additive across train→val.
  6. `g^2`            Gaussian power
  7. `log_L2_amp`     Per-scale log-L2 of `|c|` over the trailing
                      K-bar slice. Recovers the realized-vol-by-scale
                      spectrum that the rolling z-norm inside the
                      CWT strips.

`channels_per_lag(n_scales) = 7 * n_scales`. The CNN reshape
`(n, K * C) → (n, K, C)` works uniformly. No optional flags — the
prior `--include-zscore-stats / --include-returns / --include-return-sign`
channels are subsumed by the Gaussian channel + log_L2_amp + the
phase pair (which carries return direction).

Lifted out of `replay.features` so factor's scripts can
build the same input bundle without depending on apps/notebook.
"""
from __future__ import annotations

import numpy as np

from ss_features.compression import Compression, compress_tiles_2d_dwt
from ss_features.ticker import TickerData, load_prices
from ss_features.vol import log_returns, realized_vol
from ss_indicators import (
    cci, cci_strided, drawdown_from_high, macd, macd_from_fast,
    rolling_kurt, rolling_skew, rsi, rsi_strided, vol_norm_momentum,
)
from ss_wavelets import (
    DEFAULT_MORLET_OMEGA0, causal_cwt, causal_cwt_gaussian, causal_cwt_morlet,
)


# Per-scale channel count for the polar Morlet + Gaussian + log-L2
# stack. Single source of truth used by `channels_per_lag` and the
# CNN reshape.
CHANNELS_PER_SCALE: int = 7


# 9 reconstruction heads: the original 5 (price + the four canonical
# technical indicators), plus 4 lie-shape heads (momentum, drawdown, skew,
# kurt) added 2026-05-09 to test whether the rolling-z-normed CWT carries
# the cross-sectional reversal signal the lie v4 head-to-head exposed.
TARGET_NAMES = (
    'price', 'rsi', 'macd', 'vol', 'cci',
    'momentum', 'drawdown', 'skew', 'kurt',
)


def compute_scalogram(
    prices: np.ndarray,
    scales: list[int],
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(coeffs, power)` of shape `(n_scales, n_dates)` from the
    real Ricker causal CWT of raw close. Kept for research scripts that
    still consume the Ricker output directly. `causal_cwt` expects a
    2-D `(T, N)` matrix, so we add and squeeze a singleton ticker axis.
    """
    px = prices.astype(np.float32).reshape(-1, 1)
    coeffs_3d = causal_cwt(px, list(map(int, scales)), lookback=lookback)
    coeffs = coeffs_3d[:, :, 0]
    return coeffs, (coeffs ** 2).astype(np.float32)


def compute_scalogram_polar(
    prices: np.ndarray,
    scales: list[int],
    *,
    lookback: int,
    omega0: float = DEFAULT_MORLET_OMEGA0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(morlet_abs, morlet_cos, morlet_sin, gaussian)` per scale.

    Each output is `(n_scales, n_dates)` float32. The Morlet path uses
    today's rolling-z-norm of raw close (level-invariant); the Gaussian
    path uses cumulative log-returns directly (no z-norm — the lowpass
    needs DC content, and cumulative log-returns are approximately
    stationary so growth stays bounded across train→val).

    Phase is returned as the unit-circle pair `(cos, sin)` rather than
    raw `arg(c) ∈ [-π, π]` to avoid the 2π wrap discontinuity that a
    CNN cannot smooth over.
    """
    px = prices.astype(np.float32).reshape(-1, 1)
    morlet_3d = causal_cwt_morlet(
        px, list(map(int, scales)), lookback=lookback, omega0=omega0)
    c = morlet_3d[:, :, 0]
    abs_c = np.abs(c).astype(np.float32)
    # arg(0) = 0 in numpy, which gives (cos, sin) = (1, 0) at zero
    # coefficients — fine; warmup mask filters those rows out anyway.
    eps = np.float32(1e-12)
    safe_abs = np.maximum(abs_c, eps)
    cos_arg = (c.real.astype(np.float32) / safe_abs)
    sin_arg = (c.imag.astype(np.float32) / safe_abs)

    log_r = log_returns(prices).astype(np.float64)
    cum_log_r = np.cumsum(np.nan_to_num(log_r, nan=0.0)).reshape(-1, 1)
    gaussian_3d = causal_cwt_gaussian(cum_log_r, list(map(int, scales)))
    gaussian = gaussian_3d[:, :, 0].astype(np.float32)

    return abs_c, cos_arg, sin_arg, gaussian


def _rolling_log_l2_amp(
    abs_c: np.ndarray, window_cols: int,
) -> np.ndarray:
    """Per-bar per-scale log-L2 of `|c|` over the trailing K-bar slice.

    Input `abs_c` is `(n_scales, n_dates)`. Returns float32 of the same
    shape: `out[s, t] = log(sqrt(sum_{k=0..K-1} abs_c[s, t-k]^2) + eps)`.
    First `K - 1` bars are filled with NaN (warmup not satisfied); the
    valid mask downstream catches them.
    """
    n_scales, n_dates = abs_c.shape
    sq = (abs_c.astype(np.float64)) ** 2
    cs = np.cumsum(np.concatenate(
        [np.zeros((n_scales, 1)), sq], axis=1), axis=1)
    # window_sum[s, t] = sum_{k=t-K+1..t} sq[s, k] for t >= K-1.
    out = np.full((n_scales, n_dates), np.nan, dtype=np.float32)
    if window_cols <= n_dates:
        window_sum = cs[:, window_cols:] - cs[:, :n_dates - window_cols + 1]
        # window_sum has shape (n_scales, n_dates - K + 1) and aligns
        # to t = K-1, K, ..., n_dates-1.
        out[:, window_cols - 1:] = np.log(
            np.sqrt(window_sum) + 1e-12).astype(np.float32)
    return out


def rolling_zscore_stats(
    prices: np.ndarray, lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean + std over the trailing `lookback` bars.

    Retained as a standalone utility for research scripts; no longer
    consumed by `build_features_and_targets` (the Gaussian channel
    subsumes the level signal).
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


def channels_per_lag(n_scales: int) -> int:
    """Per-lag channel count used by the CNN reshape `(n, K, C)`.

    Constant `CHANNELS_PER_SCALE * n_scales` — the polar Morlet (4) +
    Gaussian (2) + log-L2-amp (1) stack has no optional channels.
    """
    return CHANNELS_PER_SCALE * n_scales


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
    feats = np.full((n_dates, window_cols, C), np.nan, dtype=np.float32)
    for k in range(window_cols):
        feats[k:, k] = channels_cn[:, :n_dates - k].T
    return feats.reshape(n_dates, window_cols * C)


def _build_per_bar_tiles(
    channels_cn: np.ndarray, window_cols: int,
) -> np.ndarray:
    """Per-bar `(K, S)` tile stack from `(S, n_dates)` channels.

    Same lag convention as `build_lagged_features`: index 0 along the
    K axis is the current bar, index `K-1` is the oldest. NaN for warmup
    rows.
    """
    if window_cols < 1:
        raise ValueError(f'window_cols must be >= 1, got {window_cols}')
    S, n_dates = channels_cn.shape
    tiles = np.full((n_dates, window_cols, S), np.nan, dtype=np.float32)
    for k in range(window_cols):
        tiles[k:, k] = channels_cn[:, :n_dates - k].T
    return tiles


def build_features_and_targets(
    prices: np.ndarray, *,
    scales: list[int], lookback: int, window_cols: int,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
    vol_window: int = 20,
    cci_n: int = 20,
    momentum_n: int = 21,
    drawdown_n: int = 63,
    skew_n: int = 63,
    kurt_n: int = 63,
    rsi_n_grid: tuple[int, ...] = (),
    rsi_w_grid: tuple[int, ...] = (),
    cci_n_grid: tuple[int, ...] = (),
    cci_w_grid: tuple[int, ...] = (),
    vol_n_grid: tuple[int, ...] = (),
    macd_fast_grid: tuple[int, ...] = (),
    momentum_n_grid: tuple[int, ...] = (),
    drawdown_n_grid: tuple[int, ...] = (),
    skew_n_grid: tuple[int, ...] = (),
    kurt_n_grid: tuple[int, ...] = (),
    omega0: float = DEFAULT_MORLET_OMEGA0,
    compression: Compression | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray,
           dict[str, np.ndarray]]:
    """Returns `(features, gt-by-target, valid-mask, target-grids)` for
    one price series.

    Per-scale channel stack (7 channels, all lag-windowed): polar Morlet
    `|c|, |c|^2, cos(arg), sin(arg)` over rolling-z-normed prices, then
    Gaussian-CWT `g, g^2` over cumulative log-returns, then a per-bar
    log-L2 of `|c|` over the trailing K-bar slice (one channel per
    scale). See module docstring.

    `valid` is the AND of: warm-up complete (max of CWT lookback and
    window-cols), feature row finite, and every target finite at that
    bar.

    `target_grids` carries the parameter-conditioning auxiliary arrays
    for conditioned targets. Both rsi and cci use the same grid shape
    convention:
      - `<x>_n_grid` only — `target_grids['<x>']` is shape
        `(n_n, n_dates)` holding x(n) for every n in the grid.
        Conditioning width p_dim=1.
      - `<x>_n_grid` + `<x>_w_grid` (both non-empty) —
        `target_grids['<x>']` is shape `(n_w * n_n, n_dates)`
        holding x on stride-w price history for every (w, n) pair,
        flattened with row index `w_idx * n_n + n_idx`. Conditioning
        width p_dim=2 = `(n / max_n, w / max_w)`.
    Empty entries when no grid is provided for that target. The 1-D
    anchor target in `gt['<x>']` is what plotting/stats compare against,
    so each conditioned head must be evaluated at its anchor during
    prediction.
    """
    morlet_abs, morlet_cos, morlet_sin, gauss = compute_scalogram_polar(
        prices, scales, lookback=lookback, omega0=omega0)
    morlet_pow = (morlet_abs ** 2).astype(np.float32)
    gauss_pow = (gauss ** 2).astype(np.float32)
    log_l2_amp = _rolling_log_l2_amp(morlet_abs, window_cols)

    # All seven per-scale channel stacks have the same `(n_scales,
    # n_dates)` shape, so the CNN reshape `(n, K, C=7*n_scales)` works
    # uniformly. Channel order matches the module docstring.
    channels: list[np.ndarray] = [
        morlet_abs,
        morlet_pow,
        morlet_cos,
        morlet_sin,
        gauss,
        gauss_pow,
        log_l2_amp,
    ]
    if compression is not None:
        if compression.kind != 'dwt':
            raise NotImplementedError(
                f'compression.kind={compression.kind!r} is not yet wired '
                'into the feature builder; only dwt is supported (DCT '
                'zigzag-keep-top-k loses the (K, C) tile structure the '
                'CNN reshape relies on and needs a separate decoder path)')
        # Each per-scale channel becomes a `(n_dates, K, S)` tile stack;
        # 2D DWT keep-LL compresses to `(n_dates, K', S')` per channel;
        # concat along S gives `(n_dates, K', CHANNELS_PER_SCALE * S')`.
        compressed_blocks = []
        for ch in channels:
            tiles = _build_per_bar_tiles(ch.astype(np.float32), window_cols)
            ll = compress_tiles_2d_dwt(tiles, compression)
            compressed_blocks.append(ll)
        compressed = np.concatenate(
            compressed_blocks, axis=-1).astype(np.float32)
        n_dates = compressed.shape[0]
        features = compressed.reshape(n_dates, -1)
    else:
        channels_cn = np.vstack([ch.astype(np.float32) for ch in channels])
        features = build_lagged_features(channels_cn, window_cols)

    rsi_gt = rsi(prices, n=rsi_n).astype(np.float64)
    macd_line, _, _ = macd(prices, fast=macd_fast, slow=macd_slow,
                          signal=macd_signal)
    macd_gt = macd_line.astype(np.float64)
    price_gt = prices.astype(np.float64)
    vol_gt = realized_vol(prices, window=vol_window)
    cci_gt = cci(prices, n=cci_n).astype(np.float64)
    momentum_gt = vol_norm_momentum(prices, n=momentum_n)
    drawdown_gt = drawdown_from_high(prices, n=drawdown_n)
    skew_gt = rolling_skew(prices, n=skew_n)
    kurt_gt = rolling_kurt(prices, n=kurt_n)
    gt = {
        'price': price_gt, 'rsi': rsi_gt, 'macd': macd_gt, 'vol': vol_gt,
        'cci': cci_gt,
        'momentum': momentum_gt, 'drawdown': drawdown_gt,
        'skew': skew_gt, 'kurt': kurt_gt,
    }

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
                [rsi(prices, n=int(n)).astype(np.float64)
                 for n in rsi_n_grid],
                axis=0)
    if cci_n_grid:
        # Same row layout as RSI grid: outer = w, inner = n.
        if cci_w_grid:
            n_n = len(cci_n_grid)
            grid_rows = []
            for w in cci_w_grid:
                for n in cci_n_grid:
                    grid_rows.append(cci_strided(prices, n=int(n), w=int(w)))
            target_grids['cci'] = np.stack(grid_rows, axis=0)
            assert target_grids['cci'].shape[0] == len(cci_w_grid) * n_n
        else:
            target_grids['cci'] = np.stack(
                [cci(prices, n=int(n)).astype(np.float64)
                 for n in cci_n_grid],
                axis=0)
    if vol_n_grid:
        # 1-D conditioning over the realized-vol window. Each row is
        # vol_gt at one window value; cond_dim=1 (n / max_n).
        target_grids['vol'] = np.stack(
            [realized_vol(prices, window=int(n)) for n in vol_n_grid],
            axis=0)
    if macd_fast_grid:
        # 1-D conditioning over MACD fast period via the canonical
        # `(fast, slow, signal)` ratio defined in `ss_indicators.macd`.
        # Holds slow ≈ 2.167*fast and signal ≈ 0.75*fast so the f=12
        # cell collapses exactly onto the textbook `(12, 26, 9)` anchor
        # — earlier the local `slow = 2 * fast` rule produced slow=24
        # at f=12, which collided with the canonical (slow=26) anchor
        # at the same FiLM cond and contaminated multi-head training.
        # cond_dim=1.
        rows = []
        for f in macd_fast_grid:
            line, _, _ = macd_from_fast(prices, fast=int(f))
            rows.append(line.astype(np.float64))
        target_grids['macd'] = np.stack(rows, axis=0)
    if momentum_n_grid:
        # 1-D conditioning over the vol-norm-momentum horizon. cond_dim=1.
        target_grids['momentum'] = np.stack(
            [vol_norm_momentum(prices, n=int(n)) for n in momentum_n_grid],
            axis=0)
    if drawdown_n_grid:
        # 1-D conditioning over drawdown lookback. cond_dim=1.
        target_grids['drawdown'] = np.stack(
            [drawdown_from_high(prices, n=int(n)) for n in drawdown_n_grid],
            axis=0)
    if skew_n_grid:
        # 1-D conditioning over the trailing-return-skew horizon. cond_dim=1.
        target_grids['skew'] = np.stack(
            [rolling_skew(prices, n=int(n)) for n in skew_n_grid],
            axis=0)
    if kurt_n_grid:
        # 1-D conditioning over the trailing-return-kurt horizon. cond_dim=1.
        target_grids['kurt'] = np.stack(
            [rolling_kurt(prices, n=int(n)) for n in kurt_n_grid],
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


def load_ticker(
    name: str, *,
    stooq_dir: str | None, kaggle_dir: str | None,
    use_yahoo: bool = False,
    start: str | None, end: str | None,
    scales: list[int], lookback: int, window_cols: int,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
    vol_window: int = 20,
    cci_n: int = 20,
    momentum_n: int = 21,
    drawdown_n: int = 63,
    skew_n: int = 63,
    kurt_n: int = 63,
    rsi_n_grid: tuple[int, ...] = (),
    rsi_w_grid: tuple[int, ...] = (),
    cci_n_grid: tuple[int, ...] = (),
    cci_w_grid: tuple[int, ...] = (),
    vol_n_grid: tuple[int, ...] = (),
    macd_fast_grid: tuple[int, ...] = (),
    momentum_n_grid: tuple[int, ...] = (),
    drawdown_n_grid: tuple[int, ...] = (),
    skew_n_grid: tuple[int, ...] = (),
    kurt_n_grid: tuple[int, ...] = (),
    omega0: float = DEFAULT_MORLET_OMEGA0,
    compression: Compression | None = None,
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
        rsi_n=rsi_n, macd_fast=macd_fast, macd_slow=macd_slow,
        macd_signal=macd_signal,
        vol_window=vol_window,
        cci_n=cci_n,
        momentum_n=momentum_n, drawdown_n=drawdown_n,
        skew_n=skew_n, kurt_n=kurt_n,
        rsi_n_grid=rsi_n_grid, rsi_w_grid=rsi_w_grid,
        cci_n_grid=cci_n_grid, cci_w_grid=cci_w_grid,
        vol_n_grid=vol_n_grid, macd_fast_grid=macd_fast_grid,
        momentum_n_grid=momentum_n_grid,
        drawdown_n_grid=drawdown_n_grid,
        skew_n_grid=skew_n_grid,
        kurt_n_grid=kurt_n_grid,
        omega0=omega0,
        compression=compression)
    return TickerData(name, prices, dates, features, targets, valid,
                      target_grids=target_grids)
