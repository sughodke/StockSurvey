"""CWT-slice feature builders shared between apps/notebook (replay
trainer) and apps/factor (cross-sectional scorer).

Per-bar features are a stack of channels lag-windowed over the trailing
`K = window_cols` bars. Channels always include the CWT (signed coeffs,
power) per scale; optionally also include the rolling z-norm stats
(mu, std) and a raw daily-return channel. Each addition is one CLI flag
and one extra entry in the channel stack, so the CNN reshape from
`(n, K * C)` → `(n, K, C)` works uniformly.

Lifted out of `ss_notebook.replay.features` so factor's scripts can
build the same input bundle without depending on apps/notebook.
"""
from __future__ import annotations

import numpy as np

from ss_features.ticker import TickerData, load_prices
from ss_features.vol import log_returns, realized_vol
from ss_indicators import cci, cci_strided, macd, rsi, rsi_strided
from ss_wavelets import causal_cwt


TARGET_NAMES = ('price', 'rsi', 'macd', 'vol', 'cci')


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
    feats = np.full((n_dates, window_cols, C), np.nan, dtype=np.float32)
    for k in range(window_cols):
        feats[k:, k] = channels_cn[:, :n_dates - k].T
    return feats.reshape(n_dates, window_cols * C)


def build_features_and_targets(
    prices: np.ndarray, *,
    scales: list[int], lookback: int, window_cols: int,
    include_zscore_stats: bool, include_returns: bool, decoder: str,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
    vol_window: int = 20,
    cci_n: int = 20,
    rsi_n_grid: tuple[int, ...] = (),
    rsi_w_grid: tuple[int, ...] = (),
    cci_n_grid: tuple[int, ...] = (),
    cci_w_grid: tuple[int, ...] = (),
    vol_n_grid: tuple[int, ...] = (),
    macd_fast_grid: tuple[int, ...] = (),
    include_return_sign: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray,
           dict[str, np.ndarray]]:
    """Returns `(features, gt-by-target, valid-mask, target-grids)` for
    one price series.

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
    # All channels kept in float32 — float64 was a 2x memory tax with no
    # accuracy benefit (downstream JAX trainer casts to float32 anyway).
    channels: list[np.ndarray] = [
        coeffs.astype(np.float32),
        power.astype(np.float32),
    ]
    if include_zscore_stats:
        mu, std = rolling_zscore_stats(prices, lookback=lookback)
        channels.append(mu[None, :].astype(np.float32))
        channels.append(std[None, :].astype(np.float32))
    if include_returns:
        channels.append(log_returns(prices)[None, :].astype(np.float32))
    elif include_return_sign:
        channels.append(log_return_signs(prices)[None, :].astype(np.float32))
    channels_cn = np.vstack(channels)
    features = build_lagged_features(channels_cn, window_cols)

    rsi_gt = rsi(prices, n=rsi_n).astype(np.float64)
    macd_line, _, _ = macd(prices, fast=macd_fast, slow=macd_slow,
                          signal=macd_signal)
    macd_gt = macd_line.astype(np.float64)
    price_gt = prices.astype(np.float64)
    vol_gt = realized_vol(prices, window=vol_window)
    cci_gt = cci(prices, n=cci_n).astype(np.float64)
    gt = {'price': price_gt, 'rsi': rsi_gt, 'macd': macd_gt, 'vol': vol_gt,
          'cci': cci_gt}

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
        # 1-D conditioning over MACD fast period. slow = 2 * fast and
        # signal = int(fast * 3 / 4) hold the canonical MACD ratios so
        # one parameter sweeps the full timescale of the indicator.
        # cond_dim=1.
        rows = []
        for f in macd_fast_grid:
            f_i = int(f)
            line, _, _ = macd(prices, fast=f_i, slow=2 * f_i,
                              signal=max(2, (f_i * 3) // 4))
            rows.append(line.astype(np.float64))
        target_grids['macd'] = np.stack(rows, axis=0)

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
    include_zscore_stats: bool, include_returns: bool, decoder: str,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
    vol_window: int = 20,
    cci_n: int = 20,
    rsi_n_grid: tuple[int, ...] = (),
    rsi_w_grid: tuple[int, ...] = (),
    cci_n_grid: tuple[int, ...] = (),
    cci_w_grid: tuple[int, ...] = (),
    vol_n_grid: tuple[int, ...] = (),
    macd_fast_grid: tuple[int, ...] = (),
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
        cci_n=cci_n,
        include_return_sign=include_return_sign,
        rsi_n_grid=rsi_n_grid, rsi_w_grid=rsi_w_grid,
        cci_n_grid=cci_n_grid, cci_w_grid=cci_w_grid,
        vol_n_grid=vol_n_grid, macd_fast_grid=macd_fast_grid)
    return TickerData(name, prices, dates, features, targets, valid,
                      target_grids=target_grids)
