"""Multi-ticker alignment of `TickerData` for cross-sectional training.

`TickerData` ships from `ss_features`; producers (apps/notebook's replay
loader, apps/factor's deterministic-indicator builder) populate one per
ticker, each with its own date range. Training a cross-sectional scorer
needs every ticker's features and prices on a common date axis so the
per-rebalance Pearson IC has well-defined cross-sections.

`align_tickers` takes a list of `TickerData`, finds the common date
range (intersection — start = max of starts, end = min of ends), and
returns aligned arrays. Tickers whose date arrays don't overlap raise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ss_features import TickerData


@dataclass(frozen=True)
class AlignedTickers:
    """Aligned multi-ticker tensors on a common date axis.

    Two modes — distinguished by whether `rebal_idx` is set:

    * **Daily mode** (`rebal_idx is None`): every field has the full
      daily date axis. `features` is `(D, N, K, F)`. Returned by
      `align_tickers`.
    * **Rebal-aligned mode** (`rebal_idx is not None`): `features`
      is pre-subsampled to `(len(rebal_idx), N, K, F)` while `dates`,
      `prices`, `valid` keep the full daily axis. Returned by
      `align_tickers_at_rebal`. Used by the supervised-`cnn` / indicator
      walk-forward pipelines so the encoder runs only on the bars
      whose latents are actually consumed (one row per rebalance),
      avoiding a full `(D, N, K, F)` panel allocation. At 3000 tickers
      that's the difference between ~786 GB and ~39 GB resident.

    Shapes (D = full daily dates, D' = `len(rebal_idx)`, N = n_tickers,
    K = window_cols, F = channels):
      - dates:     `(D,)` numpy datetime64 / object array
      - names:     `(N,)` ticker names
      - features:  `(D, N, K, F)` daily / `(D', N, K, F)` rebal — float32
      - prices:    `(D, N)` float64
      - valid:     `(D, N)` bool — replay's per-ticker valid mask, aligned
      - rebal_idx: `None` (daily) or `(D',)` int64 — daily-axis positions
                   the rebal features were sampled at
    """
    dates: np.ndarray
    names: tuple[str, ...]
    features: np.ndarray
    prices: np.ndarray
    valid: np.ndarray
    rebal_idx: np.ndarray | None = None


def align_tickers(
    tickers: list[TickerData], *, K: int, F: int,
) -> AlignedTickers:
    """Intersect ticker date ranges and stack into common-axis tensors.

    `K` and `F` are needed to reshape each ticker's `(n_dates, K*F)`
    feature matrix into `(n_dates, K, F)` before stacking. Pass them
    from a loaded `Backbone` so the reshape matches what the backbone
    was trained on.
    """
    if not tickers:
        raise ValueError('align_tickers needs at least one TickerData')
    for td in tickers:
        if td.features.shape[1] != K * F:
            raise ValueError(
                f'ticker {td.name!r}: features shape {td.features.shape} '
                f'incompatible with K*F = {K * F} (K={K}, F={F})')

    indexes = [pd.DatetimeIndex(td.dates) for td in tickers]
    common = indexes[0]
    for idx in indexes[1:]:
        common = common.intersection(idx)
    if len(common) == 0:
        raise ValueError(
            'tickers have no overlapping dates; check --start / --end')
    common = common.sort_values()

    D = len(common)
    N = len(tickers)
    features = np.empty((D, N, K, F), dtype=np.float32)
    prices = np.empty((D, N), dtype=np.float64)
    valid = np.zeros((D, N), dtype=bool)
    for j, (td, idx) in enumerate(zip(tickers, indexes)):
        loc = idx.get_indexer(common)
        if (loc < 0).any():
            missing = common[loc < 0]
            raise ValueError(
                f'ticker {td.name!r}: {len(missing)} common dates not '
                f'found in its index — duplicate dates?')
        # `features` is already preallocated as f32; assignment casts on
        # the fly. The explicit `.astype(np.float32)` here used to
        # allocate a per-ticker transient (~262 MB at full pool) only to
        # immediately copy into the f32 destination — drop it.
        features[:, j] = td.features[loc].reshape(-1, K, F)
        prices[:, j] = td.prices[loc]
        valid[:, j] = td.valid[loc]

    return AlignedTickers(
        dates=common.to_numpy(),
        names=tuple(td.name for td in tickers),
        features=features,
        prices=prices,
        valid=valid,
    )


def align_tickers_at_rebal(
    tickers: list[TickerData], *, K: int, F: int, rebal_days: int,
) -> AlignedTickers:
    """Like `align_tickers`, but only materializes features at rebalance bars.

    Computes `rebal_idx = arange(0, D, rebal_days)` over the common
    daily axis (filtered to keep `idx + rebal_days < D` so the forward-
    return target window fits), then allocates a `(len(rebal_idx), N, K, F)`
    feature panel — not the full `(D, N, K, F)`. Daily-axis prices and
    valid mask are still materialized in full because the forward-return
    helpers (`forward_log_returns` / `forward_vol_innovation`) work over
    daily bars; both are tiny (~78 MB / ~19 MB at D=6500, N=3000).

    The encoder forward in `precompute_inputs` only consumes one row per
    rebal block — running it over the full daily panel was wasted work
    AND wasted RAM. At 3000 tickers / D=6500 / K=96 / F=105 this drops
    aligned-features residency from ~786 GB to ~39 GB and the encoder
    pass from ~470 GB to ~22 GB of latent.

    Per-ticker feature reads are sparse (`td.features[loc_at_rebal]`
    where `loc_at_rebal` has D' entries, not D), so memmap-backed
    `td.features` (e.g. from a Modal worker that wrote to /tmp) only
    pages in the rebal rows.
    """
    if not tickers:
        raise ValueError('align_tickers_at_rebal needs at least one TickerData')
    if rebal_days < 1:
        raise ValueError(f'rebal_days={rebal_days} must be >= 1')
    for td in tickers:
        if td.features.shape[1] != K * F:
            raise ValueError(
                f'ticker {td.name!r}: features shape {td.features.shape} '
                f'incompatible with K*F = {K * F} (K={K}, F={F})')

    indexes = [pd.DatetimeIndex(td.dates) for td in tickers]
    common = indexes[0]
    for idx in indexes[1:]:
        common = common.intersection(idx)
    if len(common) == 0:
        raise ValueError(
            'tickers have no overlapping dates; check --start / --end')
    common = common.sort_values()

    D = len(common)
    N = len(tickers)
    rebal_idx = np.arange(0, D, rebal_days, dtype=np.int64)
    rebal_idx = rebal_idx[rebal_idx + rebal_days < D]
    if len(rebal_idx) < 4:
        raise ValueError(
            f'only {len(rebal_idx)} rebalance bars fit in {D} aligned dates '
            f'with rebal_days={rebal_days}; need >=4 for a sensible split')
    Dp = len(rebal_idx)

    features = np.empty((Dp, N, K, F), dtype=np.float32)
    prices = np.empty((D, N), dtype=np.float64)
    valid = np.zeros((D, N), dtype=bool)
    for j, (td, idx) in enumerate(zip(tickers, indexes)):
        loc_full = idx.get_indexer(common)
        if (loc_full < 0).any():
            missing = common[loc_full < 0]
            raise ValueError(
                f'ticker {td.name!r}: {len(missing)} common dates not '
                f'found in its index — duplicate dates?')
        # Sparse read: only the rebal rows from this ticker's features.
        # When `td.features` is a memmap, this pages in only the bytes
        # for D' rows — not all D.
        loc_at_rebal = loc_full[rebal_idx]
        features[:, j] = td.features[loc_at_rebal].reshape(-1, K, F)
        prices[:, j] = td.prices[loc_full]
        valid[:, j] = td.valid[loc_full]

    return AlignedTickers(
        dates=common.to_numpy(),
        names=tuple(td.name for td in tickers),
        features=features,
        prices=prices,
        valid=valid,
        rebal_idx=rebal_idx,
    )


def forward_log_returns(
    prices: np.ndarray, *, rebal_days: int,
) -> np.ndarray:
    """`(D, N)` of log-returns summed over the *next* `rebal_days` bars.

    `out[i, j] = sum(log(p[i+k+1] / p[i+k]) for k in range(rebal_days))`,
    so it's the log return realized by holding ticker j from close-of-i
    to close-of-(i+rebal_days). The trailing `rebal_days` rows are NaN
    (the future window doesn't fit).
    """
    # f32 throughout — log() of f64 prices then subtracted in-window
    # has no accumulation, so f32 precision (~7 digits) is plenty for
    # forward log returns on the [-1, 1] range typical of horizons we
    # use. Keeps a `(D, N)` panel at half the bytes.
    log_p = np.log(np.maximum(prices, 1e-12)).astype(np.float32)
    D, N = prices.shape
    fwd = np.full((D, N), np.nan, dtype=np.float32)
    if D > rebal_days:
        fwd[:D - rebal_days] = log_p[rebal_days:] - log_p[:D - rebal_days]
    return fwd


def forward_sign_demeaned(
    prices: np.ndarray, *, rebal_days: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """`(D, N)` of `sign(fwd_log_ret − cross_sectional_mean(fwd_log_ret))`.

    Per-bar cross-sectional mean is computed over the *valid* peer set
    at that bar — passing the same liquid-universe mask the IC eval will
    use keeps the demean target consistent with what the head sees.
    When `valid` is None, every finite forward return participates.

    Returns ±1 for tickers strictly above / below the per-bar peer mean,
    and 0 for ties or undefined cells (preserving the float dtype the
    downstream Pearson IC expects). Pearson IC against a ±1 target is
    point-biserial correlation — directly comparable in scale to the
    raw-fwd-return IC baseline since both are bounded in [-1, +1].

    Bars with fewer than two valid peers produce all-zero rows (no
    cross-section to demean against); they get filtered out by the
    eval mask anyway, but the explicit zeroing avoids div-by-zero.
    """
    fwd = forward_log_returns(prices, rebal_days=rebal_days)
    if valid is None:
        valid = np.isfinite(fwd)
    else:
        valid = valid & np.isfinite(fwd)
    valid_f = valid.astype(np.float64)
    counts = valid_f.sum(axis=1, keepdims=True)
    fwd_safe = np.where(valid, fwd, 0.0)
    sums = fwd_safe.sum(axis=1, keepdims=True)
    means = np.where(counts >= 2, sums / np.maximum(counts, 1.0), 0.0)
    centered = np.where(valid, fwd - means, 0.0)
    out = np.sign(centered).astype(np.float64)
    out = np.where(counts >= 2, out, 0.0)
    return out


def forward_vol_innovation(
    prices: np.ndarray, *, rebal_days: int,
) -> np.ndarray:
    """`(D, N)` of log(realized_vol_forward / realized_vol_trailing).

    Both realized vols use a window of `rebal_days` bars of squared
    log returns:
      * trailing var at `t` = mean of `r²[t-rebal_days+1 .. t]`
      * forward  var at `t` = mean of `r²[t+1 .. t+rebal_days]`

    Then innovation `= 0.5 · log(var_fwd / var_trail) = log(σ_fwd / σ_trail)`.
    Symmetric in sign (positive = vol expanding, negative = contracting),
    dimensionless, and the "trivial" piece of forward-vol prediction —
    vol persistence (clustering autocorrelation) — is structurally
    subtracted by the ratio form. The IC head's task therefore reduces
    to predicting *vol-regime change* given current features, not just
    re-emitting the trailing-vol channel it already gets as input.

    Edges: returns NaN in the leading `rebal_days` rows (no trailing
    window), the trailing `rebal_days` rows (no forward window), and
    wherever either variance is non-positive (zero-return windows).
    Caller's mask filters these via `np.isfinite` like every other
    target.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T, N = prices.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if T <= rebal_days * 2:
        return out

    log_p = np.log(np.maximum(prices, 1e-12))
    # Daily log returns aligned at the bar where the move *closes*: r[t]
    # is the move from t-1 to t. r[0] is undefined, set to NaN so the
    # cumulative path never propagates a zero.
    log_ret = np.full((T, N), np.nan, dtype=np.float64)
    log_ret[1:] = log_p[1:] - log_p[:-1]
    sq_ret = log_ret ** 2

    # Trailing var at t covers bars (t-rebal_days+1 .. t), forward var
    # at t covers bars (t+1 .. t+rebal_days). Cumulative-sum trick keeps
    # this O(T) regardless of rebal_days.
    csum = np.zeros((T + 1, N), dtype=np.float64)
    csum[1:] = np.cumsum(np.where(np.isfinite(sq_ret), sq_ret, 0.0), axis=0)
    cnt = np.zeros((T + 1, N), dtype=np.float64)
    cnt[1:] = np.cumsum(np.isfinite(sq_ret).astype(np.float64), axis=0)

    def window_mean(start: int, end: int) -> np.ndarray:
        """Mean over bars [start, end) per ticker; NaN where any input
        window had zero finite cells."""
        s = csum[end] - csum[start]
        c = cnt[end] - cnt[start]
        return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)

    eps = 1e-18
    for t in range(rebal_days, T - rebal_days):
        trail = window_mean(t - rebal_days + 1, t + 1)
        fwd = window_mean(t + 1, t + rebal_days + 1)
        good = (trail > eps) & (fwd > eps) & np.isfinite(trail) & np.isfinite(fwd)
        out[t] = np.where(good, 0.5 * (np.log(fwd) - np.log(trail)), np.nan)
    return out
