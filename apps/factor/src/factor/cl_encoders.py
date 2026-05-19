"""Fixed `(channel × compressed-length)` encoders — non-CWT alternatives.

Two deterministic encoders that produce a per-bar `(C, L)` block from
raw price, as drop-in replacements for the causal-CWT-of-price input.
They route through the *same* backbone/scoring machinery as
`indicator_features.py`: a per-bar block flattened to `C*L`, wrapped in
`identity_backbone(K=C, F=L)` with pool z-norm, fed to
`train_scorer_walkforward`. No walk-forward harness change —
`rebal_days` is already a free parameter, so the short-horizon sweep is
pure parameterization.

The flatten contract is load-bearing: `align_tickers_at_rebal` reshapes
each ticker's `td.features[loc]` via `.reshape(-1, K, F)`, so every
builder here assembles a `(T, C, L)` tensor and `.reshape(T, C*L)`, and
the matching backbone is `identity_backbone(K=C, F=L)`. `reshape` is
row-major both ways so the round-trip is exact.

Encoders
--------
* **spectral** (`SpectralGridConfig` / `build_spectral_features`) —
  per-signal causal rFFT magnitude over a trailing window, truncated to
  the `n_bins` lowest non-DC frequencies. `C = n_signals`, `L = n_bins`.
  Genuinely compressed, fixed, numpy-only. The pure-frequency contrast
  to time-frequency CWT.
* **minirocket** (`MiniRocketGridConfig` / `build_minirocket_features`)
  — the canonical MiniRocket kernel bank (84 length-9 mean-zero
  `{-1,+2}` kernels = C(9,3)) at a fixed dilation set, PPV-pooled over a
  trailing window. `C = 84 kernels`, `L = n_dilations`. Deviations from
  stock MiniRocket are deliberate and walk-forward-safe: bias is fixed
  at 0 (no training-set quantile fit → no leakage across the rolling
  split) and the dilation set is fixed (a fixed `(C, L)` rather than
  input-length-derived). Documented here so the arm is honestly named.

See `apps/docs/docs/TODO/factor-shorthorizon-representation.md` for the
pre-registered test design these feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ss_features import TickerData, load_prices, realized_vol
from factor.backbone import (
    Backbone, compute_input_stats, identity_backbone,
)
from factor.train import TrainResult, train_scorer
from factor.train_walkforward import (
    WalkForwardResult, train_scorer_walkforward,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_signals(prices: np.ndarray, vol_window: int) -> np.ndarray:
    """`(4, T)` base series derived causally from `prices`.

    Rows: [log-return, |log-return|, trailing realized vol, log-price].
    `r[0]` is NaN (no prior bar); `vol` is NaN until `vol_window` bars.
    Leading NaNs propagate through the windowed transforms below and are
    caught by the final `isfinite(...).all(axis=1)` valid mask — the
    same NaN-warmup convention as `build_indicator_features`.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = prices.shape[0]
    logp = np.log(np.maximum(prices, 1e-12))
    r = np.full(T, np.nan, dtype=np.float64)
    r[1:] = logp[1:] - logp[:-1]
    absr = np.abs(r)
    vol = realized_vol(prices, window=int(vol_window)).astype(np.float64)
    return np.stack([r, absr, vol, logp], axis=0)


_SIGNAL_NAMES = ('logret', 'abslogret', 'vol', 'logprice')


def _make_cl_backbone(
    tickers: list[TickerData], C: int, L: int, *, kind: str,
) -> Backbone:
    """Identity backbone sized to `(C, L)` with pool z-norm stats.

    Mirrors `make_indicator_backbone` but with `K=C, F=L` (the indicator
    path is the `K=1` special case). Pool z-norm matters more here than
    for indicators: rFFT magnitudes and PPV ratios live at wildly
    different scales per channel, and an un-normalized linear head's
    gradient would be dominated by the highest-variance bin.
    """
    expect = C * L
    for td in tickers:
        if td.features.shape[1] != expect:
            raise ValueError(
                f'ticker {td.name!r}: features width '
                f'{td.features.shape[1]} != C*L={expect} (C={C}, L={L}); '
                f'rebuild via load_ticker_{kind} with this cfg')
    mu, sd = compute_input_stats(tickers, K=C, F=L)
    return identity_backbone(K=C, F=L, feat_mu=mu, feat_sd=sd)


# ---------------------------------------------------------------------------
# Spectral (truncated causal rFFT magnitude)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralGridConfig:
    """Truncated causal rFFT-magnitude encoder.

    Per bar `t`: for each of the 4 base signals, take the trailing
    `window` samples, mean-center them (so the DC bin is ~0 and the
    descriptor is level-invariant), real-FFT, and keep the magnitudes of
    the `n_bins` lowest *non-DC* frequencies. Block shape `(C=4,
    L=n_bins)`. `window=128` gives `rfft` length 65, so up to 64 non-DC
    bins are available; `n_bins=16` keeps the low-frequency path shape.
    """
    window:     int = 128
    n_bins:     int = 16
    vol_window: int = 20

    def n_channels(self) -> int:
        return len(_SIGNAL_NAMES)

    def compressed_len(self) -> int:
        return self.n_bins

    def feature_width(self) -> int:
        return self.n_channels() * self.compressed_len()

    def channel_names(self) -> list[str]:
        # Row-major over (signal, bin) — must match the (C, L) assembly
        # order in build_spectral_features so a trained linear head's
        # coefficients are inspectable.
        names: list[str] = []
        for sig in _SIGNAL_NAMES:
            for b in range(self.n_bins):
                names.append(f'fft_{sig}_b{b + 1}')
        return names

    def __post_init__(self) -> None:
        max_bins = self.window // 2  # rfft len = window//2+1, minus DC
        if not (1 <= self.n_bins <= max_bins):
            raise ValueError(
                f'n_bins={self.n_bins} must be in [1, {max_bins}] for '
                f'window={self.window}')


def build_spectral_features(
    prices: np.ndarray, cfg: SpectralGridConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """`(T, C*L)` spectral block + `(T,)` valid mask.

    Vectorized over time via `sliding_window_view`. Windows that contain
    any NaN (signal warmup) produce an all-NaN rFFT row, which the valid
    mask filters — identical NaN-warmup handling to the indicator path.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = prices.shape[0]
    C = cfg.n_channels()
    L = cfg.n_bins
    W = cfg.window
    if T < W:
        return (np.empty((T, C * L), dtype=np.float32),
                np.zeros((T,), dtype=bool))

    sigs = _base_signals(prices, cfg.vol_window)        # (C, T)
    block = np.full((T, C, L), np.nan, dtype=np.float64)
    for ci in range(C):
        sw = sliding_window_view(sigs[ci], W)           # (T-W+1, W)
        swc = sw - sw.mean(axis=1, keepdims=True)        # mean-center
        mag = np.abs(np.fft.rfft(swc, axis=1))           # (T-W+1, W//2+1)
        # Skip the DC bin (≈0 post mean-center); keep the L lowest.
        block[W - 1:, ci, :] = mag[:, 1:1 + L]
    features = block.reshape(T, C * L).astype(np.float32)
    valid = np.isfinite(features).all(axis=1)
    return features, valid


# ---------------------------------------------------------------------------
# MiniRocket-style (canonical kernel bank + PPV pooling)
# ---------------------------------------------------------------------------

def _minirocket_kernels() -> np.ndarray:
    """The 84 canonical MiniRocket kernels: length-9, exactly 3 taps at
    `+2` and 6 at `-1` (mean-zero: 3·2 + 6·(−1) = 0). `C(9,3) = 84`."""
    K = np.full((84, 9), -1.0, dtype=np.float64)
    for i, idx in enumerate(combinations(range(9), 3)):
        K[i, list(idx)] = 2.0
    return K


_MR_KERNELS = _minirocket_kernels()   # (84, 9), module-level constant


@dataclass(frozen=True)
class MiniRocketGridConfig:
    """MiniRocket-style encoder: 84 fixed kernels × fixed dilation set,
    PPV-pooled over a trailing window on the log-return series.

    Block shape `(C=84, L=n_dilations)`. Bias is fixed at 0 (no
    training-set quantile fit, so nothing leaks across the walk-forward
    split) and the dilation set is fixed (so `(C, L)` is constant rather
    than derived from input length) — see module docstring.
    """
    dilations:   tuple[int, ...] = (1, 2, 4, 8)
    pool_window: int = 128

    def n_channels(self) -> int:
        return _MR_KERNELS.shape[0]      # 84

    def compressed_len(self) -> int:
        return len(self.dilations)

    def feature_width(self) -> int:
        return self.n_channels() * self.compressed_len()

    def channel_names(self) -> list[str]:
        names: list[str] = []
        for k in range(self.n_channels()):
            for d in self.dilations:
                names.append(f'mr_k{k}_d{d}')
        return names

    def __post_init__(self) -> None:
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError(f'dilations={self.dilations} must be >=1')
        if self.pool_window < 2:
            raise ValueError(f'pool_window={self.pool_window} must be >=2')


def _causal_shift(x: np.ndarray, lag: int) -> np.ndarray:
    """`out[t] = x[t-lag]`, leading `lag` entries NaN. `lag=0` → copy."""
    if lag == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    out[lag:] = x[:x.shape[0] - lag]
    return out


def build_minirocket_features(
    prices: np.ndarray, cfg: MiniRocketGridConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """`(T, C*L)` MiniRocket-style block + `(T,)` valid mask.

    For each (kernel, dilation): a causal dilated FIR over the
    log-return series, then PPV = fraction of strictly-positive
    convolution outputs over the trailing `pool_window`. A bar is valid
    only when the full pooling window of conv outputs is finite (so the
    PPV denominator is exactly `pool_window`) — anything short is NaN
    and the valid mask drops it.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = prices.shape[0]
    C = cfg.n_channels()
    L = cfg.compressed_len()
    P = cfg.pool_window
    logp = np.log(np.maximum(prices, 1e-12))
    r = np.full(T, np.nan, dtype=np.float64)
    r[1:] = logp[1:] - logp[:-1]

    block = np.full((T, C, L), np.nan, dtype=np.float64)
    if T <= P:
        return (block.reshape(T, C * L).astype(np.float32),
                np.zeros((T,), dtype=bool))

    for di, d in enumerate(cfg.dilations):
        # Pre-shift r once per tap-lag used at this dilation (taps span
        # lags {0, d, 2d, ..., 8d}); kernels then just reweight them.
        shifted = [_causal_shift(r, (8 - j) * d) for j in range(9)]
        for ki in range(C):
            kw = _MR_KERNELS[ki]
            conv = np.zeros(T, dtype=np.float64)
            for j in range(9):
                conv = conv + kw[j] * shifted[j]
            fin = np.isfinite(conv)
            pos = np.where(fin, (conv > 0.0).astype(np.float64), 0.0)
            cpos = np.concatenate(([0.0], np.cumsum(pos)))
            cfin = np.concatenate(([0.0], np.cumsum(fin.astype(np.float64))))
            # Trailing-P window [t-P+1, t] → cumsum indices (t+1) and
            # (t+1-P). Valid only where that window is fully finite.
            t = np.arange(T)
            lo = t + 1 - P
            ppv = np.full(T, np.nan, dtype=np.float64)
            ok = lo >= 0
            idx = t[ok]
            denom = cfin[idx + 1] - cfin[lo[ok]]
            numer = cpos[idx + 1] - cpos[lo[ok]]
            full = denom >= (P - 0.5)        # exactly P finite values
            sel = idx[full]
            ppv[sel] = numer[full] / P
            block[:, ki, di] = ppv

    features = block.reshape(T, C * L).astype(np.float32)
    valid = np.isfinite(features).all(axis=1)
    return features, valid


# ---------------------------------------------------------------------------
# TickerData loaders + backbone builders + walk-forward wrappers
# ---------------------------------------------------------------------------

def load_ticker_spectral(
    name: str, *,
    stooq_dir: str | None = None,
    kaggle_dir: str | None = None,
    use_yahoo: bool = False,
    start: str | None = None,
    end: str | None = None,
    cfg: SpectralGridConfig | None = None,
) -> TickerData:
    """Load one ticker and build its spectral `(C, L)` stack.

    `features` is `(T, C*L)`, compatible with `align_tickers(K=C, F=L)`.
    """
    cfg = cfg or SpectralGridConfig()
    series = load_prices(
        name, stooq_dir=stooq_dir, kaggle_dir=kaggle_dir,
        use_yahoo=use_yahoo, start=start, end=end)
    prices = series.values.astype(np.float64)
    dates = np.asarray(series.index)
    features, valid = build_spectral_features(prices, cfg)
    return TickerData(
        name=name, prices=prices, dates=dates,
        features=features, targets={}, valid=valid,
    )


def load_ticker_minirocket(
    name: str, *,
    stooq_dir: str | None = None,
    kaggle_dir: str | None = None,
    use_yahoo: bool = False,
    start: str | None = None,
    end: str | None = None,
    cfg: MiniRocketGridConfig | None = None,
) -> TickerData:
    """Load one ticker and build its MiniRocket-style `(C, L)` stack."""
    cfg = cfg or MiniRocketGridConfig()
    series = load_prices(
        name, stooq_dir=stooq_dir, kaggle_dir=kaggle_dir,
        use_yahoo=use_yahoo, start=start, end=end)
    prices = series.values.astype(np.float64)
    dates = np.asarray(series.index)
    features, valid = build_minirocket_features(prices, cfg)
    return TickerData(
        name=name, prices=prices, dates=dates,
        features=features, targets={}, valid=valid,
    )


def make_spectral_backbone(
    tickers: list[TickerData], cfg: SpectralGridConfig,
) -> Backbone:
    return _make_cl_backbone(
        tickers, cfg.n_channels(), cfg.compressed_len(), kind='spectral')


def make_minirocket_backbone(
    tickers: list[TickerData], cfg: MiniRocketGridConfig,
) -> Backbone:
    return _make_cl_backbone(
        tickers, cfg.n_channels(), cfg.compressed_len(), kind='minirocket')


def train_scorer_spectral(
    tickers: list[TickerData], cfg: SpectralGridConfig | None = None,
    **train_kwargs,
) -> TrainResult:
    """Single-split train on the spectral stack (smoke / ablation)."""
    cfg = cfg or SpectralGridConfig()
    backbone = make_spectral_backbone(tickers, cfg)
    return train_scorer(tickers, backbone, **train_kwargs)


def train_scorer_minirocket(
    tickers: list[TickerData], cfg: MiniRocketGridConfig | None = None,
    **train_kwargs,
) -> TrainResult:
    """Single-split train on the MiniRocket stack (smoke / ablation)."""
    cfg = cfg or MiniRocketGridConfig()
    backbone = make_minirocket_backbone(tickers, cfg)
    return train_scorer(tickers, backbone, **train_kwargs)


def train_scorer_spectral_walkforward(
    tickers: list[TickerData], cfg: SpectralGridConfig | None = None,
    **walkforward_kwargs,
) -> WalkForwardResult:
    """Walk-forward variant — the leaderboard-bearing entrypoint."""
    cfg = cfg or SpectralGridConfig()
    backbone = make_spectral_backbone(tickers, cfg)
    return train_scorer_walkforward(tickers, backbone, **walkforward_kwargs)


def train_scorer_minirocket_walkforward(
    tickers: list[TickerData], cfg: MiniRocketGridConfig | None = None,
    **walkforward_kwargs,
) -> WalkForwardResult:
    """Walk-forward variant — the leaderboard-bearing entrypoint."""
    cfg = cfg or MiniRocketGridConfig()
    backbone = make_minirocket_backbone(tickers, cfg)
    return train_scorer_walkforward(tickers, backbone, **walkforward_kwargs)


__all__ = [
    'SpectralGridConfig',
    'MiniRocketGridConfig',
    'build_spectral_features',
    'build_minirocket_features',
    'load_ticker_spectral',
    'load_ticker_minirocket',
    'make_spectral_backbone',
    'make_minirocket_backbone',
    'train_scorer_spectral',
    'train_scorer_minirocket',
    'train_scorer_spectral_walkforward',
    'train_scorer_minirocket_walkforward',
]
