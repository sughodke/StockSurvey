"""Deterministic-indicator alternative to the pretrained CNN backbone.

Skip the pretrained CWT encoder entirely and feed a wide stack of
parameterized technical indicators (RSI/CCI on stride grids, MACD over
fast-period grid, realized vol over window grid) directly to the scoring
head. The scoring head sees the raw indicator vector through
`identity_backbone(K=1, F=F_total)` — z-norm + flatten, no learned
compression.

Two uses:
  1. Ablation. The pretrained CNN encoder's job is scale-invariant
     signal extraction from CWT bundles. If a flat parameter sweep
     matches it on rank IC, the encoder isn't earning its keep on this
     objective.
  2. Standalone scorer. No SSL pretrain required — just price data and
     a parameter grid.

Per (date, ticker) the feature row is one channel per
(indicator, parameter-tuple). Strided RSI/CCI take the latest value at
bar `t` (history is encoded in stride `w` itself, reaching back
`(n-1)*w` bars). MACD takes line / signal / histogram at `t`. Realized
vol takes the trailing-window std of log returns at `t`. There is no
lag-window dimension — `K=1` for the identity backbone.

Channel layout is deterministic (so a trained linear head's coefficients
are inspectable via `cfg.channel_names()`):
  [rsi(w_grid x n_grid)]
  [cci(w_grid x n_grid)]
  [vol(n_grid)]
  [macd_line(fast_grid)]
  [macd_signal(fast_grid)]   (optional)
  [macd_hist(fast_grid)]     (optional)
  [coherence(window_grid)]   (optional, default-on)

The `coherence` block is the deterministic-indicator analogue of the
regime trainer's `weights_scalogram` coherence term: trailing-window
Pearson correlation between short- and long-window realized vol. With
just point-in-time RSI/CCI/vol/MACD scalars the head has no path to
recover a *time-correlation* signal, so we precompute it and feed it
in as additional channels. See `apps/regime/src/regime/trainer.py`'s
`weights_scalogram` docstring for what we're approximating.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ss_features import TickerData, load_prices, realized_vol
from ss_indicators import cci_strided, macd, rolling_pearson_corr, rsi_strided
from factor.backbone import (
    Backbone, compute_input_stats, identity_backbone,
)
from factor.train import TrainResult, train_scorer
from factor.train_walkforward import (
    WalkForwardResult, train_scorer_walkforward,
)


@dataclass(frozen=True)
class IndicatorGridConfig:
    """Parameter grids for the deterministic-indicator backbone.

    Each grid entry contributes one feature channel per
    (indicator, parameter-tuple). Final feature width is reported by
    `feature_width()`; channel-by-channel labels by `channel_names()`.

    MACD slow / signal are pinned to the canonical ratios off `fast`
    (slow = 2*fast, signal = max(2, 3*fast/4)) so one knob sweeps the
    whole indicator's timescale — matches the convention in
    `replay.features.build_features_and_targets`.
    """
    rsi_n_grid:    tuple[int, ...] = (5, 7, 10, 14, 21, 30)
    rsi_w_grid:    tuple[int, ...] = (1, 5, 10, 21, 63)
    # CCI cells dominate the warmup floor because cci_strided needs
    # (n-1)*w+1 bars before the first valid output. Capping at n=40
    # AND w=21 keeps the worst cell at (40-1)*21+1 = 820 bars (~3.25y),
    # vs the previous (n=80, w=63) which needed 4978 bars (~19.7y) and
    # left walk-forward windows over the first 19y of the dataset
    # uninformative. RSI-strided's warmup is w+n-1 (additive, not
    # multiplicative) so its widest cell stays cheap.
    cci_n_grid:    tuple[int, ...] = (10, 14, 20, 40)
    cci_w_grid:    tuple[int, ...] = (1, 5, 10, 21)
    vol_n_grid:    tuple[int, ...] = (5, 10, 20, 60, 120, 252)
    macd_fast_grid: tuple[int, ...] = (5, 8, 12, 21, 34, 55)
    include_macd_signal: bool = True
    include_macd_hist:   bool = True
    # Coherence: trailing-`N` Pearson corr between short-window and
    # long-window realized vol. Mirrors the (s=3, s=126) short/long pair
    # in `regime.weights_scalogram`'s coherence term, just with vol-
    # window proxies instead of CWT power. Setting `coherence_window_grid
    # = ()` disables the block.
    coherence_short_window: int = 5
    coherence_long_window:  int = 252
    coherence_window_grid:  tuple[int, ...] = (10, 20, 60, 120)

    def feature_width(self) -> int:
        n_macd = len(self.macd_fast_grid) * (
            1 + int(self.include_macd_signal) + int(self.include_macd_hist))
        return (
            len(self.rsi_n_grid) * len(self.rsi_w_grid)
            + len(self.cci_n_grid) * len(self.cci_w_grid)
            + len(self.vol_n_grid)
            + n_macd
            + len(self.coherence_window_grid)
        )

    def channel_names(self) -> list[str]:
        names: list[str] = []
        for w in self.rsi_w_grid:
            for n in self.rsi_n_grid:
                names.append(f'rsi_w{w}_n{n}')
        for w in self.cci_w_grid:
            for n in self.cci_n_grid:
                names.append(f'cci_w{w}_n{n}')
        for n in self.vol_n_grid:
            names.append(f'vol_n{n}')
        for f in self.macd_fast_grid:
            names.append(f'macd_line_f{f}')
        if self.include_macd_signal:
            for f in self.macd_fast_grid:
                names.append(f'macd_signal_f{f}')
        if self.include_macd_hist:
            for f in self.macd_fast_grid:
                names.append(f'macd_hist_f{f}')
        for n in self.coherence_window_grid:
            names.append(
                f'coherence_w{n}_short{self.coherence_short_window}'
                f'_long{self.coherence_long_window}')
        return names


def build_indicator_features(
    prices: np.ndarray, cfg: IndicatorGridConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(T, F)` indicator-channel matrix + `(T,)` valid mask.

    Channel order matches `cfg.channel_names()`. `valid[t]` is True iff
    every channel is finite at `t` — strided RSI/CCI return NaN until
    `(n-1)*w + 1` bars are available, realized vol is NaN until `window`
    bars, and the MACD warmup is implicit in the EMA seeding (finite
    from t=0 but only meaningful past `slow` bars; this builder does not
    artificially clip that — the scoring loop's z-norm and the
    cross-sectional valid-AND across the universe handle it).
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = prices.shape[0]
    if T == 0:
        F = cfg.feature_width()
        return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=bool)

    # Per-cell rsi_strided / cci_strided here, not their grid-vectorized
    # variants in `ss_indicators`. With the default cfg (n_grid ~6, w_grid
    # ~5) the grid versions are ~30% slower because n=6 is too small to
    # amortize numpy's per-op dispatch overhead — Python scalar Wilder is
    # faster than numpy ops on tiny 6-element arrays. The grid variants
    # become a win only when grids grow past ~30 cells, which would
    # happen in apps/replay's FiLM head training, not here.
    cols: list[np.ndarray] = []
    for w in cfg.rsi_w_grid:
        for n in cfg.rsi_n_grid:
            cols.append(rsi_strided(prices, n=int(n), w=int(w)))
    for w in cfg.cci_w_grid:
        for n in cfg.cci_n_grid:
            cols.append(cci_strided(prices, n=int(n), w=int(w)))
    for n in cfg.vol_n_grid:
        cols.append(realized_vol(prices, window=int(n)))

    macd_lines: list[np.ndarray] = []
    macd_signals: list[np.ndarray] = []
    macd_hists: list[np.ndarray] = []
    for f in cfg.macd_fast_grid:
        f_i = int(f)
        line, sig, hist = macd(prices, fast=f_i, slow=2 * f_i,
                               signal=max(2, (f_i * 3) // 4))
        macd_lines.append(line)
        macd_signals.append(sig)
        macd_hists.append(hist)
    cols.extend(macd_lines)
    if cfg.include_macd_signal:
        cols.extend(macd_signals)
    if cfg.include_macd_hist:
        cols.extend(macd_hists)

    if cfg.coherence_window_grid:
        # Compute the short/long vol pair once, then sweep window sizes.
        vol_short = realized_vol(prices, window=int(cfg.coherence_short_window))
        vol_long = realized_vol(prices, window=int(cfg.coherence_long_window))
        for n in cfg.coherence_window_grid:
            cols.append(rolling_pearson_corr(vol_short, vol_long, window=int(n)))

    features = np.stack(cols, axis=1).astype(np.float32)
    valid = np.isfinite(features).all(axis=1)
    return features, valid


def load_ticker_indicators(
    name: str, *,
    stooq_dir: str | None = None,
    kaggle_dir: str | None = None,
    use_yahoo: bool = False,
    start: str | None = None,
    end: str | None = None,
    cfg: IndicatorGridConfig | None = None,
) -> TickerData:
    """Load one ticker and build its deterministic-indicator stack.

    Returns a `TickerData` shaped for the scoring path: `features` is
    `(T, F)` with `F = cfg.feature_width()`, `targets` and
    `target_grids` are empty (this baseline doesn't pretrain heads).
    Compatible with `align_tickers(K=1, F=F)`.
    """
    cfg = cfg or IndicatorGridConfig()
    series = load_prices(
        name, stooq_dir=stooq_dir, kaggle_dir=kaggle_dir,
        use_yahoo=use_yahoo, start=start, end=end)
    prices = series.values.astype(np.float64)
    dates = np.asarray(series.index)
    features, valid = build_indicator_features(prices, cfg)
    return TickerData(
        name=name, prices=prices, dates=dates,
        features=features, targets={}, valid=valid,
    )


def make_indicator_backbone(
    tickers: list[TickerData], cfg: IndicatorGridConfig,
) -> Backbone:
    """Identity backbone sized to `cfg.feature_width()` with pool z-norm
    stats from every valid (date, ticker) feature row.

    Pool z-norm is the cheapest fix for indicators living at very
    different scales (RSI in [0,100], CCI ~±200, vol ~0.01); without it
    a linear head's gradient is dominated by whichever channel has the
    largest variance. Cross-sectional per-bar standardization would be
    stricter (matches the IC objective exactly) but requires changes to
    the training loop — pool z-norm is enough to make the head trainable.
    """
    F = cfg.feature_width()
    for td in tickers:
        if td.features.shape[1] != F:
            raise ValueError(
                f'ticker {td.name!r}: features width {td.features.shape[1]} '
                f'!= cfg.feature_width()={F}; rebuild via '
                f'load_ticker_indicators with this cfg')
    mu, sd = compute_input_stats(tickers, K=1, F=F)
    return identity_backbone(K=1, F=F, feat_mu=mu, feat_sd=sd)


def train_scorer_indicators(
    tickers: list[TickerData], cfg: IndicatorGridConfig | None = None,
    **train_kwargs,
) -> TrainResult:
    """Train the scoring head against rank IC on the deterministic
    indicator stack — no pretrained backbone involved.

    Equivalent to building an `identity_backbone(K=1, F=F)` with pool
    z-norm stats and calling `train_scorer(tickers, backbone, ...)`.
    `finetune_steps>0` is meaningless here (the identity backbone has
    no conv weights to update) — Stage 2 will run but only continue
    head training. Pass `finetune_steps=0` to skip the wasted pass.
    """
    cfg = cfg or IndicatorGridConfig()
    backbone = make_indicator_backbone(tickers, cfg)
    return train_scorer(tickers, backbone, **train_kwargs)


def train_scorer_indicators_walkforward(
    tickers: list[TickerData], cfg: IndicatorGridConfig | None = None,
    **walkforward_kwargs,
) -> WalkForwardResult:
    """Walk-forward variant of `train_scorer_indicators`.

    Builds the identity backbone the same way, then routes through
    `train_scorer_walkforward` instead of `train_scorer`. Returns a
    `WalkForwardResult` with one entry per train/val window.
    """
    cfg = cfg or IndicatorGridConfig()
    backbone = make_indicator_backbone(tickers, cfg)
    return train_scorer_walkforward(tickers, backbone, **walkforward_kwargs)


__all__ = [
    'IndicatorGridConfig',
    'build_indicator_features',
    'load_ticker_indicators',
    'make_indicator_backbone',
    'train_scorer_indicators',
    'train_scorer_indicators_walkforward',
]
