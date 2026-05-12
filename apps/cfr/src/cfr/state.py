"""Infoset bucketing — coarse regime label for tabular CFR.

The macro-regime diagnostic established that **VIX (or its equity-
vol proxy) is the strongest single regime-axis predictor of
when the pivot-arc apps deliver alpha**, with cross-sectional
dispersion as a structurally independent second axis. The tabular
CFR's infoset is a `(vol_bucket, dispersion_bucket)` tuple — 3×3 =
9 discrete states by default.

The bucket cutoffs are computed from the *training-period* sample
of each feature and held fixed for the val period — so the regime
labels reflect the training regime's distribution rather than
sliding cutoffs that would let the val period leak its own
distribution into its own labels. This matches how `apps/gate` and
`apps/vol` set thresholds: train-only quantiles, frozen for val.

For Phase 1 the features are pure-numpy aggregates over the
universe; no macro inputs required. Adding macro (VIX from FRED,
gold-VIX, etc.) is a one-feature-at-a-time extension — the bucket
infrastructure here is the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


def _ew_log_returns(prices: pd.DataFrame, *, min_active: int | None = None) -> np.ndarray:
    """Per-bar EW universe log return. NaN where < `min_active` names.

    Mirrors `apps/gate.aggregate.build_ew_aggregate` but returns a
    bare numpy array aligned to `prices.index`. Internal — the public
    API is `InfosetBuilder`.

    `min_active=None` (default) auto-scales to `max(3, n_tickers // 4)`
    — same intent as gate's `min_active=10` on a 300-name universe
    (~3%), but tracks panel size so this works on small smoke /
    test panels too.
    """
    if min_active is None:
        min_active = max(3, prices.shape[1] // 4)
    ret = prices.pct_change(fill_method=None)
    valid = ret.notna().values
    n_active = valid.sum(axis=1)
    safe_n = np.maximum(n_active, 1)
    ew_simple = np.where(valid, ret.values, 0.0).sum(axis=1) / safe_n
    ew_log = np.log1p(ew_simple)
    ew_log[n_active < min_active] = np.nan
    return ew_log


def _trailing_vol(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing stdev with ddof=1. NaN where sample < 2."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(len(x)):
        lo = max(0, t + 1 - window)
        sample = x[lo:t + 1]
        sample = sample[~np.isnan(sample)]
        if len(sample) >= 2:
            out[t] = float(np.std(sample, ddof=1))
    return out


def _cross_sectional_dispersion(prices: pd.DataFrame, window: int) -> np.ndarray:
    """Per-bar cross-sectional dispersion of trailing log returns.

    `dispersion[t] = std_over_tickers( log(p[t]) - log(p[t-window]) )`,
    computed over the names with non-NaN window-trailing returns.
    NaN until enough history exists.
    """
    p = prices.values
    log_p = np.log(p, where=(p > 0), out=np.full_like(p, np.nan, dtype=np.float64))
    T = log_p.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T):
        ret = log_p[t] - log_p[t - window]
        ret = ret[~np.isnan(ret)]
        if len(ret) >= 3:
            out[t] = float(np.std(ret, ddof=1))
    return out


def _bucket(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Map values into `[0..len(edges)]` buckets via `np.searchsorted`.

    NaN values map to `-1` so the caller can distinguish "no-state-yet"
    from "low-bucket". `edges` must be sorted ascending.
    """
    out = np.searchsorted(np.asarray(edges, dtype=np.float64),
                          values, side='right').astype(np.int64)
    out[np.isnan(values)] = -1
    return out


@dataclass
class InfosetBuilder:
    """Builds a per-bar `(infoset_id, valid_mask)` from a price panel.

    Configuration:
      `vol_window`         : trailing window for EW vol feature
      `dispersion_window`  : trailing window for cross-sectional disp
      `n_vol_buckets`      : 3 → low / mid / high
      `n_disp_buckets`     : 3 → low / mid / high

    Usage:
      builder = InfosetBuilder()
      vol_feat, disp_feat = builder.features(prices_train)
      builder.fit(vol_feat[~np.isnan(vol_feat)], disp_feat[~np.isnan(disp_feat)])
      ids_train = builder.transform(prices_train)
      ids_val = builder.transform(prices_val)
      # ids_train, ids_val are int64; -1 means "feature still warming up"

    The fit step computes the cutoff quantiles on the train sample;
    transform uses those frozen cutoffs on any panel. Fitting on val
    would leak val-regime distribution into val-regime labels.
    """
    vol_window: int = 21
    dispersion_window: int = 21
    n_vol_buckets: int = 3
    n_disp_buckets: int = 3
    vol_edges: tuple[float, ...] = field(default_factory=tuple)
    disp_edges: tuple[float, ...] = field(default_factory=tuple)
    fitted: bool = False

    @property
    def n_infosets(self) -> int:
        return self.n_vol_buckets * self.n_disp_buckets + 1  # +1 for warmup

    @property
    def warmup_id(self) -> int:
        return self.n_vol_buckets * self.n_disp_buckets

    def features(self, prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ew_log = _ew_log_returns(prices)
        vol = _trailing_vol(ew_log, self.vol_window)
        disp = _cross_sectional_dispersion(prices, self.dispersion_window)
        return vol, disp

    def fit(self, vol_sample: np.ndarray, disp_sample: np.ndarray) -> 'InfosetBuilder':
        """Compute bucket cutoffs from a training-period sample."""
        def _quantile_edges(x: np.ndarray, n_buckets: int) -> tuple[float, ...]:
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return tuple(float('inf') for _ in range(n_buckets - 1))
            qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
            return tuple(float(q) for q in np.quantile(x, qs))

        self.vol_edges = _quantile_edges(vol_sample, self.n_vol_buckets)
        self.disp_edges = _quantile_edges(disp_sample, self.n_disp_buckets)
        self.fitted = True
        return self

    def transform(self, prices: pd.DataFrame) -> np.ndarray:
        """Map a price panel to `(T,)` int infoset ids.

        Warming-up bars (either feature still NaN) get `warmup_id`.
        Otherwise the id is `vol_bucket * n_disp_buckets + disp_bucket`,
        with both buckets in `[0, n_vol_buckets)` / `[0, n_disp_buckets)`.
        """
        if not self.fitted:
            raise RuntimeError('InfosetBuilder must be fit() before transform()')
        vol, disp = self.features(prices)
        vol_b = _bucket(vol, self.vol_edges)
        disp_b = _bucket(disp, self.disp_edges)
        ids = vol_b * self.n_disp_buckets + disp_b
        warmup = (vol_b < 0) | (disp_b < 0)
        ids[warmup] = self.warmup_id
        return ids

    def fit_transform(self, prices: pd.DataFrame) -> np.ndarray:
        vol, disp = self.features(prices)
        self.fit(vol, disp)
        return self.transform(prices)


def default_infoset_builder() -> InfosetBuilder:
    """3×3 = 9 regime buckets + 1 warmup bucket."""
    return InfosetBuilder(
        vol_window=21,
        dispersion_window=21,
        n_vol_buckets=3,
        n_disp_buckets=3,
    )


__all__ = ['InfosetBuilder', 'default_infoset_builder']
