"""Continuous state-vector builder for Deep CFR.

Replaces the 9-cell discrete `InfosetBuilder` from Phase 1-2 with a
~10-dim continuous state vector per bar. Two feature blocks:

**Universe-internal (6 features, always available):**
  - `vol_21`     : trailing 21d stdev of EW universe log return
  - `disp_21`    : trailing 21d cross-sectional stdev of per-ticker log returns
  - `ret_21`     : trailing 21d mean EW log return
  - `ret_63`     : trailing 63d mean EW log return
  - `tdd_21`     : trailing 21d max drawdown of EW universe
  - `breadth`    : n_active / max(n_active) — universe coverage

**Macro (4 features, optional — if `macro` panel is provided):**
  - `vix`              : CBOE VIX (FRED VIXCLS)
  - `credit_baa`       : Moody's Baa minus 10y Treasury (BAA10Y)
  - `m2_yoy`           : 12mo % change in M2 money supply
  - `real_yield_10y`   : 10y TIPS yield (DFII10)

Per the macro-regime-diagnostic finding, these 4 of the 6 FRED
features carry directional signal across the pivot-arc apps (Pearson
r in [+0.34, +0.49] or −0.38). `slope_10y_3m` (noise) and
`fed_funds` (collinear with VIX in our window sample) are dropped.

All features are z-scored against the training-period sample stats
(mean and stdev fit on train, frozen for val) — sidesteps the
train/val distribution-shift problem that killed macro v1a in
`apps/gate`.

Public API:

```python
builder = StateVecBuilder()
builder.fit(prices_train, macro_train)            # compute per-feature stats
state_train = builder.transform(prices_train, macro_train)  # (T, n_features)
state_val   = builder.transform(prices_val,   macro_val)
```

Output is `(T, n_features) float32`, ready for the DeepCFR regret
net. Bars before warmup (insufficient history for `tdd_21` etc.)
have a separate boolean mask exposed via `valid_mask(prices, macro)`
— callers should skip those bars at training time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from cfr.state import _ew_log_returns, _trailing_vol, _cross_sectional_dispersion


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` bars; NaN where < `window` non-NaN."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(len(x)):
        lo = max(0, t + 1 - window)
        sample = x[lo:t + 1]
        sample = sample[~np.isnan(sample)]
        if len(sample) >= max(2, window // 2):
            out[t] = float(np.mean(sample))
    return out


def _trailing_max_drawdown(log_ret: np.ndarray, window: int) -> np.ndarray:
    """Max drawdown over the trailing `window` bars (positive number).

    Same logic as `gate.aggregate._trailing_max_drawdown`.
    """
    out = np.full_like(log_ret, np.nan, dtype=np.float64)
    for t in range(len(log_ret)):
        lo = max(0, t + 1 - window)
        sample = log_ret[lo:t + 1]
        sample = sample[~np.isnan(sample)]
        if len(sample) >= 2:
            cum = np.cumsum(sample)
            peak = np.maximum.accumulate(cum)
            out[t] = float(np.max(peak - cum))
    return out


@dataclass
class StateVecBuilder:
    """Per-bar state vector for Deep CFR.

    Configuration:
      `vol_window`       : trailing window for EW vol (default 21)
      `dispersion_window`: trailing window for cross-sectional dispersion (21)
      `ret_short_window` : trailing 21d EW return
      `ret_long_window`  : trailing 63d EW return
      `dd_window`        : trailing 21d EW max drawdown
      `clip_z`           : clip z-scored features to ±this many sd
                           (default 5; protects regret_net from outliers)

    Universe features always present. Macro features included iff
    `macro` is non-None (a DataFrame indexed by date with columns
    `vix`, `credit_baa`, `m2_yoy`, `real_yield_10y`).
    """
    vol_window: int = 21
    dispersion_window: int = 21
    ret_short_window: int = 21
    ret_long_window: int = 63
    dd_window: int = 21
    clip_z: float = 5.0

    feat_names: list[str] = field(default_factory=list)
    feat_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feat_std:  np.ndarray = field(default_factory=lambda: np.zeros(0))
    fitted: bool = False

    @property
    def n_features(self) -> int:
        return len(self.feat_names)

    def _raw_features(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, list[str]]:
        ew_log = _ew_log_returns(prices)
        vol = _trailing_vol(ew_log, self.vol_window)
        disp = _cross_sectional_dispersion(prices, self.dispersion_window)
        ret_s = _trailing_mean(ew_log, self.ret_short_window)
        ret_l = _trailing_mean(ew_log, self.ret_long_window)
        tdd = _trailing_max_drawdown(ew_log, self.dd_window)

        # Breadth: per-bar n_active over the panel's max
        ret = prices.pct_change(fill_method=None)
        valid = ret.notna().values
        n_active = valid.sum(axis=1).astype(np.float64)
        n_max = float(n_active.max()) if n_active.size else 1.0
        breadth = n_active / max(n_max, 1.0)

        names = ['vol_21', 'disp_21', 'ret_21', 'ret_63', 'tdd_21', 'breadth']
        cols = [vol, disp, ret_s, ret_l, tdd, breadth]

        if macro is not None and not macro.empty:
            # Reindex macro to prices.index, ffill (no-future-leak: only
            # use values whose macro release date <= bar date; the FRED
            # ffill semantics give us this).
            macro_aligned = macro.reindex(prices.index, method='ffill')
            for col in ('vix', 'credit_baa', 'm2_yoy', 'real_yield_10y'):
                if col in macro_aligned.columns:
                    names.append(f'macro_{col}')
                    cols.append(macro_aligned[col].values.astype(np.float64))

        feats = np.stack(cols, axis=1)   # (T, F)
        return feats, names

    def fit(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
    ) -> 'StateVecBuilder':
        """Compute z-score stats on the given (training) panel."""
        feats, names = self._raw_features(prices, macro)
        # Per-feature mean / std on non-NaN samples
        means = []
        stds = []
        for j in range(feats.shape[1]):
            col = feats[:, j]
            col = col[~np.isnan(col)]
            if len(col) >= 30:
                means.append(float(col.mean()))
                stds.append(float(col.std(ddof=1)))
            else:
                means.append(0.0)
                stds.append(1.0)
        self.feat_names = names
        self.feat_mean = np.array(means, dtype=np.float64)
        self.feat_std = np.where(np.array(stds) > 1e-12,
                                  np.array(stds), 1.0)
        self.fitted = True
        return self

    def transform(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Z-scored state vectors `(T, n_features) float32`.

        NaN values (warmup bars before features have history) are
        replaced with 0 (i.e., the per-feature train mean after
        z-scoring). Combine with `valid_mask()` to skip warmup bars.
        Clipping at ±`clip_z` standard deviations protects the
        regret_net from per-feature outliers.
        """
        if not self.fitted:
            raise RuntimeError('StateVecBuilder must be fit() before transform()')
        feats, names = self._raw_features(prices, macro)
        if names != self.feat_names:
            raise ValueError(
                f'feature schema mismatch: fit on {self.feat_names}, '
                f'transform got {names} (different macro columns?)')
        z = (feats - self.feat_mean) / self.feat_std
        z = np.clip(z, -self.clip_z, self.clip_z)
        z = np.where(np.isnan(z), 0.0, z)
        return z.astype(np.float32)

    def valid_mask(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """`(T,) bool` — True where every feature has a non-NaN value
        in the raw (pre-zscore, pre-fill) state vector. Used by the
        Deep CFR walkforward to skip warmup bars at training time."""
        feats, _ = self._raw_features(prices, macro)
        return ~np.isnan(feats).any(axis=1)


def default_state_vec_builder() -> StateVecBuilder:
    return StateVecBuilder()


__all__ = ['StateVecBuilder', 'default_state_vec_builder']
