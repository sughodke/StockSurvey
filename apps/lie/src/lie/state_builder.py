"""Per-date market-state feature vector ('geometric fingerprint').

The Barbour / Wheeler-DeWitt framing motivates the design constraints, not
the math:

* **Shape only, no levels.** Features are returns, correlations, and
  shape statistics -- never raw prices. A 1998 bull market and a 2019
  bull market should produce similar feature vectors if the timeless-
  geometry hypothesis is right.

* **Universe-size invariant where possible.** Top-K eigenvalues are
  normalized as fractions of the trace. Participation ratio is divided
  by N. Cross-sectional moments are unitless. This lets a future Test 2
  re-run the manifold on a different universe and check stability.

* **No calendar time leaks in.** The function takes a raw `(T, N)`
  panel; output is a `(T_out, F)` matrix where `T_out = T - warmup`.
  The caller pairs rows with dates externally; nothing inside this
  module sees a Timestamp.

Feature layout (default `MarketStateConfig` => F = 26):

    [0:8]    top-8 eigenvalues of trailing-window correlation matrix,
             each as fraction of trace (sum of all eigenvalues).
    [8]      participation ratio normalized by universe size:
             `effective_rank(C) / N`. In `[1/N, 1]`.
    [9:12]   cross-sectional MEAN of vol-normalized returns at three
             horizons (5d, 21d, 63d). `mean_i(r_i / sigma_i)`.
    [12:15]  cross-sectional STD of those vol-normalized returns.
             High = dispersion; low = the universe is moving as one
             unit (the symmetry-collapse regime).
    [15:18]  cross-sectional SKEW per horizon.
    [18:21]  cross-sectional KURT per horizon.
    [21:24]  AGGREGATE skew, kurt, and tail-fraction (P[|r|>2 sigma])
             of equal-weighted universe returns over `tail_horizon`.
    [24:26]  spectral gap features: `(lambda_1 - lambda_2) / trace` and
             `lambda_top_2_sum / trace` -- tracks how concentrated the
             leading mode is.

The leading-eigenvalue features double-encode something the participation
ratio also captures, but in a form a downstream PCA can use to extract a
non-redundant axis. Don't pre-orthogonalize -- that's what `manifold.fit`
is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from lie.correlation_network import log_returns
from lie.symmetry_rank import effective_rank


@dataclass
class MarketStateConfig:
    """Hyperparameters for `build_market_state`."""

    lookback: int = 60
    """Trailing-window size (in bars) for the rolling correlation matrix."""

    n_top_eigvals: int = 8
    """How many top eigenvalues to keep as features. Default 8 matches
    `n_components` in `ManifoldMapper`. If the universe has fewer than 8
    valid names at a date, missing slots are zero-padded."""

    momentum_horizons: tuple[int, ...] = (5, 21, 63)
    """Horizons (in bars) for vol-normalized cross-sectional return
    moments. Each horizon h consumes h+1 bars of history."""

    tail_horizon: int = 63
    """Window for aggregate skew / kurt / tail-fraction of the equal-
    weighted universe-return time series."""

    sigma_floor: float = 1e-6
    """Numerical floor on per-name realized vol to avoid divide-by-zero
    when a name has flatlined inside the horizon."""

    def warmup(self) -> int:
        """Minimum number of leading bars discarded before the first valid
        state vector. Largest of (lookback+1, max_momentum+1, tail+1)."""
        max_h = max(max(self.momentum_horizons), self.tail_horizon, self.lookback)
        return max_h + 1

    def feature_width(self) -> int:
        n_h = len(self.momentum_horizons)
        # eigvals + erank + (mean,std,skew,kurt) per horizon + agg(3) + gap(2)
        return self.n_top_eigvals + 1 + 4 * n_h + 3 + 2


def build_market_state(
    prices: np.ndarray,
    config: MarketStateConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-date market-state feature vectors from a `(T, N)` price panel.

    Returns
    -------
    (states, valid_t)
        `states` is `(T, F)` where `F = config.feature_width()`. Rows before
        `config.warmup()` and rows where the underlying correlation matrix
        couldn't be computed (fewer than 2 valid names) are filled with NaN.
        `valid_t` is a boolean array marking rows where every feature is
        finite -- callers should `prices.iloc[valid_t]` before fitting the
        manifold.
    """
    if config is None:
        config = MarketStateConfig()
    if prices.ndim != 2:
        raise ValueError(f'expected 2-D price panel, got shape {prices.shape}')

    T, N = prices.shape
    F = config.feature_width()
    states = np.full((T, F), np.nan)

    rets = log_returns(prices)
    # `rets[t-1]` corresponds to `prices[t]`; align by treating the state at
    # `prices[t]` as a function of returns up to and including `rets[t-1]`.
    # The state is computed for `t` in `[warmup, T)` -- i.e. as of the bar
    # whose price is `prices[t]`, using only data through `rets[t-1]`.

    n_top = config.n_top_eigvals
    horizons = config.momentum_horizons
    n_h = len(horizons)
    tail_h = config.tail_horizon
    sigma_floor = config.sigma_floor
    lookback = config.lookback

    # Pre-compute equal-weighted universe returns (NaN-safe via nanmean).
    with np.errstate(invalid='ignore'):
        eq_rets = np.nanmean(rets, axis=1)

    for t in range(config.warmup(), T):
        feat = np.zeros(F)
        col = 0

        # 1) Eigenvalue features from trailing correlation.
        window = rets[t - lookback: t]
        valid = ~np.isnan(window).any(axis=0)
        if int(valid.sum()) < 2:
            states[t] = np.nan
            continue
        sub = window[:, valid]
        sub = sub - sub.mean(axis=0, keepdims=True)
        std = sub.std(axis=0, ddof=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        sub = sub / std
        corr = (sub.T @ sub) / (sub.shape[0] - 1)
        corr = np.clip(corr, -1.0, 1.0)
        eigvals = np.linalg.eigvalsh(corr)[::-1]  # descending
        eigvals = np.maximum(eigvals, 0.0)
        trace = float(eigvals.sum())
        if trace <= 0:
            states[t] = np.nan
            continue

        n_valid = int(valid.sum())
        eig_frac = eigvals / trace
        feat[col: col + n_top] = 0.0
        k = min(n_top, len(eig_frac))
        feat[col: col + k] = eig_frac[:k]
        col += n_top

        # 2) Participation ratio normalized by universe size.
        feat[col] = effective_rank(corr) / max(n_valid, 1)
        col += 1

        # 3) Cross-sectional moments of vol-normalized returns per horizon.
        for h in horizons:
            r_window = rets[t - h: t]
            valid_h = ~np.isnan(r_window).any(axis=0)
            if int(valid_h.sum()) < 4:
                # not enough cross-sectional sample for skew/kurt; leave zeros
                col += 4
                continue
            r_sub = r_window[:, valid_h]
            sigma = r_sub.std(axis=0, ddof=1)
            sigma = np.maximum(sigma, sigma_floor)
            r_norm = r_sub.sum(axis=0) / (sigma * np.sqrt(h))  # vol-normalized cumulative
            mean_v = float(np.mean(r_norm))
            std_v = float(np.std(r_norm, ddof=1)) if len(r_norm) > 1 else 0.0
            # Sample skew/kurt; guarded against degenerate std.
            if std_v > 0:
                z = (r_norm - mean_v) / std_v
                skew_v = float(np.mean(z ** 3))
                kurt_v = float(np.mean(z ** 4) - 3.0)  # excess kurt
            else:
                skew_v = 0.0
                kurt_v = 0.0
            feat[col + 0] = mean_v
            feat[col + 1] = std_v
            feat[col + 2] = skew_v
            feat[col + 3] = kurt_v
            col += 4

        # 4) Aggregate tail behavior of equal-weighted universe returns.
        eq_window = eq_rets[t - tail_h: t]
        eq_window = eq_window[~np.isnan(eq_window)]
        if len(eq_window) >= 4:
            mu = float(np.mean(eq_window))
            sd = float(np.std(eq_window, ddof=1))
            if sd > 0:
                z = (eq_window - mu) / sd
                feat[col + 0] = float(np.mean(z ** 3))            # skew
                feat[col + 1] = float(np.mean(z ** 4) - 3.0)      # excess kurt
                feat[col + 2] = float(np.mean(np.abs(z) > 2.0))   # tail fraction
        col += 3

        # 5) Spectral-gap features.
        l1 = float(eigvals[0]) if len(eigvals) >= 1 else 0.0
        l2 = float(eigvals[1]) if len(eigvals) >= 2 else 0.0
        feat[col + 0] = (l1 - l2) / trace
        feat[col + 1] = (l1 + l2) / trace
        col += 2

        states[t] = feat

    valid_t = np.all(np.isfinite(states), axis=1)
    return states, valid_t


__all__ = ['MarketStateConfig', 'build_market_state']
