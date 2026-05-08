"""Per-ticker shape features for the cross-sectional v3 test.

Where `state_builder.MarketStateConfig` summarizes the *universe* into a
single feature vector per date, this module produces a `(T, N, F_t)` array
of per-ticker features -- one feature vector per `(date, ticker)` pair.
The two are concatenated downstream to form the joined embedding the
cross-sectional kNN runs over.

Same constraints as `state_builder`: shape only, no levels. Every feature
here is either a vol-normalized return, a vol level, a drawdown ratio, or
a moment of the trailing return distribution. None depend on absolute
price.

Default `TickerFeatureConfig` -> F_t = 7:

    [0:3]   vol-normalized cumulative log-return at 5d / 21d / 63d
    [3]     21d realized vol of log-returns
    [4]     drawdown from 63d high (negative number; 0 = at the high)
    [5]     trailing 63d return-distribution skew
    [6]     trailing 63d return-distribution excess kurtosis

These overlap conceptually with the cross-sectional moments in
`build_market_state` -- the difference is granularity. The state-builder
moments are universe-aggregate (one number per date for "the market
skew"); these are per-ticker (one number per (date, ticker) for
"AAPL's skew on 2024-03-15"). The cross-sectional kNN needs the latter
to discriminate between names within the same date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lie.correlation_network import log_returns


@dataclass
class TickerFeatureConfig:
    """Hyperparameters for `build_ticker_features`."""

    momentum_horizons: tuple[int, ...] = (5, 21, 63)
    vol_horizon: int = 21
    drawdown_horizon: int = 63
    moment_horizon: int = 63
    sigma_floor: float = 1e-6

    def warmup(self) -> int:
        return max(
            max(self.momentum_horizons),
            self.vol_horizon,
            self.drawdown_horizon,
            self.moment_horizon,
        ) + 1

    def feature_width(self) -> int:
        return len(self.momentum_horizons) + 1 + 1 + 2  # mom + vol + dd + (skew, kurt)


def build_ticker_features(
    prices: np.ndarray,
    config: TickerFeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-(date, ticker) feature tensor from a `(T, N)` price panel.

    Returns
    -------
    (features, valid)
        `features` is `(T, N, F_t)` with NaN where features couldn't be
        computed (warmup or NaN price gaps inside any horizon window).
        `valid` is `(T, N)` boolean -- True iff that (date, ticker) row
        is fully finite. Callers building cross-sectional embeddings
        should `valid_state[t] & valid_ticker[t, i]` for the joint mask.
    """
    if config is None:
        config = TickerFeatureConfig()
    if prices.ndim != 2:
        raise ValueError(f'expected 2-D price panel, got {prices.shape}')

    T, N = prices.shape
    F = config.feature_width()
    out = np.full((T, N, F), np.nan)
    rets = log_returns(prices)
    n_h = len(config.momentum_horizons)

    for t in range(config.warmup(), T):
        col = 0
        # Vol-normalized cumulative log-returns at each momentum horizon.
        for h in config.momentum_horizons:
            window = rets[t - h: t]                       # (h, N)
            valid = ~np.isnan(window).any(axis=0)
            sigma = np.where(valid, window.std(axis=0, ddof=1), np.nan)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0),
                             sigma, np.nan)
            mom = np.where(valid, window.sum(axis=0), np.nan)
            denom = np.where(np.isfinite(sigma) & (sigma > config.sigma_floor),
                             sigma * np.sqrt(h), np.nan)
            out[t, :, col] = mom / denom
            col += 1

        # Realized vol over `vol_horizon`.
        h_v = config.vol_horizon
        win = rets[t - h_v: t]
        valid = ~np.isnan(win).any(axis=0)
        sigma = np.where(valid, win.std(axis=0, ddof=1), np.nan)
        out[t, :, col] = sigma
        col += 1

        # Drawdown from `drawdown_horizon` high (in log-space), as a
        # NEGATIVE number: 0 == currently at the high.
        h_d = config.drawdown_horizon
        # need prices[t - h_d : t + 1] -- the +1 includes today
        if t - h_d >= 0:
            p_window = prices[t - h_d: t + 1]               # (h_d + 1, N)
            valid = ~np.isnan(p_window).any(axis=0)
            highs = np.where(valid, p_window.max(axis=0), np.nan)
            today = prices[t]
            with np.errstate(divide='ignore', invalid='ignore'):
                dd = np.log(today / highs)                  # <= 0
            out[t, :, col] = np.where(np.isfinite(dd), dd, np.nan)
        col += 1

        # Skew / kurt of trailing returns.
        h_m = config.moment_horizon
        win = rets[t - h_m: t]
        valid = ~np.isnan(win).any(axis=0)
        if int(valid.sum()) > 0:
            sub = win[:, valid]
            mu = sub.mean(axis=0)
            sd = sub.std(axis=0, ddof=1)
            sd_safe = np.where(sd > 0, sd, 1.0)
            z = (sub - mu) / sd_safe
            skew = np.mean(z ** 3, axis=0)
            kurt = np.mean(z ** 4, axis=0) - 3.0
            # Names with zero std get NaN moments (degenerate); names with
            # any NaN inside the window also get NaN.
            skew = np.where(sd > 0, skew, np.nan)
            kurt = np.where(sd > 0, kurt, np.nan)
            out[t, valid, col] = skew
            out[t, valid, col + 1] = kurt
        col += 2

    valid = np.all(np.isfinite(out), axis=2)
    return out, valid


__all__ = ['TickerFeatureConfig', 'build_ticker_features']
