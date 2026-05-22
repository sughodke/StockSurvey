"""Cross-arc-comparable OOS statistics: PSR / Deflated Sharpe Ratio.

The leaderboard ranks heterogeneous strategies — long-short equity
factor, long-only baskets, short-vol options, drawdown gates — across
different universes, eras, and frictions. A raw annualized Sharpe is not
a fair ranking key across those: it ignores higher moments (fat tails
inflate a naive Sharpe), sample length (a 5-window arc is noisier than a
6-window one), and — most importantly — *how many configurations were
tried* before the winner was reported. The leaderboard's own history
(few confirmed-OOS out of many rows) is a multiple-testing problem.

This module maps any OOS per-period **net return stream** to a single
unit-free, cross-arc-comparable number: the Deflated Sharpe Ratio (DSR)
and its underlying z-statistic (the "deflated t-stat"). DSR is the
probability that the observed Sharpe exceeds the *expected maximum*
Sharpe under a null of `n_trials` zero-skill strategies, adjusting for
the return stream's skewness and kurtosis.

References
----------
Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier"
(Probabilistic Sharpe Ratio) and (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality".

Self-contained on purpose: `ss_portfolio` is numpy-only by workspace
convention, so the normal CDF / inverse-CDF are implemented here rather
than pulling in scipy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ss_portfolio.metrics import max_drawdown

TRADING_DAYS: int = 252
_EULER_MASCHERONI: float = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Max relative error ~1.15e-9 over the open interval (0, 1). Good
    enough for the DSR's expected-maximum term, which only ever feeds
    Phi^{-1}(1 - 1/T) for T = n_trials.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f'_norm_ppf requires 0 < p < 1, got {p}')
    a = (-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0)
    b = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1)
    c = (-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0)
    d = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected maximum per-period Sharpe under `n_trials` zero-skill trials.

    SR* = sharpe_std * [(1 - g)*Z(1 - 1/T) + g*Z(1 - 1/(T*e))]

    where g is the Euler-Mascheroni constant, Z is the inverse normal
    CDF, and `sharpe_std` is the cross-trial dispersion of the
    *per-period* (non-annualized) Sharpe estimates. This is the
    benchmark SR the observed Sharpe must beat to be deemed skillful
    rather than the best of many coin flips.
    """
    if n_trials < 1:
        raise ValueError(f'n_trials must be >= 1, got {n_trials}')
    if n_trials == 1:
        return 0.0
    t = float(n_trials)
    z1 = _norm_ppf(1.0 - 1.0 / t)
    z2 = _norm_ppf(1.0 - 1.0 / (t * math.e))
    return sharpe_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)


def probabilistic_sharpe(
    sharpe_per_period: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sharpe_benchmark: float = 0.0,
) -> tuple[float, float]:
    """Probabilistic Sharpe Ratio and its z-statistic.

    All Sharpe inputs are **per-period** (non-annualized). `kurtosis` is
    the non-excess (Pearson) kurtosis — 3.0 for a normal. Returns
    `(psr, z)` where psr = Phi(z) and z is the deflated/probabilistic
    t-stat used as the cross-arc ranking key.
    """
    if n_obs < 2:
        return float('nan'), float('nan')
    denom = 1.0 - skew * sharpe_per_period + 0.25 * (kurtosis - 1.0) * sharpe_per_period ** 2
    if denom <= 0:
        return float('nan'), float('nan')
    z = (sharpe_per_period - sharpe_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _norm_cdf(z), z


@dataclass(frozen=True)
class MetricBlock:
    """Cross-arc-comparable OOS metric block for one return stream.

    `deflated_tstat` (the z behind `dsr`) is the leaderboard's primary
    ranking key: unit-free, moment- and length-aware, and penalized by
    `n_trials`. `dsr` is the probability that the Sharpe is skill, not
    the max of `n_trials` coin flips.
    """

    n_obs: int
    periods_per_year: float
    ann_sharpe: float
    sharpe_per_period: float
    skew: float
    kurtosis: float
    max_dd: float
    n_trials: int
    sharpe_std: float
    expected_max_sharpe: float
    psr: float            # PSR vs benchmark 0 (length+moments, no deflation)
    psr_tstat: float
    dsr: float            # deflated: PSR vs expected_max_sharpe
    deflated_tstat: float
    ir_vs_bench: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def standardize_oos(
    returns: np.ndarray,
    *,
    periods_per_year: float,
    n_trials: int = 1,
    trial_sharpes: np.ndarray | None = None,
    sharpe_std: float | None = None,
    benchmark: np.ndarray | None = None,
) -> MetricBlock:
    """Map an OOS per-period net return stream to a comparable MetricBlock.

    Parameters
    ----------
    returns
        1-D per-period **net** (post-cost) return stream. Block returns
        (per rebalance) or daily returns — set `periods_per_year`
        accordingly (e.g. ``TRADING_DAYS / rebal_days`` for block
        returns, ``TRADING_DAYS`` for daily).
    periods_per_year
        Annualization factor; also fixes the per-period <-> annual Sharpe
        conversion.
    n_trials
        Number of configurations tried in the arc that produced this
        row, for the deflation term. ``1`` => no deflation (DSR == PSR
        vs 0).
    trial_sharpes
        Optional per-period Sharpe estimates of all `n_trials` configs;
        their std sets the expected-max benchmark. If given, overrides
        `n_trials` length and `sharpe_std`.
    sharpe_std
        Optional explicit cross-trial dispersion of per-period Sharpe.
        Falls back to ``1/sqrt(n_obs)`` (the null s.e. of a Sharpe
        estimate) when neither this nor `trial_sharpes` is supplied.
    benchmark
        Optional same-length per-period benchmark return stream; enables
        the information ratio (annualized excess / tracking error).
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = r.size
    sd = r.std(ddof=0)
    sr_pp = float(r.mean() / sd) if sd > 0 else 0.0
    ann_sharpe = sr_pp * math.sqrt(periods_per_year)

    # Higher moments (non-excess kurtosis: normal == 3).
    if n >= 2 and sd > 0:
        z = (r - r.mean()) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean())
    else:
        skew, kurt = 0.0, 3.0

    if trial_sharpes is not None:
        ts = np.asarray(trial_sharpes, dtype=np.float64).ravel()
        n_trials = int(ts.size)
        s_std = float(ts.std(ddof=1)) if ts.size > 1 else 0.0
    elif sharpe_std is not None:
        s_std = float(sharpe_std)
    else:
        # Null s.e. of a Sharpe estimate when trial dispersion is unknown.
        s_std = 1.0 / math.sqrt(n) if n > 0 else 0.0

    sr_star = expected_max_sharpe(n_trials, s_std)
    psr, psr_z = probabilistic_sharpe(sr_pp, n, skew, kurt, sharpe_benchmark=0.0)
    dsr, dsr_z = probabilistic_sharpe(sr_pp, n, skew, kurt, sharpe_benchmark=sr_star)

    ir = None
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=np.float64).ravel()
        if b.size == n:
            excess = r - b
            te = excess.std(ddof=0)
            ir = float(excess.mean() / te * math.sqrt(periods_per_year)) if te > 0 else 0.0

    return MetricBlock(
        n_obs=n,
        periods_per_year=float(periods_per_year),
        ann_sharpe=ann_sharpe,
        sharpe_per_period=sr_pp,
        skew=skew,
        kurtosis=kurt,
        max_dd=max_drawdown(r),
        n_trials=n_trials,
        sharpe_std=s_std,
        expected_max_sharpe=sr_star,
        psr=psr,
        psr_tstat=psr_z,
        dsr=dsr,
        deflated_tstat=dsr_z,
        ir_vs_bench=ir,
    )
