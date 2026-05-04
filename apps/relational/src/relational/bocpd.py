"""N2 — Bayesian online changepoint detection over per-ticker
scalogram-energy stream.

Per-(ticker, date), summarize the high-dim CWT bundle to a 1-D mean-
power-across-scales scalar, then run Adams-MacKay (2007) BOCPD with
a Normal-Inverse-Gamma conjugate prior on the Gaussian observation
model. Score = posterior `P(run_length = 0 | data_{1..t})` at time t,
i.e. probability that t is a changepoint.

Why 1-D and not multivariate over the full fingerprint:

  * Full multivariate BOCPD requires Normal-Inverse-Wishart updates with
    `d × d` covariance terms (`d ≈ 168` for an `8 × 21` fingerprint),
    pathologically rank-deficient at the few-sample run-lengths BOCPD
    operates on.
  * Mean power across scales preserves the dominant "how much wavelet
    energy is in this scalogram right now" signal, which is what a
    regime change would shift first.

Score interpretation matches the rest of `relational/`: high score =
pick this ticker (in this case, "regime just changed → IV book hasn't
caught up"). The diagnostic dispatcher selects descending top-N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from relational.scalogram_cache import load_or_compute_cwt


def _bocpd_gaussian(
    x: np.ndarray,
    *,
    hazard: float,
    mu0: float = 0.0,
    kappa0: float = 0.01,
    alpha0: float = 0.01,
    beta0: float = 0.01,
) -> np.ndarray:
    """Adams-MacKay BOCPD with Normal-Inverse-Gamma prior on Gaussian
    observations. Returns an `(n,)` array of `P(run_length=0 | data_{1..t})`.

    Vague conjugate prior `(mu0, kappa0, alpha0, beta0)` lets the chain
    learn local variance scale within ~50 observations. `hazard` is the
    constant `1/E[run_length]` — default 1/60 in the public wrapper
    means "expect a regime change every 60 trading days", which is
    fairly liberal so the posterior responds to genuine shifts without
    over-firing on noise.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    R = np.array([1.0])
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    P_cp = np.full(n, np.nan, dtype=np.float64)
    for t in range(n):
        xt = x[t]
        if not np.isfinite(xt):
            P_cp[t] = R[0] if R.size else np.nan
            continue

        # Predictive likelihood under each run-length: Student-t.
        var_pred = beta * (kappa + 1.0) / (alpha * kappa)
        df = 2.0 * alpha
        scale = np.sqrt(np.maximum(var_pred, 1e-30))
        z = (xt - mu) / scale
        log_pred = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - (df + 1.0) / 2.0 * np.log1p(z * z / df)
        )
        # Subtract max for numerical stability before exp.
        pred = np.exp(log_pred - log_pred.max())

        # Update posterior on run-length.
        # CP branch (r_t=0): probability ∝ h × π(x_t | prior). Only the
        # prior-predictive matters; with constant hazard, summing over
        # r_{t-1} factors out of the cp branch entirely.
        # Growth branch (r_t=r+1): ∝ (1-h) × R[r] × π(x_t | r-obs posterior).
        cp = float(hazard * pred[0])
        growth = R * pred * (1.0 - hazard)
        R_new = np.concatenate(([cp], growth))
        s = R_new.sum()
        R_new = R_new / s if s > 0 else np.concatenate(([1.0], np.zeros_like(growth)))

        # Update sufficient stats (one fresh-prior slot at index 0).
        new_kappa = kappa + 1.0
        new_alpha = alpha + 0.5
        new_beta = beta + 0.5 * kappa * (xt - mu) ** 2 / new_kappa
        new_mu = (kappa * mu + xt) / new_kappa
        mu = np.concatenate(([mu0], new_mu))
        kappa = np.concatenate(([kappa0], new_kappa))
        alpha = np.concatenate(([alpha0], new_alpha))
        beta = np.concatenate(([beta0], new_beta))
        R = R_new

        P_cp[t] = R[0]

    return P_cp


def changepoint_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    hazard: float = 1.0 / 60.0,
    cache_dir=None,
) -> np.ndarray:
    """Per-(date, ticker) BOCPD changepoint posterior on mean-power.

    Reduces the cached coeffs to `mean_power[t, i] = mean_s coeffs[s, t, i]^2`
    and runs `_bocpd_gaussian` per ticker. Returns `(n_eval, n_tickers)`
    matching the other scorers' shape contract.
    """
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    summary = (coeffs ** 2).mean(axis=0).astype(np.float64)  # (n_dates, n_tickers)
    n_dates, n_tickers = summary.shape

    out = np.full(summary.shape, np.nan, dtype=np.float32)
    for i in range(n_tickers):
        col = summary[:, i]
        if np.isfinite(col).sum() < 10:
            continue
        out[:, i] = _bocpd_gaussian(col, hazard=hazard).astype(np.float32)
    return out[lookback:].copy()
