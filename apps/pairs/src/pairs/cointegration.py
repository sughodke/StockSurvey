"""Engle-Granger cointegration test.

Two-step procedure:
  1. OLS regress `y_t = β · x_t + α + ε_t` to estimate hedge ratio.
  2. Augmented Dickey-Fuller test on residuals; reject the unit-root
     null at p < threshold ⇒ residuals are stationary ⇒ pair is
     cointegrated under the EG framework.

`statsmodels.tsa.stattools.coint` does both steps internally and
returns an EG-corrected p-value (the standard ADF p-value
distribution doesn't apply because the residual is constructed
from a regression — coint uses MacKinnon's tables specific to the
two-step procedure). We use it directly rather than rolling our
own ADF.

The hedge ratio `β` from step 1 is what the spread definition uses:
`s_t = log(P_A) − β · log(P_B)` (the residual of the regression).
We return it alongside the p-value so the caller doesn't have to
re-fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class EngleGrangerResult:
    """Output of `engle_granger_test`."""
    p_value:    float    # EG-corrected ADF p-value (low ⇒ cointegrated)
    test_stat:  float    # EG ADF test statistic (more negative ⇒ stronger reject)
    hedge_beta: float    # OLS slope of `log(P_A) ~ β · log(P_B) + c`
    intercept:  float    # OLS intercept (mean spread under cointegration)
    n_obs:      int


def engle_granger_test(
    log_p_a: np.ndarray, log_p_b: np.ndarray,
) -> EngleGrangerResult:
    """Engle-Granger two-step on log-prices.

    `log_p_a` and `log_p_b` are 1-D arrays of equal length covering
    the *training* window only (no peeking). Returns the EG p-value
    and the hedge ratio for use in spread construction.

    Suppresses statsmodels' constant-column / interpolation warnings
    that fire on noisy short-history pairs — they're informational,
    not errors, and we already gate on the p-value downstream.
    """
    if log_p_a.shape != log_p_b.shape:
        raise ValueError(
            f'shape mismatch: A={log_p_a.shape} B={log_p_b.shape}')
    n = len(log_p_a)
    if n < 50:
        # ADF needs reasonable sample size to be meaningful; pairs
        # with very short history get a sentinel "not cointegrated"
        # without running the test.
        return EngleGrangerResult(
            p_value=1.0, test_stat=0.0,
            hedge_beta=float('nan'), intercept=float('nan'), n_obs=n)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            # `maxlag=1` skips the BIC lag-selection inner loop that
            # dominates `coint()` runtime (~500ms/call → ~30ms/call on
            # 1260 bars). Standard Dickey-Fuller (no augmenting lags)
            # is fine for daily-bar pairs over multi-year windows;
            # the lag selection only matters for higher-frequency
            # data with autocorrelation in the residuals.
            test_stat, p_value, _crit = coint(log_p_a, log_p_b, maxlag=1)
        except (ValueError, np.linalg.LinAlgError):
            return EngleGrangerResult(
                p_value=1.0, test_stat=0.0,
                hedge_beta=float('nan'), intercept=float('nan'), n_obs=n)

        # Hedge ratio from OLS of A on B with constant.
        x = sm.add_constant(log_p_b)
        try:
            ols = sm.OLS(log_p_a, x).fit()
            intercept = float(ols.params[0])
            beta = float(ols.params[1])
        except (ValueError, np.linalg.LinAlgError):
            return EngleGrangerResult(
                p_value=1.0, test_stat=float(test_stat),
                hedge_beta=float('nan'), intercept=float('nan'), n_obs=n)

    return EngleGrangerResult(
        p_value=float(p_value), test_stat=float(test_stat),
        hedge_beta=beta, intercept=intercept, n_obs=n)


__all__ = ['EngleGrangerResult', 'engle_granger_test']
