"""Differentiable smooth approximation of the Ledoit-Wolf studentized
Sharpe-difference test.

This module is the numpy reference for the literature-canonical
"studentized Sharpe difference" of Lo (2002) / Mertens (2002) /
Bailey-López de Prado (2014), implemented in closed form so the math
matches the (non-differentiable) bootstrap version in
`ss_portfolio.sharpe_diff` up to the parametric-Gaussian limit.

Why this module exists: `sharpe_difference_ci` is the gold-standard
INFERENCE test (Ledoit-Wolf 2008 studentized stationary-bootstrap
CI), but its index resampling and quantile extraction are not
differentiable wrt the return streams. For TRAINING — strategy
parameters that produce the returns — we need a smooth proxy. The
parametric-Gaussian / Lo-Mertens-delta-method version IS that
proxy. The two converge as `n → ∞` and `block_dep → 0`.

The tinygrad mirror lives in `apps/factor/src/factor/objectives.py`
as `block_studentized_sharpe_diff` for use as a training loss.

API
---
- `studentized_sharpe_diff(a, b, *, with_moments=False)` — point
  estimate of the t-statistic `(SR_a − SR_b) / s.e.(SR_a − SR_b)`.
  When `with_moments=True`, uses the Bailey-LdP corrected denominator
  (Probabilistic Sharpe Ratio formulation) which is more accurate at
  heavy-tailed returns.
- `parametric_ci(a, b, confidence=0.95)` — `(ΔSR, ci_lo, ci_hi)` where
  bounds are the Gaussian-approximation analogue of the bootstrap
  CI. Identical interface to `sharpe_difference_ci` modulo the CI
  computation.
- `soft_excludes_zero(t_stat, *, alpha=0.05, temperature=0.5)` — smooth
  sigmoid indicator that "the CI excludes zero on the positive side."
  Converges to a Heaviside step as `temperature → 0`.
- `p_excludes_zero(t_stat)` — `Φ(t_stat)`, the parametric
  probability that `ΔSR > 0`. Differentiable in `[0, 1]`.

All functions accept 1-D numpy arrays of equal length and return
scalar floats. No bootstrapping; no random index sampling. Same
inputs as `sharpe_difference_ci` for direct comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Cache the 1.96 default outside the hot path so the JIT-able tinygrad
# mirror can use the same constant.
NORM_QUANTILE_95 = 1.959963984540054   # Φ⁻¹(0.975)


@dataclass(frozen=True)
class ParametricSharpeDiffCI:
    """Result of `parametric_ci` — the differentiable analogue of
    `ss_portfolio.SharpeDiffCI` returned by the bootstrap version.

    Carries the same fields the bootstrap version does, minus the
    bootstrap-specific ones (`block_length`, `n_bootstraps`) and plus
    `t_stat` because it's the canonical training target.
    """
    n_obs: int
    sr_a: float
    sr_b: float
    delta_sr: float
    se_delta_sr: float
    t_stat: float
    ci_lo: float
    ci_hi: float
    confidence: float
    includes_zero: bool


def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    """Per-period Sharpe + skew + excess-kurtosis. Pearson kurtosis
    (normal = 3) used here to match `probabilistic_sharpe`."""
    n = r.size
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    if sd <= 0 or n < 2:
        return 0.0, 0.0, 0.0, 3.0
    sr_pp = mu / sd
    z = (r - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    return sr_pp, sd, skew, kurt


def _delta_method_se(a: np.ndarray, b: np.ndarray, *,
                      with_moments: bool = False) -> tuple[float, float, float, float]:
    """Lo-Mertens-LdP delta-method s.e. of the per-period Sharpe diff.

    Returns (sr_a, sr_b, delta, se).

    Plain version (Lo 2002): var(SR_i) ≈ (1 + 0.5·SR_i²) / n; covariance
    via the empirical return correlation.

    With moments (Bailey-LdP 2014): var(SR_i) ≈ (1 − γ_i·SR_i + 0.25·
    (κ_i − 1)·SR_i²) / (n − 1), where γ is skew and κ is non-excess
    kurtosis. Matches the PSR denominator term.
    """
    n = a.size
    sr_a, sd_a, sk_a, k_a = _moments(a)
    sr_b, sd_b, sk_b, k_b = _moments(b)

    if with_moments:
        var_a = (1.0 - sk_a * sr_a + 0.25 * (k_a - 1.0) * sr_a * sr_a) / max(n - 1, 1)
        var_b = (1.0 - sk_b * sr_b + 0.25 * (k_b - 1.0) * sr_b * sr_b) / max(n - 1, 1)
    else:
        var_a = (1.0 + 0.5 * sr_a * sr_a) / n
        var_b = (1.0 + 0.5 * sr_b * sr_b) / n

    if sd_a > 0 and sd_b > 0:
        rho = float(np.corrcoef(a, b)[0, 1])
    else:
        rho = 0.0
    var_diff = var_a + var_b - 2.0 * rho * math.sqrt(max(var_a * var_b, 0.0))
    se = math.sqrt(max(var_diff, 1e-24))
    return sr_a, sr_b, sr_a - sr_b, se


def studentized_sharpe_diff(
    a: np.ndarray, b: np.ndarray, *, with_moments: bool = False,
) -> float:
    """Point estimate of the studentized Sharpe-difference t-stat.

    `t = (SR_a − SR_b) / s.e.(SR_a − SR_b)`. This is the literature-
    canonical loss for training a strategy's parameters to maximize
    Sharpe difference vs a benchmark. Fully differentiable wrt the
    return arrays (under any autograd library that has mean / std /
    sqrt / corrcoef).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f'date-aligned arrays required; got n_a={a.size}, n_b={b.size}')
    if a.size < 5:
        raise ValueError(f'need n >= 5; got {a.size}')
    _, _, delta, se = _delta_method_se(a, b, with_moments=with_moments)
    return delta / se if se > 0 else 0.0


def parametric_ci(
    a: np.ndarray, b: np.ndarray, *,
    confidence: float = 0.95, with_moments: bool = False,
) -> ParametricSharpeDiffCI:
    """Differentiable parametric (Gaussian) CI for the Sharpe diff.

    Identical interface to `sharpe_difference_ci` modulo the CI being
    derived from the delta-method s.e. instead of a bootstrap
    distribution. Converges to the bootstrap CI as `n → ∞` (Lo 2002).

    For `confidence=0.95`, `ci_lo = ΔSR − 1.96·s.e.`,
    `ci_hi = ΔSR + 1.96·s.e.`. Returns both the CI and the
    studentized t-stat (= ΔSR / s.e.).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f'date-aligned arrays required; got n_a={a.size}, n_b={b.size}')
    if a.size < 5:
        raise ValueError(f'need n >= 5; got {a.size}')

    sr_a, sr_b, delta, se = _delta_method_se(a, b, with_moments=with_moments)
    # Inverse normal CDF at (1 - alpha/2). For 95% this is ≈ 1.96.
    if confidence == 0.95:
        z = NORM_QUANTILE_95
    else:
        # Acklam's rational approximation lives in deflated.py; reuse.
        from ss_portfolio.deflated import _norm_ppf
        z = _norm_ppf(1.0 - (1.0 - confidence) / 2.0)
    ci_lo = delta - z * se
    ci_hi = delta + z * se
    t_stat = delta / se if se > 0 else 0.0
    return ParametricSharpeDiffCI(
        n_obs=a.size, sr_a=sr_a, sr_b=sr_b,
        delta_sr=delta, se_delta_sr=se, t_stat=t_stat,
        ci_lo=ci_lo, ci_hi=ci_hi, confidence=confidence,
        includes_zero=(ci_lo <= 0.0 <= ci_hi),
    )


def soft_excludes_zero(
    t_stat: float, *, alpha: float = 0.05, temperature: float = 0.5,
) -> float:
    """Smooth sigmoid indicator of |t_stat| > z_{α/2}.

    Returns a value in (0, 1) that → 1 when the CI cleanly excludes 0,
    → 0 when the CI clearly straddles 0. `temperature` controls the
    sharpness; smaller is closer to a Heaviside step. Default
    `temperature=0.5` gives a soft margin appropriate for gradient
    training without vanishing-gradient pathology.

    Differentiable wrt `t_stat`. The gradient is concentrated around
    `|t_stat| ≈ z_{α/2}`.
    """
    if alpha == 0.05:
        z = NORM_QUANTILE_95
    else:
        from ss_portfolio.deflated import _norm_ppf
        z = _norm_ppf(1.0 - alpha / 2.0)
    arg = (abs(t_stat) - z) / max(temperature, 1e-9)
    return 1.0 / (1.0 + math.exp(-arg))


def p_excludes_zero(t_stat: float) -> float:
    """Parametric Pr[ΔSR > 0] = Φ(t_stat). Differentiable in [0, 1]."""
    return 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))


__all__ = [
    'NORM_QUANTILE_95',
    'ParametricSharpeDiffCI',
    'p_excludes_zero',
    'parametric_ci',
    'soft_excludes_zero',
    'studentized_sharpe_diff',
]
