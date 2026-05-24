"""Ledoit-Wolf (2008) studentized stationary-bootstrap CI for the
Sharpe difference between two return streams.

This is the literature-canonical cross-arc apples-to-apples test
recommended by the methodology agent's brief: "for each pair of arcs
(A, B), compute the Sharpe difference on the date-aligned common
window with frequency collapsed to the lower, then a studentized
stationary-bootstrap CI."

Conventions:
- Inputs are 1-D per-period **net return streams** for two arcs,
  already date-aligned to a common index.
- Stationary block bootstrap (Politis-Romano 1994) with geometric
  block length distribution. Default block length follows
  Politis-White (2004) automatic selection at b = n^(1/3); for
  short streams (n<100) we clamp to a minimum of 2.
- "Studentized" means we resample the t-statistic (ΔSR / s.e.(ΔSR))
  rather than the raw ΔSR, then map the bootstrap quantiles back to
  ΔSR space using the estimated s.e. This is more robust to skewed
  bootstrap distributions per Ledoit-Wolf.

Self-contained (numpy only; no scipy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SharpeDiffCI:
    n_obs: int
    sr_a: float
    sr_b: float
    delta_sr: float
    se_delta_sr: float
    ci_lo: float
    ci_hi: float
    confidence: float
    block_length: int
    n_bootstraps: int
    includes_zero: bool

    def __repr__(self) -> str:
        return (f'SharpeDiffCI(n={self.n_obs}, ΔSR={self.delta_sr:+.3f} '
                f'± {(self.ci_hi-self.ci_lo)/2:.3f} '
                f'[{self.ci_lo:+.3f}, {self.ci_hi:+.3f}] '
                f'{int(100*self.confidence)}%, '
                f'{"includes 0" if self.includes_zero else "excludes 0"})')


def _stationary_block_sample(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence."""
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / max(block_len, 1)
    # Start each block with a random index
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        # Geometric block length
        if block_len > 1:
            geom = int(rng.geometric(p))
        else:
            geom = 1
        end = min(i + geom, n)
        for k in range(end - i):
            idx[i + k] = (start + k) % n
        i = end
    return idx


def _sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=0)
    return float(returns.mean() / sd) if sd > 0 else 0.0


def _sharpe_diff_se(a: np.ndarray, b: np.ndarray) -> float:
    """Per-period s.e. of (SR_a - SR_b) via the Lo-Mertens-LdP delta
    method approximation: var(SR_i) ≈ (1 + 0.5*SR_i^2)/n. Cross term
    handled via the empirical correlation."""
    n = a.size
    sr_a = _sharpe(a)
    sr_b = _sharpe(b)
    var_a = (1.0 + 0.5 * sr_a * sr_a) / n
    var_b = (1.0 + 0.5 * sr_b * sr_b) / n
    # Correlation between the two return streams
    if a.std(ddof=0) > 0 and b.std(ddof=0) > 0:
        rho = float(np.corrcoef(a, b)[0, 1])
    else:
        rho = 0.0
    var_diff = var_a + var_b - 2.0 * rho * math.sqrt(var_a * var_b)
    return math.sqrt(max(var_diff, 1e-12))


def sharpe_difference_ci(
    a: np.ndarray, b: np.ndarray, *,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    block_length: int | None = None,
    seed: int | None = 42,
) -> SharpeDiffCI:
    """Studentized stationary-bootstrap CI for SR_a - SR_b.

    Parameters
    ----------
    a, b : 1-D per-period return arrays, MUST be date-aligned to the
        same length and frequency. Caller is responsible for alignment.
    n_bootstraps : default 2000, gives reasonable tail-quantile
        precision at the 95% level.
    confidence : default 0.95.
    block_length : default `max(2, int(round(n**(1/3))))`. Politis-White
        2004 automatic selection rule.
    seed : for reproducibility.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(
            f'sharpe_difference_ci needs date-aligned series; got n_a='
            f'{a.size}, n_b={b.size}')
    n = a.size
    if n < 5:
        raise ValueError(f'sharpe_difference_ci needs n >= 5; got {n}')

    if block_length is None:
        block_length = max(2, int(round(n ** (1.0 / 3.0))))

    sr_a, sr_b = _sharpe(a), _sharpe(b)
    delta = sr_a - sr_b
    se = _sharpe_diff_se(a, b)

    rng = np.random.default_rng(seed)
    boots_t = np.empty(n_bootstraps, dtype=np.float64)
    for k in range(n_bootstraps):
        idx = _stationary_block_sample(rng, n, block_length)
        a_b = a[idx]
        b_b = b[idx]
        sr_a_b, sr_b_b = _sharpe(a_b), _sharpe(b_b)
        delta_b = sr_a_b - sr_b_b
        se_b = _sharpe_diff_se(a_b, b_b)
        if se_b > 1e-12:
            boots_t[k] = (delta_b - delta) / se_b
        else:
            boots_t[k] = 0.0

    alpha = (1.0 - confidence) / 2.0
    q_lo, q_hi = np.quantile(boots_t, [alpha, 1.0 - alpha])
    # Studentized inversion — Ledoit-Wolf
    ci_lo = delta - q_hi * se
    ci_hi = delta - q_lo * se

    return SharpeDiffCI(
        n_obs=n, sr_a=sr_a, sr_b=sr_b,
        delta_sr=delta, se_delta_sr=se,
        ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        confidence=confidence, block_length=block_length,
        n_bootstraps=n_bootstraps,
        includes_zero=(ci_lo <= 0.0 <= ci_hi),
    )


__all__ = ['SharpeDiffCI', 'sharpe_difference_ci']
