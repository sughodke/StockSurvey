"""Compute the four vol-v3 predictor features from Alpaca options chains.

The predictor needs four features per (date, symbol):
  - iv_over_hv:   iv_current / hv_current
  - iv_z:         cross-sectional z-score of iv_current
  - iv_change_4w: iv_current[t] - iv_current[t - 4 weeks]
  - hv_change_4w: hv_current[t] - hv_current[t - 4 weeks]

`iv_current` is ATM 30-day implied vol. `hv_current` is trailing
realized vol of underlying log-returns (we compute it from Alpaca
underlying bars, matching the v0 convention).

Alpaca's options API gives us per-contract quotes; we synthesize
ATM 30d IV by interpolating across strikes/tenors of the chain.

Caveats:
  - Alpaca options data is delayed 15 minutes on the basic plan;
    fine for our 20-trading-day rebal cadence but not for an
    intraday strategy.
  - This module does the IV *interpolation* — actual chain-quote
    fetching is in `vol.live` because it requires an authenticated
    client and we want this module pure for testing.

This is the *connector* between the live IV feed and the frozen
predictor. Re-fitting the v3 predictor would require re-running
`apps/vol/scripts/run_walkforward_v3_dolthub_oos.py` on a wider
substrate; this file does NOT do that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Black-Scholes for IV inversion. Self-contained — vol package is
# numpy/pandas only per workspace convention.

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _naive(ts) -> pd.Timestamp:
    """Coerce any timestamp-like to a tz-naive `pd.Timestamp`. Alpaca
    contracts come back tz-naive but `pd.Timestamp.utcnow()` is
    tz-aware; mixing them raises in arithmetic."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert('UTC').tz_localize(None)
    return t


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. T in years, sigma annualized."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price (put-call parity)."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    c = bs_call_price(S, K, T, r, sigma)
    return c - S + K * math.exp(-r * T)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS vega = ∂price/∂sigma, returned per 1.00 change in sigma (NOT per 1%).
    Divide by 100 if you want vega-per-volpoint."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * _norm_pdf(d1) * math.sqrt(T)


def bs_delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return bs_delta_call(S, K, T, r, sigma) - 1.0


def implied_vol_call(
    price: float, S: float, K: float, T: float, r: float = 0.04,
    tol: float = 1e-4, max_iter: int = 100,
) -> float:
    """Invert BS call price for IV. Bisection — slower than
    Newton but robust to numeric edge cases at deep ITM/OTM."""
    if price <= 0 or T <= 0:
        return float('nan')
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price < intrinsic - 1e-6:
        return float('nan')   # arbitrage / stale quote

    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = bs_call_price(S, K, T, r, mid)
        if abs(p - price) < tol:
            return mid
        if p < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def implied_vol_put(
    price: float, S: float, K: float, T: float, r: float = 0.04,
    tol: float = 1e-4, max_iter: int = 100,
) -> float:
    """Invert BS put price for IV. Same bisection as the call path."""
    if price <= 0 or T <= 0:
        return float('nan')
    intrinsic = max(K * math.exp(-r * T) - S, 0.0)
    if price < intrinsic - 1e-6:
        return float('nan')
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = bs_put_price(S, K, T, r, mid)
        if abs(p - price) < tol:
            return mid
        if p < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class ChainQuote:
    """One row of an option chain snapshot."""
    expiration: pd.Timestamp
    strike: float
    option_type: str            # 'call' or 'put'
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    open_interest: int

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask) if self.bid > 0 and self.ask > 0 else float('nan')

    @property
    def spread_pct(self) -> float:
        m = self.mid
        if not math.isfinite(m) or m <= 0:
            return float('nan')
        return (self.ask - self.bid) / m


def atm_iv_from_chain(
    chain: list[ChainQuote], underlying_price: float,
    target_tenor_days: int = 30, tenor_tolerance_days: int = 7,
    r: float = 0.04, today: pd.Timestamp | None = None,
) -> float:
    """Synthesize a single ATM 30-day IV from a chain snapshot.

    Picks the expiration nearest to `target_tenor_days` (within
    tolerance), then picks the call+put strike pair closest to ATM,
    inverts BS on each side's mid, and returns the average. Returns
    NaN if no acceptable expiration / strike exists.
    """
    today = _naive(today) if today is not None else pd.Timestamp.utcnow().normalize().tz_localize(None)
    # Filter to acceptable tenors
    candidates = [
        q for q in chain
        if abs((_naive(q.expiration) - today).days - target_tenor_days)
            <= tenor_tolerance_days
        and math.isfinite(q.mid) and q.mid > 0
    ]
    if not candidates:
        return float('nan')

    # Pick the single expiration nearest the target
    exps = {q.expiration for q in candidates}
    best_exp = min(exps, key=lambda e: abs((e - today).days - target_tenor_days))
    candidates = [q for q in candidates if q.expiration == best_exp]
    T = max((best_exp - today).days, 1) / 365.0

    # Pick the strike nearest to underlying
    calls = [q for q in candidates if q.option_type == 'call']
    puts  = [q for q in candidates if q.option_type == 'put']
    if not calls or not puts:
        return float('nan')

    nearest_call = min(calls, key=lambda q: abs(q.strike - underlying_price))
    nearest_put  = min(puts,  key=lambda q: abs(q.strike - underlying_price))

    iv_c = implied_vol_call(nearest_call.mid, underlying_price, nearest_call.strike, T, r)
    iv_p = implied_vol_put(nearest_put.mid,   underlying_price, nearest_put.strike,  T, r)

    vals = [v for v in (iv_c, iv_p) if math.isfinite(v) and 0 < v < 5]
    if not vals:
        return float('nan')
    return float(np.mean(vals))


def realized_vol_from_bars(bars: pd.Series, window: int = 20) -> float:
    """Trailing 20-day realized log-return std, annualized (sqrt(252)).
    Returns the last value. Matches gauss314's `hv_current` convention.
    """
    if bars.size < window + 1:
        return float('nan')
    log_r = np.log(bars / bars.shift(1)).dropna()
    if log_r.size < window:
        return float('nan')
    return float(log_r.iloc[-window:].std(ddof=0) * math.sqrt(252.0))


def build_feature_row(
    iv_current: pd.Series, hv_current: pd.Series,
    iv_history: pd.DataFrame, hv_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build the four-feature predictor input cross-section.

    Parameters
    ----------
    iv_current : pd.Series  (index=symbol)   — today's ATM 30d IV
    hv_current : pd.Series  (index=symbol)   — today's trailing 20d realized vol
    iv_history : pd.DataFrame (index=date, columns=symbol)
                            — last ~6 weeks of daily-or-weekly IV; needed
                              for the 4-week change feature
    hv_history : pd.DataFrame                — same shape, for hv change

    Returns
    -------
    pd.DataFrame indexed by symbol with columns:
      iv_over_hv, iv_z, iv_change_4w, hv_change_4w.

    Rows with any NaN feature dropped at the call site, not here, so
    the caller sees what's missing.
    """
    iv_over_hv = (iv_current / hv_current.clip(lower=1e-6)).clip(-10, 10)
    iv_z = (iv_current - iv_current.mean()) / iv_current.std(ddof=0).clip(min=1e-6)

    # 4-week change = today - value 4 weeks ago.
    iv_4w_ago = (iv_history.iloc[-min(20, len(iv_history))]
                 if len(iv_history) > 0 else pd.Series(dtype=float))
    hv_4w_ago = (hv_history.iloc[-min(20, len(hv_history))]
                 if len(hv_history) > 0 else pd.Series(dtype=float))
    iv_change_4w = iv_current - iv_4w_ago.reindex(iv_current.index)
    hv_change_4w = hv_current - hv_4w_ago.reindex(hv_current.index)

    return pd.DataFrame({
        'iv_over_hv': iv_over_hv,
        'iv_z': iv_z,
        'iv_change_4w': iv_change_4w,
        'hv_change_4w': hv_change_4w,
    })


__all__ = [
    'bs_call_price', 'bs_put_price', 'bs_vega', 'bs_delta_call', 'bs_delta_put',
    'implied_vol_call', 'implied_vol_put',
    'ChainQuote', 'atm_iv_from_chain', 'realized_vol_from_bars',
    'build_feature_row',
]
