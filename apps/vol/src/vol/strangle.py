"""Build a delta-neutral short strangle from an option chain.

For each top-K pick, the deployable trade is "sell vol": sell an OTM
call and OTM put at matched |delta|, sized to a vega budget. This
module is pure — it takes a chain snapshot and a config and returns a
two-leg `Strangle` description. Order submission lives in `vol.live`.

The construction:
  1. Filter to expirations near the target tenor (±tolerance).
  2. Pick the single expiration closest to target_tenor_days.
  3. Within that expiration, find the OTM call with |Δ| ≈ target_delta_call
     and the OTM put with |Δ| ≈ target_delta_put (calls Δ > 0, puts Δ < 0).
  4. Apply liquidity gates (min_open_interest, max_spread, etc.).
  5. Size the # of contracts to deliver the requested vega budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from vol.iv_compute import (
    ChainQuote, bs_delta_call, bs_delta_put, bs_vega, implied_vol_call,
    implied_vol_put,
)
from vol.persist import StranglesConfig


@dataclass(frozen=True)
class StrangleLeg:
    """One leg of a strangle (option contract + qty + size signal).

    `qty` is the number of contracts to SELL (positive integer).
    `side` is `'sell_to_open'` for the entry — the live module
    translates that into Alpaca's `OptionLegRequest` shape.
    """
    contract_symbol: str
    option_type: str             # 'call' or 'put'
    strike: float
    expiration: pd.Timestamp
    qty: int                     # # of contracts to sell
    mid_price_at_construction: float
    iv_at_construction: float
    delta_at_construction: float
    vega_at_construction: float
    side: str = 'sell_to_open'   # convention for entry; flip on close


@dataclass(frozen=True)
class Strangle:
    """A short strangle on one underlier."""
    underlier: str
    underlier_price: float
    expiration: pd.Timestamp
    call: StrangleLeg
    put:  StrangleLeg
    net_vega: float              # total vega sold (both legs, per 1.00 sigma)
    net_delta: float             # initial net delta — should be small (≈ 0)
    construction_reason: str = ''


def build_short_strangle(
    underlier: str, underlier_price: float, chain: list[ChainQuote],
    cfg: StranglesConfig, *, r: float = 0.04,
    today: pd.Timestamp | None = None, contract_lookup: callable = None,
) -> Strangle | None:
    """Pick the strangle legs and size to vega budget.

    Returns None if the chain doesn't have enough liquidity or no
    expiration is in tolerance.

    `contract_lookup` is an optional callable that maps
    (underlier, expiration, strike, option_type) -> contract symbol
    (Alpaca OCC-formatted). If None, we synthesize the OCC symbol
    ourselves following the standard convention.
    """
    if today is None:
        today = pd.Timestamp.utcnow().normalize()

    # Step 1: tenor filter
    eligible = [
        q for q in chain
        if abs((q.expiration - today).days - cfg.target_tenor_days)
        <= cfg.tenor_tolerance_days
        and math.isfinite(q.mid) and q.mid > 0
        and q.open_interest >= cfg.min_open_interest
        and q.bid_size >= cfg.min_bid_size
        and math.isfinite(q.spread_pct) and q.spread_pct <= cfg.max_bid_ask_spread_pct
    ]
    if not eligible:
        return None

    # Step 2: pick single expiration closest to target
    exps = sorted({q.expiration for q in eligible})
    best_exp = min(exps, key=lambda e: abs((e - today).days - cfg.target_tenor_days))
    eligible = [q for q in eligible if q.expiration == best_exp]
    T = max((best_exp - today).days, 1) / 365.0

    calls = [q for q in eligible if q.option_type == 'call' and q.strike > underlier_price]
    puts  = [q for q in eligible if q.option_type == 'put'  and q.strike < underlier_price]
    if not calls or not puts:
        return None

    # Step 3: pick the strike with |Δ| closest to the target
    def _call_match(q: ChainQuote) -> tuple[float, ChainQuote, float, float, float]:
        iv = implied_vol_call(q.mid, underlier_price, q.strike, T, r)
        if not math.isfinite(iv):
            return (float('inf'), q, float('nan'), float('nan'), float('nan'))
        delta = bs_delta_call(underlier_price, q.strike, T, r, iv)
        vega = bs_vega(underlier_price, q.strike, T, r, iv) / 100.0
        return (abs(delta - cfg.target_delta_call), q, iv, delta, vega)

    def _put_match(q: ChainQuote) -> tuple[float, ChainQuote, float, float, float]:
        iv = implied_vol_put(q.mid, underlier_price, q.strike, T, r)
        if not math.isfinite(iv):
            return (float('inf'), q, float('nan'), float('nan'), float('nan'))
        delta = bs_delta_put(underlier_price, q.strike, T, r, iv)
        vega = bs_vega(underlier_price, q.strike, T, r, iv) / 100.0
        return (abs(abs(delta) - cfg.target_delta_put), q, iv, delta, vega)

    call_score, call_q, call_iv, call_delta, call_vega = min(
        (_call_match(q) for q in calls), key=lambda x: x[0])
    put_score, put_q, put_iv, put_delta, put_vega = min(
        (_put_match(q) for q in puts), key=lambda x: x[0])

    if not math.isfinite(call_iv) or not math.isfinite(put_iv):
        return None

    # Step 4: size to vega budget. Each strangle pair has vega
    # = (call_vega + put_vega) per contract per leg-pair.
    # Multiplier 100 because 1 option contract = 100 shares underlier.
    vega_per_strangle_dollar = (call_vega + put_vega) * 100.0
    if vega_per_strangle_dollar <= 0:
        return None
    qty = max(1, int(round(cfg.vega_budget_per_name_usd / vega_per_strangle_dollar)))

    occ_call = (contract_lookup(underlier, best_exp, call_q.strike, 'call')
                if contract_lookup else _occ_symbol(underlier, best_exp, call_q.strike, 'C'))
    occ_put = (contract_lookup(underlier, best_exp, put_q.strike, 'put')
               if contract_lookup else _occ_symbol(underlier, best_exp, put_q.strike, 'P'))

    call_leg = StrangleLeg(
        contract_symbol=occ_call, option_type='call', strike=call_q.strike,
        expiration=best_exp, qty=qty, mid_price_at_construction=call_q.mid,
        iv_at_construction=call_iv, delta_at_construction=call_delta,
        vega_at_construction=call_vega,
    )
    put_leg = StrangleLeg(
        contract_symbol=occ_put, option_type='put', strike=put_q.strike,
        expiration=best_exp, qty=qty, mid_price_at_construction=put_q.mid,
        iv_at_construction=put_iv, delta_at_construction=put_delta,
        vega_at_construction=put_vega,
    )

    # NET position is SHORT both legs, so net_vega is NEGATIVE for us
    # (we're short vega — profit if IV falls). Report magnitude.
    net_vega = -(call_vega + put_vega) * qty * 100.0
    # Net delta when short: −(call_Δ) − (put_Δ). Calls Δ > 0, puts Δ < 0,
    # so naturally ≈ 0 if the strikes were chosen at matched |Δ|.
    net_delta = -(call_delta + put_delta) * qty * 100.0

    return Strangle(
        underlier=underlier, underlier_price=underlier_price,
        expiration=best_exp, call=call_leg, put=put_leg,
        net_vega=net_vega, net_delta=net_delta,
        construction_reason=(
            f'tenor={(best_exp - today).days}d ; '
            f'call K={call_q.strike:.2f} Δ={call_delta:+.2f} ; '
            f'put K={put_q.strike:.2f} Δ={put_delta:+.2f} ; '
            f'qty={qty} ; vega_budget=${cfg.vega_budget_per_name_usd:.0f}'),
    )


def _occ_symbol(underlier: str, expiration: pd.Timestamp,
                strike: float, c_or_p: str) -> str:
    """Synthesize an OCC-21 option symbol.
    `underlier`: e.g. 'AAPL'. `c_or_p`: 'C' or 'P'.
    Strike encoded as 8 digits (5 dollar + 3 milli).
    """
    yymmdd = expiration.strftime('%y%m%d')
    strike_int = int(round(strike * 1000))
    return f'{underlier}{yymmdd}{c_or_p}{strike_int:08d}'


__all__ = ['StrangleLeg', 'Strangle', 'build_short_strangle']
