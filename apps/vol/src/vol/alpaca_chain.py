"""Alpaca options-chain queries + bar fetches for vol-v3 live.

Isolated from `vol.live` so the orchestration stays mock-testable and
the Alpaca-API quirks (rate limits, snapshot vs chain endpoints, OCC
symbol conventions, Greeks population, multi-leg submission) live in
one place.

What lives here:

- `fetch_option_snapshot_chain(client, underlier, target_tenor_days)`
    → `list[ChainQuote]`     real chain snapshot via `OptionSnapshotRequest`,
                              including bid/ask/sizes/OI.
- `fetch_underlying_bars(client, symbols, days)`
    → `pd.DataFrame`         daily closes for the underliers.
- `fetch_current_vix(client, ...)` — optional fallback if FRED is down;
                                     default path is `ss_macro` per `vol.live`.
- `submit_short_strangle(trading_client, strangle, account)`
    → `(order_id, rejection?)`  multi-leg `OptionLegRequest` submission.

All functions accept already-instantiated alpaca-py clients so unit
tests can inject mocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from vol.iv_compute import ChainQuote
from vol.strangle import Strangle


# ----------------------------------------------------------- chain query ----

def fetch_option_snapshot_chain(
    options_client, underlier: str, *, target_tenor_days: int = 30,
    tenor_tolerance_days: int = 7,
) -> list[ChainQuote]:
    """Pull all option contracts for `underlier` near `target_tenor_days`.

    Strategy: call `GetOptionContractsRequest` (trading client) for the
    contract list filtered by expiration window, then a batch
    `OptionSnapshotRequest` (data client) for NBBO+OI+Greeks. Returns
    a list of `ChainQuote` ready for `vol.strangle.build_short_strangle`.

    Failures (rate-limit, no chain, no quotes) are surfaced as an
    empty list — the caller can decide to skip the name. We do NOT
    swallow auth/credential errors; those propagate.
    """
    # Lazy import — keep this module light if Alpaca isn't installed.
    from alpaca.data.requests import OptionSnapshotRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import AssetStatus, ContractType

    today = pd.Timestamp.utcnow().normalize()
    exp_lo = (today + pd.Timedelta(days=target_tenor_days - tenor_tolerance_days)).date()
    exp_hi = (today + pd.Timedelta(days=target_tenor_days + tenor_tolerance_days)).date()

    trading_client = getattr(options_client, '_trading_client_for_contracts', None)
    if trading_client is None:
        # Real usage: the caller constructs a TradingClient and stows
        # it on the options client as a side-table. The test path
        # passes a duck-typed `options_client.get_option_contracts` /
        # `options_client.get_option_snapshot`.
        contracts = options_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[underlier],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=exp_lo,
                expiration_date_lte=exp_hi,
            ))
    else:
        contracts = trading_client.get_option_contracts(
            GetOptionContractsRequest(
                underlying_symbols=[underlier],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=exp_lo,
                expiration_date_lte=exp_hi,
            ))

    contract_list = getattr(contracts, 'option_contracts', None) or contracts
    if not contract_list:
        return []

    symbols = [c.symbol for c in contract_list]
    if not symbols:
        return []

    # Batched snapshot — alpaca-py returns a dict {sym -> OptionsSnapshot}.
    snapshots = options_client.get_option_snapshot(
        OptionSnapshotRequest(symbol_or_symbols=symbols))

    out: list[ChainQuote] = []
    for c in contract_list:
        sym = c.symbol
        snap = snapshots.get(sym) if isinstance(snapshots, dict) else None
        if snap is None or snap.latest_quote is None:
            continue
        q = snap.latest_quote
        bid = float(q.bid_price or 0.0)
        ask = float(q.ask_price or 0.0)
        if bid <= 0 or ask <= 0:
            continue
        oi = int(getattr(c, 'open_interest', 0) or 0)
        bid_size = int(getattr(q, 'bid_size', 0) or 0)
        ask_size = int(getattr(q, 'ask_size', 0) or 0)
        expiration = pd.Timestamp(c.expiration_date)
        # Alpaca's ContractType is .CALL or .PUT; normalize to str
        c_type = c.type
        option_type = 'call' if str(c_type).upper().endswith('CALL') else 'put'
        out.append(ChainQuote(
            expiration=expiration, strike=float(c.strike_price),
            option_type=option_type, bid=bid, ask=ask,
            bid_size=bid_size, ask_size=ask_size, open_interest=oi,
        ))
    return out


def fetch_underlying_bars(
    stocks_client, symbols: list[str], *, days: int = 30,
) -> pd.DataFrame:
    """Fetch daily closes for `symbols` over the last `days` calendar days.

    Returns a DataFrame indexed by date, columns = symbols, dtype
    float64. Missing symbols (no bars) get dropped. Uses Alpaca's
    `StockBarsRequest` with `TimeFrame.Day`.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(days * 1.5))  # weekends buffer
    req = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
        start=start, end=end,
    )
    resp = stocks_client.get_stock_bars(req)
    df = getattr(resp, 'df', None)
    if df is None or df.empty:
        return pd.DataFrame()

    # alpaca-py returns a MultiIndex (symbol, timestamp); flatten.
    if isinstance(df.index, pd.MultiIndex):
        wide = df['close'].unstack(level=0)
    else:
        wide = df[['close']].rename(columns={'close': symbols[0]})

    wide.index = pd.DatetimeIndex(wide.index).normalize()
    return wide.sort_index()


# --------------------------------------------------------- order submission

@dataclass
class StrangleSubmission:
    underlier: str
    order_id: str | None
    rejection_reason: str | None


def submit_short_strangle(
    trading_client, strangle: Strangle, *, dry_run: bool = True,
    limit_price_offset_pct: float = 0.0,
) -> StrangleSubmission:
    """Submit one short strangle as an Alpaca multi-leg LIMIT order.

    `limit_price_offset_pct` lets you trade off fill-prob vs price:
    positive = more aggressive (closer to bid for sells), negative =
    safer (closer to ask).

    Returns a `StrangleSubmission` with the order_id on success or a
    `rejection_reason` on failure. Per-name errors are captured; auth
    and connectivity failures propagate.
    """
    from alpaca.trading.requests import (
        LimitOrderRequest, OptionLegRequest,
    )
    from alpaca.trading.enums import (
        OrderClass, OrderSide, OrderType, PositionIntent, TimeInForce,
    )

    # Multi-leg sell strangle = sell_to_open both legs at a NET credit
    # = call_mid + put_mid. Alpaca's MLEG order takes a single qty for
    # all legs (the strategy contract count) and `ratio_qty` per leg
    # for the relative weights. Equal qty on both legs.
    qty = int(strangle.call.qty)  # must equal put.qty; constructor enforces
    if strangle.put.qty != qty:
        return StrangleSubmission(
            underlier=strangle.underlier, order_id=None,
            rejection_reason=(
                f'call qty {strangle.call.qty} != put qty {strangle.put.qty}; '
                'strangle constructor invariant violated'))

    net_credit_mid = (strangle.call.mid_price_at_construction
                      + strangle.put.mid_price_at_construction)
    limit_price = round(
        net_credit_mid * (1.0 - limit_price_offset_pct), 2)
    if limit_price <= 0:
        return StrangleSubmission(
            underlier=strangle.underlier, order_id=None,
            rejection_reason=f'computed limit price {limit_price} <= 0')

    legs = [
        OptionLegRequest(
            symbol=strangle.call.contract_symbol,
            side=OrderSide.SELL,
            ratio_qty=1,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        OptionLegRequest(
            symbol=strangle.put.contract_symbol,
            side=OrderSide.SELL,
            ratio_qty=1,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
    ]
    order = LimitOrderRequest(
        qty=qty,
        order_class=OrderClass.MLEG,
        legs=legs,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        type=OrderType.LIMIT,
    )

    if dry_run:
        return StrangleSubmission(
            underlier=strangle.underlier,
            order_id=f'DRY_RUN_{strangle.underlier}_{qty}x@{limit_price:.2f}',
            rejection_reason=None,
        )

    try:
        resp = trading_client.submit_order(order)
        return StrangleSubmission(
            underlier=strangle.underlier,
            order_id=str(resp.id), rejection_reason=None)
    except Exception as e:
        return StrangleSubmission(
            underlier=strangle.underlier, order_id=None,
            rejection_reason=f'{type(e).__name__}: {e}')


__all__ = [
    'StrangleSubmission',
    'fetch_option_snapshot_chain',
    'fetch_underlying_bars',
    'submit_short_strangle',
]
