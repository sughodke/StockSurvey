"""Alpaca broker adapter — shared infrastructure for live trading apps.

Wraps `alpaca-py` to expose the four operations a live-trade orchestration
loop needs: account snapshot, current positions, recent OHLC bars for a
universe, and submitting orders to reach a target dollar allocation.

Originally lived in `regime/broker.py`; promoted here so any app whose
strategy emits a target-weights series (`regime`, `relational`, …) can
share a single broker surface and the same operational risk rails.

Credentials are read from the environment:
    ALPACA_API_KEY      — required
    ALPACA_SECRET_KEY   — required
    ALPACA_BASE_URL     — optional; defaults to paper-trading endpoint

The default base URL is Alpaca's paper-trading endpoint, so a misconfigured
key will fail loudly rather than touching real capital.

`alpaca-py` is gated as the `ss-portfolio[alpaca]` optional extra so
non-trading workflows (training, backtesting) don't pull it in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


PAPER_BASE_URL: str = 'https://paper-api.alpaca.markets'


@dataclass
class Account:
    """Subset of broker account state needed for position sizing."""
    equity: float
    cash: float
    buying_power: float
    paper: bool


@dataclass
class Trade:
    """A single intended order in dollar + share terms."""
    symbol: str
    side: str
    qty: float
    notional: float
    current_weight: float
    target_weight: float
    last_price: float


@dataclass
class OrderRejection:
    """A submit_order failure that was logged-and-skipped rather than
    aborting the batch. Surfaced to the live-run caller so partial-
    submit days don't go unnoticed (one reject is normal — fractionable
    rejection on a non-fractionable; many rejects in a row may indicate
    auth or API issues)."""
    symbol: str
    qty: float
    reason: str


class AlpacaBroker:
    """Thin wrapper over alpaca-py's trading + data clients."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get('ALPACA_API_KEY')
        secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY')
        base_url = base_url or os.environ.get('ALPACA_BASE_URL', PAPER_BASE_URL)
        if not api_key or not secret_key:
            raise RuntimeError(
                'Alpaca credentials missing: set ALPACA_API_KEY and '
                'ALPACA_SECRET_KEY in the environment.')
        # 'paper' must appear in the base URL to be treated as paper trading.
        # String-equality on the URL is brittle (trailing slash, casing) and
        # silent failure here puts a TradingClient into REAL-MONEY mode.
        self._paper = 'paper' in base_url.lower()
        self._trading = TradingClient(api_key, secret_key, paper=self._paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)

    def get_account(self) -> Account:
        a = self._trading.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            paper=self._paper,
        )

    def get_positions(self) -> dict[str, float]:
        """Return {symbol: signed share quantity} for all open positions."""
        return {p.symbol: float(p.qty) for p in self._trading.get_all_positions()}

    def get_recent_bars(
        self,
        symbols: list[str],
        n_days: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Pull the last `n_days` of daily OHLC for `symbols`.

        Returns aligned (close, high, low) wide DataFrames. Calendar gaps
        (weekends, holidays) are dropped — the caller sees only trading
        days that actually have bars on Alpaca's feed.
        """
        if not symbols:
            raise ValueError('symbols list is empty')
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(n_days * 1.6) + 7)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self._data.get_stock_bars(req).df
        if bars.empty:
            raise RuntimeError(f'no bars returned for {len(symbols)} symbols')

        bars = bars.reset_index()
        bars['date'] = pd.to_datetime(bars['timestamp']).dt.tz_localize(None).dt.normalize()
        wide = bars.pivot_table(
            index='date', columns='symbol',
            values=['close', 'high', 'low'], aggfunc='last')
        prices = wide['close'].sort_index()
        highs = wide['high'].sort_index()
        lows = wide['low'].sort_index()

        common_cols = prices.columns.intersection(highs.columns).intersection(lows.columns)
        prices, highs, lows = (df[common_cols] for df in (prices, highs, lows))
        prices = prices.ffill().dropna()
        highs = highs.ffill().dropna()
        lows = lows.ffill().dropna()
        common_idx = prices.index.intersection(highs.index).intersection(lows.index)
        return prices.loc[common_idx], highs.loc[common_idx], lows.loc[common_idx]

    def build_trades(
        self,
        target_weights: pd.Series,
        last_prices: pd.Series,
        current_positions: dict[str, float],
        equity: float,
        *,
        min_notional: float = 1.0,
    ) -> list[Trade]:
        """Diff target vs current and emit fractional-share market orders.

        Symbols held but not in `target_weights` get full liquidation
        orders. Trades smaller than `min_notional` (default $1) are
        suppressed to avoid below-minimum order rejections.
        """
        all_symbols = sorted(set(target_weights.index) | set(current_positions))
        trades: list[Trade] = []
        for sym in all_symbols:
            target_w = float(target_weights.get(sym, 0.0))
            current_qty = float(current_positions.get(sym, 0.0))
            price = float(last_prices.get(sym, 0.0))
            if price <= 0:
                continue
            current_w = (current_qty * price) / equity if equity > 0 else 0.0
            target_qty = (target_w * equity) / price
            qty_diff = target_qty - current_qty
            notional = qty_diff * price
            if abs(notional) < min_notional:
                continue
            trades.append(Trade(
                symbol=sym,
                side='buy' if qty_diff > 0 else 'sell',
                qty=abs(round(qty_diff, 6)),
                notional=abs(notional),
                current_weight=current_w,
                target_weight=target_w,
                last_price=price,
            ))
        return trades

    def submit_orders(
        self, trades: list[Trade],
    ) -> tuple[list[str], list[OrderRejection]]:
        """Submit each trade as a fractional market DAY order.

        Returns `(order_ids, rejections)`. Per-order failures (most
        commonly: non-fractionable symbol rejecting a fractional qty)
        are logged, captured into `rejections`, and skipped rather than
        aborting the batch — otherwise one bad symbol mid-loop leaves
        the portfolio half-rebalanced. Callers should check
        `len(rejections)` and decide whether to alert.

        TODO(review #5): distinguish 4xx (per-symbol skip+log, current
        behavior) from 5xx/connection errors (re-raise+abort) so an
        intermittent Alpaca outage doesn't silently zero every order.
        """
        import logging
        order_ids: list[str] = []
        rejections: list[OrderRejection] = []
        for t in trades:
            req = MarketOrderRequest(
                symbol=t.symbol,
                qty=t.qty,
                side=OrderSide.BUY if t.side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            try:
                order = self._trading.submit_order(req)
                order_ids.append(str(order.id))
            except Exception as e:
                logging.warning(
                    'submit_order failed for %s qty=%g: %s', t.symbol, t.qty, e)
                rejections.append(OrderRejection(
                    symbol=t.symbol, qty=t.qty, reason=str(e)))
        return order_ids, rejections


__all__ = [
    'Account', 'Trade', 'OrderRejection', 'AlpacaBroker', 'PAPER_BASE_URL',
]
