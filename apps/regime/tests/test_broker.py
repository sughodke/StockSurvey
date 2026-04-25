"""Tests for AlpacaBroker.build_trades — pure logic, no network calls.

We instantiate the broker with stubbed env credentials so it doesn't
hit Alpaca, then exercise the diff/sizing logic directly.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest


@pytest.fixture
def broker(monkeypatch):
    monkeypatch.setenv('ALPACA_API_KEY', 'stub')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'stub')
    from regime.broker import AlpacaBroker
    return AlpacaBroker()


def test_build_trades_empty_when_aligned(broker):
    targets = pd.Series([0.5, 0.5], index=['AAPL', 'MSFT'])
    last_prices = pd.Series([100.0, 200.0], index=['AAPL', 'MSFT'])
    current = {'AAPL': 50.0, 'MSFT': 25.0}  # 50*100 + 25*200 = 5000+5000 = 10000
    trades = broker.build_trades(
        target_weights=targets, last_prices=last_prices,
        current_positions=current, equity=10000.0)
    assert trades == []


def test_build_trades_buys_to_target(broker):
    targets = pd.Series([1.0], index=['AAPL'])
    last_prices = pd.Series([100.0], index=['AAPL'])
    current: dict[str, float] = {}
    trades = broker.build_trades(
        target_weights=targets, last_prices=last_prices,
        current_positions=current, equity=1000.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.symbol == 'AAPL'
    assert t.side == 'buy'
    assert t.qty == pytest.approx(10.0, rel=1e-6)  # $1000 / $100 = 10 shares
    assert t.notional == pytest.approx(1000.0, rel=1e-6)
    assert t.target_weight == 1.0
    assert t.current_weight == 0.0


def test_build_trades_liquidates_unwanted(broker):
    targets = pd.Series([1.0], index=['AAPL'])
    last_prices = pd.Series([100.0, 50.0], index=['AAPL', 'MSFT'])
    current = {'AAPL': 5.0, 'MSFT': 8.0}  # MSFT held but not in target
    trades = broker.build_trades(
        target_weights=targets, last_prices=last_prices,
        current_positions=current, equity=1000.0)
    by_sym = {t.symbol: t for t in trades}
    assert by_sym['MSFT'].side == 'sell'
    assert by_sym['MSFT'].qty == pytest.approx(8.0, rel=1e-6)


def test_build_trades_suppresses_dust(broker):
    targets = pd.Series([0.5, 0.5], index=['AAPL', 'MSFT'])
    last_prices = pd.Series([100.0, 200.0], index=['AAPL', 'MSFT'])
    # Slightly off-balance: $10,000.50 instead of exact $10k
    current = {'AAPL': 50.0, 'MSFT': 25.0025}
    trades = broker.build_trades(
        target_weights=targets, last_prices=last_prices,
        current_positions=current, equity=10000.0,
        min_notional=1.0)
    # The MSFT delta is 0.0025 * 200 = $0.50 — below $1 threshold, suppressed.
    assert all(t.symbol != 'MSFT' for t in trades)
