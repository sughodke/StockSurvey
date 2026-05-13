"""Tests for the cadence/drift gate + risk-rail logic in live.run_live.

We mock AlpacaBroker so tests don't require credentials or network.
The gate logic is the only DCA-specific thing worth testing in
isolation; the risk-rail patterns are inherited from the
regime/relational live runners which are already tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from dca.live import _evaluate_cadence_gate, run_live
from dca.persist import CHECKPOINT_VERSION, DCACheckpoint, save_checkpoint
from dca.state import DCAState, save_state
from ss_portfolio.broker import Account


# ── Cadence/drift gate unit tests (no broker needed) ──────────────


def _make_state(last_rebal: date | None) -> DCAState:
    return DCAState(last_rebal_date=last_rebal)


def _ew_target() -> pd.Series:
    return pd.Series({'SPY': 1/3, 'TLT': 1/3, 'GLD': 1/3},
                     name='target_weight')


def test_gate_fires_on_first_run() -> None:
    state = _make_state(None)
    target = _ew_target()
    current = pd.Series({'SPY': 0.0, 'TLT': 0.0, 'GLD': 0.0})
    should, reason, drift, days = _evaluate_cadence_gate(
        state, date(2026, 5, 13), target, current,
        min_rebal_days=80, drift_threshold=0.05,
    )
    assert should is True
    assert 'no prior rebal' in reason
    assert days is None


def test_gate_holds_when_recent_and_low_drift() -> None:
    state = _make_state(date(2026, 5, 1))
    target = _ew_target()
    current = pd.Series({'SPY': 0.34, 'TLT': 0.33, 'GLD': 0.33})
    should, reason, drift, days = _evaluate_cadence_gate(
        state, date(2026, 5, 13), target, current,
        min_rebal_days=80, drift_threshold=0.05,
    )
    assert should is False
    assert 'gate held' in reason
    assert days == 12


def test_gate_fires_on_cadence_floor() -> None:
    state = _make_state(date(2026, 1, 1))
    target = _ew_target()
    current = pd.Series({'SPY': 0.34, 'TLT': 0.33, 'GLD': 0.33})
    should, reason, drift, days = _evaluate_cadence_gate(
        state, date(2026, 5, 13), target, current,
        min_rebal_days=80, drift_threshold=0.05,
    )
    assert should is True
    assert 'cadence floor met' in reason
    assert days >= 80


def test_gate_fires_on_drift_threshold() -> None:
    state = _make_state(date(2026, 5, 1))
    target = _ew_target()
    # SPY drifted way up: 50% vs target 33% → drift 17% >> 5% threshold
    current = pd.Series({'SPY': 0.50, 'TLT': 0.25, 'GLD': 0.25})
    should, reason, drift, days = _evaluate_cadence_gate(
        state, date(2026, 5, 13), target, current,
        min_rebal_days=80, drift_threshold=0.05,
    )
    assert should is True
    assert 'drift' in reason.lower()
    assert drift > 0.10


def test_gate_drift_includes_zero_target_holdings() -> None:
    """If a held name is OUTSIDE the target universe, that's max drift."""
    state = _make_state(date(2026, 5, 1))
    target = _ew_target()
    # User accidentally holds AAPL — not in target. Reindexing target
    # against current_weights' index should give AAPL target = 0,
    # so any AAPL holding shows as drift.
    current = pd.Series({'SPY': 0.30, 'TLT': 0.30, 'GLD': 0.30, 'AAPL': 0.10})
    should, reason, drift, days = _evaluate_cadence_gate(
        state, date(2026, 5, 13), target, current,
        min_rebal_days=80, drift_threshold=0.05,
    )
    assert should is True
    assert drift >= 0.10  # AAPL alone is 10% drift


# ── End-to-end run_live with a mock broker ────────────────────────


@dataclass
class _StubAccount(Account):
    pass


class _StubBroker:
    def __init__(self, *, equity=100_000.0, positions=None,
                 prices=None, last_bar_age_days=0):
        self._account = Account(equity=equity, cash=equity, buying_power=equity,
                                  paper=True)
        self._positions = positions or {}
        self._prices = prices or {sym: 100.0 for sym in
                                   ['XLB','XLE','XLF','XLI','XLK','XLP','XLU',
                                    'XLV','XLY','TLT','IEF','GLD','DBC']}
        self._last_bar_age_days = last_bar_age_days

    def get_account(self):
        return self._account

    def get_positions(self):
        return dict(self._positions)

    def get_recent_bars(self, symbols, n_days):
        idx = pd.date_range(end=pd.Timestamp.now('UTC').tz_convert(None).normalize()
                              - pd.Timedelta(days=self._last_bar_age_days),
                              periods=n_days, freq='B')
        prices = pd.DataFrame(
            [[self._prices.get(s, 100.0) for s in symbols] for _ in idx],
            index=idx, columns=symbols,
        )
        # highs/lows unused by DCA but required by the broker interface
        return prices, prices, prices

    def build_trades(self, *, target_weights, last_prices, current_positions, equity):
        trades = []
        from ss_portfolio.broker import Trade
        all_syms = sorted(set(target_weights.index) | set(current_positions))
        for sym in all_syms:
            target_w = float(target_weights.get(sym, 0.0))
            current_qty = float(current_positions.get(sym, 0.0))
            price = float(last_prices.get(sym, 0.0))
            if price <= 0:
                continue
            current_w = (current_qty * price) / equity if equity > 0 else 0.0
            target_qty = (target_w * equity) / price
            qty_diff = target_qty - current_qty
            notional = qty_diff * price
            if abs(notional) < 1.0:
                continue
            trades.append(Trade(
                symbol=sym, side='buy' if qty_diff > 0 else 'sell',
                qty=abs(round(qty_diff, 6)), notional=abs(notional),
                current_weight=current_w, target_weight=target_w,
                last_price=price,
            ))
        return trades

    def submit_orders(self, trades):
        return [f'order-{i}' for i in range(len(trades))], []


def _canonical_checkpoint(tmp_path: Path) -> Path:
    syms = sorted(['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY',
                   'TLT','IEF','GLD','DBC'])
    weight = 1.0 / len(syms)
    cp = DCACheckpoint(
        version=CHECKPOINT_VERSION,
        name='test-13etf',
        universe=syms,
        target_weights={s: weight for s in syms},
        min_rebal_days=80,
        drift_threshold=0.05,
        commission_bps=5.0,
        created_at='2026-05-13T00:00:00+00:00',
    )
    p = tmp_path / 'cp.json'
    save_checkpoint(p, cp)
    return p


def test_killswitch_aborts(tmp_path: Path) -> None:
    cp_path = _canonical_checkpoint(tmp_path)
    state_path = tmp_path / 'state.json'
    ks_path = tmp_path / 'kill'
    ks_path.touch()
    res = run_live(cp_path, broker=_StubBroker(),
                    killswitch_path=ks_path, state_path=state_path)
    assert res.aborted_reason and 'kill-switch' in res.aborted_reason


def test_stale_data_aborts(tmp_path: Path) -> None:
    cp_path = _canonical_checkpoint(tmp_path)
    state_path = tmp_path / 'state.json'
    ks_path = tmp_path / 'no-kill'
    res = run_live(
        cp_path, broker=_StubBroker(last_bar_age_days=10),
        killswitch_path=ks_path, state_path=state_path,
        max_data_age_days=3,
    )
    assert res.aborted_reason and 'stale data' in res.aborted_reason


def test_cadence_gate_holds_skips_trades(tmp_path: Path) -> None:
    cp_path = _canonical_checkpoint(tmp_path)
    state_path = tmp_path / 'state.json'
    ks_path = tmp_path / 'no-kill'
    # Simulate a recent rebal with positions exactly at target
    save_state(DCAState(last_rebal_date=date.today() - timedelta(days=10)),
                state_path)
    n = 13
    weight = 1.0 / n
    equity = 100_000.0
    price = 100.0
    positions = {s: weight * equity / price for s in [
        'XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY',
        'TLT','IEF','GLD','DBC'
    ]}
    res = run_live(
        cp_path, broker=_StubBroker(positions=positions),
        killswitch_path=ks_path, state_path=state_path,
        dry_run=True,
    )
    assert res.aborted_reason is None
    assert len(res.trades) == 0
    assert 'gate held' in res.rebal_reason


def test_first_run_dry_run_emits_trades(tmp_path: Path) -> None:
    cp_path = _canonical_checkpoint(tmp_path)
    state_path = tmp_path / 'state.json'
    ks_path = tmp_path / 'no-kill'
    res = run_live(
        cp_path, broker=_StubBroker(),
        killswitch_path=ks_path, state_path=state_path,
        dry_run=True,
    )
    assert res.aborted_reason is None
    # First run, no positions held → all 13 names get a buy
    assert len(res.trades) == 13
    assert all(t.side == 'buy' for t in res.trades)
    # Dry-run does NOT write state
    assert not state_path.exists()


def test_force_rebal_overrides_gate(tmp_path: Path) -> None:
    cp_path = _canonical_checkpoint(tmp_path)
    state_path = tmp_path / 'state.json'
    ks_path = tmp_path / 'no-kill'
    save_state(DCAState(last_rebal_date=date.today() - timedelta(days=10)),
                state_path)
    n = 13
    weight = 1.0 / n
    equity = 100_000.0
    positions = {s: weight * equity / 100.0 for s in [
        'XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY',
        'TLT','IEF','GLD','DBC'
    ]}
    res = run_live(
        cp_path, broker=_StubBroker(positions=positions),
        killswitch_path=ks_path, state_path=state_path,
        dry_run=True, force_rebal=True,
    )
    assert res.aborted_reason is None
    assert 'force_rebal=True' in res.rebal_reason
