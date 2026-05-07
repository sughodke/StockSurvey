"""Tests for regime.live.run_live risk rails + dry-run happy path.

The broker is fully stubbed — these tests verify orchestration logic
(kill-switch, staleness check, weight cap, dry-run vs submit branch),
not the Alpaca client itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from regime.broker import Account, Trade
from regime.live import LiveRunResult, run_live
from regime.persist import save_checkpoint_from_window
from regime.trainer import WindowResult


def _checkpoint(tmp_path: Path, *, lookback: int = 30, n_tail: int = 5,
                universe: list[str], top_n: int = 4) -> Path:
    """Write a synthetic optuna-mode checkpoint and return its path."""
    window = WindowResult(
        train_start=pd.Timestamp('2020-01-01'),
        train_end=pd.Timestamp('2020-12-31'),
        val_end=pd.Timestamp('2021-12-31'),
        best_params={
            'lookback': lookback, 'n_tail': n_tail, 'top_n': top_n,
            'divergence': 'kl',
            'use_short_scales': True,   # → [3, 5, 7]
            'use_mid_scales': False,
            'use_long_scales': False,
        },
        train_score=0.0, val_score=0.0,
    )
    return save_checkpoint_from_window(
        tmp_path / 'cp.json', window,
        universe=universe, rebal_days=20, max_spread=0.02, commission_bps=10)


def _stub_broker(*, last_bar_offset_days: int = 1,
                 universe: list[str] | None = None,
                 equity: float = 100_000.0,
                 positions: dict[str, float] | None = None) -> MagicMock:
    """Build a MagicMock that mimics AlpacaBroker for one rebalance pass."""
    universe = universe or ['A', 'B', 'C', 'D']
    positions = positions or {}

    n_days = 100
    end = pd.Timestamp.now('UTC').tz_convert(None).normalize() - pd.Timedelta(days=last_bar_offset_days)
    dates = pd.bdate_range(end=end, periods=n_days)
    rng = np.random.default_rng(0)
    closes = np.cumsum(rng.standard_normal((n_days, len(universe))) * 0.5, axis=0) + 100
    prices = pd.DataFrame(closes, index=dates, columns=universe)
    highs = prices + np.abs(rng.standard_normal(prices.shape)) * 0.5
    lows = prices - np.abs(rng.standard_normal(prices.shape)) * 0.5

    broker = MagicMock()
    broker.get_account.return_value = Account(
        equity=equity, cash=equity, buying_power=equity, paper=True)
    broker.get_recent_bars.return_value = (prices, highs, lows)
    broker.get_positions.return_value = positions
    broker.build_trades.side_effect = lambda **kw: [
        Trade(symbol=s, side='buy', qty=1.0, notional=100.0,
              current_weight=0.0, target_weight=float(w),
              last_price=100.0)
        for s, w in kw['target_weights'].items() if w > 0
    ][:3]  # keep small
    broker.submit_orders.return_value = (['order-1', 'order-2'], [])
    return broker


def test_run_live_killswitch_aborts(tmp_path: Path):
    """Kill-switch file present → aborted_reason set, no orders submitted."""
    cp = _checkpoint(tmp_path, universe=['A', 'B', 'C', 'D'])
    ks = tmp_path / '.killswitch'
    ks.touch()
    broker = _stub_broker()

    result = run_live(cp, broker=broker, dry_run=False,
                      killswitch_path=ks)

    assert isinstance(result, LiveRunResult)
    assert result.aborted_reason and 'kill-switch' in result.aborted_reason
    assert result.submitted_order_ids == []
    broker.submit_orders.assert_not_called()
    broker.get_recent_bars.assert_not_called()  # short-circuit before fetch


def test_run_live_stale_data_aborts(tmp_path: Path):
    """Latest bar older than max_data_age_days → abort with reason."""
    cp = _checkpoint(tmp_path, universe=['A', 'B', 'C', 'D'])
    broker = _stub_broker(last_bar_offset_days=10)

    result = run_live(cp, broker=broker, dry_run=False,
                      max_data_age_days=3,
                      killswitch_path=tmp_path / 'nope.killswitch')

    assert result.aborted_reason and 'stale' in result.aborted_reason
    broker.submit_orders.assert_not_called()


def test_run_live_dry_run_does_not_submit(tmp_path: Path):
    """Happy path with dry_run=True → trades computed, none submitted."""
    cp = _checkpoint(tmp_path, universe=['A', 'B', 'C', 'D'])
    broker = _stub_broker(last_bar_offset_days=1)

    result = run_live(cp, broker=broker, dry_run=True,
                      max_position=0.5,
                      killswitch_path=tmp_path / 'nope.killswitch')

    assert result.aborted_reason is None
    assert result.dry_run is True
    assert len(result.trades) > 0
    assert result.submitted_order_ids == []
    broker.submit_orders.assert_not_called()


def test_run_live_submits_when_live(tmp_path: Path):
    """dry_run=False with no rails tripped → submit_orders called."""
    cp = _checkpoint(tmp_path, universe=['A', 'B', 'C', 'D'])
    broker = _stub_broker(last_bar_offset_days=1)

    result = run_live(cp, broker=broker, dry_run=False,
                      max_position=0.5,
                      killswitch_path=tmp_path / 'nope.killswitch')

    assert result.aborted_reason is None
    broker.submit_orders.assert_called_once()
    assert result.submitted_order_ids == ['order-1', 'order-2']


def test_run_live_caps_per_name_weight(tmp_path: Path):
    """Verify apply_position_cap is wired into the live path. Optuna hard
    top-N produces uniform 1/top_n weights, so a binding cap only fires
    in the degenerate case top_n*cap < 1 (water-fill returns uniform
    over nonzero names). We test both regimes:

      * Non-binding (cap > 1/top_n): max stays at 1/top_n, sum=1.
      * Degenerate (cap < 1/top_n with top_n*cap < 1): max equals
        1/top_n (uniform redistribute), sum=1.
    """
    universe = ['A', 'B', 'C', 'D', 'E', 'F']
    cp = _checkpoint(tmp_path, universe=universe, top_n=4)

    # Non-binding cap: 1/4 = 0.25 < 0.5, so max=0.25.
    result_loose = run_live(
        cp, broker=_stub_broker(universe=universe), dry_run=True,
        max_position=0.5, killswitch_path=tmp_path / 'nope.killswitch')
    assert result_loose.target_weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert result_loose.target_weights.max() == pytest.approx(0.25, abs=1e-9)

    # Degenerate (4*0.20=0.80 < 1): water-fill returns uniform 1/4=0.25
    # across the 4 nonzero names — exceeds the cap intentionally because
    # the cap is infeasible.
    result_tight = run_live(
        cp, broker=_stub_broker(universe=universe), dry_run=True,
        max_position=0.20, killswitch_path=tmp_path / 'nope.killswitch')
    assert result_tight.target_weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert result_tight.target_weights.max() == pytest.approx(0.25, abs=1e-9)
