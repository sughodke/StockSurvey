"""Tests for the DCA state file (last_rebal_date persistence)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from dca.state import DCAState, load_state, save_state


def test_state_round_trip(tmp_path: Path) -> None:
    p = tmp_path / 'state.json'
    s = DCAState(last_rebal_date=date(2026, 5, 13),
                  last_rebal_checkpoint='/foo/cp.json')
    save_state(s, p)
    s2 = load_state(p)
    assert s2.last_rebal_date == date(2026, 5, 13)
    assert s2.last_rebal_checkpoint == '/foo/cp.json'


def test_state_missing_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / 'never-existed.json'
    s = load_state(p)
    assert s.last_rebal_date is None


def test_state_can_save_no_prior(tmp_path: Path) -> None:
    p = tmp_path / 'state.json'
    s = DCAState(last_rebal_date=None)
    save_state(s, p)
    s2 = load_state(p)
    assert s2.last_rebal_date is None
