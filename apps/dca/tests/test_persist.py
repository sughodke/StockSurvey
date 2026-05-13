"""Tests for DCACheckpoint serialization + validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dca.persist import (
    CHECKPOINT_VERSION,
    DCACheckpoint,
    load_checkpoint,
    save_checkpoint,
)


def _good_kwargs(**overrides) -> dict:
    base = dict(
        version=CHECKPOINT_VERSION,
        name='test-basket',
        universe=['SPY', 'TLT', 'GLD'],
        target_weights={'SPY': 1/3, 'TLT': 1/3, 'GLD': 1/3},
        min_rebal_days=80,
        drift_threshold=0.05,
        commission_bps=5.0,
        created_at='2026-05-13T00:00:00+00:00',
    )
    base.update(overrides)
    return base


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    cp = DCACheckpoint(**_good_kwargs())
    out = save_checkpoint(tmp_path / 'cp.json', cp)
    assert out.exists()

    cp2 = load_checkpoint(out)
    assert cp2.name == cp.name
    assert cp2.universe == cp.universe
    assert cp2.target_weights == cp.target_weights
    assert cp2.min_rebal_days == cp.min_rebal_days


def test_universe_target_mismatch_fails() -> None:
    with pytest.raises(ValueError, match='must match exactly'):
        DCACheckpoint(**_good_kwargs(
            universe=['SPY', 'TLT'],
            target_weights={'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2},
        ))


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match='must sum to 1.0'):
        DCACheckpoint(**_good_kwargs(
            target_weights={'SPY': 0.4, 'TLT': 0.3, 'GLD': 0.2},
        ))


def test_min_rebal_days_validation() -> None:
    with pytest.raises(ValueError, match='min_rebal_days'):
        DCACheckpoint(**_good_kwargs(min_rebal_days=0))


def test_drift_threshold_validation() -> None:
    with pytest.raises(ValueError, match='drift_threshold'):
        DCACheckpoint(**_good_kwargs(drift_threshold=1.5))
    with pytest.raises(ValueError, match='drift_threshold'):
        DCACheckpoint(**_good_kwargs(drift_threshold=-0.1))


def test_version_mismatch_fails(tmp_path: Path) -> None:
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps({**_good_kwargs(), 'version': 999}))
    with pytest.raises(ValueError, match='version mismatch'):
        load_checkpoint(p)


def test_unknown_keys_ignored_for_forward_compat(tmp_path: Path) -> None:
    raw = _good_kwargs()
    raw['mystery_future_field'] = 'whatever'
    p = tmp_path / 'fc.json'
    p.write_text(json.dumps(raw))
    cp = load_checkpoint(p)
    assert cp.name == 'test-basket'
