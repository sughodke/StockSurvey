"""Smoke tests for the EnsembleCheckpoint round-trip + the live
risk-rail shell. Real broker calls are NOT exercised here — the
`live.run_live` test uses an empty checkpoint to validate the skip
+ kill-switch paths only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ensemble.live import DEFAULT_KILLSWITCH, run_live
from ensemble.persist import CHECKPOINT_VERSION, EnsembleCheckpoint, load_checkpoint, save_checkpoint


def _make_checkpoint(**overrides) -> EnsembleCheckpoint:
    base = dict(
        version=CHECKPOINT_VERSION,
        name='test',
        w_dca=0.05,
        w_vol=2.24,
        learner='grad_sharpe',
        dca_checkpoint_path='',
        vol_checkpoint_path='',
        train_start='2023-08-02',
        train_end='2025-12-11',
        created_at='2026-05-28T00:00:00+00:00',
    )
    base.update(overrides)
    return EnsembleCheckpoint(**base)


def test_round_trip(tmp_path: Path) -> None:
    cp = _make_checkpoint(notes='hello')
    out = save_checkpoint(tmp_path / 'cp.json', cp)
    back = load_checkpoint(out)
    assert back.w_dca == cp.w_dca
    assert back.w_vol == cp.w_vol
    assert back.learner == cp.learner
    assert back.notes == 'hello'


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError):
        _make_checkpoint(w_dca=-0.1)
    with pytest.raises(ValueError):
        _make_checkpoint(w_vol=-1.0)


def test_zero_total_weight_rejected() -> None:
    with pytest.raises(ValueError):
        _make_checkpoint(w_dca=0.0, w_vol=0.0)


def test_unknown_learner_rejected() -> None:
    with pytest.raises(ValueError):
        _make_checkpoint(learner='xgboost')


def test_live_dry_run_skips_legs(tmp_path: Path) -> None:
    """With empty leg paths, both legs are skipped — no broker calls
    issued. Validates the dispatch plumbing without touching brokers.
    """
    cp = _make_checkpoint()
    out = save_checkpoint(tmp_path / 'cp.json', cp)
    fake_ks = tmp_path / 'ks'
    result = run_live(out, dry_run=True, killswitch_path=str(fake_ks))
    assert result.aborted_reason is None
    assert result.dca_result is None
    assert result.vol_result is None
    assert result.w_dca == 0.05
    assert result.w_vol == 2.24


def test_live_killswitch_aborts(tmp_path: Path) -> None:
    cp = _make_checkpoint(dca_checkpoint_path='/nonexistent/dca.json')
    out = save_checkpoint(tmp_path / 'cp.json', cp)
    fake_ks = tmp_path / 'ks'
    fake_ks.write_text('')
    result = run_live(out, dry_run=True, killswitch_path=str(fake_ks))
    assert result.aborted_reason is not None
    assert 'kill-switch' in result.aborted_reason
    assert result.dca_result is None
    assert result.vol_result is None
