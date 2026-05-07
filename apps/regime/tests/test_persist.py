"""Round-trip tests for the JSON checkpoint format."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from regime.persist import (
    Checkpoint,
    load_checkpoint,
    save_checkpoint_from_window,
)
from regime.trainer import WindowResult


def _make_window(
    *,
    strategy: str = 'regime',
    use_short: bool = False,
    use_mid: bool = True,
    use_long: bool = False,
    divergence: str | None = 'cosine',
    rsi_n: int | None = None,
) -> WindowResult:
    best_params: dict = {
        'lookback': 116, 'n_tail': 16, 'top_n': 5,
        'use_short_scales': use_short,
        'use_mid_scales': use_mid,
        'use_long_scales': use_long,
    }
    if divergence is not None:
        best_params['divergence'] = divergence
    if rsi_n is not None:
        best_params['rsi_n'] = rsi_n
    return WindowResult(
        train_start=pd.Timestamp('2015-01-01'),
        train_end=pd.Timestamp('2020-01-01'),
        val_end=pd.Timestamp('2023-01-01'),
        best_params=best_params,
        train_score=1.13,
        val_score=0.46,
        strategy=strategy,
    )


def test_save_and_load_round_trip(tmp_path: Path):
    window = _make_window()
    path = save_checkpoint_from_window(
        tmp_path / 'opt.json', window,
        universe=['AAPL', 'MSFT', 'NVDA'],
        rebal_days=20, max_spread=0.02, commission_bps=10)
    assert path.exists()

    cp = load_checkpoint(path)
    assert isinstance(cp, Checkpoint)
    assert cp.lookback == 116 and cp.n_tail == 16 and cp.top_n == 5
    assert cp.divergence == 'cosine'
    assert cp.scales == [10, 12, 15, 21, 26]  # mid scales
    assert cp.universe == ['AAPL', 'MSFT', 'NVDA']
    assert cp.train_sharpe == 1.13 and cp.val_sharpe == 0.46


def test_load_rejects_wrong_version(tmp_path: Path):
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'version': 999}))
    with pytest.raises(ValueError, match='version mismatch'):
        load_checkpoint(bad)


def test_load_ignores_extra_fields(tmp_path: Path):
    """Forward-compatibility: a v1 reader tolerates unknown extra keys.
    This also covers legacy adam-mode fields (`mode`, `scale_log_weights`,
    `log_temperature`) — they are filtered out as unknown and the rest of
    the optuna fields still load."""
    path = save_checkpoint_from_window(
        tmp_path / 'm.json', _make_window(),
        universe=['A'], rebal_days=20, max_spread=0.02, commission_bps=10)
    raw = json.loads(path.read_text())
    raw['some_future_field'] = 'whatever'
    raw['mode'] = 'adam'                       # legacy
    raw['scale_log_weights'] = [0.0, 0.0, 0.0]  # legacy
    raw['log_temperature'] = 0.0                # legacy
    path.write_text(json.dumps(raw))
    cp = load_checkpoint(path)  # must not raise
    assert cp.lookback == 116


def test_load_rejects_missing_required_field(tmp_path: Path):
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'version': 1, 'scales': [5]}))  # missing most fields
    with pytest.raises(TypeError):
        load_checkpoint(bad)


def test_load_rejects_malformed_json(tmp_path: Path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{not valid json')
    with pytest.raises(json.JSONDecodeError):
        load_checkpoint(bad)


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / 'nope.json')


def test_save_scalogram_checkpoint_round_trip(tmp_path: Path):
    """Scalogram WindowResult → checkpoint records strategy='scalogram'
    and omits the divergence field (scalogram has no divergence knob)."""
    window = _make_window(
        strategy='scalogram', use_short=True, use_mid=True, use_long=False,
        divergence=None)
    path = save_checkpoint_from_window(
        tmp_path / 'scalo.json', window,
        universe=['AAPL', 'MSFT'],
        rebal_days=20, max_spread=0.02, commission_bps=10)

    cp = load_checkpoint(path)
    assert cp.strategy == 'scalogram'
    assert cp.divergence is None  # scalogram has no divergence
    assert cp.top_n == 5
    # Short + mid scales resolved together
    assert cp.scales == [3, 5, 7, 10, 12, 15, 21, 26]


def test_save_rsi_checkpoint_round_trip(tmp_path: Path):
    """RSI WindowResult → checkpoint with rsi_n populated, scales empty."""
    window = _make_window(
        strategy='rsi', use_short=False, use_mid=False, use_long=False,
        divergence=None, rsi_n=14)
    path = save_checkpoint_from_window(
        tmp_path / 'rsi.json', window,
        universe=['AAPL', 'MSFT'],
        rebal_days=20, max_spread=0.02, commission_bps=10)

    cp = load_checkpoint(path)
    assert cp.strategy == 'rsi'
    assert cp.rsi_n == 14
    assert cp.scales == []
    assert cp.divergence is None


def test_load_legacy_checkpoint_defaults_strategy_to_regime(tmp_path: Path):
    """A checkpoint without a `strategy` field (written before scalogram
    was added) should load as strategy='regime' for back-compat."""
    path = save_checkpoint_from_window(
        tmp_path / 'm.json', _make_window(),
        universe=['A'], rebal_days=20, max_spread=0.02, commission_bps=10)
    raw = json.loads(path.read_text())
    raw.pop('strategy', None)
    path.write_text(json.dumps(raw))
    cp = load_checkpoint(path)
    assert cp.strategy == 'regime'
