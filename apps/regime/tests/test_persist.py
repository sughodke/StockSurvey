"""Round-trip tests for the JSON checkpoint format."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from regime.persist import Checkpoint, load_checkpoint, save_checkpoint
from regime.research.optimize_adam import TrainResult


def _make_train_result(n_scales: int = 4) -> TrainResult:
    return TrainResult(
        params={
            'scale_log_weights': jnp.asarray(np.linspace(-1.0, 1.0, n_scales),
                                              dtype=jnp.float32),
            'log_temperature': jnp.asarray(np.log(0.5), dtype=jnp.float32),
        },
        train_history=[0.1, 0.5, 1.2],
        val_history=[(0, 0.0), (5, 0.4), (10, 0.9)],
        train_sharpe=1.2,
        val_sharpe=0.9,
        scales=[5, 21, 90, 126],
        train_dates=(pd.Timestamp('2015-01-01'), pd.Timestamp('2020-12-31')),
        val_dates=(pd.Timestamp('2021-01-01'), pd.Timestamp('2024-12-31')),
    )


def test_save_and_load_round_trip(tmp_path: Path):
    result = _make_train_result()
    path = save_checkpoint(
        tmp_path / 'model.json', result,
        universe=['AAPL', 'MSFT', 'NVDA'],
        lookback=120, n_tail=20, rebal_days=20,
        max_spread=0.02, commission_bps=10)
    assert path.exists()

    cp = load_checkpoint(path)
    assert isinstance(cp, Checkpoint)
    assert cp.scales == [5, 21, 90, 126]
    assert cp.lookback == 120
    assert cp.n_tail == 20
    assert cp.universe == ['AAPL', 'MSFT', 'NVDA']
    assert cp.train_sharpe == 1.2
    np.testing.assert_allclose(
        cp.scale_log_weights,
        np.linspace(-1.0, 1.0, 4), rtol=1e-5)


def test_jax_params_returns_dict_for_strategy(tmp_path: Path):
    result = _make_train_result()
    path = save_checkpoint(
        tmp_path / 'm.json', result,
        universe=['A'], lookback=60, n_tail=10, rebal_days=20,
        max_spread=0.02, commission_bps=10)
    cp = load_checkpoint(path)
    params = cp.jax_params()
    assert set(params) == {'scale_log_weights', 'log_temperature'}
    assert params['scale_log_weights'].shape == (4,)


def test_load_rejects_wrong_version(tmp_path: Path):
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'version': 999}))
    with pytest.raises(ValueError, match='version mismatch'):
        load_checkpoint(bad)


def test_load_ignores_extra_fields(tmp_path: Path):
    """Forward-compatibility: a v1 reader tolerates unknown extra keys."""
    result = _make_train_result()
    path = save_checkpoint(
        tmp_path / 'm.json', result,
        universe=['A'], lookback=60, n_tail=10, rebal_days=20,
        max_spread=0.02, commission_bps=10)
    raw = json.loads(path.read_text())
    raw['some_future_field'] = 'whatever'
    raw['another_one'] = [1, 2, 3]
    path.write_text(json.dumps(raw))
    cp = load_checkpoint(path)  # must not raise
    assert cp.lookback == 60


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


def test_save_optuna_checkpoint_round_trip(tmp_path: Path):
    """Round-trip an optuna checkpoint via save_checkpoint_from_window."""
    from regime.persist import save_checkpoint_from_window
    from regime.trainer import WindowResult

    window = WindowResult(
        train_start=pd.Timestamp('2015-01-01'),
        train_end=pd.Timestamp('2020-01-01'),
        val_end=pd.Timestamp('2023-01-01'),
        best_params={
            'lookback': 116, 'n_tail': 16, 'top_n': 5,
            'divergence': 'cosine',
            'use_short_scales': False,
            'use_mid_scales': True,
            'use_long_scales': False,
        },
        train_score=1.13,
        val_score=0.46,
    )
    path = save_checkpoint_from_window(
        tmp_path / 'opt.json', window,
        universe=['AAPL', 'MSFT', 'NVDA'],
        rebal_days=20, max_spread=0.02, commission_bps=10)

    cp = load_checkpoint(path)
    assert cp.mode == 'optuna'
    assert cp.lookback == 116 and cp.n_tail == 16 and cp.top_n == 5
    assert cp.divergence == 'cosine'
    assert cp.scales == [10, 12, 15, 21, 26]  # mid scales resolved
    assert cp.scale_log_weights == [0.0] * 5  # uniform weighting flag
    assert cp.train_sharpe == 1.13 and cp.val_sharpe == 0.46


def test_load_legacy_adam_checkpoint_defaults_mode(tmp_path: Path):
    """A checkpoint without a `mode` field (written before mode existed)
    should load as adam-mode for backwards compatibility."""
    result = _make_train_result()
    path = save_checkpoint(
        tmp_path / 'm.json', result,
        universe=['A'], lookback=60, n_tail=10, rebal_days=20,
        max_spread=0.02, commission_bps=10)
    raw = json.loads(path.read_text())
    raw.pop('mode', None)
    raw.pop('top_n', None)
    raw.pop('divergence', None)
    path.write_text(json.dumps(raw))
    cp = load_checkpoint(path)
    assert cp.mode == 'adam'
    assert cp.top_n is None and cp.divergence is None


def test_save_scalogram_checkpoint_round_trip(tmp_path: Path):
    """Scalogram WindowResult → checkpoint records strategy='scalogram'
    and omits the divergence field (scalogram has no divergence knob)."""
    from regime.persist import save_checkpoint_from_window
    from regime.trainer import WindowResult

    window = WindowResult(
        train_start=pd.Timestamp('2015-01-01'),
        train_end=pd.Timestamp('2020-01-01'),
        val_end=pd.Timestamp('2023-01-01'),
        best_params={
            'lookback': 80, 'n_tail': 12, 'top_n': 10,
            # no 'divergence' key — scalogram doesn't search over it
            'use_short_scales': True,
            'use_mid_scales': True,
            'use_long_scales': False,
        },
        train_score=0.9, val_score=0.4,
        strategy='scalogram',
    )
    path = save_checkpoint_from_window(
        tmp_path / 'scalo.json', window,
        universe=['AAPL', 'MSFT'],
        rebal_days=20, max_spread=0.02, commission_bps=10)

    cp = load_checkpoint(path)
    assert cp.mode == 'optuna'
    assert cp.strategy == 'scalogram'
    assert cp.divergence is None  # scalogram has no divergence
    assert cp.top_n == 10
    # Short + mid scales resolved together
    assert cp.scales == [3, 5, 7, 10, 12, 15, 21, 26]


def test_load_legacy_checkpoint_defaults_strategy_to_regime(tmp_path: Path):
    """A checkpoint without a `strategy` field (written before scalogram
    was added) should load as strategy='regime' for back-compat."""
    result = _make_train_result()
    path = save_checkpoint(
        tmp_path / 'm.json', result,
        universe=['A'], lookback=60, n_tail=10, rebal_days=20,
        max_spread=0.02, commission_bps=10)
    raw = json.loads(path.read_text())
    raw.pop('strategy', None)
    path.write_text(json.dumps(raw))
    cp = load_checkpoint(path)
    assert cp.strategy == 'regime'
