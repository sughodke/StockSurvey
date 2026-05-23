"""Ensemble coordinator — exercises the cross-leg orchestration.

The coordinator should:
  - Call both legs even if one errors (independent strategies)
  - Capture each leg's error message rather than letting it propagate
  - Return a structured `EnsembleRunResult` either way
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vol.ensemble import EnsembleRunResult, format_ensemble, run_ensemble


def test_ensemble_captures_both_errors_when_inputs_missing(tmp_path: Path):
    """No checkpoints exist -> each leg errors with FileNotFoundError;
    coordinator returns both errors, doesn't raise."""
    result = run_ensemble(
        dca_checkpoint=tmp_path / 'no-dca.json',
        vol_checkpoint=tmp_path / 'no-vol.json',
        dry_run=True,
    )
    assert isinstance(result, EnsembleRunResult)
    assert result.dca_error is not None
    assert result.vol_error is not None
    assert result.dca_result is None
    assert result.vol_result is None
    out = format_ensemble(result)
    assert 'DCA leg' in out and 'Vol-v3 leg' in out
    assert 'ERROR' in out


def test_ensemble_one_leg_succeeds_other_fails(tmp_path: Path):
    """Build a valid vol checkpoint + simulate vol's gate-closed
    abort; DCA checkpoint is missing → DCA errors. Confirm both are
    reported independently."""
    from vol.persist import (
        LIVE_FEATURE_NAMES, StranglesConfig, VolCheckpoint, save_checkpoint,
    )
    import numpy as np

    cp = VolCheckpoint(
        feature_names=list(LIVE_FEATURE_NAMES),
        coefs=[0.01, 0.03, 0.0, 0.0, 0.005],
        feat_mean=[1.0, 0.0, 0.0, 0.0],
        feat_std=[0.4, 1.0, 0.05, 0.05],
        universe=['AAPL'], gate_fred_series='VIXCLS',
        gate_lookback_trading_days=126, top_k=1,
        strangle=StranglesConfig(),
        train_period='x', val_period='y', val_pearson_r=0.16,
        n_obs_oos=33, oos_ann_sharpe=2.8, oos_deflated_t=5.5,
    )
    vol_cp_path = tmp_path / 'vol.json'
    save_checkpoint(cp, vol_cp_path)

    # VIX gate closed (last bar BELOW median) → vol leg aborts cleanly.
    idx = pd.date_range('2025-01-01', periods=200, freq='B')
    vix = pd.Series(np.full(200, 20.0), index=idx)
    vix.iloc[-1] = 12.0

    class FlatAccount:
        equity = 100_000.0
        cash = 100_000.0
    class StubBroker:
        _paper = True
        def get_account(self):
            return FlatAccount()

    result = run_ensemble(
        dca_checkpoint=tmp_path / 'no-dca.json',
        vol_checkpoint=vol_cp_path,
        dry_run=True,
        vol_kwargs={
            'broker': StubBroker(),
            'options_data': object(),
            'bars_data': object(),
            'vix_loader': lambda: vix,
            'killswitch_path': tmp_path / 'does-not-exist',
        },
    )
    # DCA leg errors (no checkpoint)
    assert result.dca_error is not None
    # Vol leg returns a clean abort (not an error — it's a planned rail trigger)
    assert result.vol_error is None
    assert result.vol_result is not None
    assert 'VIX gate closed' in result.vol_result.aborted_reason
