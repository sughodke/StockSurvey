"""StateVecBuilder fit/transform invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfr.state_vec import StateVecBuilder, default_state_vec_builder


def _panel(n_bars: int = 400, n_tickers: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(n_bars, n_tickers))
    prices = np.cumprod(1 + rets, axis=0) * 100
    return pd.DataFrame(prices,
                        index=pd.date_range('2020-01-01', periods=n_bars, freq='B'),
                        columns=[f'T{i}' for i in range(n_tickers)])


def _macro(prices: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(prices)
    return pd.DataFrame({
        'vix':            10 + 5 * rng.random(n),
        'credit_baa':      2 + 1 * rng.random(n),
        'm2_yoy':          5 + 3 * rng.random(n),
        'real_yield_10y':  rng.random(n) - 0.5,
    }, index=prices.index)


def test_state_vec_no_macro_six_features():
    p = _panel()
    b = StateVecBuilder()
    b.fit(p)
    assert b.n_features == 6
    assert b.feat_names == ['vol_21', 'disp_21', 'ret_21', 'ret_63', 'tdd_21', 'breadth']
    state = b.transform(p)
    assert state.shape == (len(p), 6)
    assert state.dtype == np.float32


def test_state_vec_with_macro_ten_features():
    p = _panel()
    m = _macro(p)
    b = StateVecBuilder()
    b.fit(p, m)
    assert b.n_features == 10
    assert all(f.startswith('macro_') for f in b.feat_names[6:])
    state = b.transform(p, m)
    assert state.shape == (len(p), 10)


def test_state_vec_zscore_centered_on_train():
    """Z-scored features should have ~0 mean and ~1 std on the training
    panel. Features that come out near-constant on the synthetic
    panel (e.g. `breadth`, since all tickers are valid from t=0) have
    transform output std ≈ 0; we only check the non-degenerate
    features."""
    p = _panel(n_bars=600)
    b = StateVecBuilder()
    b.fit(p)
    state = b.transform(p)
    valid_mask = b.valid_mask(p)
    valid = state[valid_mask]
    means = valid.mean(axis=0)
    stds = valid.std(axis=0)
    # Filter on TRANSFORM output std (constant features will be ~0
    # regardless of the fitted std workaround).
    nondeg = stds > 0.1
    assert nondeg.sum() >= 4, f'too many degenerate features: {stds}'
    assert (np.abs(means[nondeg]) < 0.5).all(), f'features not centered: {means[nondeg]}'
    assert (stds[nondeg] > 0.3).all() and (stds[nondeg] < 3.0).all(), \
        f'stds off: {stds[nondeg]}'


def test_state_vec_clipping_bounds_outliers():
    """Clipping at ±5 sigma should bound transform output."""
    p = _panel()
    b = StateVecBuilder(clip_z=2.0)   # tight clip
    b.fit(p)
    state = b.transform(p)
    assert (state >= -2.0).all() and (state <= 2.0).all()


def test_valid_mask_excludes_warmup():
    p = _panel(n_bars=400)
    b = default_state_vec_builder()
    b.fit(p)
    mask = b.valid_mask(p)
    # Early bars have NaN for ret_63 (need 63 bars), tdd_21 (need 21+),
    # disp_21 (need 21+) — overall valid starts ~bar 63.
    assert not mask[:30].any()
    assert mask[100:].all()


def test_transform_without_fit_raises():
    p = _panel()
    b = StateVecBuilder()
    try:
        b.transform(p)
    except RuntimeError as e:
        assert 'fit' in str(e).lower()
    else:
        raise AssertionError('expected RuntimeError for unfitted transform')


def test_schema_mismatch_raises():
    """Fitting with macro but transforming without (or vice-versa) raises."""
    p = _panel()
    m = _macro(p)
    b = StateVecBuilder()
    b.fit(p, m)
    try:
        b.transform(p)   # no macro
    except ValueError as e:
        assert 'schema' in str(e).lower() or 'mismatch' in str(e).lower()
    else:
        raise AssertionError('expected ValueError for schema mismatch')
