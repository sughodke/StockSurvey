"""Infoset bucketing — train-only fit, transform on val."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfr.state import InfosetBuilder, default_infoset_builder


def _make_panel(n_bars: int = 400, n_tickers: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(n_bars, n_tickers))
    prices = np.cumprod(1 + rets, axis=0) * 100
    index = pd.date_range('2020-01-01', periods=n_bars, freq='B')
    return pd.DataFrame(prices, index=index,
                        columns=[f'T{i}' for i in range(n_tickers)])


def test_infoset_count_matches_buckets():
    b = default_infoset_builder()
    # 3 × 3 = 9 regime cells + 1 warmup = 10
    assert b.n_infosets == 10
    assert b.warmup_id == 9


def test_fit_transform_assigns_buckets():
    b = default_infoset_builder()
    p = _make_panel()
    ids = b.fit_transform(p)
    # Warmup at start; regime cells later
    assert ids[0] == b.warmup_id
    assert (ids[50:] >= 0).all()
    assert (ids[50:] < b.n_infosets).all()


def test_transform_without_fit_raises():
    b = default_infoset_builder()
    p = _make_panel()
    try:
        b.transform(p)
    except RuntimeError as e:
        assert 'fit' in str(e).lower()
    else:
        raise AssertionError('expected RuntimeError for unfitted transform')


def test_buckets_are_balanced_on_iid_data():
    """On uniform-noise prices, the 9 regime cells should each carry
    a non-trivial share of the panel (~10%). Bucket cutoffs are
    quantiles so this is by construction."""
    b = default_infoset_builder()
    p = _make_panel(n_bars=800)
    ids = b.fit_transform(p)
    valid_ids = ids[ids != b.warmup_id]
    counts = np.bincount(valid_ids, minlength=b.n_vol_buckets * b.n_disp_buckets)
    # Each of the 9 cells should have at least 3% of the valid count
    # (vol/disp are not independent — they're correlated through the
    # universe — so we don't expect perfect 11% each).
    fracs = counts / counts.sum()
    assert (fracs >= 0.03).all(), f'unbalanced bucket distribution: {fracs}'
