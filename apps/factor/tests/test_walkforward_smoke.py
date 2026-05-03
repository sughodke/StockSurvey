"""End-to-end smoke for the walk-forward training path.

Synthesizes a tiny universe (random walk, no exploitable structure),
runs `train_scorer_indicators_walkforward` over a few rolling windows,
and asserts the result aggregates correctly. Does not assert on val IC
magnitude — random walks should give val IC ≈ 0 across windows; that's
the correct behavior, not a failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor import (
    IndicatorGridConfig, build_indicator_features,
    train_scorer_indicators_walkforward,
)
from factor.train_walkforward import _generate_window_slices
from ss_features import TickerData


def _make_synthetic_universe(n_tickers: int, n_bars: int, cfg: IndicatorGridConfig,
                             ) -> list[TickerData]:
    dates = pd.bdate_range('2000-01-03', periods=n_bars).to_numpy()
    out: list[TickerData] = []
    for j in range(n_tickers):
        rng = np.random.default_rng(j)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_bars)))
        feats, valid = build_indicator_features(prices, cfg)
        out.append(TickerData(
            name=f'T{j}', prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ))
    return out


def test_window_slicer_no_overlap_when_step_eq_val():
    # 100 blocks, train=20, val=10, step=10 -> windows at [0..30), [10..40),
    # [20..50), ... last window starts at 70 ([70..100)). Total = 8 windows.
    slices = _generate_window_slices(100, train_w=20, val_w=10, step_w=10)
    assert len(slices) == 8
    # Step = val means consecutive val ranges abut without overlap.
    for i in range(len(slices) - 1):
        assert slices[i][1].stop == slices[i + 1][1].start


def test_window_slicer_drops_partial_tail_window():
    # 50 blocks, train=20, val=15 -> need 35 per window. step=15.
    # Window 0: [0..35), window 1 would need [15..50) which fits exactly.
    # Window 2 would need [30..65) — drops.
    slices = _generate_window_slices(50, train_w=20, val_w=15, step_w=15)
    assert len(slices) == 2


def test_window_slicer_validates_args():
    import pytest
    with pytest.raises(ValueError, match='must each be >= 2'):
        _generate_window_slices(50, train_w=1, val_w=10, step_w=10)
    with pytest.raises(ValueError, match='must each be >= 2'):
        _generate_window_slices(50, train_w=10, val_w=1, step_w=10)
    with pytest.raises(ValueError, match='step_window_blocks=0'):
        _generate_window_slices(50, train_w=10, val_w=10, step_w=0)


def test_walkforward_runs_and_aggregates():
    """5500 bars at rebal=20 yields ~268 rebal blocks; with train=63,
    val=39, step=39, expect (268 - 102) // 39 + 1 = 5 windows."""
    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    tickers = _make_synthetic_universe(n_tickers=4, n_bars=5500, cfg=cfg)

    res = train_scorer_indicators_walkforward(
        tickers, cfg,
        rebal_days=20,
        train_window_blocks=20,
        val_window_blocks=10,
        step_window_blocks=10,
        scorer='linear',
        n_steps=10, learning_rate=1e-2, weight_decay=1e-3,
        verbose=False,
    )

    assert res.scorer == 'linear'
    assert res.feature_width == F
    assert res.n_windows >= 2, \
        f'expected at least a couple of windows, got {res.n_windows}'
    # Indices monotonically advance, no overlap (step==val).
    for i in range(res.n_windows - 1):
        assert res.windows[i].val_block_end == res.windows[i + 1].val_block_start
    # Each window has the requested width.
    for w in res.windows:
        assert w.n_train_bars == 20
        assert w.n_val_bars == 10
        assert w.head_params['W'].shape == (F,)
        assert np.isfinite(w.train_ic)
        assert np.isfinite(w.val_ic)
    # Aggregates exist and match per-window numbers.
    np.testing.assert_allclose(
        res.mean_val_ic, np.mean([w.val_ic for w in res.windows]))
    assert 0.0 <= res.positive_val_ic_fraction <= 1.0
