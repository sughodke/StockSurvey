"""Multi-task auxiliary head — unit + smoke coverage.

Covers:
  * `factor.data.forward_robust_z` — winsorize + z-score forward log
    returns cross-sectionally; mask + edge cases.
  * `factor.objectives.masked_mse` — gradient-safe masked MSE.
  * `factor.scorers.mlp_multitask` — shared trunk + dual-head forward.
  * `train_scorer_walkforward(scorer='mlp_multitask', aux_weight=...)` —
    end-to-end smoke on a synthetic universe (random walk → val IC ≈ 0
    is the correct null result; we assert the path runs and reports
    aux MSE alongside primary IC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tinygrad.tensor import Tensor

from factor import (
    IndicatorGridConfig, build_indicator_features,
    forward_robust_z, masked_mse, train_scorer_walkforward,
)
from factor.scorers import (
    apply_mlp_multitask, init_mlp_multitask, get_scorer,
)
from ss_features import TickerData


def test_forward_robust_z_clips_outliers_and_z_scores():
    """At N=50 with winsor=(0.1, 0.9), comparing the outlier's z-score
    with vs without winsor exposes whether the clip actually fired:
    without clipping the outlier dominates the std and lands at moderate
    z; with clipping its value drops to the q_hi quantile (an interior
    return), the std contracts dramatically, and the outlier itself
    becomes one of many cells near the upper z-score plateau — its
    z-score should be *lower* than the unclipped version (because the
    std grows post-clip when the outlier no longer inflates it... wait,
    no: post-clip the outlier value is small and the std is small, so
    its z stays small). The diagnostic: |z_max| should be substantially
    smaller post-winsor than without."""
    D, N = 30, 50
    rng = np.random.default_rng(42)
    log_p = np.cumsum(
        rng.normal(0.0, 0.01, (D, N)).astype(np.float64), axis=0)
    prices = np.exp(log_p)
    # Inject a 5x gap on ticker 10 in the next-5-bar forward window
    # starting at bar 10.
    prices[15:, 10] *= 5.0

    z_clipped = forward_robust_z(prices, rebal_days=5, winsor=(0.1, 0.9))
    z_uncapped = forward_robust_z(prices, rebal_days=5, winsor=(0.0, 1.0))

    z_at_10_clipped = z_clipped[10]
    z_at_10_uncapped = z_uncapped[10]
    assert np.isfinite(z_at_10_clipped).all()
    assert np.isfinite(z_at_10_uncapped).all()
    # Without winsor, the outlier sits at the largest z (≥ 3-7 σ
    # depending on the cross-section). With winsor at (0.1, 0.9), the
    # outlier is clipped down to the 90th percentile of returns —
    # which is an *interior* return — so its z-score lies inside the
    # clipped band, comparable to ~1-2σ.
    assert abs(z_at_10_uncapped[10]) > 4.0, (
        f'expected uncapped outlier z >> 4 (got '
        f'{z_at_10_uncapped[10]:+.2f}) — control case for the winsor test')
    assert abs(z_at_10_clipped[10]) < 2.0, (
        f'winsor failed — outlier still at z={z_at_10_clipped[10]:+.2f}, '
        f'expected ≤ 2.0 after clip-to-q_hi')
    # Per-bar mean ≈ 0 and std ≈ 1 by construction (z-score on clipped).
    assert abs(z_at_10_clipped.mean()) < 0.05
    assert abs(z_at_10_clipped.std() - 1.0) < 1e-6


def test_forward_robust_z_zeroes_short_cross_sections():
    """Bars with <4 valid peers should output all-zero rows. The
    aux MSE then sees mask-times-zero diff = zero contribution — same
    behavior as `forward_sign_demeaned` does for <2 peers."""
    D, N = 30, 3   # only 3 tickers — below the 4-peer floor
    prices = 100.0 * np.exp(np.cumsum(
        np.random.default_rng(0).normal(0, 0.01, (D, N)), axis=0))
    z = forward_robust_z(prices, rebal_days=5)
    assert (z == 0).all(), 'short cross-sections should produce all-zeros'


def test_forward_robust_z_validates_winsor_args():
    prices = 100.0 * np.exp(np.cumsum(
        np.random.default_rng(0).normal(0, 0.01, (50, 10)), axis=0))
    with pytest.raises(ValueError, match='winsor'):
        forward_robust_z(prices, rebal_days=5, winsor=(0.5, 0.5))
    with pytest.raises(ValueError, match='winsor'):
        forward_robust_z(prices, rebal_days=5, winsor=(-0.1, 0.99))


def test_masked_mse_zeros_match_target():
    """Score = target on masked cells → MSE = 0."""
    target = np.array([[0.5, -0.5, 0.0],
                       [1.0, 0.0, -1.0]], dtype=np.float32)
    mask = np.ones_like(target)
    loss = masked_mse(Tensor(target), Tensor(target), Tensor(mask))
    assert abs(float(loss.item())) < 1e-6


def test_masked_mse_ignores_unmasked_cells():
    """Mask=0 cells should contribute zero gradient — putting NaN there
    should not break the loss (gradient-safe sanitization)."""
    target = np.array([[1.0, 2.0, np.nan],
                       [3.0, 4.0, np.nan]], dtype=np.float32)
    pred   = np.array([[1.0, 2.0, 999.0],
                       [3.0, 4.0, 999.0]], dtype=np.float32)
    mask   = np.array([[1.0, 1.0, 0.0],
                       [1.0, 1.0, 0.0]], dtype=np.float32)
    loss = masked_mse(Tensor(pred), Tensor(target), Tensor(mask))
    assert abs(float(loss.item())) < 1e-6


def test_masked_mse_averages_over_valid_cells():
    """4 valid cells with diff² ∈ {1, 4, 0, 1} → MSE = 6/4 = 1.5."""
    target = np.zeros((1, 4), dtype=np.float32)
    pred   = np.array([[1.0, 2.0, 0.0, -1.0]], dtype=np.float32)
    mask   = np.ones_like(target)
    loss = float(masked_mse(Tensor(pred), Tensor(target), Tensor(mask)).item())
    assert abs(loss - 1.5) < 1e-6


def test_init_mlp_multitask_builds_trunk_plus_two_heads():
    rng = np.random.default_rng(0)
    params = init_mlp_multitask(rng, hidden_flat=32, hidden=8, n_layers=2)
    # Trunk: W0 (32, 8), W1 (8, 8); heads: Wp (8, 1), Wa (8, 1).
    assert set(params.keys()) == {
        'W0', 'b0', 'W1', 'b1', 'Wp', 'bp', 'Wa', 'ba'}
    assert params['W0'].shape == (32, 8)
    assert params['W1'].shape == (8, 8)
    assert params['Wp'].shape == (8, 1)
    assert params['Wa'].shape == (8, 1)


def test_apply_mlp_multitask_returns_pair():
    rng = np.random.default_rng(0)
    params = init_mlp_multitask(rng, hidden_flat=16, hidden=4, n_layers=1)
    X = Tensor(np.random.default_rng(1).standard_normal(
        (3, 5, 16)).astype(np.float32))
    p, a = apply_mlp_multitask(params, X)
    assert p.shape == (3, 5)
    assert a.shape == (3, 5)
    # Two heads, different params, different outputs (modulo random init).
    diff = float((p - a).abs().sum().item())
    assert diff > 0.0, 'primary and aux outputs should differ at init'


def test_get_scorer_registers_mlp_multitask():
    init_fn, apply_fn = get_scorer('mlp_multitask')
    assert init_fn is init_mlp_multitask
    assert apply_fn is apply_mlp_multitask


def _make_synthetic_universe(n_tickers, n_bars, cfg):
    dates = pd.bdate_range('2000-01-03', periods=n_bars).to_numpy()
    out = []
    for j in range(n_tickers):
        rng = np.random.default_rng(j)
        prices = 100.0 * np.exp(np.cumsum(
            rng.normal(0.0002, 0.012, n_bars)))
        feats, valid = build_indicator_features(prices, cfg)
        out.append(TickerData(
            name=f'T{j}', prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ))
    return out


def test_walkforward_multitask_runs_and_records_aux_mse():
    """End-to-end smoke: random-walk universe, mlp_multitask head,
    aux_weight=0.1. Asserts the path runs, val IC is finite, and the
    aux MSE field is populated (vs NaN for the non-multitask path).
    Does not assert magnitudes — random walks should give val IC ≈ 0,
    which is the correct null result."""
    cfg = IndicatorGridConfig()
    tickers = _make_synthetic_universe(n_tickers=8, n_bars=5500, cfg=cfg)

    # Reuse the indicator-grid walkforward path to sanity-check
    # multitask wiring at a tiny universe / step count. The factor
    # __init__ exposes train_scorer_indicators_walkforward but it
    # internally uses train_scorer_walkforward via the identity
    # backbone, so assertions here exercise the multitask code path
    # in train_walkforward.py.
    from factor import train_scorer_indicators_walkforward
    res = train_scorer_indicators_walkforward(
        tickers, cfg,
        rebal_days=20,
        train_window_blocks=20,
        val_window_blocks=10,
        step_window_blocks=10,
        scorer='mlp_multitask',
        mlp_hidden=8, mlp_layers=1,
        n_steps=10, learning_rate=1e-2, weight_decay=1e-3,
        aux_weight=0.1,
        verbose=False,
    )

    assert res.scorer == 'mlp_multitask'
    assert res.n_windows >= 2
    for w in res.windows:
        assert np.isfinite(w.train_ic)
        assert np.isfinite(w.val_ic)
        # Multitask path populates the aux-MSE fields (default NaN).
        assert np.isfinite(w.train_aux_mse), (
            f'expected finite train_aux_mse, got {w.train_aux_mse}')
        assert np.isfinite(w.val_aux_mse), (
            f'expected finite val_aux_mse, got {w.val_aux_mse}')
        assert w.train_aux_mse >= 0.0
        assert w.val_aux_mse >= 0.0
        # Multitask params: trunk (W0, b0) + two heads (Wp/bp, Wa/ba).
        assert {'W0', 'b0', 'Wp', 'bp', 'Wa', 'ba'} <= set(w.head_params)


def test_walkforward_multitask_validates_aux_weight_and_scorer():
    """`aux_weight > 0` requires `scorer='mlp_multitask'`, and vice versa
    `scorer='mlp_multitask'` requires `aux_weight > 0`."""
    cfg = IndicatorGridConfig()
    tickers = _make_synthetic_universe(n_tickers=4, n_bars=5500, cfg=cfg)

    from factor import train_scorer_indicators_walkforward
    with pytest.raises(ValueError, match='requires scorer=mlp_multitask'):
        train_scorer_indicators_walkforward(
            tickers, cfg, rebal_days=20,
            train_window_blocks=20, val_window_blocks=10,
            scorer='linear', n_steps=2, aux_weight=0.1,
            verbose=False,
        )
    with pytest.raises(ValueError, match='requires aux_weight > 0'):
        train_scorer_indicators_walkforward(
            tickers, cfg, rebal_days=20,
            train_window_blocks=20, val_window_blocks=10,
            scorer='mlp_multitask', n_steps=2, aux_weight=0.0,
            verbose=False,
        )
