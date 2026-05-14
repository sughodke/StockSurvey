"""Smoke + correctness tests for the endogenous-horizon stack.

Covers:
  - `forward_log_returns_multi` shapes / edge-NaN behavior.
  - `per_bar_pearson_ic` matches scipy on a tiny case.
  - `horizon_mixture_loss` runs forward + backward (autograd path).
  - `init_mlp_horizon` + `apply_mlp_horizon` shape contract.
  - `simulate_irregular_daily_pnl` on a deterministic two-day case.
  - `train_scorer_horizon_walkforward` runs end-to-end on synthetic
    universe (no signal expected — just wiring).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tinygrad.tensor import Tensor

from factor import (
    IndicatorGridConfig, apply_mlp_horizon, build_indicator_features,
    forward_log_returns_multi, horizon_mixture_loss, init_mlp_horizon,
    make_indicator_backbone, per_bar_pearson_ic,
    simulate_fixed_horizon_daily_pnl, simulate_irregular_daily_pnl,
    train_scorer_horizon_walkforward,
)
from ss_features import TickerData


def test_forward_log_returns_multi_shape_and_edge_nans():
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (100, 4)), axis=0))
    horizons = (5, 10, 20)
    out = forward_log_returns_multi(prices, horizons=horizons)
    assert out.shape == (3, 100, 4)
    # Last h_k rows of slice k are NaN (no future window).
    for k, h in enumerate(horizons):
        assert np.isnan(out[k, -h:, :]).all(), \
            f'last {h} rows of horizon-{h} slice should be NaN'
        # Penultimate row at horizon h: log(p[h]) − log(p[0]) for k=0 etc.
        np.testing.assert_allclose(
            out[k, 0],
            np.log(prices[h]) - np.log(prices[0]),
            atol=1e-12,
        )


def test_per_bar_pearson_ic_matches_numpy():
    rng = np.random.default_rng(1)
    n_bars, n_tickers = 4, 8
    scores = rng.normal(size=(n_bars, n_tickers)).astype(np.float32)
    fwd = rng.normal(size=(n_bars, n_tickers)).astype(np.float32)
    mask = np.ones((n_bars, n_tickers), dtype=np.float32)
    # Mask out one ticker on the third bar — bar IC should still be
    # well-defined on the remaining 7.
    mask[2, 0] = 0.0

    ic_t, valid_t = per_bar_pearson_ic(
        Tensor(scores), Tensor(fwd), Tensor(mask))
    ic = ic_t.numpy()
    valid = valid_t.numpy()
    assert ic.shape == (n_bars,)
    assert valid.shape == (n_bars,)
    np.testing.assert_allclose(valid, [1, 1, 1, 1], atol=1e-6)

    # numpy reference per bar.
    for b in range(n_bars):
        m = mask[b].astype(bool)
        s = scores[b, m]
        r = fwd[b, m]
        ref = np.corrcoef(s, r)[0, 1]
        np.testing.assert_allclose(ic[b], ref, atol=1e-4)


def test_per_bar_pearson_ic_degenerate_bars_zero():
    # Single valid ticker → variance zero → IC zero, valid=0.
    scores = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    fwd    = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)  # constant target
    mask   = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    ic_t, valid_t = per_bar_pearson_ic(
        Tensor(scores), Tensor(fwd), Tensor(mask))
    assert float(ic_t.numpy()[0]) == 0.0
    assert float(valid_t.numpy()[0]) == 0.0


def test_horizon_mixture_loss_runs_forward_backward():
    rng = np.random.default_rng(2)
    n_bars, n_tickers, hidden_flat = 6, 8, 16
    K = 3
    horizons = (5, 10, 20)

    X = Tensor(rng.normal(size=(n_bars, n_tickers, hidden_flat)).astype(np.float32))
    base_mask = Tensor(np.ones((n_bars, n_tickers), dtype=np.float32))
    fwd_multi = Tensor(
        rng.normal(size=(K, n_bars, n_tickers)).astype(np.float32))
    mask_multi = Tensor(np.ones((K, n_bars, n_tickers), dtype=np.float32))

    params = init_mlp_horizon(
        rng, hidden_flat, n_horizons=K, hidden=8, n_layers=1)
    Tensor.training = True
    scores, pi = apply_mlp_horizon(params, X, base_mask)
    assert scores.shape == (n_bars, n_tickers)
    assert pi.shape == (n_bars, K)

    # Build + backward FIRST — tinygrad's `.numpy()` realizes a lazy
    # tensor and truncates its backward graph (same gotcha noted in
    # `train.py` for `.item()`). Realize for shape / value checks
    # only AFTER backward has propagated gradients into the params.
    loss = horizon_mixture_loss(
        scores, fwd_multi, mask_multi, pi, entropy_weight=0.0)
    loss.backward()
    np.testing.assert_allclose(
        pi.numpy().sum(axis=1), np.ones(n_bars), atol=1e-5)
    # All trainable params must have non-None gradients.
    for k, t in params.items():
        assert t.grad is not None, f'param {k} has no gradient'


def test_simulate_irregular_daily_pnl_deterministic_one_rebal():
    """Deterministic case: 1 ticker, 1 rebal, hold 5 days.

    With one ticker, softmax forces w=1.0 on it. Daily PnL each day is
    the daily log return. Total PnL over 5 days = sum of daily returns.
    Costs: full leverage on entry → commission_frac * 1.0 on day 0.
    """
    # 10 daily bars, 1 ticker. Daily log return = 0.01 each day.
    D, N = 10, 1
    daily_log_ret = np.full((D, N), 0.01, dtype=np.float64)
    # Single fine rebal bar at day 0.
    rebal_idx = np.array([0], dtype=np.int64)
    scores = np.array([[5.0]], dtype=np.float64)
    # One horizon option, all weight on it.
    pi = np.array([[1.0]], dtype=np.float64)
    mask = np.ones((1, N), dtype=np.float64)
    horizons = (5,)
    res = simulate_irregular_daily_pnl(
        scores=scores, pi=pi, mask=mask,
        daily_log_ret=daily_log_ret, rebal_idx=rebal_idx,
        horizons=horizons,
        daily_start=0, daily_end=10,
        commission_bps=10.0,    # 10 bps = 0.001 frac
        temperature=1.0)
    # Days 0..4 each accrue +0.01 (one ticker → w=1). Day 0 also has
    # the cost -0.001.
    assert res.n_rebals == 1
    assert res.rebal_log == [(0, 0, 5)]
    np.testing.assert_allclose(res.daily_pnl[0], 0.01 - 0.001, atol=1e-12)
    np.testing.assert_allclose(res.daily_pnl[1:5], 0.01, atol=1e-12)
    # Days 5..9: no rebal at day 5 (we exited the holding period at
    # day 5, then there are no more fine bars at day 5 or later within
    # the bar pool — so position holds at prev_w until daily_end).
    # The loop's gap-handler accrues prev_w over [5, daily_end).
    np.testing.assert_allclose(res.daily_pnl[5:], 0.01, atol=1e-12)


def test_simulate_fixed_horizon_matches_argmax_when_pi_is_onehot():
    rng = np.random.default_rng(3)
    D, N = 60, 3
    daily_log_ret = rng.normal(0, 0.01, (D, N)).astype(np.float64)
    # Fine rebal grid every 5 days.
    rebal_idx = np.arange(0, D - 20, 5, dtype=np.int64)
    n_bars = len(rebal_idx)
    scores = rng.normal(size=(n_bars, N)).astype(np.float64)
    mask = np.ones((n_bars, N), dtype=np.float64)
    horizons = (5, 10, 20)
    # pi is one-hot on horizon-20 for every bar.
    pi = np.zeros((n_bars, 3), dtype=np.float64)
    pi[:, 2] = 1.0

    irreg = simulate_irregular_daily_pnl(
        scores=scores, pi=pi, mask=mask,
        daily_log_ret=daily_log_ret, rebal_idx=rebal_idx,
        horizons=horizons,
        daily_start=0, daily_end=D,
        commission_bps=10.0, temperature=1.0)
    fixed = simulate_fixed_horizon_daily_pnl(
        scores=scores, mask=mask,
        daily_log_ret=daily_log_ret, rebal_idx=rebal_idx,
        horizon=20,
        daily_start=0, daily_end=D,
        commission_bps=10.0, temperature=1.0)
    # When pi is one-hot on h=20, the two should produce identical
    # daily PnL streams.
    np.testing.assert_allclose(irreg.daily_pnl, fixed.daily_pnl, atol=1e-12)
    np.testing.assert_allclose(irreg.sharpe, fixed.sharpe, atol=1e-9)


def test_train_scorer_horizon_walkforward_runs():
    """End-to-end smoke. Synthetic GBM universe, no signal expected.

    Asserts the pipeline runs and produces well-formed windows with
    sensible diagnostics (π distribution sums to 1, baselines present,
    no NaN Sharpes). Does not assert on whether endogenous beats
    fixed — random data shouldn't reveal an edge.
    """
    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    n_tickers = 4
    n_bars = 2000  # plenty for h_max=60 + warmup + a couple windows
    dates = pd.bdate_range('2000-01-03', periods=n_bars).to_numpy()
    tickers: list[TickerData] = []
    for j in range(n_tickers):
        rng = np.random.default_rng(j)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_bars)))
        feats, valid = build_indicator_features(prices, cfg)
        tickers.append(TickerData(
            name=f'T{j}', prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ))

    backbone = make_indicator_backbone(tickers, cfg)
    res = train_scorer_horizon_walkforward(
        tickers, backbone,
        horizons=(5, 10, 20),
        train_window_blocks=80, val_window_blocks=40, step_window_blocks=40,
        n_steps=20, learning_rate=1e-2, weight_decay=1e-3,
        mlp_hidden=8, mlp_layers=1,
        commission_bps=5.0, seed=0, verbose=False,
    )
    assert res.n_windows >= 1
    for w in res.windows:
        # Sharpes finite (not NaN).
        assert np.isfinite(w.val_endog_sharpe), \
            f'window {w.window_idx} endog Sharpe is NaN'
        assert np.isfinite(w.val_random_sharpe)
        for h in res.horizons:
            assert h in w.val_fixed_sharpes
            assert np.isfinite(w.val_fixed_sharpes[h])
        # Argmax counts sum to n_val_bars.
        total = sum(w.val_pi_argmax_counts.values())
        assert total == (w.val_block_end - w.val_block_start)
        # Mean entropy in [0, log K].
        assert 0.0 <= w.val_pi_entropy_mean <= np.log(len(res.horizons)) + 1e-3
        # Holding days reasonable.
        if w.val_endog_n_rebals > 0:
            assert min(res.horizons) <= w.val_endog_mean_holding <= max(res.horizons)
