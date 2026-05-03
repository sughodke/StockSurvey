"""End-to-end smoke for the deterministic-indicator path.

Runs the full pipeline (`build_indicator_features` →
`make_indicator_backbone` → `train_scorer_indicators`) on a synthetic
log-normal random-walk universe. The 5500-bar length is chosen to clear
the largest CCI cell's warmup at the default `IndicatorGridConfig`
(`(n-1)*w + 1 = 4978` bars for n=80, w=63).

Asserts shapes and that the optimizer ran — does not assert on IC
magnitude. Random walks have no exploitable cross-sectional structure
so val IC ≈ 0 is the correct outcome; tightening that to a numeric
bound just adds flakiness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor import (
    IndicatorGridConfig, build_indicator_features, make_indicator_backbone,
    train_scorer_indicators,
)
from ss_features import TickerData


def _make_synthetic_universe(n_tickers: int, n_bars: int, cfg: IndicatorGridConfig,
                             ) -> list[TickerData]:
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
    return tickers


def test_indicator_features_shape_and_warmup():
    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    assert F == len(cfg.channel_names()), \
        'channel_names() must enumerate exactly feature_width() channels'
    assert F == 79, \
        f'default config width drifted: expected 79, got {F}'

    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 5500)))
    feats, valid = build_indicator_features(prices, cfg)

    assert feats.shape == (5500, F)
    assert feats.dtype == np.float32
    assert valid.shape == (5500,)
    assert valid.dtype == bool
    # Largest CCI cell needs (80-1)*63 + 1 = 4978 bars; first valid index
    # at most that many bars in.
    first_valid = int(np.argmax(valid)) if valid.any() else len(valid)
    assert first_valid <= 4978, \
        f'first valid bar at {first_valid} > 4978 — warmup logic regressed'
    assert valid.any(), \
        '5500 bars should leave at least one fully-valid row'


def test_identity_backbone_shape_matches_cfg():
    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    tickers = _make_synthetic_universe(n_tickers=4, n_bars=5500, cfg=cfg)
    bb = make_indicator_backbone(tickers, cfg)

    assert bb.K == 1
    assert bb.F == F
    assert bb.hidden == F           # no compression
    assert bb.K_post == 1           # no conv layers eat lags
    assert bb.n_layers == 0
    assert bb.conv_params == ()
    assert bb.hidden_flat == F      # head input width == feature width
    assert bb.feat_mu.shape == (1, 1, F)
    assert bb.feat_sd.shape == (1, 1, F)
    # Pool z-norm produces non-degenerate stats on the synthetic pool.
    assert np.all(bb.feat_sd > 0)


def test_train_scorer_indicators_linear_runs():
    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    tickers = _make_synthetic_universe(n_tickers=4, n_bars=5500, cfg=cfg)

    res = train_scorer_indicators(
        tickers, cfg,
        rebal_days=20, train_frac=0.7, scorer='linear',
        n_steps=20, learning_rate=1e-2, weight_decay=1e-3,
        finetune_steps=0, verbose=False,
    )

    # Linear head: W:(F,), b:(1,)
    assert set(res.params) == {'W', 'b'}
    assert res.params['W'].shape == (F,)
    assert res.params['b'].shape == (1,)
    assert res.params['W'].dtype == np.float32

    # Optimizer state moved (head is not still all-zeros) and ran every step.
    assert np.any(res.params['W'] != 0.0), 'head W remained at init — Adam did not step'
    assert len(res.train_history) == 20
    assert res.n_train_bars >= 2 and res.n_val_bars >= 2

    # IC values are finite (no NaN propagation through the loss).
    assert np.isfinite(res.train_ic)
    assert np.isfinite(res.val_ic)
    assert np.isfinite(res.val_sharpe)


def test_train_scorer_indicators_mlp_runs():
    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    tickers = _make_synthetic_universe(n_tickers=4, n_bars=5500, cfg=cfg)

    res = train_scorer_indicators(
        tickers, cfg,
        rebal_days=20, train_frac=0.7, scorer='mlp',
        mlp_hidden=32, mlp_layers=1,
        n_steps=10, learning_rate=1e-3, weight_decay=1e-3,
        finetune_steps=0, verbose=False,
    )

    # 1-hidden-layer MLP: input F -> 32 -> 1.
    assert res.params['W0'].shape == (F, 32)
    assert res.params['b0'].shape == (32,)
    assert res.params['W1'].shape == (32, 1)
    assert res.params['b1'].shape == (1,)
    assert np.isfinite(res.train_ic)
    assert np.isfinite(res.val_ic)
