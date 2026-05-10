"""End-to-end smoke for the pretrained-backbone path.

Builds a `Backbone` directly with random conv weights at small but
realistic shapes (kernel=3, hidden=8, n_layers=2, K=12, F=4 →
K_post=8, hidden_flat=64). Synthesises matching `TickerData` whose
features are shape `(T, K*F)` and runs `train_scorer` for both Stage 1
(frozen backbone, head only) and Stage 2 (joint head + backbone
fine-tune).

We synthesize the backbone instead of round-tripping through an npz on
purpose — the npz I/O is owned by `ss_features.load_backbone` and
covered by its own users; this file's job is to verify the tinygrad
runtime + training loop wiring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor import (
    Backbone, TrainResult, align_tickers, apply_backbone, backbone_to_pytree,
    precompute_inputs, predict, train_scorer,
)
from ss_features import TickerData

from tinygrad.tensor import Tensor


# Compact but real shapes — conv loop runs twice, K_post stays >= 1.
KERNEL = 3
HIDDEN = 8
N_LAYERS = 2
K = 12
F = 4
K_POST = K - N_LAYERS * (KERNEL - 1)        # 8
HIDDEN_FLAT = K_POST * HIDDEN               # 64
N_BARS = 600
N_TICKERS = 6
REBAL_DAYS = 10                             # 60 rebal blocks → 42 train / 18 val


def _synthetic_backbone(seed: int = 0) -> Backbone:
    rng = np.random.default_rng(seed)
    feat_mu = rng.normal(0, 0.1, (1, K, F)).astype(np.float32)
    feat_sd = (rng.uniform(0.5, 1.5, (1, K, F))).astype(np.float32)
    chs = [F] + [HIDDEN] * N_LAYERS
    conv_params: list[tuple[np.ndarray, np.ndarray]] = []
    for in_c, out_c in zip(chs[:-1], chs[1:]):
        # He-normal init so Stage 2 fine-tuning gets a reasonable start.
        scale = float(np.sqrt(2.0 / (KERNEL * in_c)))
        W = rng.normal(0, scale, (KERNEL, in_c, out_c)).astype(np.float32)
        b = np.zeros(out_c, dtype=np.float32)
        conv_params.append((W, b))
    return Backbone(
        feat_mu=feat_mu, feat_sd=feat_sd,
        conv_params=tuple(conv_params),
        K=K, F=F, hidden=HIDDEN, K_post=K_POST,
        kernel=KERNEL, n_layers=N_LAYERS,
    )


def _synthetic_universe() -> list[TickerData]:
    dates = pd.bdate_range('2010-01-04', periods=N_BARS).to_numpy()
    tickers: list[TickerData] = []
    for j in range(N_TICKERS):
        rng = np.random.default_rng(100 + j)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, N_BARS)))
        # Random features in the (T, K*F) layout the backbone expects;
        # `align_tickers(K, F)` will reshape per ticker.
        features = rng.normal(0, 1, (N_BARS, K * F)).astype(np.float32)
        valid = np.ones(N_BARS, dtype=bool)
        tickers.append(TickerData(
            name=f'T{j}', prices=prices, dates=dates,
            features=features, targets={}, valid=valid,
        ))
    return tickers


def test_backbone_shapes_self_consistent():
    bb = _synthetic_backbone()
    assert bb.K == K and bb.F == F
    assert bb.hidden == HIDDEN
    assert bb.K_post == K_POST
    assert bb.hidden_flat == HIDDEN_FLAT
    assert len(bb.conv_params) == N_LAYERS
    # Layer 0: (kernel, F, hidden); layer 1+: (kernel, hidden, hidden).
    assert bb.conv_params[0][0].shape == (KERNEL, F, HIDDEN)
    assert bb.conv_params[0][1].shape == (HIDDEN,)
    for W, b in bb.conv_params[1:]:
        assert W.shape == (KERNEL, HIDDEN, HIDDEN)
        assert b.shape == (HIDDEN,)


def test_apply_backbone_forward_shape():
    bb = _synthetic_backbone()
    n = 7
    x = Tensor(np.random.default_rng(0).normal(0, 1, (n, K, F)).astype(np.float32))
    out = apply_backbone(bb, x)
    assert tuple(out.shape) == (n, HIDDEN_FLAT)


def test_backbone_to_pytree_roundtrip():
    bb = _synthetic_backbone()
    pt = backbone_to_pytree(bb)
    # Norm stats stay frozen, conv weights enter the gradient graph.
    assert pt['feat_mu'].requires_grad is False
    assert pt['feat_sd'].requires_grad is False
    assert len(pt['conv']) == N_LAYERS
    for layer in pt['conv']:
        assert layer['W'].requires_grad is True
        assert layer['b'].requires_grad is True
    np.testing.assert_array_equal(pt['feat_mu'].numpy(), bb.feat_mu)
    np.testing.assert_array_equal(pt['feat_sd'].numpy(), bb.feat_sd)


def test_precompute_inputs_shapes():
    bb = _synthetic_backbone()
    tickers = _synthetic_universe()
    pre = precompute_inputs(tickers, bb, rebal_days=REBAL_DAYS)
    n_blocks = pre['representation_rb'].shape[0]
    # Latent and forward-return / mask all share the same rebal axis.
    assert pre['representation_rb'].shape == (n_blocks, N_TICKERS, HIDDEN_FLAT)
    assert pre['fwd_ret_rb'].shape == (n_blocks, N_TICKERS)
    assert pre['mask_rb'].shape == (n_blocks, N_TICKERS)
    assert pre['block_log_ret_rb'].shape == (n_blocks, N_TICKERS)
    assert n_blocks >= 4, 'need >= 4 rebal blocks for a sensible train/val split'


def test_train_scorer_stage1_runs():
    bb = _synthetic_backbone()
    tickers = _synthetic_universe()
    res = train_scorer(
        tickers, bb,
        rebal_days=REBAL_DAYS, train_frac=0.7, scorer='linear',
        n_steps=20, learning_rate=1e-3, weight_decay=1e-2,
        finetune_steps=0, verbose=False,
    )
    assert isinstance(res, TrainResult)
    # Linear head over hidden_flat input.
    assert res.params['W'].shape == (HIDDEN_FLAT,)
    assert res.params['b'].shape == (1,)
    assert len(res.train_history) == 20
    assert len(res.finetune_history) == 0      # Stage 2 disabled
    # Stage-1-only path returns the unmodified backbone.
    assert res.backbone_params['feat_mu'].shape == (1, K, F)
    assert len(res.backbone_params['conv']) == N_LAYERS
    np.testing.assert_array_equal(
        res.backbone_params['conv'][0]['W'], bb.conv_params[0][0])
    assert np.isfinite(res.train_ic) and np.isfinite(res.val_ic)
    assert np.isfinite(res.val_sharpe)


def test_train_scorer_stage2_finetunes_backbone():
    bb = _synthetic_backbone(seed=1)
    tickers = _synthetic_universe()
    res = train_scorer(
        tickers, bb,
        rebal_days=REBAL_DAYS, train_frac=0.7, scorer='linear',
        n_steps=10, learning_rate=1e-3, weight_decay=1e-2,
        finetune_steps=10, finetune_lr_scale=0.1, finetune_batch_bars=4,
        verbose=False,
    )
    # Stage 2 runs and logs.
    assert len(res.finetune_history) == 10
    # Backbone weights moved (Adam stepped them) — at least one conv W differs.
    moved = any(
        not np.array_equal(res.backbone_params['conv'][i]['W'],
                           bb.conv_params[i][0])
        for i in range(N_LAYERS)
    )
    assert moved, 'Stage 2 finetune did not update any backbone conv weights'


def test_predict_returns_full_grid():
    bb = _synthetic_backbone()
    tickers = _synthetic_universe()
    res = train_scorer(
        tickers, bb,
        rebal_days=REBAL_DAYS, train_frac=0.7, scorer='linear',
        n_steps=5, learning_rate=1e-3, weight_decay=1e-2,
        finetune_steps=0, verbose=False,
    )
    # predict()'s "full grid" semantics require daily-aligned features.
    # `res.aligned` from train_scorer is now rebal-subsampled (encoder
    # only runs on bars whose latents are actually consumed); callers
    # who want daily-frequency scores should re-align with `align_tickers`.
    aligned_daily = align_tickers(tickers, K=bb.K, F=bb.F)
    scores = predict(aligned_daily, bb, res.params, scorer='linear')
    D = aligned_daily.features.shape[0]
    assert scores.shape == (D, N_TICKERS)
    # Synthetic features are all-finite, so scores should be too.
    assert np.isfinite(scores).all()
