"""Tinygrad runtime for the SSL-pretrained CNN backbone.

The on-disk npz format and `Backbone` dataclass live in
`ss_features.backbone_io` (numpy-only, no tinygrad dep) so the apps
that read these files — apps/notebook (writer + SSL probe in
`replay.reconstruct`) and apps/factor (reader for IC scoring) — can
share without depending on each other. This module owns the forward
pass: input z-norm, `n_layers` of (Conv1D + ReLU) with VALID padding,
flatten → `(n_samples, K_post * hidden)` exactly as the per-target
head saw it inside the replay trainer.

`Backbone` and `load_backbone` are re-exported here so callers can keep
doing `from factor import Backbone, load_backbone`.
"""
from __future__ import annotations

import numpy as np

from tinygrad.tensor import Tensor

from ss_features import Backbone, load_backbone
from ss_tg_ops import conv1d_nhc


def identity_backbone(
    K: int, F: int, *,
    feat_mu: np.ndarray | None = None,
    feat_sd: np.ndarray | None = None,
) -> Backbone:
    """Synthetic backbone whose `apply` is z-norm + flatten — no conv stack.

    Use as a no-encoder baseline: the scoring head reads directly off the
    flattened raw CWT bundle (`K * F` per bar), no learned compression.
    Tells you the floor — if SSL+linear can't beat it, the encoder
    isn't earning its keep.

    `feat_mu` / `feat_sd` default to 0 / 1 (skip z-norm). Pass per-cell
    stats `(1, K, F)` computed from the training pool to z-norm the
    input — Adam trains more cleanly when feature scales are matched.
    """
    if feat_mu is None:
        mu = np.zeros((1, K, F), dtype=np.float32)
    else:
        mu = np.asarray(feat_mu, dtype=np.float32).reshape(1, K, F)
    if feat_sd is None:
        sd = np.ones((1, K, F), dtype=np.float32)
    else:
        sd = np.asarray(feat_sd, dtype=np.float32).reshape(1, K, F)
    return Backbone(
        feat_mu=mu,
        feat_sd=sd,
        conv_params=(),     # empty — apply_backbone's conv loop is a no-op
        K=K,
        F=F,
        hidden=F,           # set so K_post * hidden = K * F
        K_post=K,
        kernel=1,           # unused (no conv layers)
        n_layers=0,
    )


def compute_input_stats(
    tickers, K: int, F: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool every ticker's valid feature rows and return per-cell
    `(feat_mu, feat_sd)` of shape `(1, K, F)`. Mirrors what
    `fit_cnn_multihead` does internally so the no-backbone baseline
    sees the same input distribution treatment as the supervised path.
    """
    rows = []
    for d in tickers:
        feats = d.features[d.valid]
        if feats.size == 0:
            continue
        rows.append(feats.reshape(-1, K, F))
    if not rows:
        raise ValueError('compute_input_stats: no valid feature rows across '
                         'the supplied ticker list')
    pool = np.vstack(rows).astype(np.float32)
    mu = pool.mean(axis=0, keepdims=True)
    sd = pool.std(axis=0, keepdims=True) + 1e-8
    return mu, sd


def apply_backbone(bb: Backbone, X: Tensor) -> Tensor:
    """Run the frozen backbone over `X` of shape `(n, K, F)`.

    Mirrors `fit_cnn_multihead`'s internal forward pass: input z-norm,
    `n_layers` of (Conv1D + ReLU) with VALID padding, then flatten the
    `(K_post, hidden)` activations into a single `K_post * hidden` row.

    Convolutional weights are wrapped fresh on each call — this is the
    frozen-inference path, so wrap-cost is amortized over a batch and we
    don't need to keep parameters as Tensors (save VRAM headroom for
    callers that hold many backbones).
    """
    feat_mu = Tensor(bb.feat_mu)
    feat_sd = Tensor(bb.feat_sd)
    h = (X - feat_mu) / feat_sd
    for W_np, b_np in bb.conv_params:
        W = Tensor(W_np)
        b = Tensor(b_np)
        h = conv1d_nhc(h, W, b).relu()
    return h.reshape(h.shape[0], -1)


def backbone_to_pytree(bb: Backbone) -> dict:
    """Pack a `Backbone` into a tinygrad-trainable pytree of `Tensor`s.

    Used by the fine-tuning path in `scoring.train` so the optimizer
    can update backbone weights through `loss.backward()`. Layout:
      - `'feat_mu'`, `'feat_sd'`: input z-norm, NOT optimized (kept
        fixed across fine-tuning so the backbone keeps seeing inputs
        with the same distribution it was pretrained on).
      - `'conv'`: list of `{'W', 'b'}` dicts, one per conv layer.
        Each tensor has `requires_grad=True` so they enter the
        optimizer's parameter list.
    """
    return {
        'feat_mu': Tensor(bb.feat_mu, requires_grad=False),
        'feat_sd': Tensor(bb.feat_sd, requires_grad=False),
        'conv': [{'W': Tensor(W, requires_grad=True),
                  'b': Tensor(b, requires_grad=True)}
                 for W, b in bb.conv_params],
    }


def apply_backbone_pytree(bb_params: dict, X: Tensor) -> Tensor:
    """Differentiable backbone forward over `X` of shape `(n, K, F)`.

    Same forward as `apply_backbone` but reads weights from a pytree
    dict so tinygrad autograd flows back into them. Returns a flattened
    `(n, K_post * hidden)` representation matrix.
    """
    h = (X - bb_params['feat_mu']) / bb_params['feat_sd']
    for layer in bb_params['conv']:
        h = conv1d_nhc(h, layer['W'], layer['b']).relu()
    return h.reshape(h.shape[0], -1)
