"""Load + apply the conv backbone from a replay multi-head npz.

The npz produced by `ss-replay --decoder cnn` (see
`ss_notebook.replay.cli`) stores, per target, a self-contained dict with
the shared backbone weights duplicated under that target's prefix
(`{target}__feat_mu`, `{target}__feat_sd`, `{target}__conv{i}_W`,
`{target}__conv{i}_b`, plus the per-target head + target standardizer).

For scoring we keep only the backbone (`feat_mu/sd` + `conv{i}_W/b`)
and drop the per-target head/target_mu/sd. The result is a frozen
forward function that maps `(n_samples, K, F)` → `(n_samples, K_post *
hidden)`, exactly the input the per-target head saw inside the replay
trainer.

`load_backbone` also returns the metadata blob so callers can rebuild
the input feature stack with matching scales / window_cols /
include_zscore_stats / include_returns / lookback / rsi_n / etc.

Backbone weights live as numpy arrays in the dataclass — they're
materialized as `Tensor` only at forward time (so callers can hold
many backbones cheaply, and pretrain npz layout stays the source of
truth).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad import dtypes


@dataclass(frozen=True)
class Backbone:
    """Frozen conv-backbone weights extracted from a replay multi-head npz.

    Fields are numpy arrays for portability — wrap in `Tensor` at
    forward call sites. The conv weight layout matches the JAX trainer:
    `(kernel, in_c, out_c)` ('HIO').
    """
    feat_mu: np.ndarray            # (1, K, F) input z-norm mean
    feat_sd: np.ndarray            # (1, K, F) input z-norm std
    conv_params: tuple[tuple[np.ndarray, np.ndarray], ...]   # ((W, b), ...)
    K: int                        # window_cols (input lag count)
    F: int                        # channels per lag
    hidden: int                   # conv channel count
    K_post: int                   # post-conv lag count = K - n_layers*(kernel-1)
    kernel: int
    n_layers: int

    @property
    def hidden_flat(self) -> int:
        """Flattened backbone-output width — head input dimension."""
        return self.K_post * self.hidden


def load_backbone(npz_path: str | Path) -> tuple[Backbone, dict]:
    """Load conv backbone + metadata from a replay multi-head npz.

    Verifies that every per-target prefix carries identical backbone
    tensors (the multi-head trainer duplicates the shared backbone under
    each prefix, so any drift indicates the file was hand-edited or
    produced by an incompatible writer).
    """
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z.files)
    if '_meta' not in keys:
        raise ValueError(f'{npz_path}: missing _meta blob')
    meta = json.loads(str(z['_meta']))

    prefixes = sorted({k.split('__', 1)[0] + '__' for k in keys
                       if '__' in k and not k.startswith('_')})
    if not prefixes:
        raise ValueError(f'{npz_path}: no target prefixes found')

    def _backbone_keys(prefix: str) -> list[str]:
        skip = (prefix + 'head_', prefix + 'target_')
        return sorted(
            k for k in keys
            if k.startswith(prefix) and not any(k.startswith(s) for s in skip))

    base = prefixes[0]
    base_arrays = {k[len(base):]: z[k] for k in _backbone_keys(base)}
    for p in prefixes[1:]:
        for unprefixed, arr in base_arrays.items():
            other = z[p + unprefixed]
            if not np.array_equal(arr, other):
                raise ValueError(
                    f'{npz_path}: backbone tensor {unprefixed!r} differs '
                    f'between prefixes {base!r} and {p!r}')

    feat_mu = np.asarray(base_arrays['feat_mu'], dtype=np.float32)
    feat_sd = np.asarray(base_arrays['feat_sd'], dtype=np.float32)
    conv_W_keys = sorted(k for k in base_arrays
                         if k.startswith('conv') and k.endswith('_W'))
    conv_params = tuple(
        (np.asarray(base_arrays[kw], dtype=np.float32),
         np.asarray(base_arrays[kw[:-2] + '_b'], dtype=np.float32))
        for kw in conv_W_keys
    )

    n_layers = len(conv_params)
    if n_layers == 0:
        raise ValueError(f'{npz_path}: no conv{{i}}_W tensors found')
    kernel, _, hidden = conv_params[0][0].shape
    K = feat_mu.shape[1]
    F = feat_mu.shape[2]
    K_post = K - n_layers * (kernel - 1)
    if K_post <= 0:
        raise ValueError(
            f'{npz_path}: derived K_post={K_post} <= 0 from K={K}, '
            f'n_layers={n_layers}, kernel={kernel} — backbone shapes '
            'inconsistent.')

    return (
        Backbone(
            feat_mu=feat_mu,
            feat_sd=feat_sd,
            conv_params=conv_params,
            K=K, F=F, hidden=hidden, K_post=K_post,
            kernel=kernel, n_layers=n_layers,
        ),
        meta,
    )


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


def _conv1d(x: Tensor, W: Tensor, b: Tensor) -> Tensor:
    """`x` is `(B, L, Cin)` (NHC), `W` is `(kernel, Cin, Cout)` (HIO).

    Tinygrad's `Tensor.conv2d` expects `(B, Cin, ...)`, so we transpose
    in/out and reshape weights to `(Cout, Cin, kernel)` to match.
    """
    x_bcl = x.permute(0, 2, 1)                     # (B, Cin, L)
    W_oik = W.permute(2, 1, 0)                     # (Cout, Cin, kernel)
    y_bcl = x_bcl.conv2d(W_oik)                    # (B, Cout, L_post)
    return y_bcl.permute(0, 2, 1) + b              # (B, L_post, Cout)


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
        h = _conv1d(h, W, b).relu()
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
        h = _conv1d(h, layer['W'], layer['b']).relu()
    return h.reshape(h.shape[0], -1)
