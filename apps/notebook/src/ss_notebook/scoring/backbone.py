"""Load + apply the conv backbone from a replay multi-head npz.

The npz produced by `ss-replay --decoder cnn` (see
`ss_notebook.replay.cli`) stores, per target, a self-contained dict with
the shared backbone weights duplicated under that target's prefix
(`{target}__feat_mu`, `{target}__feat_sd`, `{target}__conv{i}_W`,
`{target}__conv{i}_b`, plus the per-target head + target standardizer).

For scoring we keep only the backbone (`feat_mu/sd` + `conv{i}_W/b`)
and drop the per-target head/target_mu/sd. The result is a frozen JAX
forward function that maps `(n_samples, K, F)` → `(n_samples, K_post *
hidden)`, exactly the input the per-target head saw inside the replay
trainer.

`load_backbone` also returns the metadata blob so callers can rebuild
the input feature stack with matching scales / window_cols /
include_zscore_stats / include_returns / lookback / rsi_n / etc.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Backbone:
    """Frozen conv-backbone weights extracted from a replay multi-head npz."""
    feat_mu: jax.Array            # (1, K, F) input z-norm mean
    feat_sd: jax.Array            # (1, K, F) input z-norm std
    conv_params: tuple[tuple[jax.Array, jax.Array], ...]   # ((W, b), ...)
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

    feat_mu = base_arrays['feat_mu']
    feat_sd = base_arrays['feat_sd']
    conv_W_keys = sorted(k for k in base_arrays
                         if k.startswith('conv') and k.endswith('_W'))
    conv_params = tuple(
        (jnp.asarray(base_arrays[kw]), jnp.asarray(base_arrays[kw[:-2] + '_b']))
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
            feat_mu=jnp.asarray(feat_mu),
            feat_sd=jnp.asarray(feat_sd),
            conv_params=conv_params,
            K=K, F=F, hidden=hidden, K_post=K_post,
            kernel=kernel, n_layers=n_layers,
        ),
        meta,
    )


def _conv1d(x: jax.Array, W: jax.Array, b: jax.Array) -> jax.Array:
    return jax.lax.conv_general_dilated(
        x, W,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=('NHC', 'HIO', 'NHC'),
    ) + b


def apply_backbone(bb: Backbone, X: jax.Array) -> jax.Array:
    """Run the frozen backbone over `X` of shape `(n, K, F)`.

    Mirrors `fit_cnn_multihead`'s internal forward pass: input z-norm,
    n_layers of (Conv1D + ReLU) with VALID padding, then flatten the
    `(K_post, hidden)` activations into a single `K_post * hidden` row.
    """
    h = (X - bb.feat_mu) / bb.feat_sd
    for W, b in bb.conv_params:
        h = jax.nn.relu(_conv1d(h, W, b))
    return h.reshape(h.shape[0], -1)
