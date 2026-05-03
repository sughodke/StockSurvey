"""Frozen-backbone weight container + npz I/O.

The npz produced by `ss-replay --decoder cnn` (apps/notebook) stores,
per target, a self-contained dict with the shared backbone weights
duplicated under that target's prefix
(`{target}__feat_mu`, `{target}__feat_sd`, `{target}__conv{i}_W`,
`{target}__conv{i}_b`, plus the per-target head + target standardizer).

This module owns the on-disk format and the numpy-only data class so
both apps that touch it — apps/notebook (writer + SSL probe in
`replay.reconstruct`) and apps/factor (reader for IC scoring) — can
share without depending on each other. The runtime forward pass
(tinygrad) lives in `factor.backbone`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Backbone:
    """Frozen conv-backbone weights extracted from a replay multi-head npz.

    Fields are numpy arrays for portability — wrap in a runtime tensor
    type at forward call sites. The conv weight layout matches the JAX
    trainer: `(kernel, in_c, out_c)` ('HIO').
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
