"""Scoring heads on top of a frozen backbone.

Two variants:

- `linear`: single linear map `R^hidden_flat -> R`. ~`hidden_flat + 1`
  params. The minimal head — let it overfit fast and tell us whether
  the backbone alone carries any predictive information.
- `mlp`:    tiny ReLU MLP `hidden_flat -> hidden -> ... -> 1`. Use only
  if linear undershoots train IC noticeably; the backbone already does
  the heavy non-linear work.

Both expose the same call shape: `apply(params, X) -> scores` where `X`
is `(..., hidden_flat)` and the trailing dim is contracted away.
"""
from __future__ import annotations

import numpy as np

from tinygrad.tensor import Tensor


def _he_normal(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> Tensor:
    arr = rng.standard_normal(shape).astype(np.float32) * (2.0 / fan_in) ** 0.5
    return Tensor(arr, requires_grad=True)


def init_linear(rng: np.random.Generator, hidden_flat: int) -> dict[str, Tensor]:
    """He-init linear head — `W: (hidden_flat,)`, `b: (1,)`.

    `b` is shape `(1,)` rather than scalar because tinygrad's Adam
    materializes per-parameter momentum buffers at the param's shape and
    then assigns back; assigning a `(1,)` momentum into a `()` storage
    fails with a broadcast-to-fewer-dimensions error. The extra unit
    dim broadcasts cleanly inside `apply_linear`'s `X @ W + b`.
    """
    W = _he_normal(rng, (hidden_flat,), hidden_flat)
    b = Tensor(np.zeros((1,), dtype=np.float32), requires_grad=True)
    return {'W': W, 'b': b}


def apply_linear(params: dict[str, Tensor], X: Tensor) -> Tensor:
    """`X` shape `(..., hidden_flat)` → scores shape `(...)`."""
    out = X @ params['W']
    return out + params['b'].squeeze()


def init_mlp(
    rng: np.random.Generator, hidden_flat: int, *,
    hidden: int = 64, n_layers: int = 1,
) -> dict[str, Tensor]:
    """`n_layers` ReLU hidden blocks of width `hidden`, then a 1-d head.

    Stored as flat `W{i}/b{i}` keys so callers can treat the dict as a
    flat parameter list (matches what tinygrad optimizers consume).
    """
    if n_layers < 1:
        raise ValueError(f'n_layers must be >= 1, got {n_layers}')
    sizes = [hidden_flat] + [hidden] * n_layers + [1]
    params: dict[str, Tensor] = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f'W{i}'] = _he_normal(rng, (fan_in, fan_out), fan_in)
        params[f'b{i}'] = Tensor(np.zeros(fan_out, dtype=np.float32),
                                 requires_grad=True)
    return params


def apply_mlp(params: dict[str, Tensor], X: Tensor) -> Tensor:
    """`X` shape `(..., hidden_flat)` → scores shape `(...)`.

    Infers depth from the params dict — every `W{i}` key contributes a
    layer, the highest `i` is the linear output.
    """
    n_W = sum(1 for k in params if k.startswith('W'))
    n_layers = n_W - 1
    h = X
    for i in range(n_layers):
        h = (h @ params[f'W{i}'] + params[f'b{i}']).relu()
    h = h @ params[f'W{n_layers}'] + params[f'b{n_layers}']
    return h.squeeze(-1)


def init_mlp_multitask(
    rng: np.random.Generator, hidden_flat: int, *,
    hidden: int = 64, n_layers: int = 1,
) -> dict[str, Tensor]:
    """Shared trunk + two parallel scalar output heads (primary + aux).

    Trunk is the same `n_layers` × `hidden` ReLU MLP that `init_mlp`
    builds (`W{i}/b{i}` keys for `i in [0, n_layers)`); two extra
    `(hidden, 1)` projections (`Wp/bp` primary, `Wa/ba` aux) sit on top.

    Why a separate scorer kind: the multi-task gradient flow is the
    whole point. Aux loss flows back through `Wa/ba` into the trunk,
    shaping the representation the primary head's `Wp` then consumes.
    Two fully independent heads on the same frozen-backbone latent
    would not share gradients (Stage 1 freezes the conv stack).
    """
    if n_layers < 1:
        raise ValueError(f'n_layers must be >= 1, got {n_layers}')
    sizes = [hidden_flat] + [hidden] * n_layers
    params: dict[str, Tensor] = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f'W{i}'] = _he_normal(rng, (fan_in, fan_out), fan_in)
        params[f'b{i}'] = Tensor(np.zeros(fan_out, dtype=np.float32),
                                 requires_grad=True)
    params['Wp'] = _he_normal(rng, (hidden, 1), hidden)
    params['bp'] = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
    params['Wa'] = _he_normal(rng, (hidden, 1), hidden)
    params['ba'] = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
    return params


def apply_mlp_multitask(
    params: dict[str, Tensor], X: Tensor,
) -> tuple[Tensor, Tensor]:
    """`X` shape `(..., hidden_flat)` → `(scores_primary, scores_aux)`.

    Trunk depth inferred from `W{i}` keys excluding the two output
    projections (`Wp`, `Wa`).
    """
    n_trunk = sum(1 for k in params
                  if k.startswith('W') and k not in ('Wp', 'Wa'))
    h = X
    for i in range(n_trunk):
        h = (h @ params[f'W{i}'] + params[f'b{i}']).relu()
    p = (h @ params['Wp'] + params['bp']).squeeze(-1)
    a = (h @ params['Wa'] + params['ba']).squeeze(-1)
    return p, a


SCORERS: dict[str, tuple] = {
    'linear': (init_linear, apply_linear),
    'mlp': (init_mlp, apply_mlp),
    'mlp_multitask': (init_mlp_multitask, apply_mlp_multitask),
}


def get_scorer(name: str) -> tuple:
    """Return `(init_fn, apply_fn)` for a registered scorer kind."""
    if name not in SCORERS:
        raise ValueError(
            f'unknown scorer {name!r}; valid: {sorted(SCORERS)}')
    return SCORERS[name]
