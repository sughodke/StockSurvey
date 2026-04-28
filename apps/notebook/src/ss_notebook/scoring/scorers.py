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

import jax
import jax.numpy as jnp


def init_linear(key: jax.Array, hidden_flat: int) -> dict[str, jax.Array]:
    """He-init linear head — `W: (hidden_flat,)`, `b: ()`."""
    W = jax.random.normal(
        key, (hidden_flat,), dtype=jnp.float32) * (2.0 / hidden_flat) ** 0.5
    return {'W': W, 'b': jnp.zeros((), dtype=jnp.float32)}


def apply_linear(params: dict[str, jax.Array], X: jax.Array) -> jax.Array:
    """`X` shape `(..., hidden_flat)` → scores shape `(...)`."""
    return X @ params['W'] + params['b']


def init_mlp(
    key: jax.Array, hidden_flat: int, *,
    hidden: int = 64, n_layers: int = 1,
) -> dict[str, jax.Array]:
    """`n_layers` ReLU hidden blocks of width `hidden`, then a 1-d head.

    Stored as flat `W{i}/b{i}` keys so the dict is a clean JAX pytree
    and both `apply_mlp` and `optax.adam` see the same structure.
    """
    if n_layers < 1:
        raise ValueError(f'n_layers must be >= 1, got {n_layers}')
    sizes = [hidden_flat] + [hidden] * n_layers + [1]
    params: dict[str, jax.Array] = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, sub = jax.random.split(key)
        params[f'W{i}'] = jax.random.normal(
            sub, (fan_in, fan_out),
            dtype=jnp.float32) * (2.0 / fan_in) ** 0.5
        params[f'b{i}'] = jnp.zeros(fan_out, dtype=jnp.float32)
    return params


def apply_mlp(params: dict[str, jax.Array], X: jax.Array) -> jax.Array:
    """`X` shape `(..., hidden_flat)` → scores shape `(...)`.

    Infers depth from the params dict — every `W{i}` key contributes a
    layer, the highest `i` is the linear output. Dict structure is
    static under jit (pytree shape is part of the trace key), so this
    Python loop is fine.
    """
    n_W = sum(1 for k in params if k.startswith('W'))
    n_layers = n_W - 1
    h = X
    for i in range(n_layers):
        h = jax.nn.relu(h @ params[f'W{i}'] + params[f'b{i}'])
    h = h @ params[f'W{n_layers}'] + params[f'b{n_layers}']
    return h.squeeze(-1)


SCORERS: dict[str, tuple] = {
    'linear': (init_linear, apply_linear),
    'mlp': (init_mlp, apply_mlp),
}


def get_scorer(name: str) -> tuple:
    """Return `(init_fn, apply_fn)` for a registered scorer kind."""
    if name not in SCORERS:
        raise ValueError(
            f'unknown scorer {name!r}; valid: {sorted(SCORERS)}')
    return SCORERS[name]
