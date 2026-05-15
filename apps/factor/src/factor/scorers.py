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


def init_mlp_horizon(
    rng: np.random.Generator, hidden_flat: int, *,
    n_horizons: int, hidden: int = 64, n_layers: int = 1,
) -> dict[str, Tensor]:
    """Shared trunk + per-ticker score head + per-bar horizon-distribution head.

    Trunk: `n_layers` ReLU blocks of width `hidden`
    (`W{i}/b{i}` keys for `i in [0, n_layers)`).
    Score head: `(hidden, 1)` (`Ws/bs`) — produces a scalar per ticker
    just like `init_mlp`'s final layer.
    Horizon head: `(hidden, n_horizons)` (`Wh/bh`) — consumes the
    per-bar **cross-sectional mean** of the trunk's hidden activations
    (computed in `apply_mlp_horizon`, restricted to the masked liquid
    universe) and emits K logits which `apply_mlp_horizon` softmaxes to
    π_t.

    Why a cross-sectional mean: the horizon decision is a per-bar
    *market-state* output, not per-ticker. Pooling over the liquid
    universe builds an aggregate representation of "what does the
    market look like right now" before the K-way logit. This is the
    only way the horizon head's output dimension matches the trainer's
    per-bar π_t expectation.

    Why a separate registry slot is *not* used: the scorer apply
    contract elsewhere is `apply(params, X) -> scalar_scores`. The
    horizon head needs a mask to mean over the liquid set, so its
    apply signature is different — kept out of `SCORERS` and called
    directly by the horizon trainer.
    """
    if n_layers < 1:
        raise ValueError(f'n_layers must be >= 1, got {n_layers}')
    if n_horizons < 2:
        raise ValueError(
            f'n_horizons must be >= 2, got {n_horizons} '
            '(degenerate single-horizon model — just use train_scorer)')
    sizes = [hidden_flat] + [hidden] * n_layers
    params: dict[str, Tensor] = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f'W{i}'] = _he_normal(rng, (fan_in, fan_out), fan_in)
        params[f'b{i}'] = Tensor(np.zeros(fan_out, dtype=np.float32),
                                 requires_grad=True)
    # Score head — per-ticker scalar.
    params['Ws'] = _he_normal(rng, (hidden, 1), hidden)
    params['bs'] = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
    # Horizon head — per-bar K-way logits.
    params['Wh'] = _he_normal(rng, (hidden, n_horizons), hidden)
    params['bh'] = Tensor(np.zeros(n_horizons, dtype=np.float32),
                          requires_grad=True)
    return params


def apply_mlp_horizon_full(
    params: dict[str, Tensor], X: Tensor, mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Variant of `apply_mlp_horizon` that ALSO returns the horizon-head
    logits (pre-softmax).

    Why: in tinygrad, calling `.numpy()` on the softmax output (`pi`)
    truncates the autograd graph back to the horizon-head params
    (`Wh`/`bh`) even when the `.numpy()` call is on a `.detach()`ed
    copy. The REINFORCE training path needs to materialize the policy
    probabilities to do categorical sampling — materializing the
    pre-softmax logits is autograd-safe; the gradient still flows
    through `pi → logits → Wh/bh` correctly in subsequent loss
    computations. Callers can materialize `logits.numpy()` for sampling
    and still use `pi` in the loss.
    """
    n_trunk = sum(1 for k in params
                  if k.startswith('W') and k not in ('Ws', 'Wh'))
    h = X
    for i in range(n_trunk):
        h = (h @ params[f'W{i}'] + params[f'b{i}']).relu()
    scores = (h @ params['Ws'] + params['bs']).squeeze(-1)

    mask_b = mask.reshape(*mask.shape, 1)
    counts = mask.sum(axis=1, keepdim=True).maximum(1.0)
    pooled = (h * mask_b).sum(axis=1) / counts

    logits = pooled @ params['Wh'] + params['bh']
    logits_centered = logits - logits.max(axis=1, keepdim=True)
    exp_l = logits_centered.exp()
    pi = exp_l / (exp_l.sum(axis=1, keepdim=True) + 1e-12)
    return scores, pi, logits


def apply_mlp_horizon(
    params: dict[str, Tensor], X: Tensor, mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """`X` shape `(B, N, hidden_flat)`, `mask` shape `(B, N)`
    → `(scores: (B, N), pi: (B, K))`.

    Trunk runs over every `(bar, ticker)` cell. Score head projects to
    one scalar per ticker. Horizon head pools across the *liquid*
    cross-section per bar (masked mean of trunk hidden) and projects to
    K logits, softmaxed.

    Masked tickers contribute 0 to the cross-sectional sum and are
    excluded from the count. Bars with no liquid tickers fall back to a
    uniform-pi (loss handles via `bar_valid` zeroing; this just keeps
    softmax well-defined).

    Trunk depth is inferred from `W{i}` keys excluding `Ws`/`Wh` (the
    two output projections).
    """
    n_trunk = sum(1 for k in params
                  if k.startswith('W') and k not in ('Ws', 'Wh'))
    h = X
    for i in range(n_trunk):
        h = (h @ params[f'W{i}'] + params[f'b{i}']).relu()
    # h: (B, N, hidden)
    scores = (h @ params['Ws'] + params['bs']).squeeze(-1)   # (B, N)

    # Masked cross-sectional mean → (B, hidden). Reshape mask to (B, N, 1)
    # so multiply broadcasts over hidden dim.
    mask_b = mask.reshape(*mask.shape, 1)
    counts = mask.sum(axis=1, keepdim=True).maximum(1.0)     # (B, 1)
    pooled = (h * mask_b).sum(axis=1) / counts               # (B, hidden)

    logits = pooled @ params['Wh'] + params['bh']            # (B, K)
    # Softmax with row-max subtraction for stability.
    logits = logits - logits.max(axis=1, keepdim=True)
    exp_l = logits.exp()
    pi = exp_l / (exp_l.sum(axis=1, keepdim=True) + 1e-12)
    return scores, pi


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
