"""Φ(state, action) → predicted-deployment-Sharpe.

Tiny MLP, tinygrad runtime. Pattern mirrors apps/factor's `scorers.py`.

Important tinygrad gotcha: do NOT call `.numpy()` or `.item()` on the
loss tensor before `loss.backward()` — that materialization truncates
the backward graph. Realize after backward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam


@dataclass
class PhiTrainResult:
    params: dict[str, Tensor]
    final_train_loss: float
    losses: list[float]
    n_params: int


def _he_normal(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> Tensor:
    arr = rng.standard_normal(shape).astype(np.float32) * (2.0 / fan_in) ** 0.5
    return Tensor(arr, requires_grad=True)


def init_phi(
    rng: np.random.Generator,
    in_dim: int,
    *,
    hidden: int = 16,
    n_layers: int = 2,
) -> dict[str, Tensor]:
    """Two hidden ReLU layers + linear output. `(in_dim, hidden, hidden, 1)`."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    sizes = [in_dim] + [hidden] * n_layers + [1]
    params: dict[str, Tensor] = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"W{i}"] = _he_normal(rng, (fan_in, fan_out), fan_in)
        params[f"b{i}"] = Tensor(np.zeros(fan_out, dtype=np.float32), requires_grad=True)
    return params


def apply_phi(params: dict[str, Tensor], X: Tensor) -> Tensor:
    """`X` shape `(N, in_dim)` → predictions shape `(N,)`."""
    n_W = sum(1 for k in params if k.startswith("W"))
    n_layers = n_W - 1
    h = X
    for i in range(n_layers):
        h = (h @ params[f"W{i}"] + params[f"b{i}"]).relu()
    out = h @ params[f"W{n_layers}"] + params[f"b{n_layers}"]
    return out.squeeze(-1)


def _l2_penalty(params: dict[str, Tensor]) -> Tensor:
    terms = [(v * v).sum() for k, v in params.items() if k.startswith("W")]
    if not terms:
        return Tensor(np.zeros((1,), dtype=np.float32)).sum()
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def train_phi(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 16,
    n_layers: int = 2,
    n_steps: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> PhiTrainResult:
    """Fit Φ to `(X, y)` via MSE + L2.

    `X` is float64 numpy; converted to float32 Tensor at the boundary.
    """
    rng = np.random.default_rng(seed)
    in_dim = X.shape[1]
    params = init_phi(rng, in_dim, hidden=hidden, n_layers=n_layers)

    X_t = Tensor(X.astype(np.float32))
    y_t = Tensor(y.astype(np.float32))

    optim = Adam(list(params.values()), lr=learning_rate)

    losses: list[float] = []
    final_loss = float("nan")
    for step in range(n_steps):
        Tensor.training = True
        preds = apply_phi(params, X_t)
        mse = ((preds - y_t) * (preds - y_t)).mean()
        loss = mse + _l2_penalty(params) * weight_decay

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Only realize the loss scalar AFTER backward — calling .numpy()
        # before .backward() truncates the backward graph in tinygrad.
        loss_val = float(loss.numpy())
        losses.append(loss_val)
        if verbose and (step % 50 == 0 or step == n_steps - 1):
            print(f"  step {step:3d}  loss={loss_val:.4f}")
        final_loss = loss_val

    n_params = sum(int(np.prod(v.shape)) for v in params.values())
    return PhiTrainResult(
        params=params,
        final_train_loss=final_loss,
        losses=losses,
        n_params=n_params,
    )


def predict_phi(params: dict[str, Tensor], X: np.ndarray) -> np.ndarray:
    """Inference. Returns `(N,)` numpy array of predicted Sharpe."""
    Tensor.training = False
    preds = apply_phi(params, Tensor(X.astype(np.float32)))
    return preds.numpy()
