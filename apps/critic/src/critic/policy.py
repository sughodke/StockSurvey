"""Policy training against a pre-trained Φ value function.

Two architectures share the same `apply` interface and a single
`train_policy` entry point:

  - `vanilla`: π trained to maximize E[Φ(state, π_score(state))]. With a
    pre-trained Φ, this reduces to π imitating Φ — the policy reads the
    same state and outputs a score whose argmax/top-K is taken at
    deployment.

  - `cql`: same plus a "Conservative Q-Learning" style penalty
    `λ_cql · (mean(sigmoid(π)) − p_data)²` that anchors π's marginal
    selection probability to the empirical pair-inclusion frequency in
    the training data. The bayesian reading: penalize π from
    over-extrapolating outside the action distribution observed at
    train time.

Both train against `−Φ(state, σ(π_score(state)))` for the per-pair
case. The state is the pair's feature vector; the "action" is the
include-probability `σ(π_score)`.

Pre-registration: see `apps/docs/docs/TODO/critic-phi-value-function.md`
day-2 contingent step (note: day-1 FAILed, so this is exploratory).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam

from critic.model import init_phi, apply_phi, predict_phi


@dataclass
class PolicyTrainResult:
    params: dict[str, Tensor]
    final_loss: float
    final_phi_value: float
    final_cql_penalty: float
    losses: list[float]
    n_params: int


def init_policy(
    rng: np.random.Generator,
    in_dim: int,
    *,
    hidden: int = 16,
    n_layers: int = 2,
) -> dict[str, Tensor]:
    """Same architecture as Φ — keeps the comparison apples-to-apples."""
    return init_phi(rng, in_dim, hidden=hidden, n_layers=n_layers)


def apply_policy(params: dict[str, Tensor], X: Tensor) -> Tensor:
    """Return raw `(N,)` policy scores. Sigmoid is applied by the caller
    when computing inclusion probability vs. when ranking."""
    return apply_phi(params, X)


def _sigmoid_tensor(x: Tensor) -> Tensor:
    # Tinygrad: x.sigmoid() works in modern versions; fall back to
    # `1/(1+exp(-x))` if needed.
    return x.sigmoid()


def train_policy(
    X: np.ndarray,
    phi_params: dict[str, Tensor],
    *,
    hidden: int = 16,
    n_layers: int = 2,
    n_steps: int = 300,
    learning_rate: float = 5e-3,
    cql_weight: float = 0.0,
    empirical_inclusion_rate: float = 0.5,
    seed: int = 0,
    verbose: bool = False,
) -> PolicyTrainResult:
    """Train π to maximize E[Φ(s, σ(π(s))) · σ(π(s))] − cql_penalty.

    The Φ network is held fixed (gradients flow through to π only — Φ's
    params don't carry `requires_grad` after construction, but the tinygrad
    Adam optimizer below is given only π's parameters so Φ stays frozen
    regardless).

    `cql_weight > 0` adds the CQL-style anchor toward the empirical
    inclusion rate. `empirical_inclusion_rate` should be set to the
    fraction of pairs in the training data that the deployment policy
    historically included (e.g., 50/200 = 0.25 for top-50-of-200).
    """
    rng = np.random.default_rng(seed)
    in_dim = X.shape[1]
    policy_params = init_policy(rng, in_dim, hidden=hidden, n_layers=n_layers)

    X_t = Tensor(X.astype(np.float32))

    optim = Adam(list(policy_params.values()), lr=learning_rate)

    # Pre-compute Φ's per-sample predictions ONCE (Φ is fixed).
    Tensor.training = False
    phi_scores_np = predict_phi(phi_params, X)
    phi_scores = Tensor(phi_scores_np.astype(np.float32))

    losses: list[float] = []
    final_loss = float("nan")
    final_phi_value = float("nan")
    final_cql = float("nan")
    for step in range(n_steps):
        Tensor.training = True
        raw = apply_policy(policy_params, X_t)
        p = _sigmoid_tensor(raw)

        # Policy "value": E[p · Φ_score]. Higher is better; we minimize the
        # negative for gradient descent.
        phi_value = (p * phi_scores).mean()

        # CQL anchor: keep the marginal inclusion probability near the
        # empirical training-data rate.
        if cql_weight > 0:
            mean_p = p.mean()
            target = Tensor(
                np.array([empirical_inclusion_rate], dtype=np.float32)
            ).squeeze()
            cql_penalty = (mean_p - target) * (mean_p - target)
        else:
            cql_penalty = Tensor(np.zeros((1,), dtype=np.float32)).sum()

        loss = -phi_value + cql_weight * cql_penalty

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_val = float(loss.numpy())
        phi_val_scalar = float(phi_value.numpy())
        cql_scalar = float(cql_penalty.numpy()) if cql_weight > 0 else 0.0
        losses.append(loss_val)
        final_loss = loss_val
        final_phi_value = phi_val_scalar
        final_cql = cql_scalar
        if verbose and (step % 50 == 0 or step == n_steps - 1):
            print(
                f"  step {step:3d}  loss={loss_val:+.4f}  "
                f"⟨p·Φ⟩={phi_val_scalar:+.4f}  cql={cql_scalar:.4f}"
            )

    n_params = sum(int(np.prod(v.shape)) for v in policy_params.values())
    return PolicyTrainResult(
        params=policy_params,
        final_loss=final_loss,
        final_phi_value=final_phi_value,
        final_cql_penalty=final_cql,
        losses=losses,
        n_params=n_params,
    )


def policy_score(policy_params: dict[str, Tensor], X: np.ndarray) -> np.ndarray:
    """Return raw policy scores at inference; deployment ranks/top-K's these."""
    Tensor.training = False
    return apply_policy(policy_params, Tensor(X.astype(np.float32))).numpy()


def policy_inclusion(policy_params: dict[str, Tensor], X: np.ndarray) -> np.ndarray:
    """Return sigmoid-of-scores — inclusion probability ∈ [0, 1]."""
    Tensor.training = False
    raw = apply_policy(policy_params, Tensor(X.astype(np.float32)))
    return raw.sigmoid().numpy()
