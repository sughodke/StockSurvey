"""Deep CFR — regret_net (tinygrad MLP) over a continuous state vector.

Replaces the tabular `(infoset, action) → cumulative regret` table
from Phase 1-2 with a small neural net `regret_net(state_vec) →
predicted regret per action`. Phase 2 confirmed-null on tabular
menu enrichment because the regret-table sample density worsened
with more actions; the neural net solves this by sharing
statistical strength across (state, action) pairs.

Architecture:
  state(F=6-10) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(n_actions)
  ~7-10K params total. Tinygrad CPU/Metal backend.

Training (Deep CFR style — Brown et al. 2019):
  for each train rebal t:
      state_t = state_vec_builder.transform(prices[:t+1])[t]
      R_pred = regret_net(state_t)                     # (n_actions,)
      pi = regret_matching(R_pred, masked by avail_t)
      played = sample(pi)
      r_inst = compute_block_regrets(...)              # closed-form
      buffer.append((state_t, r_inst, avail_t))
      if len(buffer) % batch_every == 0:
          take batch_size random samples from buffer
          loss = MSE(regret_net(states) - target_regrets)[avail]
          adam.step(loss.backward())

Eval:
  for each val rebal t:
      state_t = ...
      R_pred = regret_net(state_t)                     # (n_actions,)
      pi = regret_matching(R_pred) masked by avail_t
      mixed_weights = Σ_a pi(a) * action_weights[t, a]
      apply commission, integrate portfolio return

The closed-form counterfactual regret signal (price-taker
assumption) is unchanged from tabular CFR — only the regret
representation changes. Average policy is approximated by training
the regret_net to fit *cumulative* regret over many SGD updates;
the resulting policy via regret matching converges to the no-regret
strategy at the standard CFR rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cfr.regret import regret_matching


# Tinygrad imports kept inside a function so the module can be
# imported without tinygrad if only the dataclass + numpy code is
# needed (e.g., for testing the buffer or sampling).
def _import_tinygrad():
    from tinygrad import Tensor, nn, dtypes
    from tinygrad.nn.optim import Adam
    return Tensor, nn, dtypes, Adam


class RegretNet:
    """MLP `state → regret per action`.

    Built lazily on first use so the package imports without tinygrad
    if the deep module isn't called.

    `weight_decay` (default 1e-3) is the AdamW L2 regularization
    that keeps the regret_net's output magnitudes bounded — without
    it, the smoke runs see the loss diverge to NaN as the net's
    outputs grow unboundedly fitting the running average regret.
    Regret matching is scale-invariant in the output magnitudes, so
    the policy quality is fine while training, but the loss
    reporting becomes unreadable and Adam's adaptive lr can
    destabilize.
    """

    def __init__(self, n_features: int, n_actions: int,
                 hidden: int = 64, lr: float = 1e-3,
                 weight_decay: float = 1e-3) -> None:
        Tensor, nn, dtypes, Adam = _import_tinygrad()
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self._Tensor = Tensor
        self._dtypes = dtypes
        self.layers = [
            nn.Linear(n_features, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, n_actions),
        ]
        # AdamW (Adam with decoupled weight decay) for output bounding.
        try:
            from tinygrad.nn.optim import AdamW
            self.opt = AdamW(
                nn.state.get_parameters(self.layers),
                lr=lr, weight_decay=weight_decay)
        except ImportError:
            # Older tinygrad fallback to plain Adam.
            self.opt = Adam(nn.state.get_parameters(self.layers), lr=lr)

    def __call__(self, x: 'Tensor') -> 'Tensor':
        h = self.layers[0](x).relu()
        h = self.layers[1](h).relu()
        return self.layers[2](h)

    def predict(self, state_vec: np.ndarray) -> np.ndarray:
        """Single forward pass; `state_vec` is `(F,) float32` or
        `(B, F) float32`. Returns `(n_actions,)` or `(B, n_actions)`."""
        Tensor = self._Tensor
        is_single = state_vec.ndim == 1
        x_np = state_vec[None, :] if is_single else state_vec
        x = Tensor(x_np.astype(np.float32), requires_grad=False)
        # tinygrad's eval mode is just no-grad; we don't need
        # explicit eval()/train() for inference correctness here.
        out = self(x).numpy()
        return out[0] if is_single else out

    def train_step(
        self,
        states: np.ndarray,         # (B, F) float32
        targets: np.ndarray,        # (B, n_actions) float64/32
        avail: np.ndarray,          # (B, n_actions) bool
    ) -> float:
        """One Adam SGD step on MSE loss masked by availability.

        Loss = sum over (B × A) of `mask * (pred - target)^2`,
        normalized by `mask.sum()` (per-batch mean over valid
        entries). Regret-matching is scale-invariant in the
        regret_net's outputs, so we don't pre-clip targets — that
        materially hurt the smoke result vs no-clip.

        **Footgun:** `.numpy()` / `.item()` on the loss tensor before
        `backward()` realizes the loss in tinygrad's lazy graph,
        which **severs the autograd path** from the loss back to
        the params. Calling backward on the realized loss leaves
        params with `grad=None` and the optimizer crashes with
        `assert x is not None`. Always call `backward()` BEFORE
        `numpy()`. Same trap apps/factor calls out at
        `factor.train.train_scorer` ("call .item() AFTER backward").
        """
        Tensor = self._Tensor
        Tensor.training = True
        x = Tensor(states.astype(np.float32), requires_grad=False)
        y = Tensor(targets.astype(np.float32), requires_grad=False)
        m = Tensor(avail.astype(np.float32), requires_grad=False)
        pred = self(x)
        diff = pred - y
        sq = diff * diff
        masked_sq = sq * m
        loss = masked_sq.sum() / m.sum().clip(1.0, float('inf'))
        # backward FIRST, before any .numpy()/.item() that realizes the loss.
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        loss_val = float(loss.numpy())
        Tensor.training = False
        return loss_val if np.isfinite(loss_val) else float('nan')

    def get_state_dict(self) -> dict[str, np.ndarray]:
        """Numpy state-dict for checkpoint serialization."""
        nn_local = _import_tinygrad()[1]
        params = nn_local.state.get_state_dict(self.layers)
        return {k: v.numpy() for k, v in params.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        Tensor = self._Tensor
        nn_local = _import_tinygrad()[1]
        params = nn_local.state.get_state_dict(self.layers)
        for k, v in state.items():
            params[k].assign(Tensor(v.astype(np.float32)))


@dataclass
class DeepCFRBuffer:
    """Append-only replay buffer of (state, target_regret, avail) tuples.

    No explicit running-average tracking — the SGD steps over MSE
    loss converge the regret_net to the *average* of seen targets at
    each state, which is the right object for regret matching (since
    the policy is scale-invariant).
    """
    capacity: int = 100_000
    states: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    avails: list = field(default_factory=list)

    def append(self, state: np.ndarray, target: np.ndarray,
               avail: np.ndarray) -> None:
        self.states.append(state.astype(np.float32))
        self.targets.append(target.astype(np.float32))
        self.avails.append(avail.astype(bool))
        if len(self.states) > self.capacity:
            # Reservoir-style: drop oldest. Simpler than reservoir
            # sampling and our per-step cost is dominated by SGD.
            self.states.pop(0)
            self.targets.pop(0)
            self.avails.pop(0)

    def __len__(self) -> int:
        return len(self.states)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self.states), size=batch_size)
        states = np.stack([self.states[i] for i in idx], axis=0)
        targets = np.stack([self.targets[i] for i in idx], axis=0)
        avails = np.stack([self.avails[i] for i in idx], axis=0)
        return states, targets, avails


def policy_from_predicted_regret(
    R_pred: np.ndarray,
    avail: np.ndarray,
) -> np.ndarray:
    """Regret matching on predicted regrets, masked by availability.

    Same logic as `walkforward._mask_and_renormalize` but with the
    raw regret-matching step inlined so we don't double-mask.
    """
    R = np.where(avail, R_pred, -np.inf)
    pi_pos = np.maximum(R, 0.0)
    # -inf positions become 0 after maximum
    pi_pos = np.where(np.isfinite(pi_pos), pi_pos, 0.0)
    total = pi_pos.sum()
    if total > 0:
        return pi_pos / total
    # All non-positive predicted regret on available actions →
    # uniform over available
    n_avail = int(avail.sum())
    if n_avail > 0:
        out = np.zeros_like(R_pred)
        out[avail] = 1.0 / n_avail
        return out
    out = np.zeros_like(R_pred)
    out[0] = 1.0   # cash fallback
    return out


__all__ = [
    'RegretNet', 'DeepCFRBuffer', 'policy_from_predicted_regret',
]
