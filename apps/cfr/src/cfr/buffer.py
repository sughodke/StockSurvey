"""Replay buffer for CFR (state, action, regret) tuples.

For tabular CFR (Phase 1) the buffer is overkill — regret updates
happen directly into the `(infoset, action)` table — but the same
storage shape is what Deep CFR's regret-net trainer consumes, so
the buffer lives here for forward compatibility.

The Phase 1 tabular driver currently doesn't use this; it's kept
to fix the public API across Phase 1 → Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class ReplayBuffer:
    """Append-only buffer for (state, regret_vector) tuples.

    State is whatever the encoder consumes — for tabular this is an
    int infoset id; for Deep CFR it's a feature vector. We store
    state in object dtype so either fits.

    Regret vector is `(n_actions,)`; one tuple per rebal contains the
    full vector (not just the played-action's entry) since we have
    closed-form counterfactual regret.
    """
    capacity: int | None = None
    states: list = field(default_factory=list)
    regrets: list[np.ndarray] = field(default_factory=list)
    step_count: int = 0

    def append(self, state, regret_vector: np.ndarray) -> None:
        self.states.append(state)
        self.regrets.append(np.asarray(regret_vector, dtype=np.float64))
        self.step_count += 1
        if self.capacity is not None and len(self.states) > self.capacity:
            # Reservoir-style: drop oldest. Deep CFR literature
            # alternately uses reservoir sampling so the long-run
            # average policy doesn't drift toward late states.
            self.states.pop(0)
            self.regrets.pop(0)

    def extend(self, items: Iterable[tuple[object, np.ndarray]]) -> None:
        for s, r in items:
            self.append(s, r)

    def __len__(self) -> int:
        return len(self.states)

    def regret_matrix(self) -> np.ndarray:
        """`(buffer_len, n_actions)` regret vectors."""
        if not self.regrets:
            return np.zeros((0, 0))
        return np.stack(self.regrets, axis=0)


__all__ = ['ReplayBuffer']
