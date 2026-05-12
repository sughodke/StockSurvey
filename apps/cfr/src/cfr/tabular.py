"""Tabular CFR — regret matching on a discrete (infoset, action) grid.

The Phase 1 algorithm. Cumulative regret per `(infoset, action)`,
plus a separate cumulative strategy table (sum of policies played
weighted by the reach probability — here uniform since we visit each
infoset along a single history at a time). The final policy is the
**time-averaged strategy**: cumulative strategy normalized per
infoset, which converges to the no-regret strategy at rate O(1/√T).

Two policies are exposed:
  - `current_policy(infoset)` — regret matching on cumulative regret.
    Use during training to pick actions.
  - `average_policy(infoset)` — time-averaged policy. Use at eval
    time; this is the no-regret limit and is more stable than the
    current policy.

The price-taker / counterfactual-observable property means we update
regret for *every* action at every visit, not just the sampled one
— so the regret estimator has zero sampling variance from the
played-action axis. The variance that remains is purely from
finite-T sampling of which infosets get visited.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cfr.regret import regret_matching


@dataclass
class TabularCFR:
    """Tabular cumulative-regret + cumulative-strategy table.

    Shape: `(n_infosets, n_actions)` for both tables.

    Invariants:
      - `cumulative_regret >= 0` is NOT an invariant; entries can be
        negative when an action has historically underperformed.
        Regret matching takes the positive part.
      - `cumulative_strategy >= 0` IS an invariant; we add a
        probability vector each update, never subtract.
      - The visit count `n_visits[i]` is incremented once per update
        for accountability — useful for diagnosing under-visited
        infosets.

    Iteration:
      table = TabularCFR(n_infosets, n_actions)
      for t in train_bars:
          i = infoset_at[t]
          pi = table.current_policy(i)        # for action sampling
          # ... play action, observe block, compute counterfactual
          regrets = compute_block_regrets(block_ret, action_weights, played)
          table.update(i, regrets, pi)        # both tables update
      ...
      pi_eval = table.average_policy(i)        # use at val time
    """
    n_infosets: int
    n_actions: int
    cumulative_regret: np.ndarray = field(init=False)
    cumulative_strategy: np.ndarray = field(init=False)
    n_visits: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.cumulative_regret = np.zeros(
            (self.n_infosets, self.n_actions), dtype=np.float64)
        self.cumulative_strategy = np.zeros(
            (self.n_infosets, self.n_actions), dtype=np.float64)
        self.n_visits = np.zeros(self.n_infosets, dtype=np.int64)

    def current_policy(self, infoset: int) -> np.ndarray:
        """Regret-matching policy at `infoset`. Uniform if no positive
        regret yet (cold start)."""
        if infoset < 0 or infoset >= self.n_infosets:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return regret_matching(self.cumulative_regret[infoset])

    def average_policy(self, infoset: int) -> np.ndarray:
        """Time-averaged policy at `infoset`.

        Falls back to current policy when an infoset hasn't been
        visited (cumulative_strategy still zero), then ultimately to
        uniform when there's no regret either.
        """
        if infoset < 0 or infoset >= self.n_infosets:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        s = self.cumulative_strategy[infoset]
        total = s.sum()
        if total > 0:
            return s / total
        return self.current_policy(infoset)

    def update(
        self,
        infoset: int,
        instantaneous_regret: np.ndarray,
        played_policy: np.ndarray,
    ) -> None:
        """Add one observation to the table.

        `instantaneous_regret` is `R_t(a) - R_t(σ_played)` (vector
        over actions). The cumulative regret table accumulates this
        directly. The cumulative strategy table accumulates the
        regret-matching policy (`played_policy`), so its row-
        normalization converges to the no-regret strategy.
        """
        if infoset < 0 or infoset >= self.n_infosets:
            return
        self.cumulative_regret[infoset] += instantaneous_regret
        self.cumulative_strategy[infoset] += played_policy
        self.n_visits[infoset] += 1

    def policy_table_current(self) -> np.ndarray:
        """`(n_infosets, n_actions)` current policy for every infoset."""
        return np.stack(
            [self.current_policy(i) for i in range(self.n_infosets)], axis=0)

    def policy_table_average(self) -> np.ndarray:
        """`(n_infosets, n_actions)` time-averaged policy for every infoset."""
        return np.stack(
            [self.average_policy(i) for i in range(self.n_infosets)], axis=0)


__all__ = ['TabularCFR']
