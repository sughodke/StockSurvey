"""Lie-group / hierarchical-network strategies.

Where `apps/relational` answers "who looks like whom now" in CWT-fingerprint
space (a geometric embedding), this app answers "how much symmetry is left in
the correlation structure, and how should risk flow down the hierarchy that
structure implies" (an algebraic / transformation view).

v1 surface:

* `weights_hrp` — Hierarchical Risk Parity (Lopez de Prado 2016): correlation
  -> distance metric -> single-linkage hierarchical clustering -> recursive
  bisection allocating risk top-down. Respects the actual cluster geometry
  rather than blind 1/N or mean-variance.

* `effective_rank` — participation-ratio effective rank of the correlation
  spectrum, `exp(H(lambda))`. Operationally: a falling effective rank flags a
  regime where the market is collapsing onto a low-dimensional subspace -- the
  empirical shadow of the Lie-group symmetry of the correlation structure
  collapsing toward a smaller subgroup. This is the "all correlations going to
  1" precondition of historical drawdowns.

* `LieCheckpoint` + `target_weights` — the JSON-config + dispatch plumbing
  that makes this composable with the live-trading pattern shared by
  `apps/regime` and `apps/relational`.
"""

from __future__ import annotations

from lie.persist import LieCheckpoint, load_checkpoint, save_checkpoint
from lie.inference import target_weights
from lie.symmetry_rank import effective_rank, trailing_effective_rank
from lie.hrp import weights_hrp
from lie.correlation_network import trailing_correlation, log_returns
from lie.clustering import (
    correlation_distance,
    hierarchical_linkage,
    quasi_diagonal_order,
)
from lie.state_builder import MarketStateConfig, build_market_state
from lie.manifold import ManifoldMapper, variance_explained_at_k
from lie.predictor import TimelessPredictor, information_coefficient
from lie.ticker_features import TickerFeatureConfig, build_ticker_features
from lie.cross_sectional import cross_sectional_ic_summary

__all__ = [
    # v1 -- correlation graph + HRP
    'LieCheckpoint',
    'load_checkpoint',
    'save_checkpoint',
    'target_weights',
    'effective_rank',
    'trailing_effective_rank',
    'weights_hrp',
    'trailing_correlation',
    'log_returns',
    'correlation_distance',
    'hierarchical_linkage',
    'quasi_diagonal_order',
    # v2 -- timeless manifold framework
    'MarketStateConfig',
    'build_market_state',
    'ManifoldMapper',
    'variance_explained_at_k',
    'TimelessPredictor',
    'information_coefficient',
    # v3 -- cross-sectional kNN
    'TickerFeatureConfig',
    'build_ticker_features',
    'cross_sectional_ic_summary',
]
