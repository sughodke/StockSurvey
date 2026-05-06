"""Checkpoint serialization for relational scorers.

A relational checkpoint records the choice of scorer + the
hyperparameters that won on a given universe + date range. Unlike
`regime`'s checkpoint, there are no learned continuous parameters —
the relational scorers are deterministic given their kwargs.

Strategies (each maps to a `weights_*` builder in this app):

  * **`empirical`**   — `weights_excess_regime_empirical` (k-means
    clusters of CWT fingerprints; idea-A, phase-2 Sharpe 1.07-1.13).
  * **`gmm`**         — `weights_excess_regime_gmm` (soft-cluster
    replacement, +0.03 Sharpe over `empirical`).
  * **`analog`**      — `weights_regime_analog` (k-NN analog forecast;
    idea-B).
  * **`farthest`**    — `weights_regime_farthest` (centroid distance;
    idea-C).
  * **`diversified`** — `weights_regime_diversified` (greedy farthest-
    first thinning; idea-D).
  * **`velocity`**    — `weights_velocity_magnitude` (phase-11 regime
    velocity in fingerprint space).

JSON is the on-disk format (matching `regime/persist.py`): portable,
inspectable, no arbitrary-code-execution risk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


CHECKPOINT_VERSION: int = 1

SUPPORTED_STRATEGIES: tuple[str, ...] = (
    'empirical', 'gmm', 'analog', 'farthest', 'diversified', 'velocity',
)


@dataclass
class RelationalCheckpoint:
    """In-memory representation of a saved relational config.

    `strategy_kwargs` carries the strategy-specific knobs (e.g.
    `n_tail`, `divergence`, `k_clusters`, `fp_window`, `k_neighbors`,
    `forward_horizon`, `w_delta`) — they vary per strategy and are
    threaded through to the underlying `weights_*` builder verbatim.
    """

    version: int
    strategy: str
    universe: list[str]
    lookback: int
    top_n: int
    scales: list[int]
    rebal_days: int
    max_spread: float
    commission_bps: float
    trained_at: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_sharpe: float
    val_sharpe: float
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f'unknown strategy {self.strategy!r}; supported: '
                f'{SUPPORTED_STRATEGIES}')


def save_checkpoint(path: str | Path, cp: RelationalCheckpoint) -> Path:
    """Serialize a `RelationalCheckpoint` to JSON. Caller constructs
    the dataclass — there's no `save_checkpoint_from_window` analog
    because relational scorers aren't trained, just chosen."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cp), indent=2))
    return out


def load_checkpoint(path: str | Path) -> RelationalCheckpoint:
    """Read a JSON checkpoint. Unknown keys are ignored so a v1 reader
    can tolerate forward-compatible extras; missing required keys
    surface as TypeError from the dataclass constructor."""
    raw = json.loads(Path(path).read_text())
    if raw.get('version') != CHECKPOINT_VERSION:
        raise ValueError(
            f'checkpoint version mismatch: got {raw.get("version")!r}, '
            f'expected {CHECKPOINT_VERSION}')
    known = {f.name for f in fields(RelationalCheckpoint)}
    return RelationalCheckpoint(**{k: v for k, v in raw.items() if k in known})


__all__ = [
    'CHECKPOINT_VERSION',
    'SUPPORTED_STRATEGIES',
    'RelationalCheckpoint',
    'save_checkpoint',
    'load_checkpoint',
]
