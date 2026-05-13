"""Checkpoint serialization for DCA baskets.

A DCA checkpoint is the simplest possible "model" — a fixed map
`{symbol: target_weight}` plus the rebal cadence and operational
parameters. There are no learned weights and no training procedure;
the basket is chosen once (e.g. via `scripts/build_checkpoint.py`)
and then held.

JSON on disk for the same reasons the other apps use it: portable,
inspectable, no arbitrary-code-execution surface.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


CHECKPOINT_VERSION: int = 1


@dataclass
class DCACheckpoint:
    """In-memory representation of a DCA basket."""

    version: int
    name: str
    universe: list[str]
    target_weights: dict[str, float]
    min_rebal_days: int
    drift_threshold: float
    commission_bps: float
    created_at: str
    notes: str = ''
    backtest_start: str = ''
    backtest_end: str = ''
    backtest_sharpe: float = 0.0
    backtest_cagr: float = 0.0
    backtest_max_drawdown: float = 0.0
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.target_weights) != set(self.universe):
            extra_w = set(self.target_weights) - set(self.universe)
            extra_u = set(self.universe) - set(self.target_weights)
            raise ValueError(
                f'universe and target_weights keys must match exactly; '
                f'in target_weights only: {sorted(extra_w)}; '
                f'in universe only: {sorted(extra_u)}')
        total = sum(self.target_weights.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f'target_weights must sum to 1.0 (±1e-3); got {total:.6f}')
        if self.min_rebal_days < 1:
            raise ValueError(
                f'min_rebal_days must be >= 1; got {self.min_rebal_days}')
        if not 0.0 <= self.drift_threshold <= 1.0:
            raise ValueError(
                f'drift_threshold must be in [0, 1]; got {self.drift_threshold}')


def save_checkpoint(path: str | Path, cp: DCACheckpoint) -> Path:
    """Serialize to JSON. Caller constructs the dataclass."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cp), indent=2, sort_keys=True))
    return out


def load_checkpoint(path: str | Path) -> DCACheckpoint:
    """Read a JSON checkpoint. Unknown keys are ignored for forward-compat."""
    raw = json.loads(Path(path).read_text())
    if raw.get('version') != CHECKPOINT_VERSION:
        raise ValueError(
            f'checkpoint version mismatch: got {raw.get("version")!r}, '
            f'expected {CHECKPOINT_VERSION}')
    known = {f.name for f in fields(DCACheckpoint)}
    return DCACheckpoint(**{k: v for k, v in raw.items() if k in known})


__all__ = [
    'CHECKPOINT_VERSION',
    'DCACheckpoint',
    'save_checkpoint',
    'load_checkpoint',
]
