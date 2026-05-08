"""JSON checkpoint for `lie` strategies.

Mirrors `relational.persist` so the live-trading orchestrator pattern from
`apps/regime` and `apps/relational` slots in cleanly when wired up later.

A `LieCheckpoint` records:

* `strategy`  -- the dispatch tag. v1 supports `'hrp'`. Future strategies
  (hub_premium, pairs_hierarchy, contagion) will be added here.
* `universe`  -- ordered list of tickers; the price panel passed to
  `inference.target_weights` must have these as columns in this order.
* `lookback`  -- trailing-window size in bars for both the correlation
  matrix and the effective-rank computation.
* `top_n`     -- truncate to top-N by weight after HRP (0 = keep all).
* `rebal_days`, `max_spread`, `commission_bps` -- the same risk-rail and
  cost knobs the relational strategies use.
* `use_symmetry_modulator`, `symmetry_floor` -- opt-in for the effective-
  rank gross-exposure scaling. Default off so the v1 strategy is just HRP.
* `strategy_kwargs` -- per-strategy extras (e.g. `linkage_method` for HRP).

JSON is the on-disk format -- portable, inspectable, no eval risk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


CHECKPOINT_VERSION: int = 1

SUPPORTED_STRATEGIES: tuple[str, ...] = ('hrp',)


@dataclass
class LieCheckpoint:
    """Persistent config for a `lie` strategy."""

    version: int
    strategy: str
    universe: list[str]
    lookback: int
    top_n: int = 0
    rebal_days: int = 20
    max_spread: float = 0.02
    commission_bps: float = 10.0
    use_symmetry_modulator: bool = False
    symmetry_floor: float = 0.25
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f'unknown strategy {self.strategy!r}; supported: '
                f'{SUPPORTED_STRATEGIES}')


def save_checkpoint(path: str | Path, cp: LieCheckpoint) -> Path:
    """Serialize a `LieCheckpoint` to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cp), indent=2))
    return out


def load_checkpoint(path: str | Path) -> LieCheckpoint:
    """Read a JSON checkpoint. Unknown keys are ignored for forward compat."""
    raw = json.loads(Path(path).read_text())
    if raw.get('version') != CHECKPOINT_VERSION:
        raise ValueError(
            f'checkpoint version mismatch: got {raw.get("version")!r}, '
            f'expected {CHECKPOINT_VERSION}')
    known = {f.name for f in fields(LieCheckpoint)}
    return LieCheckpoint(**{k: v for k, v in raw.items() if k in known})


__all__ = [
    'CHECKPOINT_VERSION',
    'SUPPORTED_STRATEGIES',
    'LieCheckpoint',
    'save_checkpoint',
    'load_checkpoint',
]
