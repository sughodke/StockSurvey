"""Checkpoint serialization for the learned 2-leg ensemble.

A checkpoint captures the two scalar weights and the data range they
were fit on, so a live runner can refuse to dispatch a stale fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


CHECKPOINT_VERSION: int = 1


@dataclass
class EnsembleCheckpoint:
    """In-memory representation of the learned 2-leg blend."""

    version: int
    name: str
    w_dca: float
    w_vol: float
    learner: str  # 'mv_closed_form' | 'grad_sharpe'
    dca_checkpoint_path: str
    vol_checkpoint_path: str
    train_start: str
    train_end: str
    created_at: str
    notes: str = ''
    train_sharpe: float = 0.0
    in_sample_max_dd: float = 0.0
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.w_dca < 0.0 or self.w_vol < 0.0:
            raise ValueError(
                f'ensemble weights must be non-negative; got '
                f'(w_dca={self.w_dca}, w_vol={self.w_vol})')
        if self.w_dca + self.w_vol <= 0.0:
            raise ValueError(
                f'at least one leg must have positive weight; got '
                f'(w_dca={self.w_dca}, w_vol={self.w_vol})')
        if self.learner not in {'mv_closed_form', 'grad_sharpe'}:
            raise ValueError(
                f"learner must be 'mv_closed_form' or 'grad_sharpe'; "
                f"got {self.learner!r}")


def save_checkpoint(path: str | Path, cp: EnsembleCheckpoint) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cp), indent=2, sort_keys=True))
    return out


def load_checkpoint(path: str | Path) -> EnsembleCheckpoint:
    raw = json.loads(Path(path).read_text())
    if raw.get('version') != CHECKPOINT_VERSION:
        raise ValueError(
            f'checkpoint version mismatch: got {raw.get("version")!r}, '
            f'expected {CHECKPOINT_VERSION}')
    known = {f.name for f in fields(EnsembleCheckpoint)}
    return EnsembleCheckpoint(**{k: v for k, v in raw.items() if k in known})


__all__ = [
    'CHECKPOINT_VERSION',
    'EnsembleCheckpoint',
    'save_checkpoint',
    'load_checkpoint',
]
