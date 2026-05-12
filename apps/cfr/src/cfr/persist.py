"""Checkpoint serialization for a trained tabular CFR.

JSON on-disk (matching the pattern in `regime/persist.py` and
`relational/persist.py`): portable, inspectable, no arbitrary-code-
execution risk.

A checkpoint captures:
  - The action menu definition (modes + gross levels)
  - The infoset builder configuration + fitted bucket cutoffs
  - The cumulative regret + cumulative strategy tables
  - The train/val windows that produced the table
  - The per-window summary stats

Phase 2 will extend this with the deep-CFR weight artifacts
(regret_net, policy_net) — Phase 1's tabular tables are small enough
(`O(n_infosets × n_actions)` ≈ 100 floats) to stay inline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cfr.state import InfosetBuilder
from cfr.tabular import TabularCFR


CHECKPOINT_VERSION: int = 1


@dataclass
class CFRCheckpoint:
    """In-memory representation of a trained tabular CFR."""
    version: int
    universe: list[str]
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    rebal_days: int
    commission_bps: float
    action_keys: list[str]
    infoset_n_vol_buckets: int
    infoset_n_disp_buckets: int
    infoset_vol_window: int
    infoset_disp_window: int
    infoset_vol_edges: list[float]
    infoset_disp_edges: list[float]
    cumulative_regret: list[list[float]]    # (n_infosets, n_actions)
    cumulative_strategy: list[list[float]]  # (n_infosets, n_actions)
    n_visits: list[int]                     # (n_infosets,)
    summary: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    *,
    table: TabularCFR,
    builder: InfosetBuilder,
    universe: list[str],
    action_keys: list[str],
    train_start: str, train_end: str,
    val_start: str, val_end: str,
    rebal_days: int, commission_bps: float,
    summary: dict[str, Any] | None = None,
) -> Path:
    cp = CFRCheckpoint(
        version=CHECKPOINT_VERSION,
        universe=list(universe),
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        rebal_days=rebal_days, commission_bps=commission_bps,
        action_keys=list(action_keys),
        infoset_n_vol_buckets=builder.n_vol_buckets,
        infoset_n_disp_buckets=builder.n_disp_buckets,
        infoset_vol_window=builder.vol_window,
        infoset_disp_window=builder.dispersion_window,
        infoset_vol_edges=list(builder.vol_edges),
        infoset_disp_edges=list(builder.disp_edges),
        cumulative_regret=table.cumulative_regret.tolist(),
        cumulative_strategy=table.cumulative_strategy.tolist(),
        n_visits=table.n_visits.tolist(),
        summary=summary or {},
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cp), indent=2))
    return out


def load_checkpoint(path: str | Path) -> CFRCheckpoint:
    raw = json.loads(Path(path).read_text())
    if raw.get('version') != CHECKPOINT_VERSION:
        raise ValueError(
            f'checkpoint version mismatch: got {raw.get("version")!r}, '
            f'expected {CHECKPOINT_VERSION}')
    return CFRCheckpoint(**raw)


def restore_table_and_builder(
    cp: CFRCheckpoint,
) -> tuple[TabularCFR, InfosetBuilder]:
    """Rebuild in-memory `TabularCFR` + `InfosetBuilder` from a checkpoint."""
    builder = InfosetBuilder(
        vol_window=cp.infoset_vol_window,
        dispersion_window=cp.infoset_disp_window,
        n_vol_buckets=cp.infoset_n_vol_buckets,
        n_disp_buckets=cp.infoset_n_disp_buckets,
        vol_edges=tuple(cp.infoset_vol_edges),
        disp_edges=tuple(cp.infoset_disp_edges),
        fitted=True,
    )
    n_infosets = builder.n_infosets
    n_actions = len(cp.action_keys)
    table = TabularCFR(n_infosets=n_infosets, n_actions=n_actions)
    table.cumulative_regret = np.array(cp.cumulative_regret, dtype=np.float64)
    table.cumulative_strategy = np.array(cp.cumulative_strategy, dtype=np.float64)
    table.n_visits = np.array(cp.n_visits, dtype=np.int64)
    return table, builder


__all__ = [
    'CHECKPOINT_VERSION',
    'CFRCheckpoint',
    'save_checkpoint',
    'load_checkpoint',
    'restore_table_and_builder',
]
