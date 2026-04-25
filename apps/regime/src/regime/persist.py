"""Checkpoint serialization for trained regime models.

A checkpoint captures everything `live.py` needs to score the universe
at a future date: the learned params, the scale grid, the strategy
hyperparameters, and the training-time universe + metadata.

Stored as JSON (not pickle) so checkpoints are portable across Python
versions, inspectable in a text editor, and safe to load from disk
without arbitrary-code-execution risk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from regime.trainer import TrainResult


CHECKPOINT_VERSION: int = 1


@dataclass
class Checkpoint:
    """In-memory representation of a saved regime model."""

    version: int
    scales: list[int]
    scale_log_weights: list[float]
    log_temperature: float
    lookback: int
    n_tail: int
    rebal_days: int
    max_spread: float
    commission_bps: float
    universe: list[str]
    trained_at: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_sharpe: float
    val_sharpe: float

    def jax_params(self) -> dict[str, jnp.ndarray]:
        """Return params in the dict form expected by `ss_indicators.symmetric_kl_divergence`."""
        return {
            'scale_log_weights': jnp.asarray(self.scale_log_weights, dtype=jnp.float32),
            'log_temperature': jnp.asarray(self.log_temperature, dtype=jnp.float32),
        }


def save_checkpoint(
    path: str | Path,
    result: TrainResult,
    *,
    universe: list[str],
    lookback: int,
    n_tail: int,
    rebal_days: int,
    max_spread: float,
    commission_bps: float,
) -> Path:
    """Serialize a `TrainResult` + run hyperparams to a JSON checkpoint."""
    cp = Checkpoint(
        version=CHECKPOINT_VERSION,
        scales=list(result.scales),
        scale_log_weights=np.asarray(result.params['scale_log_weights']).tolist(),
        log_temperature=float(result.params['log_temperature']),
        lookback=lookback,
        n_tail=n_tail,
        rebal_days=rebal_days,
        max_spread=max_spread,
        commission_bps=commission_bps,
        universe=list(universe),
        trained_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        train_start=result.train_dates[0].date().isoformat(),
        train_end=result.train_dates[1].date().isoformat(),
        val_start=result.val_dates[0].date().isoformat(),
        val_end=result.val_dates[1].date().isoformat(),
        train_sharpe=result.train_sharpe,
        val_sharpe=result.val_sharpe,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cp), indent=2))
    return out


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read a JSON checkpoint and return a `Checkpoint` dataclass.

    Unknown keys in the JSON are ignored so a v1 reader can tolerate a
    forward-compatible v1.x file with extra metadata; missing required
    keys still surface as a TypeError from the dataclass constructor.
    """
    raw = json.loads(Path(path).read_text())
    if raw.get('version') != CHECKPOINT_VERSION:
        raise ValueError(
            f'checkpoint version mismatch: got {raw.get("version")!r}, '
            f'expected {CHECKPOINT_VERSION}')
    known = {f.name for f in fields(Checkpoint)}
    return Checkpoint(**{k: v for k, v in raw.items() if k in known})
