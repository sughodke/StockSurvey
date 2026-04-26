"""Checkpoint serialization for trained regime models.

A checkpoint captures everything `live.py` needs to score the universe
at a future date: the model parameters, the scale grid, the strategy
hyperparameters, and the training-time universe + metadata.

Two checkpoint *modes* are supported, distinguished by the `mode`
field:

  * **`adam`** — JAX-Adam output: 13 continuous `scale_log_weights`
    (softmaxed at inference) + a learned `log_temperature` for soft
    top-N. Produced by `regime.research.optimize_adam.train()`.
  * **`optuna`** — Optuna+vectorbt output: discrete `top_n` count +
    `divergence` choice + scale subset (encoded directly in the
    `scales` field). Produced by `regime.trainer.train()` via
    `save_checkpoint_from_window()`.

Old `mode`-less checkpoints default to `adam` so they keep loading.

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

from regime.research.optimize_adam import TrainResult


CHECKPOINT_VERSION: int = 1


@dataclass
class Checkpoint:
    """In-memory representation of a saved regime model.

    Fields below `val_sharpe` are populated only for `mode == 'optuna'`
    checkpoints; `adam` checkpoints leave them at the default.
    """

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
    # Optuna-mode-only fields (with defaults for back-compat with adam
    # checkpoints written before this schema existed).
    mode: str = 'adam'
    top_n: int | None = None
    divergence: str | None = None
    # Which weight builder produced this checkpoint. Defaults to
    # 'regime' so checkpoints written before scalogram was added still
    # load correctly (they were all regime).
    strategy: str = 'regime'
    # Whether CWT input was log-returns (vs raw close) at train time.
    # Defaults to False — empirically raw close has higher val Sharpe
    # for the cross-sectional ranking objective; see the comment block
    # above `regime.trainer._log_returns`. Persisted on the checkpoint
    # so live inference scores with the same input the trainer used.
    use_log_returns: bool = False
    # RSI period — populated only for `strategy == 'rsi'`. None for the
    # CWT-based strategies (regime, scalogram) which don't use RSI.
    rsi_n: int | None = None

    def jax_params(self) -> dict[str, jnp.ndarray]:
        """Return params in the dict form expected by the JAX divergence
        functions. Used by the adam-mode inference path; for optuna mode
        the scale weights default to zeros (uniform softmax = equal
        per-scale weighting, which matches Optuna's search semantics)."""
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
    """Serialize a JAX-Adam `TrainResult` + run hyperparams to JSON."""
    cp = Checkpoint(
        version=CHECKPOINT_VERSION,
        mode='adam',
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
    return _write_checkpoint(path, cp)


def save_checkpoint_from_window(
    path: str | Path,
    window,  # regime.trainer.WindowResult
    *,
    universe: list[str],
    rebal_days: int,
    max_spread: float,
    commission_bps: float,
) -> Path:
    """Serialize an Optuna `WindowResult` to a checkpoint file.

    Sets `mode='optuna'` and `strategy=window.strategy`. Strategy-
    specific fields are populated only when the strategy uses them:

      * regime    — `scales` (from subset flags), `divergence`
      * scalogram — `scales` (from subset flags)
      * rsi       — `rsi_n`; `scales` left empty, `divergence` None

    `scale_log_weights` stays at zeros (length matches `scales`) so the
    inference-time CWT divergence sees uniform per-scale weighting,
    matching what the Optuna search used. RSI checkpoints carry an
    empty `scale_log_weights` since they never invoke the CWT path.
    """
    from regime.trainer import _resolve_scales

    strategy = getattr(window, 'strategy', 'regime')
    if strategy == 'rsi':
        scales: list[int] = []
    else:
        scales = _resolve_scales(window.best_params)
    cp = Checkpoint(
        version=CHECKPOINT_VERSION,
        mode='optuna',
        strategy=strategy,
        scales=scales,
        scale_log_weights=[0.0] * len(scales),
        log_temperature=0.0,
        lookback=int(window.best_params['lookback']),
        n_tail=int(window.best_params['n_tail']),
        top_n=int(window.best_params['top_n']),
        divergence=(str(window.best_params['divergence'])
                    if 'divergence' in window.best_params else None),
        rsi_n=(int(window.best_params['rsi_n'])
               if 'rsi_n' in window.best_params else None),
        rebal_days=rebal_days,
        max_spread=max_spread,
        commission_bps=commission_bps,
        universe=list(universe),
        trained_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        train_start=window.train_start.date().isoformat(),
        train_end=window.train_end.date().isoformat(),
        val_start=window.train_end.date().isoformat(),
        val_end=window.val_end.date().isoformat(),
        train_sharpe=float(window.train_score),
        val_sharpe=float(window.val_score),
        use_log_returns=bool(getattr(window, 'use_log_returns', False)),
    )
    return _write_checkpoint(path, cp)


def _write_checkpoint(path: str | Path, cp: Checkpoint) -> Path:
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
