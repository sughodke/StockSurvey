"""Live orchestration for the learned 2-leg ensemble.

The ensemble runner is a thin orchestrator over the existing
`dca.live.run_live` and `vol.live.run_live` rails:

  1. Load the EnsembleCheckpoint to read (w_dca, w_vol).
  2. Dispatch the DCA leg with `gross_scale=w_dca`.
  3. Dispatch the vol_v3 leg with `vega_scale=w_vol`.
  4. Apply ensemble-level risk rails (kill switch + dry-run gate)
     in addition to each leg's own four rails.

Rationale: keeping the legs independent at the broker layer means
each leg's existing four-rail discipline (kill switch, freshness,
cadence/drift, position cap, dry-run, plus vol's rail #6 friction
monitor) carries over. The ensemble layer adds one shared kill
switch + dry-run gate so an operator can halt both legs with one
file touch / one flag.

Live mode is OFF by default. The user must explicitly pass
`--live` to submit orders on either leg. Even with `--live`, the
ensemble kill switch overrides everything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ensemble.persist import EnsembleCheckpoint, load_checkpoint


DEFAULT_KILLSWITCH: str = '~/.ensemble-killswitch'


@dataclass
class EnsembleRunResult:
    timestamp: str
    checkpoint_path: str
    w_dca: float
    w_vol: float
    dry_run: bool
    dca_result: Any = None
    vol_result: Any = None
    aborted_reason: str | None = None
    notes: list[str] = field(default_factory=list)


def _killswitch_active(path: str) -> bool:
    return Path(path).expanduser().exists()


def run_live(
    ensemble_checkpoint: str | Path,
    dry_run: bool = True,
    killswitch_path: str = DEFAULT_KILLSWITCH,
    skip_dca: bool = False,
    skip_vol: bool = False,
    dca_kwargs: dict | None = None,
    vol_kwargs: dict | None = None,
) -> EnsembleRunResult:
    """Run one ensemble rebal pass.

    Parameters
    ----------
    ensemble_checkpoint : path to a JSON `EnsembleCheckpoint`.
    dry_run : default True. Set to False to submit orders on both legs.
    killswitch_path : abort if this file exists. Default
        `~/.ensemble-killswitch`. Per-leg kill switches still apply.
    skip_dca / skip_vol : let an operator dispatch only one leg for
        smoke testing or staged rollout.
    dca_kwargs / vol_kwargs : passthrough to each leg's `run_live`.
        Sensible defaults are applied if missing.
    """
    cp = load_checkpoint(ensemble_checkpoint)
    ts = datetime.now(UTC).isoformat()
    notes: list[str] = []

    if _killswitch_active(killswitch_path):
        return EnsembleRunResult(
            timestamp=ts,
            checkpoint_path=str(ensemble_checkpoint),
            w_dca=cp.w_dca, w_vol=cp.w_vol, dry_run=dry_run,
            aborted_reason=(
                f'ensemble kill-switch present at {killswitch_path}; '
                f'both legs aborted'),
        )

    dca_kwargs = dict(dca_kwargs or {})
    vol_kwargs = dict(vol_kwargs or {})

    dca_result = None
    if not skip_dca and cp.w_dca > 0.0 and cp.dca_checkpoint_path:
        from dca.live import run_live as dca_run_live
        dca_kwargs.setdefault('dry_run', dry_run)
        dca_kwargs.setdefault('gross_scale', cp.w_dca)
        try:
            dca_result = dca_run_live(cp.dca_checkpoint_path, **dca_kwargs)
        except TypeError as e:
            if 'gross_scale' in str(e):
                notes.append(
                    'dca.live.run_live does not yet accept gross_scale; '
                    'falling back to unscaled DCA dispatch — operator must '
                    'verify the DCA checkpoint targets the intended gross.')
                dca_kwargs.pop('gross_scale', None)
                dca_result = dca_run_live(cp.dca_checkpoint_path, **dca_kwargs)
            else:
                raise
    elif cp.w_dca <= 0.0:
        notes.append('w_dca <= 0; DCA leg skipped this rebal.')

    vol_result = None
    if not skip_vol and cp.w_vol > 0.0 and cp.vol_checkpoint_path:
        from vol.live import run_live as vol_run_live
        vol_kwargs.setdefault('dry_run', dry_run)
        vol_kwargs.setdefault('vega_scale', cp.w_vol)
        try:
            vol_result = vol_run_live(cp.vol_checkpoint_path, **vol_kwargs)
        except TypeError as e:
            if 'vega_scale' in str(e):
                notes.append(
                    'vol.live.run_live does not yet accept vega_scale; '
                    'falling back to unscaled vol dispatch — operator must '
                    'verify the vol checkpoint targets the intended scale.')
                vol_kwargs.pop('vega_scale', None)
                vol_result = vol_run_live(cp.vol_checkpoint_path, **vol_kwargs)
            else:
                raise
    elif cp.w_vol <= 0.0:
        notes.append('w_vol <= 0; vol leg skipped this rebal.')

    return EnsembleRunResult(
        timestamp=ts,
        checkpoint_path=str(ensemble_checkpoint),
        w_dca=cp.w_dca, w_vol=cp.w_vol,
        dry_run=dry_run,
        dca_result=dca_result,
        vol_result=vol_result,
        notes=notes,
    )


def format_run(r: EnsembleRunResult) -> str:
    lines = [
        f'=== ensemble live run @ {r.timestamp} ===',
        f'checkpoint:      {r.checkpoint_path}',
        f'learned weights: w_dca = {r.w_dca:.4f}  w_vol = {r.w_vol:.4f}',
        f'dry_run:         {r.dry_run}',
    ]
    if r.aborted_reason:
        lines.append(f'ABORTED: {r.aborted_reason}')
        return '\n'.join(lines)
    if r.dca_result is None:
        lines.append('DCA leg: skipped')
    else:
        lines.append('DCA leg: dispatched')
    if r.vol_result is None:
        lines.append('vol leg: skipped')
    else:
        lines.append('vol leg: dispatched')
    if r.notes:
        lines.append('notes:')
        for n in r.notes:
            lines.append(f'  - {n}')
    return '\n'.join(lines)


__all__ = [
    'DEFAULT_KILLSWITCH',
    'EnsembleRunResult',
    'run_live',
    'format_run',
]
