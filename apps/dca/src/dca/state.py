"""Local state file for DCA — last-rebal-date tracking.

Lightweight JSON file at `~/.dca-state.json` (override-able). Records
the timestamp of the last actual (non-dry-run) rebal so the cadence
gate in `live.run_live` knows when the next quarterly rebal is due.

If the file is missing the gate treats it as "no prior rebal" → due
now (subject to the drift threshold also being satisfied).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


DEFAULT_STATE_PATH: str = '~/.dca-state.json'


@dataclass
class DCAState:
    """Persisted across runs."""
    last_rebal_date: date | None
    last_rebal_checkpoint: str = ''


def load_state(path: str | Path = DEFAULT_STATE_PATH) -> DCAState:
    p = Path(path).expanduser()
    if not p.exists():
        return DCAState(last_rebal_date=None)
    raw = json.loads(p.read_text())
    d_raw = raw.get('last_rebal_date')
    d = date.fromisoformat(d_raw) if d_raw else None
    return DCAState(
        last_rebal_date=d,
        last_rebal_checkpoint=raw.get('last_rebal_checkpoint', ''),
    )


def save_state(state: DCAState, path: str | Path = DEFAULT_STATE_PATH) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'last_rebal_date': (
            state.last_rebal_date.isoformat()
            if state.last_rebal_date else None),
        'last_rebal_checkpoint': state.last_rebal_checkpoint,
        'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


__all__ = ['DEFAULT_STATE_PATH', 'DCAState', 'load_state', 'save_state']
