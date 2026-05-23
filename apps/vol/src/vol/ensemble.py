"""Coordinator for the DCA + vol-v3 ensemble live deployment.

Runs DCA's equity rebalance and vol-v3's short-vol overlay against
the same Alpaca paper account in one pass. They don't compete for
capital — DCA is long equity ETFs, vol is short individual-name
options — but they share rails:

  - Single kill-switch (~/.ensemble-killswitch or the per-leg files
    if you want them independent; default is the per-leg files so an
    operator can disable just one)
  - Same Alpaca account (uses the same env credentials)
  - Combined audit-trail summary

Rationale: the cross-arc ladder showed DCA + vol-v3 × 3 overlay
deflated-t = +5.35 over the comparable 29-block overlap (vs DCA-only
on the same overlap +1.39, vs DCA-full +1.93). The ensemble is the
**deployment position #2** from the leaderboard prose: vol-v3-alone
is +5.55 but carries regime-tailwind risk; DCA stays the long-term
robustness backing. The coordinator is the operational expression of
that.

Order of operations matters:
  1. DCA rebalance first (long equity exposure is the base book).
  2. Vol-v3 overlay second (only fires when VIX gate is open; doesn't
     compete for capital since options use margin separately).
  3. Both produce structured results; the coordinator emits a
     combined `EnsembleRunResult`.

A failure in one leg does NOT abort the other — they're independent
strategies that happen to share rails. Each `LiveRunResult` carries
its own `aborted_reason` if applicable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Avoid binding the LiveRunResult types at import time (the two are
# different dataclasses) — coordinator just holds them as `object`.


@dataclass
class EnsembleRunResult:
    timestamp: str
    dca_result: object | None
    vol_result: object | None
    dca_error: str | None = None
    vol_error: str | None = None


def run_ensemble(
    *,
    dca_checkpoint: str | Path,
    vol_checkpoint: str | Path,
    dry_run: bool = True,
    broker=None,
    options_data=None,
    bars_data=None,
    dca_kwargs: dict | None = None,
    vol_kwargs: dict | None = None,
) -> EnsembleRunResult:
    """Run both legs of the ensemble.

    `broker`/`options_data`/`bars_data` may be passed for testing;
    in production they're constructed from env vars by each leg.
    """
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    dca_kwargs = dca_kwargs or {}
    vol_kwargs = vol_kwargs or {}

    # --- DCA leg ---
    dca_result = None
    dca_error = None
    try:
        from dca.live import run_live as dca_run_live
        dca_kw = dict(dca_kwargs)
        # If a broker was passed in, route it through (test path).
        if broker is not None and 'broker' not in dca_kw:
            dca_kw['broker'] = broker
        dca_result = dca_run_live(
            dca_checkpoint, dry_run=dry_run, **dca_kw)
    except Exception as e:
        dca_error = f'{type(e).__name__}: {e}'

    # --- Vol-v3 leg ---
    vol_result = None
    vol_error = None
    try:
        from vol.live import run_live as vol_run_live
        vol_kw = dict(vol_kwargs)
        if broker is not None and 'broker' not in vol_kw:
            vol_kw['broker'] = broker
        if options_data is not None and 'options_data' not in vol_kw:
            vol_kw['options_data'] = options_data
        if bars_data is not None and 'bars_data' not in vol_kw:
            vol_kw['bars_data'] = bars_data
        vol_result = vol_run_live(
            vol_checkpoint, dry_run=dry_run, **vol_kw)
    except Exception as e:
        vol_error = f'{type(e).__name__}: {e}'

    return EnsembleRunResult(
        timestamp=timestamp,
        dca_result=dca_result, vol_result=vol_result,
        dca_error=dca_error, vol_error=vol_error,
    )


def format_ensemble(result: EnsembleRunResult) -> str:
    lines = [f'=== ensemble live @ {result.timestamp} ===']
    lines.append('\n--- DCA leg ---')
    if result.dca_error:
        lines.append(f'  ERROR: {result.dca_error}')
    elif result.dca_result is not None:
        from dca.live import format_run as dca_format_run
        lines.append(dca_format_run(result.dca_result))
    else:
        lines.append('  (no result)')

    lines.append('\n--- Vol-v3 leg ---')
    if result.vol_error:
        lines.append(f'  ERROR: {result.vol_error}')
    elif result.vol_result is not None:
        from vol.live import format_run as vol_format_run
        lines.append(vol_format_run(result.vol_result))
    else:
        lines.append('  (no result)')
    return '\n'.join(lines)


__all__ = ['EnsembleRunResult', 'run_ensemble', 'format_ensemble']
