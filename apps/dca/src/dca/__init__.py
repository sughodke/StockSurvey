"""DCA — Dollar-Cost-Averaging multi-asset rebalancer.

Canonical live strategy as of 2026-05-13 — see
`apps/docs/docs/findings/cfr-vs-dca-realistic.md` for the
deployment-decision finding that made this the deployable winner
over the CFR Phase 4d active strategy.

Single-purpose: hold a fixed target-weight basket, rebalance at a
cadence floor (default quarterly) or whenever any single name has
drifted past `drift_threshold` from target — whichever fires first.

The four risk rails on `live.run_live` mirror `regime live` and
`ss-relational live`:

  1. Kill-switch file (`~/.dca-killswitch` by default)
  2. Data freshness (latest bar age ≤ `max_data_age_days`)
  3. Per-name weight cap (water-fill via `ss_portfolio.apply_position_cap`)
  4. Dry-run by default (`--live` is opt-in)

Plus a fifth, DCA-specific rail:

  5. Cadence + drift gate — skip the rebal if both
     `now - last_rebal_date < min_rebal_days` AND
     `max(|w_current - w_target|) < drift_threshold`.
"""

from dca.persist import (
    CHECKPOINT_VERSION,
    DCACheckpoint,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    'CHECKPOINT_VERSION',
    'DCACheckpoint',
    'load_checkpoint',
    'save_checkpoint',
]
