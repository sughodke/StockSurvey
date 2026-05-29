"""Learned 2-leg ensemble (DCA + vol_v3).

The recipe: target portfolio is `w_dca * dca_basket + w_vol * vol_v3_overlay`
with (w_dca, w_vol) fit via mean-variance on the joint (DCA daily,
vol_v3 daily-aligned) return stream over a strictly-prior training
window. See `findings/learned-ensemble-beats-deterministic.md` for the
OOS validation (beats deterministic `(1, 2)` recipe by ann ΔSR +3.0
to +4.9 across 4 splits, every CI excludes 0, max-DD tighter).
"""

from ensemble.persist import (
    CHECKPOINT_VERSION,
    EnsembleCheckpoint,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    'CHECKPOINT_VERSION',
    'EnsembleCheckpoint',
    'load_checkpoint',
    'save_checkpoint',
]
