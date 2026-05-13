"""Build the canonical multi-asset DCA checkpoint.

Universe: 13 ETFs from Phase 4d's PASS run — 9 SPDR sector ETFs +
2 bond ETFs (TLT, IEF) + 2 commodity ETFs (GLD, DBC). Equal-weight
target = 1/13 per name. Quarterly rebal cadence with a 5% per-name
drift threshold for off-cadence rebals.

Backtest stats are from the full-panel 2005-2025 EW reconstruction
(see `findings/cfr-vs-dca-realistic.md`).

    uv run python apps/dca/scripts/build_checkpoint.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dca.persist import CHECKPOINT_VERSION, DCACheckpoint, save_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output' / 'dca-multiasset.json'


# 9 SPDR sector ETFs + 2 bond ETFs + 2 commodity ETFs.
# Same universe as Phase 4d. Pinned to ensure live trading uses
# exactly the basket the backtest validated.
UNIVERSE: list[str] = sorted([
    'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY',  # sectors
    'TLT', 'IEF',                                                    # bonds
    'GLD', 'DBC',                                                    # commodities
])


def main() -> Path:
    n = len(UNIVERSE)
    weight = 1.0 / n
    target_weights = {sym: weight for sym in UNIVERSE}

    cp = DCACheckpoint(
        version=CHECKPOINT_VERSION,
        name='multiasset-13etf-ew',
        universe=UNIVERSE,
        target_weights=target_weights,
        # Quarterly: 63 trading days ≈ 91 calendar days. Use 80
        # trading-day floor so a once-a-quarter cron still triggers
        # even if a few days are skipped (holidays, weekends).
        min_rebal_days=80,
        drift_threshold=0.05,   # 5% absolute weight deviation triggers off-cadence rebal
        commission_bps=5.0,     # ETF-level estimate
        created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        notes=(
            'Multi-asset 13-ETF equal-weight basket from Phase 4d. '
            'Canonical live strategy after the cfr-vs-dca-realistic '
            'finding (2026-05-13) showed CFR active alpha collapsed '
            'to +0.015 net of friction vs DCA, with worse worst-window '
            'Sharpe. DCA wins on operational simplicity + tax efficiency.'),
        backtest_start='2005-02-25',
        backtest_end='2025-12-11',
        backtest_sharpe=0.673,             # FULL panel including GFC
        backtest_cagr=0.084,
        backtest_max_drawdown=-0.407,      # GFC worst-case
        provenance={
            'source_universe': 'apps/cfr Phase 4d (cfr-phase4d.json)',
            'finding': 'apps/docs/docs/findings/cfr-vs-dca-realistic.md',
            'val_only_sharpe_2010_2025': 0.805,
            'bias_corrected_sharpe_post_bond_tailwind': 0.55,
        },
    )
    out = save_checkpoint(OUTPUT, cp)
    print(f'wrote {out}  ({n} symbols, EW {weight*100:.4f}% each)')
    return out


if __name__ == '__main__':
    main()
