"""Dump vol-v3-DoltHub stream with realistic options friction.

Step 2 of TODO/ladder-methodology-rewrite.md. The on-disk stream
`Output/vol-v3-dolthub-oos-returns.npz` uses commission_bps=0 because
the vol-points accounting was originally upstream of friction. For
actual paper-trade options-strangle deployment, realistic round-trip
friction is ~100-500 bps (spread + fees + vega-hedging slippage).

This script applies a per-fired-rebal cost in vol-points (since the
stream is in iv_rv_gap units) and dumps a new NPZ for each commission
level in {50, 100, 200, 400} bps. The 200 bps stream is the canonical
"realistic" deployable version that lands on the DSR ladder; the
others are sensitivity context for the finding page.

Run from repo root:
    uv run python apps/vol/scripts/dump_vol_v3_realistic_cost.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
IN = REPO / 'Output/vol-v3-dolthub-oos-returns.npz'

# Per-fired-rebal vol-point cost = commission_bps * 1e-4. At 200 bps
# = 0.02 vol points subtracted from each fired rebal's alpha. This
# is the deployment proxy for "strangle round-trip friction in
# vol-payoff units" — bps-of-NAV translates 1:1 to bps-of-vol-payoff
# when vega is sized to a fixed-NAV target (the standard recipe).
COMMISSION_GRID = [50, 100, 200, 400]


def main() -> None:
    src = np.load(IN, allow_pickle=True)
    full_alpha = np.asarray(src['full_panel_alpha'], dtype=np.float64).copy()
    fired_alpha = np.asarray(src['fired_only_alpha'], dtype=np.float64).copy()
    fire_flags = np.asarray(src['fire_flags'], dtype=bool)
    rebal_dates = src['rebal_dates']
    ppy = float(src['periods_per_year'])

    n_fired = int(fire_flags.sum())
    n_total = full_alpha.size
    print(f'source stream: {n_total} rebals, {n_fired} fired '
          f'({100*n_fired/n_total:.1f}%), ppy={ppy:.2f}')
    print(f'source full_panel ann Sharpe (commission=0): '
          f'{full_alpha.mean()/full_alpha.std(ddof=0)*math.sqrt(ppy):+.3f}')
    print()
    print(f'{"comm_bps":>10s} {"net full":>10s} {"net fired":>11s} '
          f'{"full annSh":>11s} {"fired annSh":>12s}')

    for comm_bps in COMMISSION_GRID:
        cost = comm_bps * 1e-4
        # Apply cost to fired rebals only (no trade on closed-gate
        # rebals, so no friction).
        full_net = full_alpha.copy()
        full_net[fire_flags] -= cost
        fired_net = fired_alpha - cost

        full_sd = full_net.std(ddof=0)
        fired_sd = fired_net.std(ddof=0)
        full_sh = (full_net.mean() / full_sd * math.sqrt(ppy)
                   if full_sd > 0 else 0.0)
        fired_sh = (fired_net.mean() / fired_sd * math.sqrt(ppy)
                    if fired_sd > 0 else 0.0)
        print(f'{comm_bps:>10d} {full_net.mean():>+10.5f} '
              f'{fired_net.mean():>+11.5f} {full_sh:>+11.3f} '
              f'{fired_sh:>+12.3f}')

        out_path = REPO / f'Output/vol-v3-dolthub-oos-c{comm_bps}-returns.npz'
        np.savez(out_path,
                 full_panel_alpha=full_net,
                 fired_only_alpha=fired_net,
                 rebal_dates=rebal_dates,
                 fire_flags=fire_flags,
                 periods_per_year=np.float64(ppy),
                 commission_bps=np.float64(comm_bps),
                 source=np.str_('vol-v3-dolthub-oos with post-hoc realistic friction'))
        print(f'  → {out_path.name}')


if __name__ == '__main__':
    main()
