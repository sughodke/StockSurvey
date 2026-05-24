# NPZ explicit-dates backfill — operational plumbing

**Status: `pending`. No eval, no falsification bar. This is a small,
one-pass plumbing fix that converts the tail-aligned cross-arc
comparisons into proper date-aligned ones. Effort: ~half a day of
work, ~5–10 LOC per producing script.**

---

## Motivation — the cost has already been paid once

The per-window meta-allocator probe (2026-05-23, brief at
`.research-per-window-meta-allocator.md`) ran the H1 oracle on the
full five-arc set including the date-less NPZs (`relational-returns`,
`walkforward-linear-s200-wd0.001-windows` factor LO, etc.) by
**tail-aligning** them to DCA's calendar. That choice inflated H1
oracle ΔSR_ann to **+2.21** with CI **[−2.50, +3.28]** — a clearly
preposterous point estimate produced by the oracle picking arcs
whose distributions happen to have the fattest right tail across
the tail-aligned blocks, not the arcs that actually won those
calendar blocks in reality.

Restricting to the three date-bearing arcs (DCA, gate-v0,
vol-v3-dolthub-c200) collapsed H1 oracle to the honest **+1.43,
CI [+1.03, +2.05]**. The +0.78 point-estimate gap (and the
6× wider CI under tail-alignment) is exactly the calibration error
of the missing dates.

The same problem shows up in the leaderboard's Ledoit-Wolf
cross-arc Sharpe-difference table: several rows are flagged
`tail-aligned (arc NPZ lacks dates)` because
`apps/docs/scripts/compute_sharpe_diff_vs_dca.py` can't pull the
gold-standard `'block20-dates'` alignment for them and has to fall
back to `'tail'` / `'block-tail'` / `'daily-tail'`. Those rows are
honestly caveated but they're **not comparable** to the date-aligned
rows in the same table.

The fix is mechanical: every producing script already KNOWS the
date list at the point it dumps the return stream (the prices panel
is date-indexed, the rebal-block schedule is constructed from it).
What's missing is plumbing the date list through to the
`np.savez(...)` call.

## Scope — streams that need a `rebal_dates` / `dates` key

Audit ran 2026-05-23 via:

```bash
uv run python -c "
import numpy as np, pathlib
for p in sorted(pathlib.Path('Output').glob('*returns*.npz')):
    d = np.load(p, allow_pickle=True)
    has = ('rebal_dates' in d.files) or ('dates' in d.files)
    print(f'{p.name:55s} has_dates={has}  keys={[k for k in d.files]}')"
```

### Already date-bearing (do nothing)

| NPZ | date key | producing script |
|---|---|---|
| `gate-returns.npz` | `dates` | `apps/gate/scripts/...` |
| `pairs-returns.npz` | `dates` | `apps/pairs/scripts/...` |
| `vol-v3-dolthub-oos-returns.npz` (+ c50/c100/c200/c400 variants) | `rebal_dates` | `apps/vol/scripts/run_walkforward_v3_dolthub.py` |

### Needs the fix

| NPZ | stream key | producing script | line of `np.savez` | source of dates in scope |
|---|---|---|---|---|
| `relational-returns.npz` | `val_daily_ret` (daily) | `apps/relational/scripts/modal/relational_dwt_phase2.py` | L275 | `val_equity.index` at L272 (already a `DatetimeIndex` slice `[VAL_START:VAL_END]` of the equity panel) |
| `relational-returns.npz` | — also dump `dates` after the `pct_change().dropna()` (same line) | same | same | `val_equity.pct_change().dropna().index` |
| `walkforward-linear-s200-wd0.001-windows.npz` (factor LO + LS baseline) | `oos_block_returns` / `oos_block_returns_long_short` | `apps/factor/scripts/modal/train_indicator.py` | L676 (and the L372 single-cell variant for the head-only npz) | per-window OOS block schedule is constructed in `train_scorer_walkforward`; the `WalkForwardResult.windows[i]` carries the (train, val) block indices into the source daily panel — needs a small accessor surface change in `factor.train_walkforward` to expose `oos_block_dates` |
| `sh-indicator-r5-s1-windows.npz` (factor 5d LO + LS — the one tail-aligned in the meta-allocator brief) | `oos_block_returns` / `oos_block_returns_long_short` | `apps/factor/scripts/modal/train_indicator.py` (same path, different cell) | L676 | same — single accessor change covers both |
| `lie-shape-knn-returns.npz` | `ls_block_returns` | `apps/lie/scripts/shape_knn_longshort.py` | L181 | block schedule is constructed from the source price panel at the top of `main()`; needs to be threaded down with `net` |
| `lie-shape-knn-wide-returns.npz` | `ls_block_returns` | `apps/lie/scripts/modal/shape_knn_longshort_modal.py` | L207 | same shape as the local variant; date list available in the same scope as `blr`/`scores` |
| `momentum-12-1-returns.npz` | `ls_block_returns` | `apps/factor/scripts/momentum_12_1.py` | L111 | block schedule is the monthly rebal index from the formation/hold loop; already constructed earlier in the same function as the `(R, N)` blocks |
| `low-vol-bab-returns.npz` | `ls_block_returns` | `apps/factor/scripts/low_vol_bab.py` | L114 | same shape as `momentum_12_1` — monthly rebal index already in scope |
| `dca-winner-4etf-returns.npz` | `daily_ret` | `apps/dca/scripts/dump_winner_basket_returns.py` | L56 | `prices.index` at L34 (already a `DatetimeIndex`); just dump alongside `daily` |
| `vol-returns.npz` (gauss314 v3 stream — distinct from `vol-v3-dolthub-oos-returns.npz`) | `full_panel_alpha` / `fired_only_alpha` | `apps/vol/scripts/run_walkforward_v3_regime_gated.py` | L494 | per-rebal date list is constructed during the walkforward (the gauss314 path already produces `rebal_dates` for the DoltHub variant — copy the same plumbing) |

Bonus (date-less but not yet consumed by any cross-arc tool — only
fix if cheap):

| `regime-scaled-dca-returns.npz` | `passive`/`vol_target`/`dd_gate` | `apps/dca/scripts/regime_scaled_dca.py` | L175 | val_index already constructed at the top of the function |

`dca-returns.npz` is daily and produced by
`apps/dca/scripts/dca_returns_dump.py` (L43) — the DCA arc is the
calendar reference for tail-alignment, so it's the one stream that
*doesn't* strictly need dates. But dumping them defensively unifies
the schema (one key everywhere, no special-case for DCA) and is
the same one-liner.

## Acceptance criteria

1. Every NPZ in the "needs the fix" table above has either a
   `rebal_dates` (`(R,)` block schedule) or a `dates` (`(N,)` daily
   index) key, populated from the actual price-panel index used at
   save time. The key is `np.datetime64[D]` (or its `pd.Timestamp`
   `.values` form) — not a string-pickled index.
2. `apps/docs/scripts/compute_sharpe_diff_vs_dca.py`'s SPECS table
   gets its tail-aligned rows promoted from `'tail'` / `'block-tail'`
   / `'daily-tail'` to `'block20-dates'` / `'daily'`. Re-running the
   script should NOT produce a `caveat` containing
   "tail-aligned (arc NPZ lacks dates)" for any of the previously
   broken arcs.
3. `apps/docs/scripts/per_window_meta_allocator.py` should be able
   to include `relational-analog` and `factor-5d-LO-skip1` in the
   honest H1 oracle table without a per-arc tail-alignment fallback.
   The arcs' "valid blocks" should reflect their *actual* calendar
   overlap with the DCA reference, not 261 or 234 by construction.
4. Smoke test: a one-shot `uv run python -c "import numpy as np;
   [print(p, np.load(p)['dates' if 'dates' in np.load(p).files
   else 'rebal_dates'][:3]) for p in ...]"` shows real dates, not
   integer indices, for every fixed NPZ.

## Backward-compatibility plan

- Existing NPZs **stay on disk** until each producing script is
  re-run. Downstream consumers (`compute_sharpe_diff_vs_dca.py`,
  `compute_dsr.py`, `per_window_meta_allocator.py`,
  `ensemble_discovery.py`) already guard with
  `('rebal_dates' in d.files) or ('dates' in d.files)` and fall
  back; no consumer breaks if the producer hasn't been re-run yet.
- The fix is "re-run each producer once, atomic per-NPZ overwrite",
  not a destructive migration.
- Each producer's re-run is independent and can be scheduled when
  the corresponding arc is touched anyway (next leaderboard refresh,
  next sister experiment, etc.). No blocking dependency between them.

## Follow-up (separate workstream — not part of this TODO)

Once the NPZs are fixed, **re-run the Ledoit-Wolf cross-arc
Sharpe-difference table** in `apps/docs/docs/leaderboard.md` and
replace the tail-aligned rows with proper `block20-dates`-aligned
numbers. That re-run IS a leaderboard-row-producing experiment
(the numbers will change, possibly materially for the inflated
tails), so it gets its own pre-reg + finding page. Spawned-from
chain: this plumbing → re-run LW table → row(s) on leaderboard.

The same applies to the per-window meta-allocator probe: H1/H4
should be re-evaluated on the corrected calendar set. The H1
ΔSR_ann is unlikely to clear the bar even with the corrected
relational + factor arcs included (the 2026-05-23 brief argues the
*real* lift is bounded by the per-arc individual-vs-DCA lift, which
is +1.68 for vol-v3 alone — and vol-v3 is the only arc with a
post-2023 OOS slice that drives most of the oracle alpha). But the
honest number should be on record.

## Effort estimate

| script | est. LOC delta | risk |
|---|---|---|
| `apps/relational/scripts/modal/relational_dwt_phase2.py` | +2 (one kwarg, one index pluck) | low — date index already in scope |
| `apps/factor/scripts/modal/train_indicator.py` | +5–10 (needs `WalkForwardResult.windows[i].oos_block_dates` accessor surfaced from `factor.train_walkforward`) | medium — touches a re-used eval module, but additive (no behavior change) |
| `apps/lie/scripts/shape_knn_longshort.py` | +3 (thread `block_dates` from the block-construction loop down to `np.savez`) | low |
| `apps/lie/scripts/modal/shape_knn_longshort_modal.py` | +3 (same shape as local) | low |
| `apps/factor/scripts/momentum_12_1.py` | +2 | low |
| `apps/factor/scripts/low_vol_bab.py` | +2 | low |
| `apps/dca/scripts/dump_winner_basket_returns.py` | +1 | trivial |
| `apps/vol/scripts/run_walkforward_v3_regime_gated.py` | +3 (copy the rebal_dates plumbing from the DoltHub-variant cousin script) | low |
| **total** | **~25 LOC** | half-day including re-run + leaderboard re-row |

## Out of scope

- The actual re-run of the Ledoit-Wolf cross-arc CI table after the
  fix lands. That's the next workstream and its own leaderboard row.
- The per-window meta-allocator's H1–H4 re-evaluation on the
  corrected calendar set. Same reasoning.
- Migrating the producing scripts to a shared `dump_returns_stream`
  helper. There are 8 producers; centralizing them is a separate
  refactor PR and not the bottleneck — each producer already has its
  own canonical dump line.
- Fixing `regime-scaled-dca-returns.npz`. Not currently consumed by
  any cross-arc tool. Defer until a consumer needs it.
- Adding a schema-version key (`schema_version=2`). A nice-to-have
  but not required — the duck-type check
  `('rebal_dates' in d.files) or ('dates' in d.files)` is sufficient.

## Cross-links

- Triggering finding: `.research-per-window-meta-allocator.md` (the
  H1 inflation from +1.43 honest to +2.21 tail-aligned).
- Consumer: `apps/docs/scripts/compute_sharpe_diff_vs_dca.py`
  (the alignment-strategy enum lives in its `SPECS` table).
- Consumer: `apps/docs/scripts/per_window_meta_allocator.py`
  (currently restricts to the date-bearing arc subset by hand).
- Downstream leaderboard table: Ledoit-Wolf cross-arc Sharpe-diff
  rows in `apps/docs/docs/leaderboard.md` flagged
  "tail-aligned (arc NPZ lacks dates)".
