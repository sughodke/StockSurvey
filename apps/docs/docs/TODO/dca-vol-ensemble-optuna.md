# DCA × vol overlay joint Optuna search — pre-registered arc

**Status: `pending` — pre-registration locked before the eval runs.**
Sister arc to [`dca-basket-optuna`](dca-basket-optuna.md), which
adjudicated the DCA-basket-only question and found a `partial-OOS`
(effectively `confirmed-null`) verdict. This arc extends the same
discipline to the **joint** decision: which (DCA basket × vol overlay
sizing) combination — if any — materially beats the canonical
13-ETF + vol×3 ensemble that was incidentally surfaced at
[`vol-v3-dolthub-oos`](../findings/vol-v3-dolthub-oos.md) (deflated-t
+5.35 on 29 overlap blocks at n_trials=21).

---

## Motivation

The vol-v3 DoltHub OOS extension surfaced an implicit "best ensemble"
of **canonical-13-ETF DCA + vol-v3 stream × 3.0** with date-aligned
ensemble deflated-t **+5.35** over 29 non-overlapping 20-day blocks.
That number was the peak of a small grid (vega_scale ∈
{0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0}) reported in the finding —
**fixed basket, varied scale only**. We never asked the joint
question: is there a **basket × overlay-scale combination** that
materially beats the canonical DCA + vol×3 ensemble after honest
deflation for the joint search cost?

This pre-reg locks the joint search and a falsification bar before
running, mirroring the d49c672 → a6898c8 pattern from
[`dca-basket-optuna`](dca-basket-optuna.md).

## Search space (locked — bucketed, capital-free overlay)

**Capital-free-overlay semantics.** The vol-v3 stream is in per-rebal
alpha (vol points × notional). Adding it to DCA returns is:

```
r_ensemble[t] = r_dca_block[t] + vega_scale × r_vol_alpha[t]
```

`vega_scale` represents options-notional sizing per name; it does not
compete with the DCA cash book. We search over (DCA-cash-book
composition) × (overlay-scale).

DCA basket buckets (identical to the
[`dca-basket-optuna`](dca-basket-optuna.md) search space — equal
weight within each bucket; the only lever is which buckets are
present):

| bucket | choices | n |
|---|---|---:|
| `equity_core` | `{9-spdr-sectors-EW, SPY-only, VTI-only, top-3-by-trailing-sharpe}` | 4 |
| `intl` | `{none, EFA+EEM, VEU}` | 3 |
| `bonds` | `{none, TLT-only, TLT+IEF, AGG, TLT+IEF+TIP}` | 5 |
| `commodities` | `{none, GLD-only, GLD+DBC, GLD+DBC+USO}` | 4 |
| `reits` | `{none, VNQ}` | 2 |
| `rebal_trading_days` | `{21, 63, 80, 126, 252}` | 5 |
| `vega_scale` | `{0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0}` | 7 |

Grid: 4 × 3 × 5 × 4 × 2 × 5 × 7 = **16,800 combinations**.
`vega_scale=0.0` is explicitly included to give the search an honest
"no overlay" option.

**Excluded** (data-quality / launch-date / scope constraints):
- `XLRE` (2015 launch), `XLC` (2018 launch) — train window pre-2020
- Leveraged ETFs (UPRO/TMF) — different strategy class
- Currency-hedged equivalents — not the question under test
- Pre-2023-08 vol stream data — the DoltHub OOS window starts there
  by construction

## Trial budget (locked)

**N_TRIALS = 200**. This is the deflation count regardless of how many
additional trials we *would have run* given more compute. Same budget
as the basket-only arc — keeps deflation comparable across the two
arcs.

Under `sharpe_std_ann=0.25`:
`E[max Sharpe | 200-trial null] ≈ 0.25 × Z(1 − 1/200) ≈ 0.65`
annualized. The 33-obs sample's null s.e. on `sharpe_per_period` is
`1/sqrt(33) ≈ 0.174`, which dominates. The combined effect is the
deflated-t penalty applied to any winner the search surfaces.

## Walk-forward split (locked — thin by design)

The vol stream has only 33 non-overlapping 20-trading-day rebals
(2023-08 → 2026-03), so the split is **thin**:

- **Train** (Optuna objective): vol-rebal index 0..19 — first 20 obs
  (~2023-08 → 2025-01-21 / 2025-02-19)
- **Val** (OOS verdict): vol-rebal index 20..32 — last 13 obs (~13
  months, 2025-02 → 2026-03)

Optuna sees ONLY the train slice. Val is touched exactly once at the
end, when reporting the winner's OOS deflated-t. No re-tuning, no
peeking.

**Acknowledged caveat (recorded for honesty):** 13 val obs is **thin**.
The entire vol-v3 OOS sample is a **regime-tailwind calm-bull window
with no vol crisis** (see
[`vol-v3-dolthub-oos`](../findings/vol-v3-dolthub-oos.md) §"Three
caveats"). The most likely outcome is `partial-OOS` or
`confirmed-null` — the search will likely surface marginal
improvements within the noise band. The pre-reg bar must be high
enough that noise alone does not trigger `confirmed-OOS`.

## Objective (locked)

For each candidate (basket, vega_scale):

1. Build DCA daily-return stream over full Stooq history via
   `cfr.baselines.PassiveEW(rebal_days=basket['rebal_trading_days'],
   commission_bps=10.0)`.
2. At each vol rebal_date, compound the DCA daily stream over the
   forward 20-trading-day window → `dca_block[t]`.
3. Form ensemble per-block:
   `r_ens[t] = dca_block[t] + vega_scale × full_panel_alpha[t]`
   from `Output/vol-v3-dolthub-oos-returns.npz`.
4. Compute deflated-t via `ss_portfolio.standardize_oos(r_ens_train,
   periods_per_year=12.6, n_trials=200, sharpe_std=0.25/sqrt(12.6))`
   on the train portion (20 obs).
5. **Optuna maximizes train-period deflated-t.**

`periods_per_year=12.6` matches the vol-v3 finding's calibration
(33 rebals over ~32 calendar months ≈ 12.6/yr).

## Canonical reference point (computed under identical method)

`canonical-13-ETF DCA + vol-v3 × 3.0` is the implicit current
incumbent. Compute its train-deflated-t and val-deflated-t under the
identical method (same date alignment, same DSR calibration, same
n_trials=200 deflation penalty) and report alongside the winner.

## Falsification bar (locked — before eval runs)

Let `t_canon_val` = val-deflated-t of canonical-13-ETF + vol×3 under
identical method (n_trials=200, 13 val obs).
Let `t_winner_val` = val-deflated-t of Optuna's train-winner applied
OOS.

| outcome | verdict | action |
|---|---|---|
| `t_winner_val > t_canon_val + 1.0` AND val max-DD ≤ canon max-DD + 5pp | **`confirmed-OOS`** — adopt new mix | Update live ensemble defaults, append leaderboard row, dump deployable return stream |
| `t_winner_val ∈ [t_canon_val, t_canon_val + 1.0]` | **`partial-OOS`** — interesting but not enough to switch | Document; keep canonical as live |
| `t_winner_val < t_canon_val` OR val max-DD > canon max-DD + 5pp | **`confirmed-null`** — canonical stands | Document the search as defense of the current recipe |

**Critical:** the canonical reference is re-computed under identical
method. The +5.35 number reported in
[`vol-v3-dolthub-oos`](../findings/vol-v3-dolthub-oos.md) used
n_trials=21 (the implicit grid the vol arc tried); this arc applies
n_trials=200 (the joint search cost), so the canonical reference
deflated-t under THIS arc's accounting will be lower. The fair
comparison is winner-val vs canonical-val under identical method.

## What does NOT count as a result

- A train-period winner — Optuna will produce one by construction.
  This pre-reg only counts the **val-period deflated-t against
  canonical-val**.
- A winner with worse val max-DD than canonical by more than +5pp,
  EVEN IF deflated-t is higher.
- An ensemble where `vega_scale=0.0` wins — that's the basket-only
  question already adjudicated; ties to `vega_scale > 0` configurations
  are broken in favor of the canonical-basket arm to avoid relitigating
  the basket-only verdict.

## Datasets / dependencies (locked)

- DCA cash book: Stooq daily closes (`./StooqData`), continuous coverage
  2023-08-02 → 2026-03-04 (the vol rebal-date span).
- Vol stream: `Output/vol-v3-dolthub-oos-returns.npz` — 33 obs of
  `full_panel_alpha` with `rebal_dates` for date alignment.
- `cfr.baselines.PassiveEW` for the EW rebal engine.
- `ss_portfolio.standardize_oos` for the deflated-t.

## Reproduction

```bash
uv run python apps/dca/scripts/optuna_dca_vol_ensemble.py \
    --n-trials 200 \
    --out Output/dca-vol-ensemble-optuna.json
```

Artifacts: `Output/dca-vol-ensemble-optuna.json` (full Optuna study
serialization, top-10, canonical reference, verdict).

## Expected outcome (recorded for honesty)

My honest prior: **`partial-OOS` or `confirmed-null`**. The canonical
DCA + vol×3 ensemble already posts deflated-t +5.35 at n_trials=21;
the +1.0-deflated-t bar on top of the joint-search-deflated baseline
is high. The most plausible *real* improvement, if any:

- A `vega_scale` between 2-5 that marginally beats 3.0 — the
  vol-v3-dolthub-oos finding noted the peak was "broad and flat"
  across 1-5x. Marginal lift expected, not +1.0.
- A simpler basket (the basket-only arc's 4-ETF winner: VTI+TLT+IEF+GLD)
  delivering same overlap-Sharpe with slightly better max-DD —
  matching the basket-only arc's finding under joint search.

Most plausible nulls: the joint search just confirms the existing
recipe; the 13-obs val window can't statistically distinguish
adjacent vega_scale values (knife-edge between 2.0 and 5.0 is below
the noise floor at n=13).

## Why this matters

This is the last unresolved joint-search question before
`ss-vol live` engineering investment. If the search confirms-null,
the canonical 13-ETF + vol×3 recipe is the defensible deployment.
If it confirms-OOS, the new mix is the deployment. If partial-OOS,
the canonical stands and the small lift is noted as scope
contingency. Either way, the deployment decision becomes more
defensible than the prose-only argument that motivated the current
ensemble.

## Cross-links

- Sister arc: [`dca-basket-optuna`](dca-basket-optuna.md)
- Vol stream provenance: [`vol-v3-dolthub-oos`](../findings/vol-v3-dolthub-oos.md)
- Methodology: [`deflated-sharpe-leaderboard`](../findings/deflated-sharpe-leaderboard.md)
- Verdict labels: [leaderboard](../leaderboard.md#verdict-labels)
