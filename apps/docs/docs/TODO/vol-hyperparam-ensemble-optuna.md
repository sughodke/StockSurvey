# Vol-hyperparam × DCA ensemble joint Optuna search — pre-registered arc

**Status: `pending` — pre-registration locked before the eval runs.**
Sister arc to [`dca-vol-ensemble-optuna`](dca-vol-ensemble-optuna.md),
which adjudicated the DCA-basket × vega_scale joint search with the
vol stream **frozen** at the canonical v3-DoltHub recipe. The
`partial-OOS` verdict there flagged that **the search did not touch
vol-internal hyperparams** — top-K, gate lookback, gate threshold
quantile, rebal cadence. This arc extends the discipline to the
**vol axis**: sweep the vol-stream construction itself, jointly with
the overlay-sizing knob, with the DCA basket fixed at canonical 13-ETF.

---

## Motivation

The vol-v3 stream at `Output/vol-v3-dolthub-oos-returns.npz` was
produced by ONE recipe (top-K=100, VIX > 126d-median gate, 20-trading-day
non-overlapping rebal, 4-feature OLS predictor refit once on 2019-10 →
2023-07). The recipe was inherited from gauss314-v3 — not chosen by
a systematic search on the DoltHub substrate. The basket-side arc
proved the DCA cash leg is not the binding lever; the open question
is: **does a different (top-K, gate, cadence) configuration on the
same vol predictor and substrate produce a materially better
ensemble?**

This is the cleanest test before `ss-vol live` engineering investment:
if the vol-recipe is search-robust at fixed substrate, we deploy v3.
If a different recipe materially wins, we deploy that recipe instead.
If nothing wins, the v3 recipe is the defensible deployment under
joint search.

## Search space (locked — vol-side joint, DCA-side fixed)

**DCA cash leg: FIXED** at canonical 13-ETF (XLB-Y + TLT/IEF + GLD/DBC)
at `rebal_days=80`, `commission_bps=10`. The basket axis was
adjudicated in [`dca-vol-ensemble-optuna`](dca-vol-ensemble-optuna.md);
varying it here would just relitigate the basket-only verdict.

**Vol-side recipe knobs (all swept by Optuna):**

| knob | choices | n |
|---|---|---:|
| `top_k` | `{25, 50, 100, 200, 400}` | 5 |
| `gate_lookback` (trading days for VIX rolling-median window) | `{21, 63, 126, 252, 504}` | 5 |
| `rebal_trading_days` (constrained ≥ forward_days = 20 for non-overlap) | `{20, 40, 60}` | 3 |
| `gate_quantile` (VIX > quantile of rolling window fires the gate) | `{0.40, 0.50, 0.60, 0.70}` | 4 |
| `vega_scale` (overlay sizing) | `{0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0}` | 7 |

Grid: 5 × 5 × 3 × 4 × 7 = **2,100 combinations**.
N_TRIALS = **200**, TPE sampler.

**Fixed (not swept — locked from v3-DoltHub recipe):**
- Forward window for realized vol = **20 trading days** (matches v3's
  forward-RV target horizon; sweeping it would force a 2D non-overlap
  constraint with rebal_trading_days and explode the grid).
- Predictor: **frozen** 4-feature OLS (`iv_over_hv, iv_z, iv_change_4w,
  hv_change_4w`) trained 2019-10 → 2023-07 on the same merged DoltHub
  + Stooq panel — refit once, applied to every trial's val window. The
  predictor itself is not part of this search (changing features is
  a different arc).
- DoltHub `volatility_history.parquet` snapshot (date range
  2023-08-01 → 2026-04-30 for val).
- Universe = full DoltHub cohort (no OI restriction; matches v3-DoltHub).
- Commission on vol leg = **0 bps** (matches v3 stream; deployable
  options-broker friction is a deployment overlay, not part of this
  search).

**Excluded** (out of scope by design):
- Multi-feature predictor variants (any change to feature set is a
  different arc).
- Multi-asset / non-VIX gate (different arc).
- Per-name vol overlay vs index-overlay (different arc).
- DCA-basket sweep (already adjudicated in
  [`dca-vol-ensemble-optuna`](dca-vol-ensemble-optuna.md) and
  [`dca-basket-optuna`](dca-basket-optuna.md)).

## Trial budget (locked)

**N_TRIALS = 200**. Same as the two sister arcs to keep deflation
comparable. Under `sharpe_std_ann=0.25`:
`E[max Sharpe | 200-trial null] ≈ 0.65 ann`. The joint deflation
penalty applies to any winner.

## Walk-forward split (locked)

The vol stream's natural unit is per-rebal alpha. Because
`rebal_trading_days` is itself a sweep dimension, the number of obs
per trial **varies**: at 20d cadence ≈ 33 rebals; at 40d ≈ 17 rebals;
at 60d ≈ 11 rebals. To keep the split fair across trials, we use a
**date-based split** rather than a count-based split:

- **Train**: rebal dates from 2023-08-01 to 2024-12-31
- **Val** (OOS verdict): 2025-01-01 to 2026-03-15

At the canonical 20d cadence this is ~20 train / ~13 val obs (same
as the sister arc). At 60d cadence it's ~6 train / ~6 val — which
the pre-reg explicitly accepts as a structural cost of including
longer rebal cadences. The DSR formula's `sqrt(n_obs - 1)` term
penalizes thin samples automatically.

**Acknowledged caveat (recorded for honesty):** thin val samples at
longer cadences. The pre-reg bar is high enough that noise alone
will not trigger `confirmed-OOS` at n_val=6.

## Objective (locked)

For each trial:

1. Build the vol-alpha stream under `(top_k, gate_lookback,
   rebal_trading_days, gate_quantile)` using the v3-DoltHub pipeline
   (frozen predictor + universe + forward-RV target).
2. For each vol rebal_date, compound the **canonical-13-ETF DCA** daily
   stream over the forward `rebal_trading_days` (≥ 20) → `dca_block[t]`.
3. Form ensemble per-block:
   `r_ens[t] = dca_block[t] + vega_scale × full_panel_alpha[t]`
4. Compute train-deflated-t via `standardize_oos(r_ens_train,
   periods_per_year=252/rebal_trading_days, n_trials=200,
   sharpe_std=0.25/sqrt(periods_per_year))`.
5. **Optuna maximizes train-period deflated-t.**

Note `periods_per_year` is trial-dependent (varies with
`rebal_trading_days`); this is the honest annualization, not a fudge.

## Canonical reference point (computed under identical method)

`canonical v3-DoltHub recipe + canonical-13-ETF + vol × 3.0`:
- `top_k=100, gate_lookback=126, rebal_trading_days=20,
  gate_quantile=0.50, vega_scale=3.0` (the existing v3-DoltHub recipe
  + vega=3.0 ensemble that the sister arc proved is currently the
  incumbent).

Train- and val-deflated-t computed under the same method (date-based
split, n_trials=200, identical DSR calibration).

## Falsification bar (locked — before eval runs)

Let `t_canon_val` = val-deflated-t of the canonical recipe under
identical method (n_trials=200).
Let `t_winner_val` = val-deflated-t of Optuna's train-winner applied OOS.

| outcome | verdict | action |
|---|---|---|
| `t_winner_val > t_canon_val + 1.0` AND val maxDD ≤ canon + 5pp AND winner uses `vega_scale > 0` | **`confirmed-OOS`** — adopt new vol recipe | Update `apps/vol/scripts/run_walkforward_v3_dolthub_oos.py` defaults, append leaderboard row, re-dump vol stream NPZ |
| `t_winner_val ∈ [t_canon_val, t_canon_val + 1.0]` AND winner uses `vega_scale > 0` | **`partial-OOS`** — interesting but not enough to switch | Document; keep v3-DoltHub canonical |
| `t_winner_val < t_canon_val` OR winner uses `vega_scale = 0` (winner is DCA-only) OR maxDD blow-out | **`confirmed-null`** — v3 recipe is defensible | Document the search as defense of v3 |

## What does NOT count as a result

- A train-period winner — Optuna will produce one. This pre-reg only
  counts **val-period deflated-t against canonical-val**.
- A winner with `vega_scale = 0` — that's DCA-only, which is the
  basket-only-arc question already adjudicated. If Optuna's winner has
  `vega_scale = 0`, the verdict defaults to `confirmed-null` for this
  arc (the vol recipe doesn't help).
- A winner whose stream has `< 6 val obs` after applying the date
  split. Configurations that survive the train split but produce a
  degenerate val should be filtered before verdict — but this is a
  structural filter, not a post-hoc edit (recorded here so it can't be
  used to manufacture a winner).
- A winner whose val max-DD exceeds canon by > 5pp.

## Datasets / dependencies (locked)

- DoltHub: `.iv-cache/volatility_history.parquet` (already cached).
- Stooq daily closes for forward-RV target + DCA cash leg
  (`./StooqData`).
- FRED `VIXCLS` via `ss_macro.load_fred_series` (auto-fetched, cached
  in `.macro-cache/`).
- DCA closes for the canonical 13-ETF (loaded fresh; the existing
  `Output/cfr_phase4d_multiasset_close.pkl` covers it).

## Compute notes

Each trial requires:
1. Vol-alpha stream construction (re-walk val dates, apply gate,
   compute top-K minus universe). Heavy lift = **once per trial**.
2. DCA-block compounding under the trial's `rebal_trading_days`.
   Cheap (numpy slice + prod).
3. DSR computation on ≤ ~33-obs train series. Cheap.

The predictor, panel merge, feature matrix, forward-RV, VIX series,
DCA daily stream are **all built once and cached in memory** before
the Optuna loop. Per-trial cost is the per-rebal accounting only —
expected ~50-200ms per trial → 200 trials in ~30-60s. **Local laptop;
no Modal.**

## Reproduction

```bash
uv run python apps/vol/scripts/optuna_vol_hyperparam_ensemble.py \
    --n-trials 200 \
    --out Output/vol-hyperparam-ensemble-optuna.json
```

Artifacts: `Output/vol-hyperparam-ensemble-optuna.json` (Optuna study,
canonical reference, top-10, verdict).

## Expected outcome (recorded for honesty)

My honest prior: **`partial-OOS` or `confirmed-null`**. Two reasons:

1. The basket-only arc found the vol-overlay sizing axis (`vega_scale`)
   is robust at 3.0 — every top-10 picked it. The remaining vol-recipe
   knobs may produce small lifts but probably not +1.0 deflated-t over
   the v3 recipe.
2. The vol-v3-DoltHub OOS sample is a regime-tailwind calm-bull window
   — see [`vol-v3-dolthub-oos`](../findings/vol-v3-dolthub-oos.md)
   §"Three caveats". In a calm regime, gate variations matter less
   (the gate fires often enough on any reasonable threshold); top-K
   variations dilute / concentrate the same calm-regime VRP windfall.

Most plausible *real* improvement, if any: a slightly **wider top-K**
(top-200 instead of top-100) could reduce per-rebal variance via
diversification within the gated picks; or a **shorter gate lookback**
(21d or 63d) could fire more often in calm regimes where 126d-median
is sticky.

Most plausible nulls: the v3 recipe is search-robust on this substrate
because the binding constraint is the calm-bull regime, not the recipe.

## Why this matters

This is the last unresolved joint-search question before `ss-vol live`
engineering investment. After this arc:

- If `confirmed-OOS`: deploy the new vol recipe + canonical-13-ETF
  ensemble. Strongest empirical case yet for `ss-vol live`.
- If `partial-OOS`: deploy v3-DoltHub recipe; note the marginal lift
  as scope contingency.
- If `confirmed-null`: deploy v3-DoltHub recipe with full confidence
  that no joint-search alternative beats it.

In all three cases, the next test is the same: **wait for the next vol
crisis** and re-run the chosen recipe through it. The joint search
adjudicates the *substrate choice*; the crisis adjudicates the
*durability*.

## Cross-links

- Sister arc: [`dca-vol-ensemble-optuna`](dca-vol-ensemble-optuna.md)
- Basket arc: [`dca-basket-optuna`](dca-basket-optuna.md)
- Vol stream provenance: [`vol-v3-dolthub-oos`](../findings/vol-v3-dolthub-oos.md)
- Methodology: [`deflated-sharpe-leaderboard`](../findings/deflated-sharpe-leaderboard.md)
- Verdict labels: [leaderboard](../leaderboard.md#verdict-labels)
