# Ladder methodology rewrite — full migration

**Status: `done` (2026-05-23). All five steps landed. Closing finding
at [`findings/ladder-methodology-rewrite.md`](../findings/ladder-methodology-rewrite.md).
Headline: under Ledoit-Wolf, ZERO arcs beat DCA at 95% CI; four arcs
significantly worse. DSR demoted from cross-arc rank key to within-
stratum gate. Pre-registration text below is unchanged — locked
design.**

---

## Why this is happening

Three background agents converged independently on the same problem:

| agent brief | finding |
|---|---|
| [`.research-walkforward-apples-to-apples.md`](../../../../.research-walkforward-apples-to-apples.md) | **DSR is a within-arc selection-bias correction, NOT a cross-arc rank key.** Bailey-LdP never claimed it was. The literature-canonical cross-arc test is Ledoit-Wolf (2008) studentized stationary-bootstrap CI on Sharpe-difference of common-window-aligned streams. |
| [`.research-findings-assumptions-audit.md`](../../../../.research-findings-assumptions-audit.md) | `sharpe_std_ann=0.25` calibrated on one arc class only; `commission_bps=10` unrealistic for vol options (real 100-500bps); `n_trials` under-counted in several arcs; Z>2 anti-conservative for 17 arcs (Bonferroni→2.71, FDR→2.4); DSR conflates standalone vs overlay arcs. |
| `findings/dca-vol-ensemble-optuna.md` | Even within the existing framework, the ensemble Optuna search couldn't distinguish basket configurations at n_val=13 — confirming the methodology agent's point that the rank-by-DSR magnitude is noise-dominated. |

The three findings together imply: **the ladder is internally
consistent under its documented framework, but the framework is
methodologically insufficient for the question we're using it for.**

The migration below replaces the methodology in five atomic commits.

## Sequencing

### Step 1 — sharpe_std null-noise floor (`standardize_oos` math fix)

**Findings driving this**:
- Synthetic test confirmed `sharpe_std_pp = sharpe_std_ann/sqrt(ppy)`
  is mechanically correct as a conversion.
- BUT for short-stream arcs the workspace 0.25 is *smaller than the
  null estimation noise `1/sqrt(n_obs-1)`*, which under-deflates.

| arc | n_obs | workspace sharpe_std_pp | null floor `1/√(n−1)` |
|---|---:|---:|---:|
| vol-v3-dolthub-oos | 33 | 0.0704 | **0.1768** (2.5× under-deflates) |
| lie-shape-knn | 60 | 0.0722 | **0.1302** (1.8× under-deflates) |
| relational | 1241 | 0.0157 | 0.0284 (1.8× under-deflates) |

**Variance decomposition** of the workspace 0.245 empirical:
- Total observed cross-trial std (39 factor arms): 0.245 ann
- Pure null component at typical arm n_obs=234: ≈ 0.234 ann
- **Structural-only residual: √(0.245² − 0.234²) ≈ 0.072 ann**

**Fix**: `standardize_oos` computes effective `sharpe_std_pp` as:

```
sharpe_std_pp_effective = sqrt(
    (1/sqrt(n_obs-1))**2          +    # null estimation noise floor
    (struct_only_ann / sqrt(ppy))**2   # workspace structural component
)
```

where `struct_only_ann = 0.072` (decomposed from the 0.245 empirical).

**Effect**: every short-stream arc's deflated-t drops materially.
Expected reordering — vol-v3-dolthub-oos's published +5.55 likely
falls to ~+2 range; lie-shape-knn-LS becomes more decisively
confirmed-null.

### Step 2 — `commission_bps` realism for options arcs

**Finding**: vol-v3 currently uses `commission_bps=0` in its stream
(vol-points accounting), and the DSR ladder treats every other arc
at `commission_bps=10`. For actual options-strangle deployment,
typical round-trip frictions are 100-500 bps (paper Alpaca: closer
to 100-200 bps on liquid OPRA names).

**Fix**: re-run vol-v3 stream with `commission_bps=200` applied at
the per-rebal level. Add a sensitivity grid `{50, 100, 200, 400}` to
the leaderboard finding row so readers see the dependence.

**Effect**: vol-v3-dolthub-oos's ann Sharpe likely halves under
realistic options costs. Combined with Step 1's null-floor fix, the
arc drops from rank #1 to mid-pack.

### Step 3 — per-arc-class `sharpe_std_ann` empirical re-measurement

**Finding**: the workspace 0.25 was measured on one arc class (factor
sweeps). Other arc classes plausibly have different cross-config
dispersion (vol gate sweeps, relational arm sweeps, etc.).

**Approach**: walk the actual Optuna study artifacts where they
exist (DCA basket search, DCA × vol ensemble, factor sweeps,
relational arms). For each, compute the actual cross-trial std of
*annualized* Sharpes from the trials. Replace the single global 0.072
structural-only with per-arc-class values:

| arc class | candidate sharpe_std (ann, structural-only) | source |
|---|---|---|
| factor walk-forward | 0.072 (existing) | 39 arms (the only empirical we have) |
| vol gate / surface | TBD | tbd from vol-v0..v3 sweep history |
| relational | TBD | tbd from 8-arm sweep |
| DCA basket | TBD | 200-trial Optuna study just landed |
| ensemble | TBD | 200-trial joint search just landed |

For arc classes without empirical data: keep 0.072 as workspace
default, flag as inheriting calibration.

### Step 4 — Ledoit-Wolf common-window Sharpe-difference column

**Finding**: the methodology agent's headline recommendation. The
literature-canonical cross-arc test is:

> For each pair of arcs (A, B) — compute the Sharpe difference
> `ΔSR = SR_A - SR_B` on the date-aligned common window with frequency
> collapsed to the lower (block) frequency. Compute the studentized
> stationary-bootstrap CI via Ledoit-Wolf (2008). Report `ΔSR ± CI`.

**New column**: for each arc on the ladder, compute `ΔSR vs DCA-canonical`
on the common-window overlap. Provide 95% CI. Arcs whose CI excludes 0
are *significantly different* from DCA; arcs whose CI includes 0 are
statistically indistinguishable.

**Effect**: the ladder still has a deflated-t column (DSR-as-gate),
but the new primary "is X better than DCA?" question is answered by
the Ledoit-Wolf CI column directly. Most current "above-DCA" claims
likely fail this test.

### Step 5 — stratified ladder

**Finding**: DSR conflates standalone vs overlay arcs (different
scales), heterogeneous walk-forward structures (rolling vs
single-split vs block-aggregated), and arc classes (factor / gate /
relational / etc.).

**Fix**: stratify the ladder by:
- arc class (cross-sectional factor, regime overlay, multi-asset
  passive, options surface, ensemble)
- sample structure (rolling walk-forward, single train/val split,
  block-aggregated single-pass)
- universe (mega-cap, broad equity, multi-asset ETF, single-name
  options)

Within each stratum, rank by deflated-t and add the Ledoit-Wolf ΔSR
column. **Cross-stratum ranking is explicitly NOT meaningful** — the
ladder will document this.

## Falsification bar (locked, before re-publish)

After all five steps land, the **republished ladder** must satisfy:

1. **Within-arc consistency**: same arc evaluated under the new
   methodology must give a deflated-t within ±50% of the prior value
   (i.e., we're not accidentally inverting verdicts).
2. **DCA stays the canonical anchor**: any methodology change that
   demotes DCA's existing partial-OOS claim to confirmed-null breaks
   too much downstream; flag for separate investigation if it happens.
3. **No new confirmed-OOS claims** from the methodology change alone.
   If an arc that was partial-OOS becomes confirmed-OOS purely from the
   per-arc-class sharpe_std re-calibration, that's a data-snooping red
   flag — pre-register a tighter calibration test first.
4. **The Ledoit-Wolf CI column must exclude 0 for AT MOST one arc**
   under realistic friction assumptions. If multiple arcs claim
   statistically-significant lift over DCA after Steps 1-3 corrections,
   the calibration was over-aggressive.

## Cross-links

- Methodology brief: `.research-walkforward-apples-to-apples.md`
- Assumptions audit: `.research-findings-assumptions-audit.md`
- Ensemble Optuna result: `findings/dca-vol-ensemble-optuna.md`
- DSR baseline: `findings/deflated-sharpe-leaderboard.md`

## Commit sequencing

| step | files touched | new ladder behavior |
|---|---|---|
| 1 | `packages/portfolio/src/ss_portfolio/deflated.py` + tests; `compute_dsr.py` re-run | every short-stream arc's deflated-t drops |
| 2 | new vol-v3-realistic-cost dump + ArcSpec; finding update | vol-v3 falls from rank #1 |
| 3 | `compute_dsr.py` per-arc-class `sharpe_std_ann`; ArcSpec entries | per-arc calibration; documented |
| 4 | new Ledoit-Wolf CI script + ladder column | primary cross-arc test added |
| 5 | leaderboard.md restructure into strata; nav update | ladder communicates what it actually measures |

Each step is atomic. After step 5, write a closing-finding page
`findings/ladder-methodology-rewrite.md` that summarizes the before/after
state and the rules for new arcs.
