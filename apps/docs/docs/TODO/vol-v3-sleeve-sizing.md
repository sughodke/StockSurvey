# Vol_v3 sleeve sizing — what fraction of DCA capital should overlay vol_v3?

**Status:** **superseded 2026-05-24** by
[`findings/vol-sleeve-sizing`](../findings/vol-sleeve-sizing.md).
Verdict: [`partial-OOS`](../leaderboard.md#verdict-labels).

**Headline result:** recommended `vega_scale = 2.0, c_options_bps ≤ 200`.
Combined Sharpe +2.46 (vs DCA-only +1.30), ΔSR_ann +1.174 with
LW 95% CI [+0.028, +2.600] (barely excludes 0), combined max-DD
−4.9% (better than DCA-only −6.8%), deflated-t +2.74 (BELOW the
+3.0 hurdle locked in the pre-reg). At c_options_bps = 400 every
cell's CI includes 0. Per the locked verdict logic this lands at
`partial-OOS` not `confirmed-OOS` — clears at 200 bps, collapses at
400 bps. See finding for the full heatmap + per-cell numbers.

**Next-experiment chain (per CLAUDE.md `partial-OOS` rule —
"stratify the windows"):**

1. **Sleeve-vs-DCA correlation regime stratification.** The +0.276
   correlation between vol_v3 and DCA is the population number across
   the 33-rebal sample; what does the per-rebal correlation look like
   in VIX-high vs VIX-low subsamples? If correlation rises in
   crisis (which is when the sleeve is meant to fire), the +2.46
   combined Sharpe overestimates crisis behaviour.
2. **Options-broker integration ticket.** The friction-grid result
   says "if you can hit c_options_bps ≤ 200 in practice, deploy
   vega_scale=2.0." The unknown is whether free-tier Alpaca options
   actually hits that bar at top-200 OI. Pre-reg the dry-run
   measurement: run `apps/vol/scripts/sleeve_live_dryrun.py` daily
   for one VIX-gate-fired cycle, log realized bid-ask spreads, see
   if mean realized c_bps clears 200 with paper-tier rate limits.
3. **A sixth risk rail in `vol/live.py` — realized friction monitor.**
   "Abort if rolling-3-rebal realized c_options_bps > 250" (25%
   cushion over the recommended 200). Required for safe deployment;
   not yet built.

---

## Original pre-reg (locked before eval ran)

**Status:** pre-reg, open.

## Provenance

Spawned by the 2026-05-24
[meta-allocator-no-vol-v3 finding](../findings/meta-allocator-no-vol-v3.md).
The 2026-05-23 meta-allocator analysis found "inverse-arc-vol beats
1/N" but the 2026-05-24 falsification showed that result was vol_v3-
carried — once vol_v3 is excluded, the 5-arc inverse-vol allocator
ties 1/N to within Ledoit-Wolf 95% CI (ΔSR_ann +0.039 [−0.152,
+0.220], `confirmed-null`).

The verdict-implied next experiment per CLAUDE.md's `confirmed-null`
rule is *not* "test another allocator over the arc bundle" — that
lever is exhausted on this panel. The orthogonal lever is **direct
sleeve sizing**: how much capital should overlay vol_v3 on top of
canonical DCA?

This dovetails with the already-landed
[`dca-vol-ensemble-optuna` finding](../findings/dca-vol-ensemble-optuna.md)
which collapsed onto vega_scale=3.0 (8/10 top trials) on the 33-rebal
OOS slice. That result is the joint Optuna-Optuna's best guess; this
TODO is the falsifiable validation that 3.0 actually clears a locked
bar against the canonical-13 + vol×3 baseline on the orthogonal axis
of *sleeve-fraction stability across realistic-friction stress*.

## Question

What is the maximum vega_scale (effective vol_v3 sleeve overlay
fraction on top of DCA capital) such that the resulting ensemble
clears Ledoit-Wolf 95% CI over canonical DCA *and* maintains positive
deflated-t under 400 bps options friction (the upper-bound friction
stress documented in
[`ladder-methodology-rewrite`](../findings/ladder-methodology-rewrite.md))?

## Steel-manned mechanism — why this should work

Vol_v3's deflated-t under realistic 200 bps friction is +3.43 (from
the methodology-rewrite Step 2). Its date-aligned correlation with
DCA is +0.276 (positive but small enough that diversification still
applies). Per Markowitz / sleeve-sizing theory, if `μ_vol, σ_vol`
characterize vol_v3 alpha and `μ_dca, σ_dca` characterize DCA, the
optimal sleeve fraction under quadratic utility is
`f* = (μ_vol × σ_dca² − ρ × μ_dca × σ_vol × σ_dca) /
(μ_vol × σ_vol² × (1−ρ²))` — which at our measured numbers (vol_v3
ann Sharpe ~+2.0 post-friction, DCA ~+0.67, ρ +0.276) yields a
sleeve fraction in the 50%–150% range *under perfect information*.
vega_scale=3.0 in the joint Optuna selected this neighborhood
empirically.

The mechanism for "this could clear the bar at a friction harsher
than 200 bps" is that vol_v3's pre-friction edge is large enough
(+5.86 unrestricted Sharpe per the v1 finding) that even at 400 bps
the post-friction Sharpe stays well above +1.0, well within the
sleeve-sizing optimum range.

## Test design (PRE-REGISTERED)

**Universe:** vol-v3-DoltHub-OOS 33-rebal stream (frozen artifact
`Output/vol-v3-dolthub-oos-returns.npz`) × DCA canonical-13 daily
stream (frozen artifact `Output/cfr_phase4d_multiasset_close.pkl`).

**Windowing:** same single-split walk-forward as
[`dca-vol-ensemble-optuna`](../findings/dca-vol-ensemble-optuna.md) —
train rebals 0–19, val rebals 20–32.

**Arms:** `vega_scale ∈ {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}`
× `c_options_bps ∈ {0, 100, 200, 400}` — pre-frozen vol-v3 returns
re-deflated by the friction stress.

**Decision metric (pre-reg locked):**
1. Pooled OOS Sharpe_ann of the ensemble `r_ens = r_dca + vega ×
   r_vol_after_friction`.
2. Ledoit-Wolf studentized stationary-bootstrap CI (n=2000, seed=42)
   on `Sharpe(r_ens) − Sharpe(r_dca_only)`.
3. Deflated-t under workspace `sharpe_std_ann=0.072` + null floor.

**Pre-reg verdicts:**
- `confirmed-OOS` = there exists a `(vega, c_bps)` configuration with
  ΔSR_ann ≥ +0.5 AND CI excludes 0 AND deflated-t > +3.0 at c_bps =
  400 (the harshest friction).
- `partial-OOS` = same numbers at c_bps = 200 but not 400.
- `confirmed-null` = no configuration clears CI excludes 0 at any
  c_bps tested, i.e. ensemble Sharpe is within noise of DCA-only
  across the entire (vega, c) grid.

**Pre-reg expected outcome (steel-man side):** I expect `partial-OOS`
or `confirmed-OOS` at c_bps=200 with optimal vega in [2, 4]. The
joint Optuna landing on vega=3.0 is consistent with this prior; the
test is whether the result survives the 400 bps stress.

**Cost of being wrong:** if `confirmed-null` lands, vol_v3 as a
sleeve overlay is operationally fragile to friction — the deployment
recipe collapses back to DCA-only and `apps/vol live` infrastructure
is on hold pending a quote-source upgrade.

## Driver

Extend `apps/dca/scripts/optuna_dca_vol_ensemble.py` into a
grid-search script (`apps/dca/scripts/vol_sleeve_friction_grid.py`)
that doesn't use Optuna — just a 9 × 4 = 36-cell grid. Local-only
(no Modal, < 30s). Reuses the frozen vol-v3 returns artifacts at
each c_bps level
(`Output/vol-v3-dolthub-oos-c{0,100,200,400}-returns.npz` per the
methodology rewrite).

Output:
- `Output/vol-sleeve-friction-grid.json` — full 36-cell table.
- Leaderboard row + findings page on conclusion.
