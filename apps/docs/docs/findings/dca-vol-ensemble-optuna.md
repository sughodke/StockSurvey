# DCA × vol overlay joint Optuna search — `partial-OOS`

**Operational rule.** A pre-registered Optuna search over 16,800
(DCA-basket × vega_scale) combinations (N_TRIALS = 200, train rebal
0-19 / val 20-32 of the 33-obs vol-v3-DoltHub OOS sample,
deflated-Sharpe objective) **failed to find an ensemble that beats
the canonical 13-ETF + vol × 3.0 incumbent by the pre-reg's +1.0
deflated-t bar.** The winner — a stripped-down **SPY + GLD + vol × 3.0**
mix at 80d rebal — beats canonical by Δ val deflated-t = **+0.612**,
which the locked rule labels `partial-OOS`. The vol overlay × 3.0
sizing is consistent across the entire top-10 (TPE collapsed onto
vega_scale=3.0 as the joint search's preferred sizing). **The
canonical 13-ETF + vol × 3.0 recipe is defensible; the search did
not find a defensible alpha lift over it.** A minor operational note:
the winner is structurally identical to the basket-only arc's finding
— a simpler basket delivers near-identical val Sharpe; complexity in
the cash leg is replaceable.

## Pre-registration

Locked at [`TODO/dca-vol-ensemble-optuna.md`](../TODO/dca-vol-ensemble-optuna.md)
in commit `1a06966` (2026-05-23) BEFORE the eval ran. The
falsification bar, search space, walk-forward split, trial budget,
and DSR calibration were all fixed before any Optuna trial fired.

## Result

### Headline numbers (winner vs canonical, evaluated under identical method)

| metric | canonical 13-ETF + vol × 3.0 | Optuna winner |
|---|---:|---:|
| basket | 9 SPDR sectors + TLT/IEF + GLD/DBC (13) | **SPY + GLD** (2) |
| vega_scale | 3.0 | **3.0** |
| rebal_days | 80 | **80** |
| train deflated-t (n=20) | +3.892 | +4.080 (search target) |
| **val deflated-t (n=13)** | **+4.083** | **+4.695** |
| val ann Sharpe | +2.282 | +2.769 |
| val max-DD | −0.010 | −0.008 |
| val skew | +2.54 | +2.51 |
| val kurtosis | 8.53 | 8.54 |

### Verdict per pre-reg

- Δ val deflated-t (winner − canonical) = **+0.612**
- Δ val max-DD                          = **+0.002** (winner marginally better)
- Pre-reg bar: confirmed-OOS requires Δ deflated-t > +1.0 AND Δ
  max-DD > −0.05
- Pre-reg bar: partial-OOS requires Δ deflated-t ≥ 0.0 AND Δ max-DD
  > −0.05
- **Locked verdict: `partial-OOS`** (Δ deflated-t is positive but
  below the +1.0 confirmed-OOS bar; the canonical recipe is
  defensible and the winner represents simplification rather than
  alpha lift)

### Why the winner is so close to canonical

The top-10 trials by train deflated-t **all carry vega_scale = 3.0**.
Eight of the top-10 are exactly the same config (SPY + GLD + 80d).
TPE collapsed onto the vega axis hard — the basket axis is
near-degenerate at this sample size:

| rank | equity | intl | bonds | commod | reit | rebal | vega | train t |
|---:|---|---|---|---|---|---:|---:|---:|
| 1-8 | SPY-only | none | none | GLD-only | none | 80 | **3.0** | +4.080 |
| 9-10 | SPY-only | none | none | GLD-only | none | 21 | **3.0** | +4.074 |

The vol overlay × 3.0 dominates the variance budget at scale=3.0
applied to per-period alpha of ~0.05-0.15 vol-points: the DCA cash leg
contributes ~+0.20-0.40 train Sharpe; the vol × 3.0 contributes the
remaining +3.0+ Sharpe. **The basket choice is structurally below the
noise floor** at this n_train=20 / n_val=13 sample size — any
diversifying basket that doesn't blow up max-DD ties on deflated-t.

## What we learned

### vega_scale=3.0 is robust across the joint search

The pre-reg's honest prior was right on this: the
[`vol-v3-dolthub-oos`](vol-v3-dolthub-oos.md) finding noted the peak
ensemble deflated-t was "broad and flat" across vega ∈ {1, 2, 3, 5}.
The joint search confirms it — every top-10 config picks vega=3.0,
none of the basket variations move the joint optimum off that scale.
**Sizing at 3.0 is the load-bearing choice; basket is a sideshow at
this sample size.**

### The 2-ETF (SPY + GLD) winner is structurally identical to the basket-only arc's finding

The [`dca-basket-optuna`](dca-basket-optuna.md) arc (basket-only,
20-year sample) found a 4-ETF winner (VTI + TLT + IEF + GLD)
delivering near-identical val Sharpe to canonical-13. This joint
search collapses further to a 2-ETF mix (SPY + GLD) — because:

1. The 33-obs OOS window is **calm-bull regime**, so the bond ballast
   (TLT/IEF) and commodity diversifier (DBC) are not contributing
   variance reduction that matters at the 20-obs train sample.
2. SPY ≈ VTI ≈ 9-sectors-EW in this short window — exposure is the
   same; the choice between them is below noise.

This is the joint-search version of the basket arc's finding: **the
canonical complexity is replaceable without loss when the sample is
thin and calm; the joint search didn't find a basket that *adds*
alpha.**

### The pre-registered bar was the right discipline

Without the +1.0 bar locked in advance:
- +0.612 looks like a "real lift" — 50% bigger absolute and would be
  tempting to call `confirmed-OOS`
- We'd snoop the 2-ETF winner and switch live deployment
- The +5.35 ensemble-peak claim from the vol-v3 finding (n_trials=21)
  would compound with a +4.70 joint-search winner claim (n_trials=200),
  inflating the documented edge

With the bar locked: +0.612 is correctly priced as "below the threshold
that distinguishes real edge from noise at 13 val obs." The canonical
recipe — or the winner — both pass; the search did not find a
defensible alpha lift.

### What the search did NOT test (acknowledged in pre-reg)

- **Regime sensitivity.** The 33-obs sample is the
  [vol-v3-dolthub-oos](vol-v3-dolthub-oos.md) calm-bull window with
  no vol crisis. The same search re-run after a vol crisis could
  produce a different winner (bond ballast suddenly load-bearing,
  vega_scale optimum lower for tail-risk management).
- **Dynamic vega_scale.** All trials use a fixed vega_scale across
  the OOS span. A regime-conditional scale (e.g. vega smaller when
  VIX > X) is a higher-dimensional search not in pre-reg scope.
- **Vol stream variants.** Frozen at `full_panel_alpha` from
  `vol-v3-dolthub-oos-returns.npz`. Alternative vol-overlay streams
  (fired-only, different gate cadences) would be a different arc.
- **Sub-ETF cash leg.** Bucketed ETF families only; arbitrary-ticker
  cash books out of scope by design.

## Honest caveats

1. **The +0.612 lift is within the noise band at n_val=13.** The
   null s.e. on a deflated-t at 13 obs is ~0.5-0.8 depending on
   moments; +0.612 sits inside one standard error.
2. **Regime-tailwind dominates everything.** Both canonical (+4.08)
   and winner (+4.70) val deflated-t are **inflated relative to
   crisis-inclusive expectations** by the vol-v3-dolthub-oos finding's
   regime-tailwind caveat (see §"The three caveats"). Read both as
   relative comparisons, not absolute deployable numbers.
3. **No transaction costs on the vol leg.** The vol-v3 stream is
   `commission_bps=0`. Options-broker frictions will reduce the
   deployable Sharpe of the *ensemble* — equally for canonical and
   winner — but the relative comparison stands.
4. **DCA-leg sample bias.** The DCA daily returns are computed over
   a 4-year buffer window (2022-01 → 2026-04) including the 252d
   trailing-Sharpe lookback for the top-3-by-trailing-sharpe
   dynamic basket. This is sufficient for the per-block compounding
   but the trailing-Sharpe basket may be sensitive to the lookback
   start; not load-bearing because the winner is `SPY-only`.

## Implications

### For the live DCA + vol ensemble

- **Canonical 13-ETF + vol × 3.0 stays defensible** — the search did
  not find a defensible alpha lift over it.
- **Optional simplification**: SPY + GLD + vol × 3.0 delivers
  near-identical val deflated-t with one-sixth the holdings on the
  cash leg. This is a maintenance / operational decision, not an
  alpha decision. **Do not deploy the winner over canonical** —
  the joint search proved they're equivalent under the bar, not that
  the winner is better.
- **vega_scale = 3.0 is the robust sizing.** Every top-10 picks it;
  this is the load-bearing finding of the arc.

### For future ensemble searches

- The +1.0 deflated-t bar is high but defensible at thin val
  samples. Future joint searches with n_val < 20 should set similar
  bars; otherwise the search becomes a snooping device on a tiny
  sample.
- Joint (basket × overlay-scale) search with 13 val obs is **basket-
  insensitive**. Future work should focus on the overlay-side levers
  (vega_scale schedule, regime gate variants) rather than basket
  composition until the vol stream accumulates more OOS obs.
- N_TRIALS = 200 is fine — TPE converged in ~30 trials; the marginal
  gain from trial 30-200 was noise.

## Reproduction

```bash
uv run python apps/dca/scripts/optuna_dca_vol_ensemble.py
```

Inputs: `StooqData/` (DCA cash leg), `Output/vol-v3-dolthub-oos-returns.npz`
(vol-v3 alpha stream, 33 rebals 2023-08 → 2026-03).
Outputs: `Output/dca-vol-ensemble-optuna.json`. Runs in <10 seconds
on a local laptop (Optuna TPE + thin numpy block-return compounding).

## Master walk-forward log

[Cross-arc DSR ladder](../leaderboard.md#cross-arc-deflated-sharpe-ranking)
— this search's val deflated-t (winner +4.70, canonical-on-val
+4.08, both at n_trials=200) is **not directly comparable** to the
vol-v3 row's +5.55 (n_trials=12, full 33-obs sample) — the trial
counts and obs counts differ. The comparison that matters is the
winner-val (+4.70) vs canonical-val (+4.08) under identical method.
Verdict label
[`partial-OOS`](../leaderboard.md#verdict-labels).
