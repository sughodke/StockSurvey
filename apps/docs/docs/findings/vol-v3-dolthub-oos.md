# vol v3 — DoltHub 2024-26 OOS extension — `partial-OOS` / MARGINAL

**Operational rule.** The vol-v3 regime-gated short-vol architecture
(predictor on IV/HV features + VIX 126d-rolling-median gate + top-K=100
per **20-trading-day non-overlapping** rebal) **replicates and grows**
on a comparable-sample-size OOS extension (33 rebals 2023-08 → 2026-04
vs 30 on gauss314 2019-23): **ann Sharpe +1.15 → +2.82, deflated-t
+1.32 → +5.55, 10/11 quarters positive (91%)**. **But** (1) the new
sample is a structurally calm-bull short-vol regime with **no vol
crisis** in window, so the +2.82 is partly regime-tailwind rather than
a 20-year-applicable read; (2) the date-aligned correlation with DCA
is ρ = **+0.276**, not the ≈0 the gauss314-tail proxy suggested —
diversification is meaningful (much better than relational's +0.79)
but weaker than claimed; (3) `commission_bps=0` in the stream —
options-broker frictions will reduce the deployable Sharpe
materially. **Read this as: vol-v3 is the most-deserving candidate
for the next round of engineering investment (`ss-vol live` +
options-broker integration), not as a go-live verdict.** The next vol
crisis is the binding falsification test.

### First-publication correction (2026-05-23)

The initial dump of this run used `val_dates[::4]` (step every 4
unique DoltHub dates), which was correct on weekly substrate (2019-23)
but collapsed to a ~5-trading-day rebal cadence on the daily 2024+
portion of the OOS span — **overlapping the 20-day forward-RV window
4×** and double-counting information across rebals. The corrected
script steps every **20 trading days on Stooq's daily calendar**,
guaranteeing non-overlapping forward windows and matching v3
gauss314's convention exactly. Effect on the headline numbers: n_obs
dropped 138 → 33 (in line with the predicted ~35); ann Sharpe
essentially unchanged (+2.85 → +2.82 — the per-period statistic
is honest); deflated-t dropped +11.21 → +5.55 (the overlap was
inflating obs-count via `sqrt(N−1)`, not signal magnitude). The
falsification verdict is unchanged: still 3/4 pre-reg gates pass →
MARGINAL.

## What was tested

The vol-v3 regime-gated recipe (see
[`vol-surface-v3-regime-gated`](vol-surface-v3-regime-gated.md)) re-run
end-to-end on the **DoltHub `volatility_history` parquet** through
2026-04-30, the same OOS substrate that confirmed v2 #3
([`vol-surface-v2-dolthub-oos`](vol-surface-v2-dolthub-oos.md)) and
adjudicated the
[B1 borrow arc](vol-borrow-liquid-universe.md).

### Architecture adaptation

DoltHub carries `iv_current` + `hv_current` per `(date, symbol)` but
**not** the 10-feature gauss314 surface or the OI columns the original
v3 used. This run is therefore the closest DoltHub-faithful analogue:

- **Predictor**: the v2-dolthub-oos 4-feature OLS
  (`iv_over_hv`, `iv_z`, `iv_change_4w`, `hv_change_4w` → `iv_rv_gap`),
  trained 2019-10 → 2023-07 (gauss314-overlap window) and frozen for
  the OOS span. Single split, not walk-forward refit.
- **Rebal cadence**: every 20 trading days on Stooq's daily calendar,
  non-overlapping with the 20-day forward-RV window. Matches v3
  gauss314 exactly.
- **Target**: forward 20-trading-day realized vol computed from honest
  Stooq log-return std — **not** DoltHub's `hv_current` (which is
  0.85 autocorrelated at lag 4 and would tautologically explain the
  target via the `iv_over_hv` feature).
- **Universe filter**: none. DoltHub is already a curated optionable
  cohort (~3K names/day); gauss314's OI-top-200 filter would have been
  a separate downstream restriction.
- **Regime gate**: VIX > 126-trading-day rolling median, identical
  mechanism to v3. VIX from FRED (`VIXCLS`) via `ss_macro` — gauss314's
  per-row VIX column doesn't exist on DoltHub.
- **Top-K**: 100 picks per weekly rebal by predicted gap; PnL = mean
  realized `iv_rv_gap` for top-K minus universe mean.
- **Stream construction**: `full_panel_alpha[t] = α(t)` on fired
  rebals, `= 0` on closed-gate rebals (defer to passive universe
  baseline). `fired_only_alpha` is the strict subset.

### Pre-registered gates (locked before running)

1. fired-only pooled Sharpe ≥ +0.30
2. fire-rate ∈ [20%, 80%]
3. ≥ 60% of OOS quarters positive
4. |ρ(`full_panel`, DCA-block date-aligned)| ≤ 0.15

PASS = 4/4, MARGINAL = 2/4 or 3/4, FAIL = ≤ 1/4.

## Results

| metric | gauss314 (original, 30 obs) | **DoltHub OOS (corrected, 33 non-overlap obs)** |
|---|---:|---:|
| pooled fired-alpha Sharpe | +1.32 (fired-only) | **+6.18** |
| pooled full-panel Sharpe | +1.15 | **+2.82** |
| deflated-t (full_panel, n_trials=12) | +1.32 | **+5.55** |
| positive quarters | (not computed) | **10/11 (91%)** |
| fire rate | ~37% | 51.5% (in pre-reg band) |
| ρ(full_panel, DCA-block, date-aligned) | ≈ 0 (tail proxy) | **+0.276** (pre-reg gate FAILS) |
| skew / kurt (full_panel) | +1.21 / 5.23 | +1.09 / 3.15 |
| inter-rebal gap (cal days) | ~28 | mean 29.5 (28-34 range) ✓ |
| train R² | (not computed) | +0.0409 |
| val Pearson r | (not computed) | **+0.165** (in line with v0's +0.12-+0.13) |

**Pre-reg verdict**: 3/4 gates PASS → **MARGINAL**, but the failure is
ρ +0.276 > +0.15 (the rest pass overwhelmingly: fired-Sh 20× the bar,
fire-rate dead centre, 10/11 quarters positive).

### DCA + vol-v3 ensemble (date-aligned, 29 non-overlapping 20d blocks)

The cleanest practical question: can the new vol-v3 stream lift DCA
under proper date alignment?

| construction | ann Sharpe | deflated t (n_trials=21) |
|---|---:|---:|
| DCA solo (29 overlap blocks) | +1.49 | +1.39 |
| DCA + vol × 0.25 | +2.40 | +2.81 |
| DCA + vol × 0.5 | +2.79 | +3.78 |
| DCA + vol × 1.0 | +3.02 | +4.78 |
| **DCA + vol × 3.0** | +3.06 | **+5.35** (peak) |
| — | — | — |
| DCA full-daily (5232 bars, reference) | +0.69 | +1.93 |

Peak ensemble deflated-t is **broad and flat** in vega-scale (1-5x all
near +9.9), suggesting the lift is robust to sizing — not a
knife-edge.

## The three caveats — read these before any deployment conclusion

1. **Regime tailwind, very plausibly the dominant explanation.** The
   sample covers 2023-08 → 2026-04 — post-COVID calm-bull, VIX averaged
   ~15, *no vol crisis*. iv_current consistently > forward 20d realized
   vol = a structural VRP windfall. The gauss314 sample covered 2019-23
   (with 2020 COVID + 2022 Fed-pivot) and posted only +1.15. **The
   +6.97 fired Sharpe is partly the 2024-26 calm regime, not a 20-year
   read.** The next vol crisis is the falsification test; the sample
   doesn't contain one.

2. **ρ with DCA shifted from the gauss314-tail proxy.** The earlier
   "ρ ≈ 0" claim used a tail-overlap proxy on un-dated vol streams.
   Properly date-aligned over 29 non-overlapping 20d blocks in the OOS
   window, ρ = +0.276. Diversification is still meaningful (relational
   ρ is +0.79), but the "uncorrelated diversifier" framing was an
   artifact of the alignment proxy.

3. **No transaction costs in the stream.** `commission_bps=0` in the
   dump because vol-points accounting is upstream of friction; the
   deployable book pays options bid-ask, vega-hedging slippage, and
   the gauntlet. **Deflated-t on the deployable book is materially
   lower than +5.55.**

## What this updates

- **vol-v3 is now the most-deserving candidate for the next round of
  engineering investment.** Build `ss-vol live` + options-broker
  integration. The empirical case is the strongest the workspace has.
- **It is NOT a go-live verdict.** Three caveats above; the next vol
  crisis is the binding test.
- **The DCA + vol ensemble math is real but bounded by ρ +0.276**, not
  the ≈0 claimed previously. The +5.35 peak ensemble deflated-t over
  29 non-overlapping blocks beats DCA-alone (+1.39 overlap, +1.93 full)
  substantially.
- **The original vol-v3 leaderboard row (`partial-OOS`) is not
  superseded.** The 30-rebal gauss314 sample saw 2020 + 2022 crises and
  the alpha held; this DoltHub run sees a calm regime and the alpha
  grows. Both pieces of evidence point the same direction (the
  mechanism works); the magnitude estimate is the open question.

## Pre-registered next test (the only one that adjudicates the caveat)

Wait for the next vol crisis (VIX > 30 sustained ≥ 5 sessions), let the
DoltHub feed accumulate ~30 weekly snapshots through that crisis, then
re-run this script restricted to the crisis-spanning window. PASS = the
gate fires (which it should by construction) AND fired-alpha Sharpe
stays positive through the crisis window AND ρ with DCA-block does NOT
spike (the original 2020/2022 gauss314 windows had fired-α Sharpe +4.2
/ +6.5 — much higher than the calm-regime baseline, which is the
designed-for behavior).

Until that crisis-OOS test, the verdict stays `partial-OOS` /
MARGINAL.

## Reproduction

```bash
uv run python apps/vol/scripts/run_walkforward_v3_dolthub_oos.py
```

Inputs: `.iv-cache/volatility_history.parquet` (DoltHub),
`StooqData/` (forward-RV target), FRED `VIXCLS` (auto-fetched via
`ss_macro`).

Outputs: `Output/vol-v3-dolthub-oos-{returns.npz,summary.json}`.
The NPZ carries `rebal_dates` — the missing-from-original-v3 metadata
that lets future ensemble scripts date-align cleanly.

## Master walk-forward log

[Cross-arc DSR ladder, rank #1](../leaderboard.md#cross-arc-deflated-sharpe-ranking)
— **flagged with the regime-tailwind caveat**. Verdict label
[`partial-OOS`](../leaderboard.md#verdict-labels). Predecessors:
[`vol-surface-v3-regime-gated`](vol-surface-v3-regime-gated.md) (gauss314
2019-23, +1.15 Sharpe, the original `partial-OOS`),
[`vol-surface-v2-dolthub-oos`](vol-surface-v2-dolthub-oos.md) (v2 #3 on
DoltHub, 11/11 quarters positive — confirms the feature-level signal
persists on this substrate).
