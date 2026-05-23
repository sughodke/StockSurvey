# vol v3 — DoltHub 2024-26 OOS extension — `partial-OOS` / MARGINAL

**Operational rule.** The vol-v3 regime-gated short-vol architecture
(predictor on IV/HV features + VIX 126d-rolling-median gate + top-K=100
per weekly rebal) **replicates and grows** on a 4.6× larger never-seen
sample (138 weekly rebals 2023-08 → 2026-04 vs 30 on gauss314 2019-23):
**ann Sharpe +1.15 → +2.85, deflated-t +1.32 → +11.21, 11/11 quarters
positive**. **But** (1) the new sample is a structurally calm-bull
short-vol regime with **no vol crisis** in window, so the +2.85 is
partly regime-tailwind rather than a 20-year-applicable read; (2) the
date-aligned correlation with DCA is ρ = **+0.215**, not the ≈0 the
gauss314-tail proxy suggested — diversification is meaningful (much
better than relational's +0.79) but weaker than claimed; (3)
`commission_bps=0` in the stream — options-broker frictions will reduce
the deployable Sharpe materially. **Read this as: vol-v3 is the
most-deserving candidate for the next round of engineering investment
(`ss-vol live` + options-broker integration), not as a go-live
verdict.** The next vol crisis is the binding falsification test.

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

| metric | gauss314 (original, 30 obs) | **DoltHub OOS (new, 138 obs)** |
|---|---:|---:|
| pooled fired-alpha Sharpe | +1.32 (fired-only) | **+6.97** ⚠️ |
| pooled full-panel Sharpe | +1.15 | **+2.85** |
| deflated-t (full_panel, n_trials=12) | +1.32 | **+11.21** ⚠️ |
| positive quarters | (not computed) | **11/11 (100%)** |
| fire rate | ~37% | 48.6% (in pre-reg band) |
| ρ(full_panel, DCA-block, date-aligned) | ≈ 0 (tail proxy) | **+0.215** (pre-reg gate FAILS) |
| skew / kurt (full_panel) | +1.21 / 5.23 | +1.01 / 2.89 |
| train R² | (not computed) | +0.0409 |
| val Pearson r | (not computed) | **+0.165** (in line with v0's +0.12-+0.13) |

**Pre-reg verdict**: 3/4 gates PASS → **MARGINAL**, but the failure is
ρ +0.215 > +0.15 (the rest pass overwhelmingly: fired-Sh 23× the bar,
fire-rate dead centre, all 11 quarters positive).

### DCA + vol-v3 ensemble (date-aligned, 115 overlap blocks)

The cleanest practical question: can the new vol-v3 stream lift DCA
under proper date alignment?

| construction | ann Sharpe | deflated t (n_trials=21) |
|---|---:|---:|
| DCA solo (115 overlap blocks) | +1.27 | +2.12 |
| DCA + vol × 0.25 | +2.21 | +4.54 |
| DCA + vol × 0.5 | +2.61 | +6.19 |
| DCA + vol × 1.0 | +2.83 | +8.22 |
| **DCA + vol × 3.0** | +2.86 | **+9.87** (peak) |
| DCA + vol × 5.0 | +2.84 | +9.97 |
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
   Properly date-aligned over 115 blocks in the OOS window, ρ = +0.215.
   Diversification is still meaningful (relational ρ is +0.79), but
   the "uncorrelated diversifier" framing was an artifact of the
   alignment proxy.

3. **No transaction costs in the stream.** `commission_bps=0` in the
   dump because vol-points accounting is upstream of friction; the
   deployable book pays options bid-ask, vega-hedging slippage, and
   the gauntlet. **Deflated-t on the deployable book is materially
   lower than +11.21.**

## What this updates

- **vol-v3 is now the most-deserving candidate for the next round of
  engineering investment.** Build `ss-vol live` + options-broker
  integration. The empirical case is the strongest the workspace has.
- **It is NOT a go-live verdict.** Three caveats above; the next vol
  crisis is the binding test.
- **The DCA + vol ensemble math is real but bounded by ρ +0.215**, not
  the ≈0 claimed previously. The +9.87 peak ensemble deflated-t over
  115 blocks beats DCA-alone (+2.12 overlap, +1.93 full) substantially.
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
