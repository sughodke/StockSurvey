# Higher-EV move — borrow-stress conditioning on the *liquid* vol universe (the un-reached novel-data leg, re-pointed)

> **CLOSED 2026-05-22 — `reversed-OOS`.** Stage 0 PASSED (coverage 99%,
> real dispersion). Phase A found a strong *in-sample* premium-amplifier
> (high-borrow-stress hi-tercile Sharpe +3.80, direction flipped vs the
> squeeze-tail assumption below). Phase B OOS-tested the flipped hypothesis
> on never-seen DoltHub 2024-26 (552 weeks): **it did NOT replicate** —
> high-stress VRP +0.011 ≤ low-stress +0.014, hi>lo in 36% of quarters. The
> in-sample +3.80 was selection on 11 rebals. Borrow leg closes; do not
> re-attempt without a paid securities-lending feed. Finding:
> [`vol-borrow-liquid-universe`](../findings/vol-borrow-liquid-universe.md).
> The rest of this page is the original (pre-result) plan, kept for record.

> **Why this exists.** The
> [`vol-borrow-illiquid-vrp`](vol-borrow-illiquid-vrp.md) arc died
> `reversed-OOS` purely on **quote-availability**: vendor coverage
> tracks liquidity, so 92.5% of the thin cohort was un-priceable
> ([finding](../findings/vol-borrow-illiquid-vrp-falsified.md);
> [concept](../notes.md#quote-availability-is-a-deployability-gate)).
> The **borrow leg** of that arc — SEC FTD + FINRA daily short-volume
> + short interest as a mechanism-linked conditioning variable — was
> *never reached* ("not falsified, just unreached"). This page
> re-points that exact, un-mined novel-data leg away from the dead
> cohort and onto the **liquid optionable universe**, where the
> quote-availability wall that killed the last attempt
> **structurally cannot bite** (liquid names → DoltHub Stage-0
> already confirmed free per-contract NBBO; gauss314 carries OI+IV).
> It is the orthogonal **novel-data axis** the standing strategic
> frame and the research director selected over more
> cleverer-model-on-standard-data variants (which are
> `confirmed-null` 49× **at the prediction layer with forecast loss
> on standard price/volume data** — see Scope clarification below;
> sizing-layer direct-Sharpe learners are a separate, partially-
> positive story per
> [`findings/learned-ensemble-beats-deterministic.md`](../findings/learned-ensemble-beats-deterministic.md)
> and
> [`notes.md#learner-layer-matters-more-than-learner-complexity`](../notes.md#learner-layer-matters-more-than-learner-complexity))
> and just produced one more benchmark-artifact
> `partial-OOS` in the
> [endogenous-horizon arc](../findings/factor-endogenous-horizon-mixture.md)).

## Scope clarification (2026-05-29)

The "novel-data > cleverer-model-on-standard-data" framing motivating
this TODO is correct, but the supporting "49× confirmed-null"
shorthand applies to **prediction-layer learners with forecast loss
and simplex action space** on standard price/volume data, not to
learners categorically. Sizing-layer learners with direct-Sharpe
loss have been shown to clear the deterministic-recipe bar on the
arc-blend panel
([`findings/learned-ensemble-beats-deterministic.md`](../findings/learned-ensemble-beats-deterministic.md)).
This TODO's "deliberately boring locked-OLS model + novel data"
choice still stands — the data lever is the higher-EV move for this
specific arc — but the rationale is "novel data here, locked model
to avoid burning DOF on a thin dataset," not "learners are
disproven." Framing distinction in
[`notes.md#learner-layer-matters-more-than-learner-complexity`](../notes.md#learner-layer-matters-more-than-learner-complexity).

## The mechanism (not a bolted-on second signal)

The VRP persists because a market-maker who is short vol must
delta-hedge and unwind. **A hard-to-borrow underlying directly
raises that hedging cost** — the MM short a put cannot freely short
the underlying when borrow is scarce/expensive, so demands a fatter
premium. Securities-lending stress is therefore a *measurable proxy
for the exact friction that generates the premium*, not an
orthogonal factor. It is two-edged:

- **Premium amplifier.** High borrow fee / low locate / FTD pressure
  → costlier MM hedging → richer VRP → *go more short vol here*.
- **Tail predictor.** Hard-to-borrow + FTD spike + utilization near
  100% = squeeze setup. Short vol = short gamma; a violent squeeze
  blows up the position *and* the same scarcity means no exit →
  *this rich-looking cell is lethal, skip it*.

The research question: can a borrow-stress-conditioned model
separate **rich-and-safe** from **rich-and-lethal** on the *liquid*
universe? That conjunction is a microstructure interaction —
structurally un-publishable (academia isolates one clean effect),
slow-moving (FTD/SI publish with a multi-day lag — favours a patient
operator over HFT), and on the universe the repo already has a
`confirmed-OOS`/MARGINAL deployable recipe for.

## Stage 0 — borrow-data feasibility gate (HARD PRE-CHECK — run before any Modal)

The strategic memory's HARD PRE-CHECK (learned from the
quote-availability kill): *before sinking a novel-data arc, verify
the novel data is actually retrievable, free, and coverage-complete
over the exact universe and span the signal needs.* The
quote-availability analogue here is **borrow-data-availability**.
Probe, offline, no Modal:

1. **SEC fails-to-deliver** — semi-monthly flat files at
   `sec.gov/data/foiadocsfailsdatahtm` (free, no auth). Confirm:
   parses; spans ≥ 2010 → 2026; keyed by CUSIP/symbol joinable to
   the gauss314 + DoltHub symbol sets.
2. **FINRA daily short-sale volume** — daily consolidated files at
   `cdn.finra.org/equity/regsho/daily/` (free, no auth). Confirm:
   parses; per-symbol daily short vol + total vol; spans the v1/v2
   walk-forward + the DoltHub 2024–26 OOS window.
3. **FINRA bi-monthly short interest** — settlement-date short
   interest (free). Confirm: parses; joinable; lag documented.
4. **Coverage join test.** For a sample rebal date in each of the 5
   v1/v2 windows + 2 DoltHub-OOS quarters: what fraction of the
   *liquid* universe's selected names have all three borrow signals
   present? **Gate: ≥ 90% coverage on the liquid universe** (the
   inverse of the 7.5% that killed the illiquid arc — if liquid-name
   borrow coverage is also thin, this arc dies here, cheaply, and
   that is itself a clean verdict).

**Stage 0 verdict is a gate, not a leaderboard row.** PASS → Phase A
unblocked. FAIL (any source missing, or < 90% liquid-universe
coverage) → record the gate outcome in this page, do **not** spend
Modal, and the novel-data leg is closed for free-data the same way
the illiquid leg was.

### Stage 0 result (2026-05-22): **PASS** → Phase A unblocked

Probe `apps/vol/scripts/borrow_stage0_probe.py` (local, free, ~2 min) over
6 sample dates spanning the gauss314 v1/v2 span (2019-11 → 2023-03),
reconstructing each date's top-200-OI liquid cohort from the local
`.iv-cache/data_IV_USA.csv` and pulling FINRA daily short-volume + SEC FTD:

- **Coverage: 99.3%** mean of the top-200-OI names present in FINRA
  consolidated short-volume (covers all NMS symbols, as expected); SEC FTD
  present for ~99% of the cohort (1 semi-month file 404'd — non-fatal).
- **Dispersion: short-volume-ratio hi−lo tercile gap = 0.266 mean** (0.24–0.28
  every date) → the liquid cohort is **not** degenerate; there is real,
  splittable borrow-stress variation. This retires the pre-registered
  kill-risk (liquid-name borrow-stress too flat to form terciles — the mirror
  of the illiquid coverage death).

**Caveats carried into Phase A:** (1) short-*volume* ratio is the most
available but *noisiest* borrow proxy (much short volume is MM/HFT intraday,
not directional borrow demand); the cleaner free signals are SEC FTD
*magnitude* (not just presence — Stage 0 measured presence) and FINRA
bi-monthly short *interest*. (2) The pure borrow-*cost* signal (loan fee /
utilization) remains paid-data (Markit/IHS); Stage 0 confirms the free
*proxies* are available + dispersed, and Phase A tests whether they actually
condition the VRP. Artifact `Output/vol-borrow-stage0.json`.

## Falsifiable hypotheses

- **H1 (premium):** within the liquid cohort, mean per-rebal VRP
  alpha is higher in the high-borrow-stress sub-cohort than the
  low-borrow-stress one.
- **H2 (tail):** worst single-rebal drawdown is *worse* in the
  high-borrow-stress sub-cohort (squeeze blowups) — defined-risk
  structures must cap this.
- **H3 (separability — the load-bearing cut):** a
  borrow-stress-conditioned selection produces higher net Sharpe
  than the **unconditioned v3 baseline** by ≥ **+0.10**.

H1 ∧ ¬(H2 fatal) ∧ H3 = the deployable edge.

## Phase A — gauss314, the locked v3 recipe + borrow split

**Substrate.** gauss314 full schema (has OI + IV;
`ss_iv.load_gauss314_full`), span ≈ 2019–2023. This is the *only*
substrate that reproduces the repo's one deployable vol recipe, so
the borrow-conditioning delta attributes cleanly against a **locked
`confirmed-OOS`/MARGINAL baseline** rather than a fresh cohort.

**Baseline (locked, do not re-tune — from
[`vol-surface-v3-regime-gated`](../findings/vol-surface-v3-regime-gated.md)):**

| Component | Choice |
|---|---|
| Universe | top-200 OI per date |
| Predictor | v1 10-feature surface, linear OLS (no MLP, no new features — change *data*, not model) |
| Gate | per-rebal-bar VIX[t] > 126d-rolling-median(VIX) (v3 operational best) |
| Top-K | 50 picks per fired rebal |
| Rebal | 20 trading days |
| Sizing | equal-$-vega (v2 #1 confirmed) |
| Structure | **defined-risk vertical spreads only** (never naked short vol — H2 tail cap) |

**Test arm.** Each *fired* rebal: split the 50 picks by a Stage-0
borrow-stress composite (FTD pressure + short-vol ratio + SI
utilization) into low/mid/high terciles. Evaluate H1 (per-tercile
alpha), H2 (per-tercile worst-rebal DD), and the H3 conditioned
selection (e.g. drop the high-stress squeeze-tail tercile, overweight
the high-premium-but-safe band) vs the unconditioned 50-pick v3
baseline.

**Windowing.** The v1/v2 5-window walk-forward (300/120/120, 20d
rebal) — identical to v1/v2/v3 so the baseline is bit-comparable.

**Phase-A pre-registered cuts (fixed verdict vocabulary):**

- **PASS — `confirmed-OOS`:** H3 conditioned net Sharpe beats
  unconditioned v3 by ≥ **+0.10**, H1 holds (high-stress alpha >
  low-stress), H2 not fatal (defined-risk DD ≤ 1.5× baseline
  worst-rebal), positive in ≥ 4/5 windows.
- **MARGINAL — `partial-OOS`:** H3 delta in [+0.05, +0.10), or
  positive in 3/5 windows, or H1 holds but H3 < +0.05.
- **FAIL — `confirmed-null` on the conjunction:** H3 delta < +0.05
  with H1 still holding → borrow-stress is real but doesn't separate
  safe/lethal at deployable granularity; record VRP-alone, drop the
  borrow leg.
- **FAIL — `reversed-OOS`:** conditioned ≤ unconditioned AND H1
  fails → the borrow thesis is wrong on the liquid universe too;
  the whole borrow leg closes.

## Phase B — DoltHub OOS extension (contingent on Phase A)

**Trigger (pre-registered):** run **only if** Phase A H3 delta ≥
**+0.10** (PASS) *or* ∈ [+0.05, +0.10) with H1 holding (MARGINAL
worth OOS-confirming). If Phase A FAILs, Phase B does not run.

**Substrate.** DoltHub `option_chain` (free per-contract NBBO,
Stage-0 confirmed), 2024–2026 — the same span that confirmed v2 #3
signal continuity. OI absent on DoltHub → liquid cohort defined by
the v2 #3 4-feature proxy + bottom-tercile quoted relative spread
(the *observed* transaction cost, strictly more faithful than OI).

**Phase-B cut:** the Phase-A borrow-conditioned selection, applied
unchanged out-of-sample on 2024–26, retains ≥ 60% of its Phase-A H3
delta and stays positive in ≥ 7/11 OOS quarters → `confirmed-OOS`
on the full arc. Decay below that → `partial-OOS` (in-sample real,
OOS-fragile). Sign-flip → `reversed-OOS`.

## Expected delta / honest prior

The +5.86 / +2.01 fired-α numbers are **not inherited** — Phase A
re-applies the borrow split inside the locked v3 recipe and the
honest prior on the *conditioning delta* is +0.05–0.20 (sharper
cohort selection + squeeze-tail avoidance), with real downside that
borrow-stress on liquid names is too weak/too-arbitraged to separate
safe from lethal (liquid = more eyes on the borrow data — the
mirror-image risk of the illiquid arc's coverage problem). The
headline deliverable is the **H3 conditioned-vs-unconditioned
delta**, pre-registered above; everything else is diagnostic.

## Compute placement

- **Stage 0**: local, offline, free (flat-file pulls + coverage
  join). No Modal.
- **Phase A / B**: the borrow-tercile split multiplies the
  walk-forward → **Modal** per the heavy-work rule. Borrow-data
  prep (FTD/short-vol/SI pulls + universe join) is local prep that
  pickles the input; the `local_entrypoint` ships the pickle via
  RPC (pattern: `apps/factor/scripts/modal/prep_universe_pivot_data.py`).

## Where the result lands

- One leaderboard row per Phase (A; B if triggered).
- `findings/vol-borrow-liquid-universe.md` per the
  after-every-experiment protocol; cross-link to the illiquid
  falsified finding and the v3 recipe.
- Update this TODO with the verdict → next-experiment chain; if the
  arc closes, mark superseded with a finding pointer.

## Cross-links

- Re-points the un-reached borrow leg of
  [`vol-borrow-illiquid-vrp`](vol-borrow-illiquid-vrp.md);
  inherits the mechanism, drops the dead cohort.
- Baseline locked from
  [`vol-surface-v3-regime-gated`](../findings/vol-surface-v3-regime-gated.md);
  OOS span from
  [`vol-surface-v2-dolthub-oos`](../findings/vol-surface-v2-dolthub-oos.md).
- Gated by the standing strategic frame: novel data + a deliberately
  boring (locked v1 OLS) model, not a cleverer model on standard
  data.
