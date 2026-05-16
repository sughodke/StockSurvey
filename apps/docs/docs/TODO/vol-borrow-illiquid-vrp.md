# Joint v0 — small-capacity illiquid-options VRP × securities-lending stress

!!! failure "CLOSED 2026-05-15 — `reversed-OOS`, killed upstream of the spread question"

    Ran Stage-1 (cheap offline gates first). **Gate A**: the
    unrestricted edge is *real* — +0.1408 of premium/cycle survives
    artifact-cleaning (halt contamination only 1.8%; val r unchanged).
    **Gate B-pre**: only **7.5%** of clean predicted picks are
    quote-available in the only free per-contract source (DoltHub
    `option_chain`); the quote-available 7.5% has **negative** mean
    edge (−0.092) while the absent 92.5% holds the +0.199. The
    re-frame fails *before* the borrow-conditioning and
    breakeven-spread stages — the cohort carrying the edge cannot be
    priced (historically or live) on free data, and the priceable
    subset is the liquid cohort
    [`vol-surface-v2-oi-restriction`](../findings/vol-surface-v2-oi-restriction.md)
    already falsified. Full write-up:
    [`vol-borrow-illiquid-vrp-falsified`](../findings/vol-borrow-illiquid-vrp-falsified.md).
    Durable concept extracted →
    [notes.md: quote-availability is a deployability gate](../notes.md#quote-availability-is-a-deployability-gate).
    Borrow leg (Stage C3) never reached; not falsified, just unreached.
    **Re-open only with a paid microcap option-quote source.**

**Operational framing (the rule this would establish if it PASSES):**
the variance risk premium concentrated in thin-OI options is a
*capacity-constrained* edge — uneconomical for desks that need size,
harvestable by a small operator for whom illiquidity is a moat rather
than a kill — and securities-lending stress on the underlying is the
mechanism-linked variable that predicts both *where the premium is
fattest* and *where the short-vol tail is lethal*.

**Verdict → next-experiment chain (stated up front):**

- **PASS (`confirmed-OOS`)** → the discarded `confirmed-OOS` vol signal
  is real net of honest thin-name friction for a small operator. Next:
  defined-risk live infra (`ss-vol live` + options broker), tiny-
  notional paper trade on the borrow-conditioned cohort.
- **MARGINAL (`partial-OOS`)** → survives only above an implausible
  spread assumption, or borrow-conditioning helps in some windows only.
  Next: stratify on the surviving windows; report breakeven spread as
  the deliverable, not a Sharpe.
- **FAIL, VRP dies on spreads (`reversed-OOS`)** → the arc's original
  kill was correct for small operators too. The re-frame is closed.
- **FAIL, VRP survives but borrow adds nothing (`confirmed-null` on the
  joint thesis)** → small-cap VRP is the finding; the *conjunction* is
  falsified. Record VRP-alone separately; drop the borrow leg.

The joint thesis is independently falsifiable via the
borrow-conditioning delta (cut C3 below) — this is what makes it a
real hypothesis and not "try VRP on illiquid names."

## Why this workstream exists

The vol arc closed `partial-OOS`
([`vol-arc-synthesis`](../findings/vol-arc-synthesis.md)) after
[`vol-surface-v2-oi-restriction`](../findings/vol-surface-v2-oi-restriction.md)
showed alpha Sharpe decays monotonically with the OI floor: **+5.86
unrestricted → −0.48 at top-200 OI**. The arc read "alpha lives where
options are illiquid" as a deployment-killer and v3 spent effort
clawing back a MARGINAL +2.01 on the *liquid* subset via a regime gate.

That move embedded an institutional-scale assumption — a strategy must
absorb size — and let it veto the only `confirmed-OOS` signal in 146
logged runs. The assumption does not bind a small operator. This
workstream re-asks the question the arc never asked: *does the thin-OI
VRP survive honest thin-name friction for someone small enough that
their fills don't move the print?* — and tests it jointly with the one
data source whose persistence mechanism is the same.

## The joint hypothesis (mechanism, not a bolted-on second signal)

The thin-OI VRP persists because a market-maker who sells vol on a thin
name must hedge and unwind in a market with no depth; their own
inventory/illiquidity cost is high, so they demand a fat premium. **A
hard-to-borrow underlying directly raises that hedging cost** — the MM
who is short the put cannot freely short the underlying to delta-hedge
when borrow is scarce/expensive. So securities-lending stress is not an
orthogonal factor stapled on; it is a *measurable proxy for the exact
friction that generates the premium*.

The same variable is two-edged:

- **Premium amplifier.** High borrow fee / low locate availability →
  costlier MM hedging → richer VRP. Predicts *go more short vol here*.
- **Tail predictor.** Hard-to-borrow + FTD spike + utilization near
  100% = squeeze setup. Short vol = short gamma; a violent squeeze
  up-move blows up the short-vol position *and* the same scarcity means
  no exit. Predicts *this rich-looking cell is lethal — skip it*.

The whole research question is whether a borrow-conditioned model can
separate **rich-and-safe** from **rich-and-lethal**. If it can, that
conjunction is structurally un-publishable (academia isolates one clean
effect; this is a microstructure interaction) and persistent.

## Falsifiable hypotheses

- **H1 (premium):** within the thin-OI cohort, mean per-rebal VRP alpha
  is higher in the high-borrow-stress sub-cohort than the
  low-borrow-stress one.
- **H2 (tail):** worst single-rebal drawdown is *worse* in the
  high-borrow-stress sub-cohort (squeeze blowups).
- **H3 (separability — the joint thesis):** a borrow-stress-conditioned
  selection produces higher net-of-spread, tail-truncated alpha Sharpe
  than the unconditioned thin-OI baseline by ≥ +0.10.

H3 is the load-bearing cut. H1 ∧ ¬H2 ∧ H3 = the deployable edge.

## Stage 0 — data-feasibility gate — **RESOLVED 2026-05-15: PASS (free)**

Probed the DoltHub `post-no-preference/options` HTTP API directly
(`SHOW TABLES` / `DESCRIBE option_chain` / single-symbol single-date
sample pulls).

- **`option_chain` table exists** — the `ss_iv` loaders only ever query
  `volatility_history`. Per-contract schema: `date, act_symbol,
  expiration, strike, call_put, bid, ask, vol (implied vol), delta,
  gamma, theta, vega, rho`.
- ✅ **Per-contract NBBO (`bid`/`ask`) is populated and free**, on the
  *wide* DoltHub universe (2,276 US tickers incl. thin small-caps —
  verified PLCE has chains), with greeks. Vega comes directly — no
  `0.3989·S·√T` approximation needed.
- ❌ **No open-interest column anywhere in DoltHub.** The original
  v2 #2 cohort was OI-defined; that exact split is not reproducible
  from free data.
- ✅ **Blocker is moot — cohort redefined by quoted relative spread.**
  The re-frame's question is "does VRP survive honest thin-name
  *spread*"; we now observe the spread directly. Define the illiquid
  cohort by `rel_spread = (ask − bid) / mid` (top band = illiquid)
  instead of by OI. Strictly *more* faithful than OI — OI is a
  liquidity proxy; the quoted spread is the actual transaction cost
  that determines whether the edge survives.
- Borrow leg unchanged: free FTD + FINRA short-volume + short-interest
  triplet; true borrow fee deferred to v1.

**Foreshadowing (single-date probe, 2022-06-17).** Thin-name spreads
are enormous: PLCE Jul-15 30-strike call quoted 13.40 / 15.90
(**≈17% of mid**); the matching put 0.25 / 2.15 (**≈160% of mid**).
Liquid AAPL contracts quote ~2-3%. Crossing a 17-160% spread *twice*
(enter + exit) on a short-vol position will obliterate most plausible
edges. The breakeven-spread deliverable may be a *fast kill* — itself
a clean, cheap, valuable verdict (it would confirm the arc's original
discard was correct for small operators too).

**Stage-1 data-prep note.** The DoltHub HTTP API aborts on broad
queries (~50-row deadline cap); the existing loader chunks by calendar
year per symbol. `option_chain` pulls must follow the same pattern.
`option_chain` date span unconfirmed (companion `volatility_history` is
2019-02-09 → 2026-04-30); confirm via chunked probe during prep.

**Verdict:** Stage 0 **PASS** — v0 is unblocked and runnable for free
with one design change (spread-defined cohort, not OI-defined).

## Stage 1 — v0 test design

- **Signal — fixed, reused, not rebuilt.** The v1 10-feature surface
  predictor (linear OLS) → forward IV-RV gap. Per the standing
  research-strategy frame, we change *data / universe / cost*, not the
  model. No MLP, no new features in v0.
- **Universe.** Thin-liquidity tradable cohort defined by *quoted
  relative spread* `(ask−bid)/mid` (top band = illiquid; OI unavailable
  per Stage 0): US-listed equities with listed options, enough strikes
  to build a defined-risk spread, underlying price > $5. This is the
  +5.86 regime, not the −0.48 one.
- **Borrow conditioning.** Split the cohort each rebal by the Stage-0
  borrow-stress score into low / high sub-cohorts (terciles). Test
  H1/H2 directly; build the H3 conditioned selection.
- **Construction — the honest change vs v1/v2/v3.** Friction symmetry
  *removed* (v1-v3 cancelled friction alpha-vs-EW by symmetry — exactly
  the assumption a thin-name strategy violates hardest). Defined-risk
  structures only (vertical spreads, never naked short vol) so the tail
  is capped. Per-name notional cap. v3 126d-VIX gate retained as a
  tail-avoidance overlay (no new short vol into a rising-VIX regime).
- **Spread treatment.** Stage 0 resolved per-contract NBBO is free →
  model the *measured* round-trip spread directly (no parametric proxy
  needed). Still report the **breakeven spread** as the headline
  deliverable — the single number that says whether any small operator
  can trade this at all.
- **Windowing.** Same walk-forward as v1/v2 (5 windows) + the DoltHub
  2024-2026 OOS extension (the wide-universe span that already
  confirmed signal continuity in
  [`vol-surface-v2-dolthub-oos`](../findings/vol-surface-v2-dolthub-oos.md)).
- **Compute.** Thin-OI universe is wide and the spread-sensitivity
  sweep multiplies the walk-forward → **Modal** per the heavy-work
  rule. Data prep (FTD/short-vol/SI pulls, OI-cohort join, DoltHub
  table check) is local prep that pickles the input.

## Pre-registered cuts (mapped to the fixed verdict vocabulary)

- **C1 — VRP survives honest friction.** Net alpha Sharpe ≥ **+0.30**
  on the thin-OI cohort after modeled thin-name round-trip spread + tail
  truncation, positive in ≥ 4/5 walk-forward windows, AND survives the
  DoltHub 2024-26 OOS extension → `confirmed-OOS`.
- **C2 — partial.** Net alpha positive but < +0.30, or survives only
  above an implausibly tight spread, or positive in 3/5 windows →
  `partial-OOS`.
- **C3 — joint thesis (H3).** Borrow-conditioned selection beats the
  unconditioned thin-OI baseline by ≥ **+0.10** net Sharpe. C3 is
  evaluated *independently* of C1/C2: if C1 PASSES but C3 fails →
  `confirmed-null` on the conjunction (small-cap VRP alone is the
  finding, drop the borrow leg). If C1 fails but C3 PASSES (borrow
  conditioning rescues an otherwise-dead cohort) → the *conjunction* is
  the finding, the strongest possible outcome.
- **C4 — full fail.** Net alpha ≤ 0 once realistic thin-name spread is
  applied and borrow-conditioning adds nothing → `reversed-OOS`; the
  arc's original kill was correct for small operators too. Re-frame
  closed.

## Expected delta / honest prior

+5.86 gross is **not inherited** — it was measured under the
friction-cancelling construction this test deliberately breaks.
Realistic prior: low-single-digit net Sharpe or worse; plausibly
negative once thin-name spreads are crossed twice. The borrow
conditioning is hypothesized to add +0.10–0.30 by sharpening cohort
selection and dodging squeeze-tail blowups. **The headline deliverable
is the breakeven round-trip spread**, not a point Sharpe — that single
number tells us whether any small operator can trade this at all.

## Cross-links

- Re-frames the discard in
  [`vol-arc-synthesis`](../findings/vol-arc-synthesis.md) /
  [`vol-surface-v2-oi-restriction`](../findings/vol-surface-v2-oi-restriction.md).
- Reuses the v1 predictor + short-vol PnL from
  [`vol-surface-v1`](../findings/vol-surface-v1.md) and `ss_iv`.
- Borrow-leg proxy is the securities-lending / settlement-stress data
  class (SEC FTD + FINRA short-volume + short interest).
- Gated by the standing research-strategy frame: pursue
  structurally-persistent edges (capacity-constrained / novel-data),
  not published anomalies.
