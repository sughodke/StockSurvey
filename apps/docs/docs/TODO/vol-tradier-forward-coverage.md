# Vol — Tradier forward structural coverage probe

**Verdict → next-experiment chain (stated up front):**

The falsified [`vol-borrow-illiquid-vrp`](vol-borrow-illiquid-vrp.md) arc
killed the small-capacity re-frame on **DoltHub**-specific coverage:
12% symbol / 7.5% pick-weighted coverage of the v1 clean-pick cohort
in `option_chain`, with mean `edge_frac` on the quotable subset
**−0.092** (negative before any spread cost). The open structural
question that gates the entire paid-data spend decision:

> Is the missing 88% of the cohort **absent from DoltHub specifically**
> (because the publisher chose a narrower universe) or **absent from
> OPRA itself** (because the names aren't options-listed at all)?

If the names *are* OPRA-listed but DoltHub just doesn't ship them, then
**paid OPRA-sourced vendors (Polygon Options Advanced ~$199/mo, CBOE
DataShop, ORATS) will carry them** and the paid-data spend question is
live. If the names are *not* OPRA-listed at all, **no vendor can help
at any price** and the illiquid arc is definitively closed.

The cheapest source of truth on OPRA listings is the Tradier sandbox
(paper API key, $0, no funding) — it returns delayed-but-real chains
from the consolidated OPRA feed, with **greeks + IV included via
ORATS** in sandbox (same source as the funded tier).

## Hypothesis (falsifiable)

The v1 microcap pick cohort is structurally absent from **OPRA itself**,
not just from DoltHub's narrower publisher cut.

Null: the cohort is broadly OPRA-listed (≥50% of unique pick-symbols
have populated chains today on Tradier sandbox) and the falsified arc's
DoltHub-specific coverage gap is not the OPRA-coverage gap.

## Test design

- **Cohort:** unique pick-symbols from
  `Output/vol-v0-breakeven-picks.pkl` (the breakeven-script sample of
  v1's top-K clean picks across 8 evenly-spaced rebal dates; random
  sample is unbiased for measuring coverage). Comparison is
  symbol-weighted (matches the falsified arc's 12% figure) and
  pick-weighted (matches its 7.5% figure).
- **Probe (per symbol):** Tradier sandbox `/v1/markets/options/{expirations,chains}`,
  `greeks=true`. Record: chain present (T/F); near-ATM straddle relative
  spread on the nearest expiry ≥14 days out; greeks/IV populated (T/F).
  Cache JSON per (symbol, expiration) to `.iv-cache/tradier_chains/`.
- **Auth:** `TRADIER_TOKEN` env var only (never paste in chat).
  Endpoint: `https://sandbox.tradier.com` (paper). Throttle ~1 s
  between calls.
- **Caveat (foregrounded):** Tradier sandbox returns *today's* chains,
  not 2019–2023 historical. A symbol absent from Tradier-today doesn't
  *prove* it wasn't OPRA-listed in 2021 — it might have been delisted /
  acquired since. So **Tradier-today coverage is a lower bound on
  historical OPRA coverage.** That's fine for the decision: if
  Tradier-today already covers ≥50%, paid OPRA vendors definitely carry
  the names; if Tradier-today is ~12% (≈ DoltHub's number), the
  cohort wasn't broadly OPRA-listed even then.

## Pre-registered bands (matched to the falsified arc's bands for
direct comparability)

| Symbol-weighted coverage with finite ATM bid/ask | Verdict | Implication |
|---|---|---|
| **≥ 50%** | **COVERED** — OPRA carries the cohort | Paid historical NBBO (ORATS $99/mo or Polygon Advanced ~$199/mo) is now viable; the spend question is live and the next gate is historical replay on a paid source. |
| **15 – 50%** | **SEVERELY CONSTRAINED** | Matches the falsified arc's liquid-subset reading; no new information; do not pay. |
| **< 15%** | **KILLED** — OPRA itself doesn't carry these names | **Definitive close** of the illiquid arc. No vendor at any price helps. Lock the deployable strategy as v3 regime-gated liquid (+2.01 fired-Sharpe). |

Secondary signals (reported, not gating):
- Pick-weighted coverage (for direct comparability with the falsified
  arc's 7.5%).
- ATM straddle relative-spread distribution on the covered subset
  (p25/p50/p75/p90) — first cheap forward-feasibility look at observed
  spreads, supplementing the falsified arc's never-reached breakeven
  stage.
- Greeks/IV population rate (data-quality check on the ORATS-sourced
  fields in sandbox).

## Pointers

- Parent (resolved): [`vol-borrow-illiquid-vrp-falsified`](../findings/vol-borrow-illiquid-vrp-falsified.md);
  [`TODO/vol-borrow-illiquid-vrp`](vol-borrow-illiquid-vrp.md).
- Vendor sweep that motivated this: established that OPRA-tape vendors
  (Tradier/IBKR/Alpaca/Polygon/CBOE) structurally cover the full US
  listed-options universe, so OPRA listing is the precondition that
  determines whether *any* paid spend has a chance.
- Driver: `apps/vol/scripts/run_v0_tradier_forward_coverage.py`.
- Artifacts: `Output/vol-v0-tradier-forward-coverage.json`,
  `.iv-cache/tradier_chains/<symbol>_<exp>.json`.
