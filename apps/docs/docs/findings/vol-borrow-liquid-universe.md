# Borrow-stress conditioning on the liquid vol universe — `reversed-OOS`, arc closed

**Operational rule.** Securities-lending stress (FINRA short-volume + SEC
fails-to-deliver) is **freely available and well-dispersed on the liquid
optionable universe** (Stage 0 PASS, the inverse of the illiquid arc's
coverage death) — but conditioning the v3 short-vol recipe on it produces
**no out-of-sample edge**. A strong in-sample premium-amplifier signal
(high-borrow-stress → richer VRP, hi-tercile Sharpe +3.80) **failed to
replicate** on 50× more out-of-sample data. The borrow leg closes; do not
re-attempt without a paid securities-lending feed (loan fee / utilization),
and treat any in-sample borrow-VRP signal as a data-snooping candidate until
OOS-confirmed.

## The arc, stage by stage

This re-pointed the *unreached borrow leg* of the illiquid-VRP arc (which died
`reversed-OOS` on quote-availability) onto the **liquid** universe, where the
quote-availability wall structurally cannot bite.

**Stage 0 — feasibility gate: PASS.** Over 6 sample dates (2019-11→2023-03),
FINRA short-volume covered 99.3% of the top-200-OI cohort and the short-ratio
hi−lo tercile gap was 0.266 — real, splittable dispersion. The pre-registered
kill-risk (liquid-name borrow-stress too flat to form terciles) was retired.

**Phase A — borrow split on the locked v3 recipe (in-sample, gauss314
2019-2023, 11 fired rebals).** The mechanism looked *strong*:

| tercile | mean alpha (vol pts) | worst-rebal | in-sample Sharpe |
|---|---:|---:|---:|
| low borrow-stress | +0.002 | −0.108 | +0.11 |
| high borrow-stress | **+0.042** | −0.026 | **+3.80** |

High-borrow-stress names carried ~20× the low tercile's VRP **and** the
*safest* tail — the **opposite** of the scoping page's squeeze-tail
assumption. So the pre-registered conditioned rule (drop high-stress) *failed*
(H3 −2.17) because it discarded the alpha; the implied rule was the inverse
(overweight high-stress).

**The discipline that mattered:** that flip was read off 11 in-sample fired
rebals, post-hoc — exactly the [Sullivan-Timmermann-White](leaderboard.md)
data-snooping setup. The +3.80 was treated as a **hypothesis, not a result**,
and gated to a pre-registered OOS test.

**Phase B — OOS confirmation (DoltHub 2024-26, 552 weekly snapshots, 11
quarters): FAIL.** On never-seen out-of-sample data, high-borrow-stress
carried **marginally lower** realized VRP (+0.011) than low-stress (+0.014),
beating it in only **36% of quarters** (pre-reg PASS needed ≥60%). The
in-sample +3.80 **did not replicate**. It was selection on 11 rebals.

## Why this is the cleanest demonstration of the workstream's value

The borrow premium-amplifier is the single most spectacular in-sample signal
the whole DSR investigation produced (hi-tercile Sharpe +3.80). Under a naive
process it would have been shipped. The pre-registered OOS test — on a sample
50× larger — killed it. This is the entire point of the deflated-Sharpe /
pre-registration discipline: **a high in-sample number is a hypothesis; only
out-of-sample survival is evidence.** The mechanism (hard-to-borrow → costlier
MM hedging → richer VRP) is economically plausible, but the *free* proxies
(short-volume, FTD) do not isolate it cleanly enough to survive OOS on the
liquid universe — and the clean signal (loan fee / utilization) remains paid
data.

## What this closes

- The **B1 borrow arc** (`reversed-OOS`).
- The **free-borrow-proxy novel-data leg** entirely: the illiquid leg died on
  coverage; the liquid leg passes coverage but fails OOS confirmation.
- It leaves **DCA as the only deflated-t-confident edge** on the
  [cross-arc ladder](../leaderboard.md#cross-arc-deflated-sharpe-ranking), and
  pushes the next real lever to *paid data* (securities-lending feed; or the
  field's frontier — fundamentals/value, text/events) rather than another
  free-data conditioning of the existing recipe.

## Honest caveats

- Phase B's substrate (DoltHub weekly, decimal IV) is not bit-identical to
  Phase A's (gauss314 daily) — but the OOS sample is 50× larger and the
  mechanism should be substrate-agnostic, so the OOS FAIL is credible.
- A units bug (DoltHub IV decimal vs percent forward-vol) was caught and fixed
  mid-run; the corrected magnitudes are realistic and the verdict is unchanged.

## Master walk-forward log

[Phase A](../leaderboard.md) (`partial-OOS`, in-sample) → [Phase B]
(../leaderboard.md) (`reversed-OOS`, arc closed). Re-points
[`vol-borrow-illiquid-vrp-falsified`](vol-borrow-illiquid-vrp-falsified.md);
baseline from [`vol-surface-v3-regime-gated`](vol-surface-v3-regime-gated.md).
