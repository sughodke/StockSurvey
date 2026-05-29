---
tags:
  - cfr
  - dca
  - sensitivity
  - audit-followup
  - diagnostic
  - audit-followup
---

# CFR Phase 4d sensitivity follow-up — friction × gate stress-test (diagnostic)

**Operational rule (refined):** *The CFR Phase 4d closure stands at the
deployment-threshold level (alpha < +0.10 across the entire friction ×
gate space) but the specific FAIL framing in
[`cfr-macro-gate-final`](cfr-macro-gate-final.md) is overstated.*
Always-deploy CFR posts **3/5 positive windows, not 1/5**, at every
friction level tested. The gated **1/5** count was a denominator
artifact — windows where the gate held closed contributed exactly
zero alpha by construction. **The honest verdict for the entire CFR
Phase 4d body of work is best described as **alpha exists but below the
deployment threshold**, which is closer to `partial-OOS` than to
`confirmed-null` under the established verdict vocabulary. The
sensitivity check itself is structurally a `diagnostic`.**
The deployment call (use DCA, not CFR) is unchanged because no tested
configuration clears the +0.10 paper-trade threshold; but the
load-bearing claims that closed the arc were less robust than the
phrasing suggested.

## Scope clarification (2026-05-29)

The grid result — "no single hand-coded VIX lookback (60d / 90d /
126d) dominates across apps, and 90d worsens alpha" — stands
verbatim. It is in fact the cleanest motivation on the docs site
for the orthogonal lever: a **learned memory-window selection from
multi-lookback inputs**, with a direct-Sharpe objective and an
action space allowing `gross > 1` so the deterministic recipe is
representable. Hand-coding the lookback is the failure mode this
sensitivity grid documents; learning it is the untested path.

The pre-registration that formalizes the learner-side test is
[`TODO/e2e-portfolio-v4-learned-regime-gate.md`](../TODO/e2e-portfolio-v4-learned-regime-gate.md);
the framing distinction is in
[`notes.md#learner-layer-matters-more-than-learner-complexity`](../notes.md#learner-layer-matters-more-than-learner-complexity)
and the empirical proof-of-concept at the sizing layer is
[`findings/learned-ensemble-beats-deterministic.md`](learned-ensemble-beats-deterministic.md).

## Why this experiment

The 2026-05-14 audit of `apps/docs/**` (see `.audit-research-directions.md`)
flagged the CFR arc closure as the single most consequential decision in
this body of work and argued it rests on three load-bearing choices:

1. A single un-bracketed friction number (`CFR_DRAG_BPS_YR = 50.0`) in
   [`cfr-vs-dca-realistic`](cfr-vs-dca-realistic.md), with no
   sensitivity table.
2. A FAIL verdict in [`cfr-macro-gate-final`](cfr-macro-gate-final.md)
   that triggered on "positive in only 1/5 windows" — but the gate held
   closed in 4/5 windows, so alpha was exactly zero by construction in
   those windows. The 1/5 denominator conflates "did the gate fire" with
   "was deployment profitable when it fired".
3. A 1-year rolling-median VIX lookback that the finding itself partly
   admits was wrong (it inflates post-stress and gates the bot off
   during early-recovery regimes).

This follow-up stress-tests all three. No new training required — the
per-window raw Sharpes for CFR and EW are already on disk in
`Output/cfr-phase4d.json`. Friction enters as a Sharpe drag
(`drag = bps_per_yr / 10_000 / annualized_window_vol`), and gates are
recomputed from `Output/cfr_phase3_macro.pkl` (VIX) and
`Output/cfr_phase4d_multiasset_close.pkl` (in-universe dispersion).

## Setup

- Driver: `apps/cfr/scripts/sensitivity_analysis.py`
- Wall: ~3 seconds local CPU.
- Output: `Output/cfr-sensitivity.json`.

Per-window inputs (annualized window vol on the multi-asset EW basket,
used to convert bps/yr → Sharpe drag):

| Window | val_start | val_end | EW window vol |
|---:|---|---|---:|
| 0 | 2010-03-01 | 2013-04-05 | 0.124 |
| 1 | 2013-04-08 | 2016-05-10 | 0.094 |
| 2 | 2016-05-11 | 2019-06-17 | 0.082 |
| 3 | 2019-06-18 | 2022-07-21 | 0.168 |
| 4 | 2022-07-22 | 2025-08-29 | 0.116 |

Raw per-window CFR alpha vs passive EW (from `cfr-phase4d.json`):

| Window | CFR raw Sh | EW raw Sh | CFR − EW |
|---:|---:|---:|---:|
| 0 | +1.367 | +0.945 | **+0.422** |
| 1 | +0.622 | +0.647 | −0.026 |
| 2 | +0.495 | +1.002 | **−0.508** |
| 3 | +0.780 | +0.690 | +0.090 |
| 4 | +1.041 | +0.740 | **+0.301** |

Mean +0.056; 3/5 positive; binding loss in w2.

## Section 1 — friction sensitivity (no gate, always-deploy)

| CFR drag (bps/yr) | DCA drag (bps/yr) | Mean alpha | Positive wins | Verdict |
|---:|---:|---:|---:|:---:|
| 10 | 2  | +0.0487 | 3/5 | MARGINAL |
| 10 | 5  | +0.0514 | 3/5 | MARGINAL |
| 10 | 10 | +0.0559 | 3/5 | MARGINAL |
| 15 | 2  | +0.0441 | 3/5 | MARGINAL |
| 15 | 5  | +0.0469 | 3/5 | MARGINAL |
| 15 | 10 | +0.0514 | 3/5 | MARGINAL |
| 25 | 2  | +0.0350 | 3/5 | MARGINAL |
| 25 | 5  | +0.0378 | 3/5 | MARGINAL |
| 25 | 10 | +0.0423 | 3/5 | MARGINAL |
| 35 | 2  | +0.0260 | 3/5 | MARGINAL |
| 35 | 5  | +0.0287 | 3/5 | MARGINAL |
| 35 | 10 | +0.0332 | 3/5 | MARGINAL |
| 50 | 2  | +0.0123 | 3/5 | MARGINAL |
| **50** | **5** | **+0.0150** | **3/5** | **MARGINAL** (matches `cfr-vs-dca-realistic`) |
| 50 | 10 | +0.0196 | 3/5 | MARGINAL |

**Findings:**

1. **Mean alpha swings 5× across the friction grid** (+0.015 → +0.056)
   but never clears the +0.10 PASS threshold. The audit's framing
   "at 25 bps drag, realistic alpha is +0.045 — comfortably above noise"
   is arithmetically correct but operationally insufficient: the
   relevant threshold is +0.10 (from `passive-ew-benchmark`), not
   +0.000.
2. **Positive-window count is 3/5 across the entire grid**, not 1/5.
   The 1/5 in `cfr-macro-gate-final` came from gating with a denominator
   that included 4 windows where the gate was closed (and therefore
   alpha was zero by construction). The honest always-deploy count is
   3/5 — MARGINAL by the pre-registered cuts, not FAIL.
3. **The original 50/5 bps assumption is the worst-case corner** of the
   sensitivity space, not a best estimate. The audit was right that no
   bracketing was done.

## Section 2 — alternative gate variants

All run at CFR=50 bps / DCA=5 bps friction (worst case, to isolate
gate effect from friction effect):

| Gate | Fires in | Mean alpha vs DCA | Pos/n | Alpha on fired | Pos/fired |
|---|:---:|---:|:---:|---:|:---:|
| `vix_60d` (audit: "shorter lookback") | (none) | +0.0000 | 0/5 | +0.0000 | 0/0 |
| `vix_90d` (audit: "shorter lookback") | w3 | +0.0126 | 1/5 | +0.0629 | 1/1 |
| `vix_126d` | (none) | +0.0000 | 0/5 | +0.0000 | 0/0 |
| `vix_252d` (baseline) | w4 | +0.0525 | 1/5 | +0.2627 | 1/1 |
| `disp_60d_vs_252d_med` (audit: "in-universe dispersion") | w1, w2, w3 | **−0.1146** | 1/5 | −0.1910 | 1/3 |
| `disp_30d_vs_252d_med` | w1, w2, w3 | **−0.1146** | 1/5 | −0.1910 | 1/3 |
| `always_cfr` (no gate, reference) | all 5 | +0.0150 | 3/5 | +0.0150 | 3/5 |

**The audit's two specific hypotheses are falsified:**

### Hypothesis falsified #1: shorter-lookback VIX gates do not fix the w0 miscall

The 60d and 126d gates never fire across any val_start in the panel. The
90d gate fires only in w3 (+0.09 raw alpha), capturing a smaller signal
than the 252d gate (which fires in w4 at +0.30 raw alpha). Mean alpha
under any shorter VIX lookback is *lower*, not higher, than the
baseline. The "memory-heavy gate fights regime transitions" diagnosis in
`cfr-macro-gate-final` correctly identifies the w0 miscall (VIX 19.3 vs
post-GFC-inflated median 25.4) but the cure (shorter lookback) makes
the rest of the panel worse.

### Hypothesis falsified #2: in-universe dispersion gate is actively counter-productive

The intuition was: CFR's edge is cross-asset rotation; rotation alpha
is largest when assets are weakly correlated; so a "fire when 60d avg
pairwise correlation is below trailing 252d median" gate should select
for CFR's edge. **Empirically, this gate fires in exactly the wrong
windows**:

| Window | CFR vs EW alpha | Dispersion gate decision |
|---:|---:|:---:|
| 0 (2010-03) | **+0.422** | CLOSED (correlation high in post-GFC recovery) |
| 1 (2013-04) | −0.026 | OPEN |
| 2 (2016-05) | **−0.508** | OPEN ← catches the worst loss |
| 3 (2019-06) | +0.090 | OPEN |
| 4 (2022-07) | **+0.301** | CLOSED (correlation high during 2022 inflation cycle) |

The dispersion gate is anti-correlated with CFR's actual alpha pattern.
Mechanism guess: in the 2016-19 calm-bull regime, equity sectors
decorrelated *within* the equity bucket but the cross-asset bucket
(bonds, commodities) was tightly correlated with equities (low rates →
all assets up). The dispersion measure picked up sector-level
decorrelation as "rotation opportunity" but the bonds/commodities
buckets didn't deliver. In 2010-03 and 2022-07, the asset classes were
moving together at a high level (recovery / inflation) so average
correlation was high — but *within* that the rotation across asset
classes mattered, which is what CFR captured. **Average pairwise
correlation is the wrong dispersion measure for cross-asset rotation
alpha.**

## Section 3 — best gate (252d VIX) × friction sweep

| CFR drag (bps/yr) | DCA drag (bps/yr) | Mean alpha | Pos/n | Verdict |
|---:|---:|---:|---:|:---:|
| 10 | 2  | +0.0589 | 1/5 | FAIL |
| 10 | 5  | +0.0594 | 1/5 | FAIL |
| 10 | 10 | +0.0603 | 1/5 | FAIL |
| 25 | 2  | +0.0563 | 1/5 | FAIL |
| 25 | 5  | +0.0568 | 1/5 | FAIL |
| 50 | 5  | +0.0525 | 1/5 | FAIL |

Across the entire friction grid, the gated strategy (252d VIX) is
strictly dominated by the always-deploy strategy on mean alpha (max
+0.060 vs always-deploy's max +0.056 — within noise of each other) but
*scored* worse on the positive-windows metric (1/5 vs 3/5) because the
gate is closed in 4 windows where alpha is forced to zero by
construction. **The pre-registered FAIL cut in `cfr-macro-gate-final`
mixes two different deployment policies into one comparison**: it
counts a "gate closed" window as a deployment outcome of "alpha=0",
which is a fair description of what the policy does, but produces a
positive-window count that's worse than the no-gate alternative
without making the underlying alpha worse.

## Refined verdict for the CFR arc

| Original claim | Refined claim |
|---|---|
| "Bot is fully dead" (`cfr-macro-gate-final`) | Bot is **MARGINAL** — alpha exists, signed positive, but below the +0.10 deployment threshold at every tested friction level. |
| "Realistic alpha is +0.015" (`cfr-vs-dca-realistic`) | Realistic alpha is **+0.015 (worst case at 50/5 bps) to +0.056 (best case at 10/10 bps)**. Original number was an unbracketed worst-case. |
| "FAIL on positive in 1/5" (`cfr-macro-gate-final`) | Without a gate, **positive in 3/5**. The 1/5 was an artifact of mixing the gate's deployment policy into the count. |
| "Memory-heavy regime gates fight you across regime boundaries" (operational rule) | True for the 252d gate's w0 miscall, but **shorter VIX lookbacks don't fix it** (60d/126d never fire; 90d fires in a worse window). The rule should be qualified: memory-heavy gates fight regime transitions, *but no candidate VIX lookback tested has better discrimination*. |
| "Use DCA as canonical live" (`cfr-vs-dca-realistic`) | **Unchanged.** CFR doesn't clear the deployment threshold at any tested friction level; DCA has higher worst-window Sharpe; operational labor on the bot side is real. |

## What's left to try (orthogonal levers, low priority)

The CFR closure is no longer arguable on friction or gate-lookback
grounds; both axes are now stress-tested and the answer is the same.
Three orthogonal levers remain genuinely untested:

1. **Absolute-VIX threshold gate** rather than relative-to-rolling-median.
   The pattern in the per-window data is suggestive: positive-alpha
   windows have VIX 19, 15, 23 (mean ~19); negative-alpha windows have
   VIX 13, 15 (mean ~14). A "deploy when VIX > 17" gate would fire in
   w0 and w4 (capturing +0.42 + +0.30 = +0.72 raw alpha across 2/5
   windows). But the threshold is chosen-after-seeing-data; pre-registering
   it on the first 3 windows and re-evaluating on the last 2 is the only
   honest way to test this — and you only get 2 OOS windows, which is
   under-powered to land a verdict.
2. **Better dispersion measure.** Not average pairwise correlation (that
   was the falsified arm) but e.g. *cross-bucket* dispersion only
   (corr(equities, bonds, commodities) at the bucket level, not within
   the equity bucket). This isolates the rotation regime CFR actually
   trades. Cheap to test (~1 hour).
3. **CFR with the gate baked into training** rather than as a post-hoc
   composition. The current pipeline trains CFR on all bars then applies
   the gate at deployment; an alternative trains CFR only on bars where
   the gate is open, so the regret-net specializes for the deployment
   regime. This is ~1 day of Modal time to test and breaks the "no
   walkforward retraining" property of this follow-up.

None of these promote CFR above the +0.10 deployment threshold with high
prior probability — the binding constraint is still that CFR's alpha
on this universe at this horizon, even at best-case friction, is
+0.05 to +0.06 mean Sharpe. To clear +0.10 you need a gate that fires
in 3 of CFR's 3 positive-alpha windows (w0, w3, w4) while staying
closed in w1 and w2 — a 3-bit classifier on 5 examples, which is at
the edge of what 5 OOS windows can identify without overfitting.

## Master walk-forward log

This row is appended to [`leaderboard.md`](../leaderboard.md) as the
sensitivity follow-up to the 2026-05-13 closure rows. Verdict label:
[`diagnostic`](../leaderboard.md#verdict-labels) — this is a stability
check that audits prior verdicts, not a fresh hypothesis test with
its own pre-registered cuts. The substantive finding is that the
prior `confirmed-null` on `cfr-macro-gate-final` was overstated (the
underlying alpha description fits `partial-OOS` better — 3/5 windows
positive, real-signed alpha, but below deployment threshold) while
the prior `confirmed-null` on `cfr-vs-dca-realistic` was correct on
the deployment decision but under-bracketed on the alpha estimate.

Artifacts:

- Driver: `apps/cfr/scripts/sensitivity_analysis.py`
- Output: `Output/cfr-sensitivity.json` (full matrices: friction grid,
  gate decisions per window, friction × gate sweep)
- Source audit: `.audit-research-directions.md` (the 2026-05-14 audit
  that motivated this re-test)
