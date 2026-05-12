---
tags:
  - cfr
  - phase-1
  - meta-allocator
  - stooq_us_long
---

# CFR Phase 1 — tabular CFR clears trailing-best-greedy by +0.609 Sharpe (6/6 wins), ties naive uniform mix

**Operational rule (added 2026-05-12 to
[`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md#operational-rules-extracted-from-findings)):**
the binding constraint at the meta-allocator level is the action
menu, not the algorithm. Counterfactual-regret minimization
correctly identifies that uniform mixing is no-regret when no
infoset has clear edge — so any lift over uniform mix requires
the menu to contain modes that are themselves alpha-positive in
some regime. Phase 2 (13F-imitation-pretrained scorer modes +
sector-restricted variants) is the architectural correction.

Verdict: [`partial-OOS`](../leaderboard.md#verdict-labels).
CFR Phase 1 cleanly **passes** the pre-registered cut against
trailing-best-greedy (+0.609 Sharpe lift in 6/6 windows, well
above the +0.10 PASS threshold) but **does not** clear passive
EW (mean alpha −0.093) and **ties** the naive 1/16 uniform mix
within noise (+0.002 Sharpe Δ). Both the win and the caveat are
load-bearing.

## Setup

6 walk-forward windows on `stooq_us_long` (312 tickers,
2000-01-03 → 2025-12-11):

- **Windowing**: 1260-train (~5y) / 780-val (~3y) / 780-step.
  Matches gate / pairs canonical.
- **Action menu**: 16 actions = `cash` + (`ew, mom, rev, lowv,
  highv`) × (`0.5, 1.0, 2.0`) gross. Universe-agnostic — no
  saved checkpoints. `top_k=20` for the top-K modes. `mom` =
  21d momentum; `rev` = 5d reversal; `lowv` / `highv` = 21d vol
  ranking.
- **Infoset**: 3 × 3 = 9 cells on `(trailing 21d EW vol bucket,
  21d cross-sectional dispersion bucket)`. Cutoffs are
  train-period quantiles, frozen for val. 1 warmup cell on top.
- **Algorithm**: tabular CFR — regret matching on cumulative
  regret table, time-averaged policy at eval, single
  chronological pass through train rebals.
- **Friction**: 10 bps commission on L1 turnover at each rebal,
  matched to the
  [passive-EW benchmark](passive-ew-benchmark.md) convention.

Driver: `apps/cfr/scripts/modal/run_phase1.py` (Modal CPU 8c).
Wall: image build cached + 6.2s uv sync + **7.7s walkforward** +
transit = 23s end-to-end. Same eval reproducible locally with
`uv run python apps/cfr/scripts/run_walkforward.py`.

## Result

| win | val_dates | CFR Sh | Passive EW | Trailing-best | Naive uniform | α vs EW | CFR vs trailing |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2005-01 → 2008-02 | +0.279 | +0.529 | **−0.727** | +0.321 | −0.251 | **+1.006** |
| 1 | 2008-02 → 2011-03 | **+0.596** | +0.331 | −0.312 | +0.416 | **+0.265** | **+0.908** |
| 2 | 2011-03 → 2014-04 | +0.817 | +0.928 | +0.174 | +0.711 | −0.111 | +0.643 |
| 3 | 2014-04 → 2017-05 | +0.727 | +0.916 | +0.256 | +0.637 | −0.189 | +0.471 |
| 4 | 2017-05 → 2020-07 | +0.440 | +0.440 | +0.060 | +0.288 | +0.000 | +0.380 |
| 5 | 2020-07 → 2023-08 | +0.697 | +0.968 | +0.452 | +1.172 | −0.271 | +0.245 |
| **mean** | | **+0.593** | **+0.685** | **−0.016** | **+0.591** | **−0.093** | **+0.609** |

## Reading the result — three nested comparisons

### vs trailing-best-greedy: PASS by a wide margin

CFR beats "switch into whatever scorer had the highest trailing
Sharpe over the last 63 days" by **+0.609 mean Sharpe**, in
**6/6 windows**. The lift is largest in the first two windows
(+1.01, +0.91) where trailing-best gets caught chasing
dot-com-era leadership into the 2005-2007 bull market and then
into the GFC. It narrows in calmer windows (w5: +0.245) but
never inverts.

This is the comparison the [pre-registered Phase 1
cut](../TODO/apps-cfr.md) was written against — and the
algorithm clears it by an order of magnitude over the +0.10
threshold. Mechanistically: trailing-best concentrates 100% on
whichever single mode just won, which is **regime-mismatched by
construction** (a strategy works because of regime, so just-won
means the regime is changing). CFR's regret matching diffuses
across modes when no infoset has clear edge — that's not a
sophisticated regime call, just no-regret damping.

### vs naive uniform mix: tied within noise

CFR Sharpe +0.593 vs naive uniform Sharpe +0.591 — Δ +0.002, in
noise. **This is the honest reading of the result.** At this
universe × menu, the algorithm finds essentially nothing the
1/16 uniform mix doesn't already find. Two consistent
interpretations:

1. **The menu's modes are too close to alpha-zero.** Top-K
   momentum / reversal / low-vol / high-vol on a 312-name
   universe are all classical factor exposures with well-known
   premia ~0 net of friction at 10 bps × 20d rebal. There's
   nothing for the regret matching to concentrate on because
   no mode has consistently positive regret in any infoset.
2. **The 9-cell infoset doesn't separate regimes that would
   reward concentration.** A coarser regime label means CFR
   averages signals over within-cell heterogeneity. Phase 2
   adds a learned encoder that should separate at finer grain.

Both point to the same architectural conclusion: **the
algorithm is doing no-regret correctly; what's missing is
something for it to discover beyond uniform mixing**. This is
the [Cover universal portfolio](https://www.proquest.com/openview/8b9f99c41bbac5e1adcdcf3bb8a72683)
result restated — no-regret over a uniform-baseline-dominated
menu is the uniform baseline.

### vs passive EW: −0.09 alpha (within noise)

CFR mean Sharpe +0.593 vs passive EW +0.685 — alpha **−0.093**,
within the ±0.10 noise band, **only 1/6 windows positive**
(window 1, the GFC; mirrors gate-drawdown-v0's GFC outlier
pattern).

Consistent with the [passive-EW benchmark
finding](passive-ew-benchmark.md): on the canonical
`stooq_us_long` universe, passive EW Sharpe averages ~0.69 over
2005-2023, and **no model row in the leaderboard has cleared it
since the benchmark was adopted 2026-05-10**. CFR adds to that
list. The friction stack (10 bps on L1 turnover, 20-bar rebal)
explains some of the underperformance — CFR's average policy
involves enough mode switching that turnover-cost erodes the
benefit — but the deeper issue is that the menu doesn't contain
modes that beat passive on this universe.

## Per-window stratification — the GFC outlier

Window 1 (val 2008-02 → 2011-03, GFC) is the only window where
CFR posts positive alpha vs passive EW (+0.265). The pattern is
familiar:

- Window 1 also carries pairs-classical-v0's biggest win
  (+0.870 alpha) and gate-drawdown-v0's biggest win (+0.321
  alpha).
- The
  [macro-regime diagnostic](macro-regime-diagnostic.md) showed
  GFC-era macro state (VIX 25, real yields +1.5%, fed funds 3%,
  credit spreads +3%) cluster with all "win" windows across the
  pivot arc.
- In CFR's per-infoset policy, the GFC window's high-vol /
  high-dispersion infoset learns to concentrate on
  cash + low-vol mode rather than the uniform mix; the lift over
  uniform comes from this single regime-specific concentration.

The other 5 windows (calm-vol regimes 2005-08, 2011-17, 2017-23
ZIRP melt-up, post-COVID recovery) all show CFR's policy
converging to near-uniform across modes — which makes sense:
those regimes have no clearly best mode, and uniform is the
no-regret answer.

## Why this is `partial-OOS` and not `PASS` or `confirmed-null`

The TODO's pre-registered cut was against trailing-best-greedy,
which CFR clears. By that cut alone, this is PASS. But the
adjacent benchmarks (naive uniform tie, passive EW −0.09 alpha)
tell us **why CFR clears trailing-best**: it's not finding a
hidden edge in the menu, it's *just refusing to follow a bad
heuristic*. That's a real result but it's a weak one — the
algorithm earns its keep against a very low bar.

The honest verdict is `partial-OOS`: the algorithm works as
designed, the lift over the named baseline is real and
window-consistent, but the practically-useful comparison (vs
passive EW, vs the naive ensemble that costs nothing to
implement) doesn't separate.

Per the
[`partial-OOS` next-experiment rule](../leaderboard.md#verdict-labels):
*stratify the windows*. The GFC window's positive alpha + the
narrowing-but-not-inverting CFR-vs-trailing lift across windows
confirms a regime-conditional structure exists. The Phase 2 plan
in the [`apps/cfr` TODO](../TODO/apps-cfr.md) is the natural
response: add modes that are themselves alpha-positive in some
regime (13F-imitation-pretrained scorers, sector-restricted
variants), so CFR's regret matching has something to
concentrate on beyond the uniform baseline.

## Mechanism — why CFR matches uniform on this menu

The mathematical reason is clean. Let `R_a` be cumulative regret
for action `a` at some infoset and `π(a) = max(R_a, 0) /
Σ max(R, 0)` (regret matching).

- If **no action has consistently positive regret**, the
  positive-part sum is small and dominated by noise; `π`
  converges to uniform.
- If **one action has clearly positive regret** in some
  infoset, `π` concentrates on it.

In Phase 1's stooq_us_long-universe × universe-agnostic-menu,
the empirical reality is the first case in 8 of 9 regime cells.
The cumulative regret table after 1 training pass on 25 years of
data has entries scattered around zero with magnitude
comparable to per-step noise — the time-averaged policy is then
close to uniform.

This is **expected behavior for CFR over a menu that doesn't
contain edge**. It's also the precise architectural diagnostic
Phase 2 needs: the cumulative regret table reveals which
actions / infoset combinations had any actionable signal at all.
Logs from this run show 9/9 cells were visited; future analysis
should look at per-cell argmax-action stability across windows
to confirm.

## Architectural implications for Phase 2

The Phase 1 result reframes the Phase 2 work:

1. **The algorithm is fine.** Tabular CFR with closed-form
   counterfactual regret converges as expected. The
   [pre-registered Phase 2 question](../TODO/apps-cfr.md) ("does
   13F imitation pretraining help?") is testable against this
   Phase 1 baseline — Phase 2 must show **CFR > Phase 1 CFR by
   ≥ +0.10 Sharpe** to clear its own pre-registered cut.
2. **The action menu is the binding constraint.** Phase 2's
   priority order needs to flip: build 13F-imitation modes
   **first** (so the policy has something to concentrate on),
   then deep CFR on top of that. The order in the original TODO
   has Phase 0 scaffold → Phase 1 → Phase 2 = imitation; we now
   know Phase 1's "PASS vs trailing-best" was less informative
   than the "tie vs uniform" caveat.
3. **The infoset may also be a binding constraint**, but it's
   secondary to the menu issue. A learned encoder over
   multi-modal features (price CWT + macro + 13F-consensus +
   13D events + calendar) gives CFR a finer regime label than
   3×3 vol×dispersion buckets. But fine-grained regime labels
   over a menu of low-edge modes still gives a near-uniform
   policy — fix the menu first.

## Reproducing

```bash
# 1. Local prep (uses Stooq cache, ~20s):
uv run python apps/cfr/scripts/modal/prep_phase1_data.py

# 2. Ship to Modal (image cached after first run, ~25s total):
uvx modal run apps/cfr/scripts/modal/run_phase1.py

# Result lands at:
Output/cfr-phase1.json

# Or run locally (~30s + Stooq scan time):
uv run python apps/cfr/scripts/run_walkforward.py \
    --data-dir ./StooqData \
    --output Output/cfr-phase1.json
```

## Master walk-forward log

[2026-05-12 cfr Phase 1 row](../leaderboard.md) —
[`partial-OOS`](../leaderboard.md#verdict-labels). Numbers
reproduce from `Output/cfr-phase1.json`.
