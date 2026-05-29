# HRP asymmetric two-sided modulator — pre-registered

**Status: `pending` — pre-registration locked before the eval runs.**
Follow-up to the [`lie-hrp-baseline`](../findings/lie-hrp-baseline.md)
window-5 decomposition (commit `81337e0` / 2026-05-28), which falsified
the symmetric `eff_rank/n_active` modulator as a DCA overlay candidate.
The headline w5 lift (+0.586) was a Sharpe-arithmetic artifact:
sub-period mod-lifts sum to **+0.043**, the gate stays floored through
the 2023 recovery, and 2023 alpha is *destroyed*, not preserved. The
modulator detects regime change on the *down* leg (gate activated 100%
of 2022 rebals) but is **regime-blind on the bounce**. This pre-reg
locks the asymmetric rescue test before running it.

---

## Why this pre-reg exists

Per the [verdict→next-experiment rule](../../CLAUDE.md) for
`partial-OOS` (which is the honest read of the w5 lift once
decomposed): "stratify the windows ... if a feature splits them, you
have a regime gate." The w5 decomposition produced exactly that
feature: **direction of the regime shift**. The current modulator is
symmetric in time but the regime it gates is asymmetric — symmetry-
breaks last 1-3 months, recoveries last 4-12 months. A symmetric gate
holds the floor through both legs and prints the recovery as alpha
destruction.

The proposed two-sided gate (mentioned but not pre-reg'd in the
finding's "Operational implication" paragraph) is the orthogonal
lever: same detection signal, **asymmetric release**.

---

## Mechanism — the steel-man

Why this could clear the bar (not why it won't):

1. **The detection side already works.** w5 sub-period table shows
   the symmetric modulator caught the 2022 drawdown — gate active
   100% of rebals, eff_rank dropped from ~31 (2021) to ~24 (2022).
   That is *real* symmetry-break detection on a 312-name universe,
   not a numerical artifact. The mechanism is sound on the down leg.

2. **The release side has a clean orthogonal trigger.** Cumulative
   trailing-N-day EW return is the canonical positive-regime
   indicator. When it turns positive after a sustained drawdown,
   the symmetry-break is resolving — return correlation has
   re-decoupled from crisis-mode lockstep. The current modulator
   ignores this and waits for eff_rank to recover (which lags by
   months because the same names that crashed dominate the
   re-clustering).

3. **The 2023 recovery cell is the test surface, not the training
   surface.** The +0.586 headline was a 2023 *cost*; replacing the
   2023 cell with the unmodulated HRP stream (gate fully open)
   would have produced a strictly better window. The asymmetric
   rescue is therefore a mechanically clean hypothesis: it can
   only fix the failing cell, not the passing ones.

If the per-sub-period alpha attribution becomes additive (i.e.
sub-period lifts roughly sum to the full-window lift instead of
cancelling), the gate is real. If the full-window lift evaporates
because 2022 detection alpha was always Sharpe-arithmetic, the
arc closes.

---

## Search space (locked)

Two-sided gate (one open-rule, one close-rule) parameterized by:

| param | values | mechanism |
|---|---|---|
| `close_rule` | `eff_rank_drop` (current symmetric) | gate closes (scalar → floor) when `eff_rank / n_active` falls below floor |
| `open_rule` | `none` (control = current symmetric), `ewma_return_positive`, `eff_rank_recovery`, `trailing_max_return` | gate re-opens (scalar → 1.0) on positive-regime trigger |
| `open_lookback` | {20, 40, 60, 100} trading days | window for the open-rule indicator |
| `open_threshold` | {0.0, +0.5σ, +1.0σ} of trailing-3y EW return | strictness of the release trigger |
| `floor` | {0.25 (current), 0.50} | how much the gate throttles when closed |

Grid size: 1 (close) × 4 (open_rule) × 4 (lookback) × 3 (threshold)
× 2 (floor) = **96** trials. n_trials for deflation = 96.

`open_rule = none` reproduces the published symmetric modulator;
every other cell is asymmetric.

---

## Datasets + windowing (locked)

| field | value |
|---|---|
| universe | `stooq_us_long` 312-name long-history panel (matches the lie-hrp baseline finding) |
| span | 2005-01-01 → 2025-10-16 |
| windowing | same 6-window walk-forward used in the baseline (windows w1-w6) |
| primary cell | window 5 (2020-07 → 2023-08) — the cell whose decomposition motivated the test |
| rebal | every 21 trading days (matches lie-hrp baseline) |
| friction | 10 bps per \|Δw\| (matches lie-hrp baseline) |
| metric | per-sub-period α vs EW, per-window block-Sharpe, **full-window α decomposed into sub-period α** |
| sub-period split | identical to the w5 decomposition: 2020-rebound-tail / 2021-calm-bull / 2022-rate-cycle / 2023-recovery (so the additivity check uses the same buckets that already falsified the symmetric version) |
| comparison | (a) unmodulated HRP, (b) symmetric modulator (current published version, `open_rule = none`), (c) asymmetric winner |

---

## Pre-locked verdict bar

The 96-trial grid produces an in-search winner `cell*` on the **w1-w4
training windows** (i.e. the symmetric modulator's training windows
that *don't* include the w5 cell that motivated this test). The
verdict is locked on the **w5 OOS behaviour at `cell*`**:

| condition | verdict |
|---|---|
| (1) w5 full-window α-lift ≥ +0.10 vs unmodulated HRP AND (2) sub-period α-lift sum ≥ +0.08 (the additivity check that killed the symmetric version) AND (3) 2023-recovery sub-period α ≥ 0 (the "no alpha destruction" requirement) | **confirmed-OOS** — ship as DCA overlay candidate |
| (1) holds but (2) or (3) fails | **partial-OOS** — gate detects but doesn't preserve; archive without ship |
| (1) fails OR `cell*` is `open_rule = none` (the symmetric control wins) | **confirmed-null** — the asymmetric lever doesn't rescue; HRP modulator arc closes for good |

**Sample-size honesty.** w5 has 39 rebals across 4 sub-periods —
fewer than the n=198 fold that constrained the follow-consensus arc.
A 96-trial deflation against this sample is hostile. If the
stationary-bootstrap 95% CI on the w5 OOS α-lift is wider than
±0.40, the verdict is **automatically downgraded one tier**
(confirmed → partial, partial → null). The additivity check (2) is
the load-bearing falsifier — a high full-window lift that doesn't
sum across sub-periods reproduces exactly the artifact the baseline
finding already caught.

---

## What out-of-scope means here

- **New detection signals.** This pre-reg tests *release* asymmetry
  against the same `eff_rank/n_active` detection signal that already
  works on the down leg. Adding a new detection signal compounds the
  search space and obscures whether the asymmetry alone is enough.
  If the asymmetric rescue lands `confirmed-null`, a separate pre-reg
  can replace the detection signal.
- **Cross-universe generalization.** Test on `stooq_us_long` only.
  If w5 clears, a second pre-reg can port to Phase-2 / factor-narrow.
  Adding universe variation here would re-open the universe-shift
  arc inside a w5-specific test.
- **Optuna parameter search.** The 96-cell grid is enough resolution
  given the w5 sample length. Optuna here would burn deflation
  budget on noise within the same training fold.

---

## Acceptance criteria

1. Driver script `apps/lie/scripts/run_hrp_asymmetric_modulator.py`
   computes the 96-cell grid over the 6-window walk-forward, picks
   `cell*` on w1-w4, reports w5 OOS metrics including the sub-period
   decomposition table.
2. Verdict label lands in `apps/docs/docs/leaderboard.md` per the
   table above.
3. Finding (if confirmed-OOS or partial-OOS) writes to
   `apps/docs/docs/findings/lie-hrp-asymmetric-modulator.md` with
   `cell*` parameters and the additivity-corrected α attribution.
4. If `confirmed-null`, append a closing paragraph to
   [`lie-hrp-baseline`](../findings/lie-hrp-baseline.md) marking the
   HRP-modulator arc closed and remove this TODO entry.

---

## Pointers

- Parent finding that motivated this: [`lie-hrp-baseline`](../findings/lie-hrp-baseline.md) (window-5 decomposition appended 2026-05-28).
- Verdict vocab: [`leaderboard.md#verdict-labels`](../leaderboard.md#verdict-labels).
- Existing driver to extend: `apps/lie/scripts/hrp_w5_decomposition.py`.
- Sub-period buckets to reuse: 2020-rebound-tail / 2021-calm-bull / 2022-rate-cycle / 2023-recovery.
