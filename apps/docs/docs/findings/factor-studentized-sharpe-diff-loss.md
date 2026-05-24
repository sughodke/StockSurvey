# Factor head trained against studentized Sharpe-diff loss — `confirmed-null`

**Operational rule.** Training a factor head directly against the
differentiable studentized Sharpe-difference loss
(`block_studentized_sharpe_diff_vs_ew`, the literature-canonical
analogue of the Ledoit-Wolf bootstrap test the cross-arc ladder uses)
**does NOT beat the existing `ir_vs_ew` loss** on factor-narrow at 5d
cadence. The candidate's pooled-OOS t-stat is +0.69 vs the baseline's
+1.29 — **Δt = −0.60**, candidate is materially WORSE. The baseline
`ir_vs_ew` is the only arm whose pooled-OOS bootstrap CI excludes 0
(ann ΔSR +0.46, [+0.02, +0.91]). The honest expectation locked in the
pre-reg held: direct training against ΔSR/s.e. doesn't escape the
cross-sectional null; in this head-to-head it actively underperforms
the canonical Sharpe-aligned loss.

## Pre-registration

Locked at [`TODO/factor-studentized-sharpe-diff-loss`](../TODO/factor-studentized-sharpe-diff-loss.md)
in commit `fdab384` BEFORE the eval ran. The falsification bar,
walk-forward design, and arm definitions were not editable post-hoc.

## Result

### Headline numbers (pooled OOS across windows, bootstrap 95% CI)

| arm | n_obs | per-period t | pooled ΔSR ann | 95% CI ann | excludes 0? |
|---|---:|---:|---:|---|:---:|
| rank_ic (reference) | 1000 | +0.87 | +0.28 | [−0.14, +0.68] | no |
| **`ir_vs_ew` (baseline)** | 800 | **+1.29** | **+0.46** | **[+0.02, +0.91]** | **YES** |
| `studentized_sharpe_diff_vs_ew` (candidate) | 800 | +0.69 | +0.25 | [−0.26, +0.73] | no |

### Verdict per pre-reg

- Δ t_stat (candidate − baseline) = **−0.60**
- Candidate's CI excludes 0: **False**
- Pre-reg bar:
  - `confirmed-OOS` requires Δt ≥ +1.0 AND CI excludes 0 → fails (both)
  - `partial-OOS` requires Δt ≥ +0.3 → fails
- **Locked verdict: `confirmed-null`**

### Per-window val IC

| arm | mean val IC | pos-IC fraction |
|---|---:|---:|
| rank_ic | +0.0109 | 0.50 |
| ir_vs_ew | +0.0012 | 0.40 |
| studentized_sharpe_diff_vs_ew | +0.0044 | 0.60 |

Note: per-window mean val Sharpe was reported as NaN for `ir_vs_ew`
and the candidate due to a pre-existing nanmean aggregation bug in
`train_walkforward.py` (window-0 NaN for the candidate; one
window with degenerate softmax for `ir_vs_ew`). The per-window values
print cleanly in the training log; only the aggregate display is
affected. The pooled bootstrap CI is the load-bearing column above
and is computed correctly from the per-window OOS streams.

## What we learned

### 1. The candidate is not just null — it's materially worse

Both arms see the same data, same windows, same hyperparameters,
same seed. The only difference is the loss function. `ir_vs_ew`
optimizes `(SR_LO − SR_EW) / TE` (information ratio); the candidate
optimizes `(SR_LO − SR_EW) / s.e.(diff)`. They're closely related
but **the candidate's s.e. denominator can collapse to near-zero**
when the LO portfolio is very correlated with EW (which it is when
the head is poorly initialized or the softmax is concentrated near
uniform). That instability hurt training, especially window 0 where
the candidate produced NaN gradients on the first step.

### 2. The training stability gap is real

The candidate's window 0 hit `logT=+nan, sq=+nan` on iteration 1 —
the Lo-Mertens delta-method s.e. `var_diff = var_a + var_b − 2ρ√(var_a·var_b)`
went near-zero in the initial near-uniform softmax regime, blowing
up the gradient. Adam recovered for subsequent windows (window-1
through 9 trained finitely) but window 0 was lost.

**Recommendations for future use** of the loss:
- Use `with_moments=True` (Bailey-LdP denominator is more stable)
- Add a stability floor: `se = max(se, 1e-3)` or similar
- Start at a higher `init_log_temperature` so the initial softmax
  isn't degenerate
- Skip the loss for the first ~N warm-up steps; train with
  `ir_vs_ew` for warmup, then switch to studentized for fine-tuning

### 3. The methodology rewrite finding is reinforced

The methodology rewrite established that under proper apples-to-
apples testing (Ledoit-Wolf CI), zero arcs on the current ladder beat
DCA. The implicit follow-up question was: *can directly training
against that test escape the null?* This arc answers **no** — on the
factor-narrow universe at 5d cadence, even the literature-canonical
training target doesn't surface a beat-baseline result.

The honest read: **the binding constraint is the data, not the
optimization method**. Cross-sectional alpha on this universe at this
horizon is at-or-below the noise floor; no loss function can extract
information that isn't there.

### 4. `ir_vs_ew` remains the recommended factor loss

The baseline didn't just survive — it produced the **only positive
result on the ladder** (CI excluding 0 at +0.02 to +0.91 ann ΔSR).
This is consistent with [`factor-loss-pivot`](factor-loss-pivot.md)
which established `ir_vs_ew` as a reasonable alternative to rank-IC
under the appropriate horizon. The new finding sharpens that
recommendation: at 5d cadence on factor-narrow, `ir_vs_ew` is
*specifically* the loss that produces a 95%-CI-excluding-zero result
on the pooled OOS sample.

## Reproduction

```bash
# Modal eval (~5-10 min on T4, < $0.20):
uvx modal run apps/factor/scripts/modal/train_studentized_sharpe_diff.py

# Local verdict (applies locked pre-reg bar):
uv run python apps/factor/scripts/verdict_studentized_sharpe_diff.py
```

Inputs: `apps/notebook/data/stooq_us_long/` (factor-narrow universe,
310 tickers at `min_history_bars=6500`). Outputs:
`Output/factor-stud-sh-diff-{rank_ic,ir_vs_ew,studentized_sharpe_diff_vs_ew}-windows.npz`,
`Output/factor-studentized-sharpe-diff-verdict.json`.

## Master walk-forward log

[Cross-arc ladder](../leaderboard.md#cross-arc-ranking--primary-ledoit-wolf-Δsr-vs-dca)
— the candidate's pooled OOS stream is in `Output/factor-stud-sh-diff-studentized_sharpe_diff_vs_ew-windows.npz`
and could be added as a new ladder row at `n_trials=1` (single
pre-registered hypothesis, no Optuna search), but with `confirmed-null`
verdict and Δt = −0.60 there's no operational reason to surface it
above the baseline. The interesting row to potentially add is the
**baseline `ir_vs_ew` stream as a new ladder candidate**: it
genuinely excludes 0 vs zero-benchmark on pooled OOS. Whether it
beats DCA under Ledoit-Wolf CI is a separate (likely null, but
worth verifying) question. Verdict label
[`confirmed-null`](../leaderboard.md#verdict-labels) per locked pre-reg.

## Cross-links

- Loss landed: commit `cb8f84e`
- Pre-reg: commit `fdab384`
- Modal port: commit `82a0637`
- Methodology rewrite: [`ladder-methodology-rewrite`](ladder-methodology-rewrite.md)
- Strategy-selection brief: `.research-best-candidate-for-studentized-loss.md`
- Factor short-horizon baseline: [`factor-shorthorizon-representation`](factor-shorthorizon-representation.md)
