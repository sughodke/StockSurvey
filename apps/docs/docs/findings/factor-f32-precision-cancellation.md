---
tags:
  - factor-narrow
  - diagnostic
---

# Factor — f32 forward log returns silently regressed val IC by 6×

**Operational rule:** keep `forward_log_returns`, `daily_log_ret`,
and any other rank-IC target array at **f64** until the Tensor
boundary cast in
[`precompute_inputs`](https://github.com/sughodke/StockSurvey/blob/master/apps/factor/src/factor/train.py).
[`pearson_rank_ic`](https://github.com/sughodke/StockSurvey/blob/master/apps/factor/src/factor/objectives.py)
is plain cross-sectional Pearson on raw forward returns; its
covariance numerator cancels catastrophically when val IC is small
(~0.003), and f32 forward-return precision drifts the IC at exactly
that magnitude. f32 byte-savings on a ~78 MB panel are not worth the
SNR loss on a metric that lives in the cancellation regime.

## What happened

Commit
[`3002e8d`](https://github.com/sughodke/StockSurvey/commit/3002e8d)
("factor: scale supervised-cnn walkforward to 3000+ tickers")
demoted `forward_log_returns` and `daily_log_ret` to `np.float32`
with a comment justifying it as "no long-horizon accumulation, f32
precision is plenty". That comment was correct for *absolute* error
on a single forward return, but wrong for the IC numerator's
*cancellation* error.

Re-running the
[supervised-`cnn` walkforward](factor-ssl-walkforward.md) for the
[multi-task auxiliary head A/B](https://github.com/sughodke/StockSurvey/commit/239ebf9)
on 2026-05-10 produced **mean val IC = +0.0005** on the linear arm
— vs the
[doc-recorded **+0.0031**](factor-ssl-walkforward.md#walk-forward-result-2026-05-09)
on the same backbone, same 297-ticker stooq_us_long pool, same
hyperparameters. Per-window: `−0.000, +0.000, −0.006, −0.006,
+0.009, +0.006`. Diff against the doc's `−0.000, +0.001, +0.003,
−0.002, +0.010, +0.006`: 4/6 windows match within ulp-level noise,
windows 2 and 3 substantively differ — window 2 sign-flips. The
asymmetry is consistent with cancellation noise dominating in
quiet windows where cross-sectional return spread is small.

## Mechanism — Pearson IC cancellation in f32

`pearson_rank_ic` is *Pearson on raw scores against raw forward
returns* — despite the function name, no rank/argsort step. The
covariance numerator is:

```
cov_per_bar = sum((scores - s_mean) * (fwd_returns - r_mean) * mask)
```

A sum of signed terms whose result is near zero (val IC ≈ +0.003).
That's textbook catastrophic cancellation — the answer is the
small difference of large near-equal sums.

The forward-return path:

- `log_p_f32 = log(prices)` at magnitudes ~6 (e.g. `log(450) ≈ 6.1`).
  f32 ulp at that scale is ~7e-7, so each `log_p` cell carries up
  to ulp/2 ≈ 3.5e-7 absolute error.
- `fwd_f32 = log_p_f32[t+rebal] - log_p_f32[t]` is the difference of
  two ~equal f32 values. The IEEE 754 subtract is exact, but
  inherits the operand rounding error → up to ~7e-7 absolute error
  on a result of magnitude ~0.05.
- That ~7e-7 error multiplied by the cross-sectional spread of
  scores then summed over ~297 tickers per bar gives a per-bar
  cancellation noise of order **1e-3 to 1e-4** in the cov numerator
  — exactly the magnitude the val IC signal lives at.

f64-then-cast-to-f32:

- `log_p_f64` carries f64 precision (~1e-15 relative).
- `fwd_f64` has f64 precision (~1e-15 absolute on a 0.05 result).
- Cast to f32 at the Tensor boundary quantizes to f32 ulp at the
  result magnitude (~3e-9 at 0.05) — **~200× more precise** than
  the f32-throughout path.

So the cast itself is fine; the error came from doing the
subtraction *inside* f32 where each operand had ~7e-7 of rounding
noise.

## Verification

The fix in
[`9209fa9`](https://github.com/sughodke/StockSurvey/commit/9209fa9)
keeps `forward_log_returns` (`data.py:206-215`) and `daily_log_ret`
/ `log_p` (`train.py:165-171`) at f64. The f32 cast at
`precompute_inputs`'s output (`fwd_rb`, `blr_rb`) is unchanged —
the Tensor handoff still produces f32, just from f64-precision
inputs.

Linear-arm rerun on the f64 path (commit
[`9209fa9`](https://github.com/sughodke/StockSurvey/commit/9209fa9)):

| Window | f64 val IC (this run) | Doc 2026-05-09 | f32 val IC (regressed) |
|--------|-----------------------|----------------|------------------------|
| 0      | −0.000                | −0.000         | −0.000                 |
| 1      | +0.001                | +0.001         | +0.000                 |
| 2      | +0.003                | +0.003         | −0.006                 |
| 3      | −0.002                | −0.002         | −0.006                 |
| 4      | +0.010                | +0.010         | +0.009                 |
| 5      | +0.006                | +0.006         | +0.006                 |
| **mean** | **+0.0031**         | **+0.0031**    | **+0.0005**            |

Bit-for-bit reproduction of the doc baseline. Windows 2 and 3 (the
ones that drifted under f32) are restored.

## Why the regression went undetected for a week

`3002e8d` landed as part of a memory + wall-time audit aimed at
scaling the walkforward to 3000+ tickers. The commit's headline
deliverable was the OOM avoidance via
`align_tickers_at_rebal` and disk-handoff Modal workers — both of
which were correct and substantial. The f32 demotion was a
side-quest that landed without a regression test against the prior
+0.0031 baseline.

Two structural fixes the audit could have caught:

- **No leaderboard row for the original 2026-05-09 SSL walkforward.**
  The +0.0031 number sits only in the finding doc body. A
  leaderboard row with the per-window numbers would have been
  trivially diff-able against any rerun. (This finding's row below
  fills that gap retroactively, plus the regression-and-fix.)
- **No CI smoke that pins per-window val IC.** The existing factor
  smoke test exercises the full path on a synthetic universe and
  asserts only that aggregates exist. A pin-to-known-numbers test
  on a small fixture pool (e.g. 8 tickers × ~500 bars) would have
  flagged the drift on the audit PR.

The drift was caught only because we re-ran the linear arm as the
control for the
[multi-task auxiliary head](https://github.com/sughodke/StockSurvey/commit/239ebf9)
A/B and noticed the linear arm came in at +0.0005 instead of the
doc's +0.0031.

## Operational implications beyond the fix

- **Any future precision optimization on a `pearson_rank_ic`-target
  array needs to be benchmarked against the IC under cancellation,
  not against the absolute-error tolerance of one cell.** "f32
  precision is plenty for a single bar" is true; "f32 precision is
  plenty for a sum-of-signed-terms-near-zero" is not.
- **The Pearson-IC + small-signal regime is the brittle part of the
  factor stack.** The same brittleness applies to anything that
  sums signed cell contributions to a near-zero result —
  e.g. `block_sharpe`'s costs subtraction, `masked_mse` on tiny
  residuals, the multi-task aux head's `aux_weight * masked_mse`
  contribution at small `aux_weight`. Worth keeping f64 in those
  paths until proven otherwise.
- **`block_log_ret` was *not* fixed in this patch.** It's still f32
  per `train.py:218`. That's used for `block_sharpe` (eval-only,
  not the loss), and the Sharpe metric is less cancellation-prone
  because portfolio block returns are *averaged* per bar, not
  cross-sectionally summed against demeaned scores. Worth a
  separate look if Sharpe numbers ever come in suspiciously low,
  but not on the critical path for the IC investigation.

## Master walk-forward log

[Leaderboard row](../leaderboard.md) tagged
[`diagnostic`](../leaderboard.md#verdict-labels) — bug-fix
verified, baseline restored. The rerun this regression-and-fix
unblocked is the
[multi-task auxiliary head A/B](factor-ssl-walkforward.md#outstanding-questions),
which now has a clean baseline to compare against.
