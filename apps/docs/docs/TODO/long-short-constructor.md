# Long-short constructor — close the rank-IC / long-only-top-N mismatch

Diagnostic motivating this TODO:
[Rank-IC trains a signed signal that long-only top-N can only
half-execute](../findings/factor-rankic-long-only-mismatch.md).

The cheap test before any retraining: take an existing
factor-narrow rank-IC checkpoint and run two portfolio
constructors over the same walk-forward windows. If the
sign-symmetric metric was being half-discarded, long-short
will lift Sharpe materially. If not, the deployment-mismatch
hypothesis is falsified and the next move is the
[different prediction problem](different-prediction-problem.md)
TODO instead.

## Test design

Universe: factor-narrow (297 tickers, the same list used in
[factor-indicator-baseline](../findings/factor-indicator-baseline.md)).
Walk-forward: existing 6-window setup, train=63 / val=39 /
step=39 blocks at `rebal_days=20`.

Two arms, same head, same windows:

- **Long-only top-N (existing)** — `softmax(score / τ)` with
  the temperature already in the checkpoint, `apply_position_cap`
  with the existing 0.25 cap. Re-run the existing eval for the
  apples-to-apples baseline.
- **Long-short market-neutral (new)** — z-score the per-bar
  score vector, clip at ±3σ, normalize so `sum(w) = 0` and
  `sum(|w|) = leverage` (start at `leverage = 1.0`). No spread
  gate on the short side for v1 — adds confound; track turnover
  and add a gate later if needed.

Costs: 10 bps per side on L1 turnover at each rebal. Long-short
roughly doubles turnover vs long-only (you turn over both legs)
so this is a meaningful sensitivity dimension. Keep an extra
20 bps and 50 bps row in the summary.

## Pre-registered pass / fail

The verdict the next leaderboard row should declare:

| Outcome | Long-short val Sharpe (mean over 6 windows, 10 bps) | Verdict | Next move |
|---|---|---|---|
| **Pass** | ≥ +0.20 with ≥ 4/6 windows positive | `confirmed-OOS` | Retrain head with Sharpe-aligned loss that bakes in the long-short constructor; sweep leverage. |
| **Inconclusive** | +0.10 to +0.20 or 3/6 windows | `partial-OOS` | Stratify windows by vol regime / dispersion before deciding. Possibly a `rebal_days` sweep before retrain. |
| **Fail** | < +0.10 or ≤ 2/6 windows | `confirmed-null` | Long-short cannot rescue the head. Pivot to [different prediction problem](different-prediction-problem.md) — this falsifies the "discarded short signal" hypothesis. |

The leverage row (`sum(|w|) = 1`) is what the comparison gate
reads. A leverage = 2 row would inflate Sharpe roughly linearly
before costs and is informative as a sensitivity check, not a
shippable result.

## Implementation scope

~80 LoC plus a walk-forward column. Touched files:

- New: `apps/factor/src/factor/portfolio/long_short.py` —
  `long_short_weights(scores, mask, leverage=1.0, clip_sigma=3.0)`
  returning a per-bar signed weight vector.
- `apps/factor/src/factor/train_walkforward.py` — extend
  `WalkForwardResult` with a `val_sharpe_long_short` column;
  call the new constructor on the held-out val block; rebal /
  cost accounting matches the long-only path.
- `apps/factor/scripts/eval_long_short.py` — small driver
  that loads an existing checkpoint and runs the walk-forward
  with both constructors, writes a single JSON.

No tinygrad changes. No retraining. No new universe build.
The factor-narrow walk-forward already runs in <2 min local on
a cached panel, so this is a local job, not a Modal job.

## What this TODO is *not* a test of

- Not testing rank-IC vs Sharpe-aligned training. That's the
  *next* experiment if pass.
- Not testing different feature classes / wider universes /
  different rebal cadences. The point is to isolate the
  constructor change while holding everything else fixed.
- Not testing live-tradability. Long-short adds borrow-cost +
  short-availability complications that would need a paper-
  trade dry-run before going live; the v1 test is OOS Sharpe
  only.

## Implementation order

1. Implement `long_short_weights` + the walk-forward column.
2. Run on the existing linear-head checkpoint
   (`Output/walkforward-linear-s200-wd0.001-windows.npz`) —
   the +0.012 mean val IC baseline.
3. Run on the MLP-head and multi-task aux-head checkpoints
   for the same windows (see
   [factor-multitask-aux-head finding](../findings/factor-multitask-aux-head.md)).
4. Land 2-3 leaderboard rows (one per checkpoint × constructor),
   write the closing finding page that points back here.
5. If pass: write the retrain-with-Sharpe-loss follow-on TODO.

The retrain step is gated on the cheap test passing — no point
sinking GPU hours into a Sharpe-aligned objective if the
constructor swap alone doesn't move the needle.
