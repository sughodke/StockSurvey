# Loss-pivot eval — Sharpe and IR-vs-EW losses underperform rank-IC by 0.37 Sharpe

The cheap test motivated by the user's diagnosis after the
long-short result: rank-IC is scale-invariant and doesn't see costs;
maybe a loss aligned with the EW gate (Sharpe directly, or
Information Ratio vs the EW benchmark) finds a *different head*
that beats long-only's val Sharpe. Result: it found a different
head, but a materially worse one.

Verdict: [`confirmed-null`](../leaderboard.md#verdict-labels) for
the loss-mismatch hypothesis. The new losses didn't fail to help;
they actively destroyed ~0.37 of val Sharpe. This is independent
confirmation of the
[long-short constructor](factor-rankic-long-only-mismatch.md)
finding's downstream conclusion: the cross-sectional return
prediction problem is not bottlenecked on portfolio construction
or loss choice — it's bottlenecked on signal magnitude. Pivot
to a different prediction problem per the
[`TODO/different-prediction-problem`](../TODO/different-prediction-problem.md).

## Setup

Same factor-narrow universe (297 stooq_us_long names,
`min_history_bars=6500`), same 6-window walk-forward (train=63,
val=39, step=39 blocks at `rebal_days=20`), same linear head, same
`n_steps=200 lr=1e-2 wd=1e-3 commission_bps=10`. Three loss
arms, head re-initialized per window per arm:

- **`rank_ic`** — existing baseline. `pearson_rank_ic` (per-bar
  Pearson averaged across bars). Scale-invariant; temperature
  frozen at `log_temperature=0.0` (temp=1.0). Reproduces the
  [`factor-indicator-baseline`](factor-indicator-baseline.md)
  numbers bit-for-bit.
- **`block_sharpe`** — `block_sharpe(scores, log_temp, blr, mask,
  rebal_days, commission_frac)` from `factor.objectives`. The
  exact eval-time differentiable Sharpe wired as a training loss.
  Temperature trainable (added to AdamW alongside head weights).
- **`ir_vs_ew`** — new `block_ir_vs_ew` in `factor.objectives`.
  Same softmax-LO constructor as `block_sharpe`, but the per-bar
  return subtracts the universe's EW return before the Sharpe
  reduction. The optimum is alpha-per-tracking-error rather
  than total-return-per-volatility — directly aligned with the
  `alpha = model_val_sharpe − passive_val_sharpe` rule in
  [`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md).
  Temperature trainable.

All three arms eval'd on **all four metrics** per window — val
IC, val Sharpe (long-only), val IR vs EW, val Sharpe (long-short)
— so the cross-loss comparison is apples-to-apples regardless of
which loss trained the head.

## Result (2026-05-10)

| arm | mean val IC | mean val Sh (LO) | mean val IR | mean val Sh (LS) | mean logT |
|---|---:|---:|---:|---:|---:|
| `rank_ic` | +0.0055 | **+0.278** | **−0.486** | −0.067 | 0.00 (frozen) |
| `block_sharpe` | +0.0066 | **−0.097** | −0.415 | −0.017 | **−1.60** |
| `ir_vs_ew` | +0.0006 | **−0.109** | **−0.394** | −0.243 | **−1.50** |

Headline: **block_sharpe loss val Sharpe = −0.097 vs rank_ic +0.278**
→ delta **−0.375**. **ir_vs_ew loss val Sharpe = −0.109 vs rank_ic
+0.278** → delta **−0.387**. Both new losses underperformed by
roughly 0.37 of Sharpe — well outside the ±0.10 noise band.

Pre-registered cuts (per
[`TODO/long-short-constructor`](../TODO/long-short-constructor.md)
applied here as well):

| Cut | Threshold | Observed | Verdict |
|---|---|---|---|
| Pass: max delta vs rank_ic ≥ +0.20 | ≥ +0.20 | **−0.375** | fail |
| Fail (clean): all new arms within ±0.10 of rank_ic | ≤ 0.10 | 0.375–0.387 | fail (worse than the band predicts) |

The script reported "INCONCLUSIVE" because its asymmetric verdict
logic only treats large positive deltas as PASS. The honest read
of the test is **confirmed-null with negative-direction signal**:
the loss isn't the binding constraint, and aligning the loss with
the deployment metric makes it worse, not better.

## Per-window detail

```
=== rank_ic ===
win     tr_ic    val_ic    val_sh    val_ir  val_sh_ls    logT
  0  +0.1447   +0.0039     -0.985    -1.311     -0.336   +0.00
  1  +0.0951   -0.0101     +0.855    -0.216     -0.382   +0.00
  2  +0.1255   +0.0227     +0.732    -0.733     +0.273   +0.00
  3  +0.1245   +0.0007     +0.235    -0.734     -0.212   +0.00
  4  +0.1148   -0.0088     +0.418    -0.451     -0.212   +0.00
  5  +0.1063   +0.0246     +0.411    +0.531     +0.466   +0.00

=== block_sharpe ===
win     tr_ic    val_ic    val_sh    val_ir  val_sh_ls    logT
  0  +0.0178   +0.0009     -0.633    -0.474     -0.225   -1.22
  1  +0.0080   -0.0249     +0.803    +0.020     -0.717   -1.60
  2  +0.0252   +0.0251     -0.275    -0.569     +0.568   -1.70
  3  +0.0243   +0.0340     -0.198    -0.628     +0.579   -1.76
  4  +0.0241   -0.0172     -0.559    -0.935     -0.542   -1.62
  5  +0.0338   +0.0213     +0.282    +0.096     +0.235   -1.71

=== ir_vs_ew ===
win     tr_ic    val_ic    val_sh    val_ir  val_sh_ls    logT
  0  -0.0037   +0.0007     -0.498    -0.338     -0.270   -1.11
  1  +0.0181   -0.0098     +0.587    +0.130     -0.436   -1.22
  2  +0.0363   +0.0061     -0.219    -0.516     -0.042   -1.71
  3  +0.0223   +0.0196     -0.187    -0.394     +0.337   -1.61
  4  +0.0354   -0.0118     +0.233    -0.011     -0.693   -1.73
  5  +0.0256   -0.0014     -0.569    -1.235     -0.351   -1.14
```

## Mechanism — why the Sharpe-aligned losses *underperform*

Three observations from the per-window data, in order of
load-bearingness:

### 1. Temperature collapses to ~0.2 in both new arms.

`logT` settles between **−1.1 and −1.8** in every window of both
new arms (temp = exp(logT) ≈ 0.16 to 0.33). The default rank-IC
temperature is `temp=1.0` — a "warm" softmax that spreads weight
across the universe with mild tilt. The Sharpe-aligned losses
push toward `temp ≈ 0.2`, a near-argmax distribution that
concentrates the portfolio on a handful of top-scoring names per
bar.

This is not the gradient acting on a calibration error. The
optimizer is *correctly* learning that, given a small per-bar
mean-return signal, the highest in-sample Sharpe comes from
maximizing weight on the top scores. In sample (train), this
works — the train Sharpe under `block_sharpe` is positive
(unreported in the table but visible from the loss curve). On
val, the pattern reverses: concentration amplifies variance
without amplifying mean, because the IC at ~+0.005 doesn't
provide enough lift to overcome the noise of a 1-3 name bet.

### 2. Train IC collapses too — by ~5×.

`rank_ic` arm gets mean train IC ~+0.115. `block_sharpe` and
`ir_vs_ew` both get mean train IC ~+0.022 — a 5× drop. The
Sharpe-aligned losses don't even fit the training distribution
as well as the IC-aligned loss does. This is the gradient
signal making the head *less* good at cross-sectional ordering
(rank-IC is now ~0 on train) in exchange for being more
concentrated on whichever names happened to win in-sample. The
new heads are overfitting on extreme picks rather than learning
broad signal — and on val, those extreme picks don't generalize.

### 3. The "rank-IC is wasteful" intuition was wrong in this regime.

The user's earlier diagnosis (made before this test fired) was
that rank-IC's spread-thin behavior was throwing away
information by failing to concentrate. The data here says the
opposite: rank-IC's spread-thin behavior is *the right policy*
in low-signal regimes because it acts as inadvertent risk
control. A 297-name approximately-equal-weight portfolio has
low variance; the small IC delivered modest mean return; result
is a positive (if small) Sharpe.

When the optimizer is told "go optimize Sharpe directly," it
correctly finds that high *in-sample* Sharpe requires
concentration. But concentration in a low-IC regime is a poor
*risk-adjusted* trade out-of-sample — variance balloons, mean
doesn't follow.

## The general principle

**Sharpe-as-loss requires sufficient per-name signal-to-noise
to clear the variance penalty of concentration.** The math:
under independent-bet assumptions and Markowitz-optimal
weights, optimal portfolio variance scales as ~1/IR², and
optimal concentration scales as ~1/IC. At IC=+0.005, the
optimizer says "concentrate on a few names" but the OOS
realized Sharpe doesn't grow because the +0.005 doesn't deliver
enough mean return to dominate the new variance. Rank-IC's
spread-thin behavior was acting as a Bayesian shrinkage prior
toward EW — appropriate for a low-confidence regime even though
the metric itself doesn't know about EW.

This is a known failure mode in quant: *Sharpe optimization
without sufficient signal produces concentration without
payoff*. We just confirmed it for our specific setup.

## Connection to the EW gate and the long-short result

Putting the three findings on this leaderboard arc together:

1. [`passive-ew-benchmark`](passive-ew-benchmark.md) — no model
   row clears its universe's passive EW Sharpe.
2. [`factor-rankic-long-only-mismatch`](factor-rankic-long-only-mismatch.md)
   — long-short constructor doesn't rescue the rank-IC head
   (val Sharpe **−0.067**); the "discarded short signal"
   hypothesis is falsified.
3. **This finding** — Sharpe-aligned and IR-aligned training
   losses both *worsen* val Sharpe by ~0.37; the "wrong loss"
   hypothesis is falsified, in the opposite-direction sense.

These three independent tests converge on the same diagnosis:
**at the +0.005 to +0.012 cross-sectional IC scale our
indicator stack delivers, no portfolio construction or loss
choice can produce a head that clears EW after costs**. The
binding constraint is signal magnitude, not how that signal is
optimized or deployed.

## Implication

Per the `confirmed-null` next-move rule: stop testing
variations of the same lever. We have now exhausted three
levers (constructor — long-short; loss — Sharpe and IR; and
the original metric — long-only top-N) without lifting val
Sharpe above its rank-IC baseline. The next test must change
the underlying *prediction problem* — what the head is being
asked to predict, not how the prediction is monetized.
[`TODO/different-prediction-problem`](../TODO/different-prediction-problem.md)
is now the top-priority research thread.

The `block_sharpe` and `block_ir_vs_ew` losses themselves
remain in `factor.objectives` for any future experiment where
the underlying signal is large enough to reward Sharpe-aligned
optimization. They're the right tool when IC is high; they're
the wrong tool when IC is below the friction floor.

## Operational rule (added to CLAUDE.md)

> **Don't train on Sharpe / IR losses for cross-sectional return
> heads with mean val IC < +0.02.** The 2026-05-10 loss-pivot eval
> (`apps/factor/scripts/loss_pivot_eval.py`) showed both
> `block_sharpe` and `block_ir_vs_ew` losses underperformed rank-IC
> by ~0.37 of val Sharpe (rank-IC +0.278 vs block_sharpe −0.097 vs
> ir_vs_ew −0.109) on the factor-narrow indicator stack. Mechanism:
> Sharpe-aligned losses train temperature to ~0.2 (near-argmax
> concentration), which amplifies variance without amplifying mean
> in the low-IC regime. Rank-IC's scale-invariance acts as
> inadvertent shrinkage toward EW — appropriate when IC is small.

## Master walk-forward log

[2026-05-10 loss-pivot block_sharpe row](../leaderboard.md) and
[2026-05-10 loss-pivot ir_vs_ew row](../leaderboard.md) —
both [`confirmed-null`](../leaderboard.md#verdict-labels).
