# `apps/factor` — bilevel horizon objective (IC for score head, deployment reward for π)

**Status**: pre-registered 2026-05-15. Sweep not yet run.

## Hypothesis

The factor endogenous-horizon mixture's score head is trained on
rank-IC (well-behaved, stable, established) and the horizon head π
is trained on *the same* mixture-IC signal. The
[2026-05-14 sweep](../findings/factor-endogenous-horizon-mixture.md)
closed with mean Δ-fix +0.048 (`partial-OOS`) and the
[2026-05-14 oracle diagnostic](../leaderboard.md) revealed a +0.112
ceiling on horizon selection — the mixture captures ~42% of it.

The bilevel objective decouples the two heads' training signal:

```
L = -mean_t Σ_k π_t[k] · IC_k_t                                  ← score head's gradient
    -λ · mean_t Σ_k π_t[k] · per_bar_net_daily_return_k_t        ← π head's gradient only
```

**Score-head gradient flows from term 1 only** (term 2 uses
`scores.detach()` so the score head never sees the realized-return
signal). **π head gradient flows from both terms** — it can now weight
horizons by the realized PnL of the score-weighted portfolio, not
just by their rank stability.

Why this might lift Δ-fix above +0.048:

1. **Score head retains rank-IC's stability**. The 2026-05-12 sizing-input
   v0 finding showed `mse_alpha` is calibrated but adds zero signal-quality
   over `rank_ic` — rank-IC is the right training objective for the
   score head. Bilevel preserves that.
2. **π head sees the supervision signal it actually cares about** at
   deployment: per-day realized return. Currently π weights horizons
   by their per-bar IC, which is a proxy for return that ignores
   return magnitude and post-cost realization. Direct supervision on
   the actual deployment metric should give π more state-conditional
   discrimination — especially across horizons where IC and realized
   return decouple (e.g., when a high-rank-IC bar at h=60 has small
   realized return due to commission drag).

## Why this isn't `apps/critic` v0.2 redux

v0.2's policy training collapsed to rank-by-Φ because π and Φ shared
input space and architecture. The bilevel objective is structurally
different: the two terms in the loss are **not collinear** —
rank-IC is a per-bar rank statistic; per-day net return is a per-bar
PnL with cost-aware scaling that varies across horizons. π has real
degrees of freedom: it can weight a horizon DOWN even when IC favors
it (because cost amortization at short horizons hurts net return) or
UP (because the score's magnitude-tilted bet pays off at that horizon
despite mediocre IC).

## Test design

### Loss specification

Both terms are normalized to per-batch std (detached, so the std is a
constant in the autograd graph — pure scale factor):

```
L_IC_norm  = -mean_t Σ_k π_t[k] · IC_k_t * valid_k_t / std(IC_kt)
L_RET_norm = -mean_t Σ_k π_t[k] · ret_k_t * valid_k_t / std(ret_kt)
L = L_IC_norm + λ · L_RET_norm + entropy_reg
```

where `ret_k_t = (centered_score_t · fwd_log_return_k_t · mask_k_t) /
n_valid_t / horizon_k - commission_frac / horizon_k`. The
score-centering matches the IC computation's centering convention.
Division by `horizon_k` converts the cumulative log return into a
per-day rate (so the commission term is comparable across horizons).

`scores.detach()` is applied inside `L_RET` so the deployment-return
gradient flows only into π, not into the score head.

### Sweep

λ ∈ {0.0, 0.25, 0.5, 1.0, 2.0} (5 arms, including α=0 which reproduces
the existing 2026-05-14 `partial-OOS` baseline). Normalization by
per-batch std means λ is a dimensionless balance between the two
training signals; small values nudge π toward deployment-aware mixing,
large values push it to dominate.

### Universe / windowing

Identical to the 2026-05-14 entropy sweep:
- `factor-narrow`: 297 stooq_us_long tickers, `min_history_bars=6500`.
- 6-window walk-forward at `h_min=5` fine grid.
- Train 252 fine bars × val 156 × step 156 (~5y train / ~3y val).
- Horizons `(5, 10, 20, 40, 60)`.
- Commission 10 bps; temperature 1.0; AdamW; n_steps 200.
- Seed 0.

This makes the sweep directly comparable to the entropy sweep's per-α
rows.

### Pre-registered cuts

Apply per-arm; the **sweep verdict is the best arm's verdict**:

- **STRONG-PASS** (`confirmed-OOS`): mean Δ-fix ≥ +0.10 AND 6/6 positive
  windows (captures ≥ ~90% of the +0.112 oracle ceiling).
- **PASS** (`confirmed-OOS`): mean Δ-fix ≥ +0.10 AND ≥ 5/6 positive.
- **MARGINAL** (`partial-OOS`): mean Δ-fix ≥ +0.07 AND ≥ 4/6 positive.
- **FAIL** (`confirmed-null`): otherwise. (Includes the case where the
  λ=0 arm is the only positive arm — that just reproduces the
  existing `partial-OOS` baseline; we want a *lift* from the bilevel
  objective specifically.)

The pre-reg cuts deliberately raise the Δ-fix threshold from +0.048
(the 2026-05-14 baseline) to +0.07 (MARGINAL) and +0.10 (PASS). Any
new arm has to **move the needle** vs the existing baseline; a tied
result is FAIL.

### Honest acknowledgements before running

1. **Per-bar realized return is noisier than IC.** At high λ the
   horizon head will overfit to in-sample returns and OOS could collapse.
   The +0.10 PASS cut + 5/6 positive cut filters this — overfit
   policies generally win in-sample but fail on at least one OOS
   window.
2. **The score head's stability under the bilevel loss is untested.**
   The detach is meant to preserve rank-IC training signal, but if
   the optimizer is sensitive to the loss's overall scale (Adam's
   second-moment normalization couples weight updates across the
   shared trunk), high λ could still perturb the score head
   indirectly. The λ=2.0 arm tests this — if score-head's per-window
   train IC plummets at λ=2 vs λ=0, the detach isn't doing what we
   want and we need a stronger separation (e.g., two optimizers).
3. **"Per-day" approximation**: the deployment-reward term uses
   `score · forward_log_return / horizon` rather than the full
   softmax-top-N portfolio with proper commission. A genuine
   deployment-aware loss would go through the
   `simulate_irregular_daily_pnl` pipeline at training time, which is
   too expensive for autograd. The simpler proxy keeps the gradient
   cheap while preserving the directional signal (score tilt times
   realized return per day). If the sweep PASSes, the natural follow-up
   is the heavier full-portfolio variant.

## Compute

Modal T4, ~50 min wall, ~$0.30 — same harness as the 2026-05-14
entropy sweep (shared cold-start + feature build, one walk-forward
per λ). Driver:

```bash
uvx modal run apps/factor/scripts/modal/horizon_mixture.py \
    --entropy-weights '0.0' \
    --deployment-reward-weights '0.0,0.25,0.5,1.0,2.0'
```

The existing `--entropy-weights` sweep continues to work — the two
sweeps are independent and can be combined later (cross-product) if
the bilevel arm PASSes.

## Where to land the result

- One leaderboard row for the sweep (mirrors the 2026-05-14 entropy
  row structure).
- Extension to
  [`findings/factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
  under a new "Bilevel objective" section, with the per-λ Δ-fix
  table + the score-head-stability check.
- Update CLAUDE.md's apps/factor section IF the sweep PASSes
  (operational rule: "train horizon head with separate
  deployment-aware loss; score head stays rank-IC").
- Close this TODO with `confirmed-OOS` / `partial-OOS` / `confirmed-null`
  verdict pointer.

## Concept link

If this works, it instantiates an empirical pattern that's been
floating across the codebase: **train each head with the loss that
matches the decision it makes**. Score head decides cross-sectional
ranking → rank-IC. Horizon head decides "when to rebalance" → realized
deployment return. The bilevel split is a clean implementation of
that pattern; if the result generalizes, it suggests other multi-head
architectures in the codebase (factor multi-task aux, replay's
multi-head reconstruction, regime's regime + scalogram dual scorers)
should be reviewed for analogous decoupling.
