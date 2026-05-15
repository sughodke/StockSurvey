# `apps/factor` — horizon-aligned IndicatorGridConfig variant

## Closed 2026-05-15

[`confirmed-null`](../leaderboard.md#verdict-labels) on the
input-side rescue hypothesis. Sweep ran on Modal T4 (~12 min wall):

| arm | mean endog | best-fix(h) | Δ-fix | h=60 argmax | verdict |
|---|---:|---:|---:|---:|---|
| (default, λ=0) baseline (cached) | +0.448 | +0.401 (h60) | **+0.048** | 80% | partial-OOS |
| (default, λ=0.25) (cached) | +0.453 | +0.405 (h60) | +0.047 | 78% | partial-OOS |
| (horizon-aligned, λ=0) | +0.401 | **+0.437 (h10)** | **−0.036** | **94%** | confirmed-null |
| (horizon-aligned, λ=0.25) | +0.396 | +0.431 (h10) | −0.035 | 94% | confirmed-null |

**The key diagnostic**: the score head IS using the new
horizon-aligned channels — under the new grid, **fixed-h10 Sharpe
lifts to +0.437** (vs default's fixed-h60 at +0.401), and the
per-horizon Sharpe profile genuinely flattens. But π collapses
HARDER on h=60 (94% argmax) than under default (80%) because π's
training signal is rank-IC and per-bar IC SNR is highest at long
horizons regardless of feature stack. The architecture has a
fundamental misalignment: π is trained to maximize per-bar IC
(rewards h=60); deployment metric is Sharpe (favors h=10 under
horizon-aligned grid).

**w3 canary fires for the third time**: w3 (2015-06-30) drops
from +0.84 (default λ=0) → +0.55 (horizon-aligned λ=0), Δ −0.29.
Same pattern as:

- Entropy reg α=0.05: w3 from +0.84 → +0.66 (Δ −0.18)
- Bilevel λ=2: w3 from +0.84 → +0.57 (Δ −0.27)
- Horizon-aligned grid: w3 from +0.84 → +0.55 (Δ −0.29)

Three independent rescue attempts. Three drops on the same window.
The architecture's state-conditional skill at w3 is fragile to ANY
perturbation.

**Side-finding worth documenting**: `(horizon-aligned, fixed-h10)`
deployment recipe = +0.437 Sharpe is a comparable, operationally
simpler alternative to `(default, mixture)` = +0.448 (gap 0.011,
within noise). Not pre-registered as a deployment but a real
fallback if the mixture architecture is retired.

**Doesn't close**: output-side restructure (per-horizon score heads,
mixture *over* specialized heads), target-side intervention
(REINFORCE-style π training against realized Sharpe instead of
rank-IC). Both are genuinely new architectures, not feature swaps.

See extended findings page
[`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
("Horizon-aligned feature grid" section) for the per-horizon
fixed-Sharpe profile table, per-window w3-canary repeat, and the
updated operational rule.

Leaderboard row: 2026-05-15 horizon-aligned-grid sweep
(`confirmed-null`).

---

## Original pre-registration (preserved for the record)

**Status at time of writing**: pre-registered 2026-05-15. Sweep not
yet run.

### Hypothesis

The 74-channel default `IndicatorGridConfig` produces score head IC
profiles that peak at h=60 even though channels span a range of
timescales. The bilevel sweep (2026-05-15) confirmed this is *not* a
training-objective issue — adding deployment-return supervision
monotonically hurts. The remaining hypothesis is that the **feature
stack's coverage of short horizons is sparse**, biasing the score
head's linear combination toward long-horizon signal where SNR is
highest.

The horizon set the architecture mixes over is `{5, 10, 20, 40, 60}`,
but the default config's channel grids don't cleanly align:

- `rsi_n_grid = (5, 7, 10, 14, 21, 30)` — has 5, 10 but jumps from 30
  to nothing. No channels at n=40 or 60.
- `cci_n_grid = (10, 14, 20, 40)` — covers 10, 20, 40 but no n=5 or 60.
- `vol_n_grid = (5, 10, 20, 60, 120, 252)` — has 5, 10, 20, 60 but
  *no n=40*.
- `macd_fast_grid = (5, 8, 12, 21, 34, 55)` — has 5 but no 10, 20,
  40, 60 directly (closest are 8, 21, 34, 55).
- `coherence_window_grid = (10, 20, 60, 120)` — has 10, 20, 60 but
  no 5, 40.

The hypothesis: adding **horizon-aligned cells** at the action-space
periods will give the score head explicit information at each
horizon, shifting its per-horizon IC profile such that short
horizons (h=5, h=10) become competitive with h=60 — opening up the
action space the 2026-05-14 oracle diagnostic wants (h=5: 27%,
h=10: 22%, h=20: 12%, h=40: 11%, h=60: 28%).

## What changes

Expand each grid to explicitly include every horizon in `{5, 10, 20,
40, 60}`:

```python
horizon_aligned = IndicatorGridConfig(
    rsi_n_grid=(5, 7, 10, 14, 20, 21, 30, 40, 60),       # +20, +40, +60
    rsi_w_grid=(1, 5, 10, 21, 63),                       # unchanged
    cci_n_grid=(10, 14, 20, 40),                         # unchanged (n=60×w=21 warmup too long)
    cci_w_grid=(1, 5, 10, 21),                           # unchanged
    vol_n_grid=(5, 10, 20, 40, 60, 120, 252),            # +40
    macd_fast_grid=(5, 8, 10, 12, 20, 21, 34, 40, 55, 60), # +10, +20, +40, +60
    coherence_window_grid=(5, 10, 20, 40, 60, 120),      # +5, +40, +60
)
```

Channel count: `9×5 + 4×4 + 7 + 10×3 + 6 = 45 + 16 + 7 + 30 + 6 =
104 channels` (was 74). This is a **strict superset** of the default
config — every existing cell is retained; 30 new horizon-aligned
cells are added. If the new cells are noise, the head can learn to
weight them to zero (falls back to current behavior). If they carry
short-horizon signal, the head can use them.

CCI's `n=60` is intentionally NOT added because the worst cell
`(n=60, w=21)` needs `(60-1)*21 + 1 = 1240` bars warmup, which would
shrink the walk-forward training window by ~30%. The horizon
coverage at h=60 instead comes from the RSI n=60 / MACD fast=60 /
vol n=60 / coherence window=60 cells.

## Test design

### Arms

Cross-product of `config_variant ∈ {default, horizon-aligned}` × `λ
∈ {0.0, 0.25}` = 4 arms.

Two of these are **cached from the 2026-05-15 bilevel sweep** and
don't need re-running:

- `(default, λ=0.0)`: mean endog +0.448, Δ-fix +0.048, partial-OOS.
- `(default, λ=0.25)`: mean endog +0.453, Δ-fix +0.047, partial-OOS.

Two new arms to run on Modal:

- `(horizon-aligned, λ=0.0)`: tests whether the expanded grid lifts
  Δ-fix above +0.048 by itself.
- `(horizon-aligned, λ=0.25)`: tests whether expanded grid + light
  deployment-reward supervision compound. (λ=0.25 chosen because it
  was the only non-baseline bilevel arm that tied baseline within
  noise rather than degrading — the natural pairing if the new
  config opens up new state-conditional structure.)

### Universe / windowing

Identical to the 2026-05-14 entropy sweep + 2026-05-15 bilevel sweep:
factor-narrow (297 stooq_us_long, min_history=6500), 6-window
walk-forward at h_min=5 fine grid, commission 10 bps, n_steps=200,
seed=0.

### Pre-registered cuts (apply per arm; best arm sets the sweep
verdict)

- **STRONG-PASS** (`confirmed-OOS`): mean Δ-fix ≥ +0.10 AND 6/6
  positive windows.
- **PASS** (`confirmed-OOS`): mean Δ-fix ≥ +0.10 AND ≥ 5/6 positive.
- **MARGINAL** (`partial-OOS`): mean Δ-fix ≥ +0.07 AND ≥ 4/6
  positive.
- **FAIL** (`confirmed-null`): otherwise — feature-space is not the
  binding constraint at this architecture / dataset scale.

### Per-horizon IC profile diagnostic (auxiliary)

In addition to the Δ-fix metric, report the per-horizon mean IC
across the 6 windows for each arm:

| arm | mean IC h=5 | h=10 | h=20 | h=40 | h=60 |

If the horizon-aligned arms show flatter IC profiles (less h=60
dominance, more h=5/h=10 mass) than the default arms, that's
evidence the feature swap is doing what's predicted — even if the
deployment Sharpe doesn't move. A flatter IC profile + null Sharpe
would point to "the score head IS using short-horizon information
but π isn't able to act on it", which would re-prioritize output-
or target-side experiments (dimensions #2 and #3 from the cross-
app feature-design matrix).

### Honest acknowledgements before running

1. **30 added channels is 40% more parameters in the score head**.
   With only 252 fine-grid bars in the train window, the head will
   overfit harder. The 5-fold walk-forward design partially controls
   for this — but if the head overfits to the new channels and
   they don't generalize, Δ-fix could DROP below +0.048.
2. **Same warmup floor** as default. The CCI n=40 w=21 cell still
   dominates at 820 bars; min_history=6500 keeps every fold valid.
3. **Two-arm sweep is faster than a full re-sweep**. The default
   arms at λ ∈ {0, 0.25} are cached from 2026-05-15. We pay one
   feature build (the horizon-aligned variant takes ~30% longer
   per ticker than default) + 2 walk-forwards. Expected wall:
   ~20-25 min on Modal T4, vs the bilevel sweep's ~50 min.
4. **Feature width is now 104**. Score head with linear or
   1-layer MLP × 104 inputs has ~100-200 params. Still tiny, but
   the increase in capacity is exactly what we want IF the new
   channels carry signal, and exactly what we don't want if they
   don't.

## Compute

```bash
uvx modal run apps/factor/scripts/modal/horizon_mixture.py \\
    --config-variant horizon-aligned \\
    --deployment-reward-weights '0.0,0.25'
```

Modal T4, ~25 min wall, ~$0.15.

## Where to land the result

- One leaderboard row for the sweep (two new arms, referenced
  against the cached default-config arms).
- Extension to
  [`findings/factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
  under a new "Horizon-aligned feature grid" section, with the
  per-arm + per-horizon-IC tables.
- Update CLAUDE.md's apps/factor section IF the sweep PASSes
  (operational rule: "use horizon-aligned IndicatorGridConfig for
  endogenous-horizon training").
- Close this TODO with verdict pointer.

## Concept link

If this works, it instantiates the same pattern as the
[2026-05-14 oracle diagnostic's "score-head specialization"
finding](../findings/factor-endogenous-horizon-mixture.md#arc-closure-revised):
**match the feature stack to the action space**. Score head trained
at h=60 specialized to h=60 because that's where the IC was; if the
feature stack carries information at h=5, the IC profile shifts and
the policy has new room to maneuver.

If it doesn't work — if the horizon-aligned channels add noise
without lifting deployment Sharpe — that's strong evidence the
binding constraint is **information content in the price series at
short horizons**, not feature-stack coverage. At that point the
remaining levers are SSL backbone (different feature representation
entirely), output-side restructure (per-horizon score heads), or
target-side intervention (different prediction target).
