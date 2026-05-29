# Learned 2-leg blend beats the deterministic (DCA + 2x vol_v3) ensemble

**Operational rule.** *A 2-parameter mean-variance learner fit on
**any** strictly-prior slice of (DCA daily, vol_v3 daily-aligned)
beats the canonical deterministic recipe on OOS Sharpe **and** max-DD
across every test window evaluated. The deterministic (1, 2) recipe
under-allocates vol_v3 relative to mean-variance optimality at the
vol_v3 substrate's measured σ/α structure. Use the learned weights as
the deployment recipe, not the deterministic 2x scale.*

## Status

Direct executable answer to the user's challenge after
[`meta-allocator-internal-features`](meta-allocator-internal-features.md)
landed `confirmed-null`: that arc framed the problem as
"forecast cross-arc returns → softmax weights on simplex," which
(a) optimized the wrong loss and (b) physically could not represent
the deterministic ensemble's gross > 1.0 vol-sleeve scaling. The
deterministic ensemble's alpha is mean-variance, not forecastable —
so a mean-variance learner with the right action space finds it
trivially.

Verdict: **confirmed-OOS** vs the deterministic recipe, across four
independent train/test splits, every CI excludes zero on the positive
side, every max-DD tighter than the deterministic.

## Why my prior arc missed this

Three mismatches in the internal-features arc that this rewrite fixes:

1. **Objective.** The prior arc trained to forecast next-quarter
   per-arc returns. The deterministic ensemble isn't a forecaster —
   it captures a fixed σ/α mismatch. Forecast accuracy and Sharpe of
   weighted streams are different losses; only the second matters
   for portfolio selection.
2. **Action space.** Softmax-of-forecasts lives on the probability
   simplex (weights ≥ 0, sum to 1). The deterministic recipe has
   `w_vol = 2.0` — gross > 1.0 — and is physically un-representable
   in the simplex. The learner could not match the recipe even with
   perfect forecasts.
3. **Constancy is the answer.** The optimal policy at this σ/α
   structure is approximately constant in time. The prior learner
   spent capacity hunting time-varying signal that isn't there and
   ended up noisier than a constant.

This rewrite uses a 2-parameter unconstrained learner over the
joint (DCA, vol_v3) daily stream. The objective is portfolio Sharpe
directly. No simplex, no forecasting, no overfit-capacity:
**closed-form mean-variance, plus a numerical-confirmation gradient
ascent on Sharpe.** The two implementations converge to different
weights but the same OOS portfolio behavior — both dominate vol_v3
relative to DCA.

## Eval setup

| field | value |
|---|---|
| streams | DCA daily (PassiveEW rebal_days=80, 10 bps friction on 13-ETF Phase 4d basket from `Output/cfr_phase4d_multiasset_close.pkl`) + vol_v3 daily-aligned (per-rebal alpha spread evenly across days until next rebal, source `Output/vol-v3-dolthub-oos-c200-returns.npz`) |
| vol_v3 active range | 2023-08-02 → 2025-12-11 |
| splits | four (train → test): 2023-08–2023-12 → 2024+; 2023-08–2024-03 → 2024-04+; 2023-08–2024-06 → 2024-07+; 2023-08–2024-12 → 2025+ |
| learners | (a) closed-form diagonal MV: `w_vol/w_dca = (μ_vol/σ_vol²)/(μ_dca/σ_dca²)`; (b) gradient ascent on Sharpe (numerical confirmation) |
| baseline | deterministic `1.0 * r_dca + 2.0 * r_vol` (the post-2020 ranking arc's load-bearing recipe) |
| metric | OOS annualized Sharpe; Ledoit-Wolf studentized ΔSR CI vs deterministic; max-DD; CI from `ss_portfolio.sharpe_difference_ci` stationary bootstrap |

## Results — every split, every metric

| Test window | n | DCA only Sh | vol_v3 only Sh | **Det (1,2) Sh** | **Learned MV Sh** | **Learned grad Sh** | ΔSR_ann MV vs Det 95% CI | Det max-DD | **Learned grad max-DD** |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 2024-01+ | 489 | +1.32 | +12.09 | +7.53 | **+12.11** | +12.11 | **[+2.25, +6.97]** | −9.55% | **−0.08%** |
| 2024-04+ | 428 | +1.12 | +12.40 | +7.51 | **+12.41** | +12.41 | **[+2.64, +7.52]** | −9.55% | **−0.30%** |
| 2024-07+ | 365 | +1.26 | +11.58 | +7.23 | **+11.61** | +11.61 | **[+2.12, +7.10]** | −9.55% | **−0.16%** |
| 2025-01+ | 237 | +1.40 | +7.72 | +4.75 | **+7.75** | +7.75 | **[+0.92, +5.88]** | −9.55% | **−0.34%** |

**Every learned-vs-deterministic CI excludes zero on the positive
side.** The headline ΔSR vs deterministic ranges +3.01 to +4.91
annualized across splits; the *narrowest* (2025-only test, n=237)
still posts ΔSR +3.01 with CI [+0.92, +5.88].

Max-DD is tighter for the learned model in every split too —
*not because* of an explicit DD constraint, but because optimal-
Sharpe weighting at the measured σ/α structure shifts the portfolio
away from DCA's equity-beta drawdowns toward vol_v3's near-monotone
PnL stream.

## Learned weights — what the optimizer found

The closed-form MV and the gradient-ascent learner converge to
different weights but the same effective portfolio:

| Split | MV (w_dca, w_vol) | Grad (w_dca, w_vol) |
|---|---|---|
| 2023-08–2023-12 train | (1.000, 177.87) | (0.017, 2.241) |
| 2023-08–2024-03 train | (1.000, 40.32) | (0.062, 2.239) |
| 2023-08–2024-06 train | (1.000, 53.47) | (0.033, 2.240) |
| 2023-08–2024-12 train | (1.000, 45.60) | (0.069, 2.239) |

Both effectively allocate near-100% of risk to vol_v3 — the closed-
form MV does it by raising w_vol; the gradient ascent does it by
lowering w_dca. The *ratio* w_vol / w_dca is what matters for the
portfolio Sharpe, and both learners agree: vol_v3's Sharpe contribution
swamps DCA's at this substrate. The deterministic (1, 2) was *too
conservative* relative to the data's MV-optimal blend.

The gradient-ascent learner's `w_vol ≈ 2.24` is the more
interpretable answer: anchoring on DCA at gross ≈ 0 (or very low) and
overlaying ~2.24x vol_v3. Practically equivalent to vol_v3-only
deployment.

## Why this isn't a leakage artifact

Three reasons to take this at face value:

1. **The learner has 2 free parameters.** Not 14 features, not a
   neural net — two scalars over a closed-form objective. Overfit on
   2 parameters across 100-300 training observations is not the
   failure mode here.
2. **The training slice is strictly prior to the test slice.** The
   smallest train window is n=105 (2023-08 to 2023-12), which is
   ~5 months. The largest test window is n=489 (2024-01+). Earlier
   train → larger OOS window, all four splits.
3. **The vol_v3 daily-aligned stream is the same one the
   deterministic ensemble eats.** Both candidates see the identical
   data; the learner just sizes it differently.

## Caveats — what this does NOT prove

This finding is a **methodological correction**, not a green light
to ship vol_v3 standalone. The four caveats that the
[`vol-v3-sleeve-sizing`](vol-v3-sleeve-sizing.md) +
[`post-2020-arc-ranking`](post-2020-arc-ranking.md) work already
established still apply identically:

- **vol_v3's measured Sharpe is academic-clean.** Zero options-
  broker friction, zero bid-ask, zero exchange fees beyond the
  declared 10 bps DoltHub friction model. Rail #6 (realized friction
  monitor at `~/.vol-friction-history.csv`) is the deployment honesty
  gate; it must hold against measured spreads before any of these
  Sharpes are ship-eligible.
- **Single-substrate concentration.** vol_v3 is one short-vol recipe.
  Allocating 100% of risk there means 100% capacity exposure to
  vol_v3's specific failure mode. DCA's presence in the deterministic
  recipe was partly a risk-diversification choice, not a Sharpe-
  optimization choice. The learner doesn't get the risk-diversification
  credit because Sharpe doesn't penalize concentration directly.
- **vol_v3's data window is short.** 33 rebal points across 2023-08
  → 2026-03. The MV optimum at this sample length is genuinely the
  MV optimum, but the σ/α structure could shift; the learner has no
  built-in robustness to non-stationarity.
- **vol_v3 is currently un-shippable.** Per the friction rail and
  the broker-pickup TODO. The learned recipe sits in the same
  pre-ship state as the deterministic recipe — the right read is
  "when vol_v3 is live, use the learned weights, not the deterministic
  (1, 2)."

## Operational implication

The deployment recipe **updates** from "DCA + 2x vol_v3" to
"DCA at gross approximately 0.05 + vol_v3 at gross approximately
2.24" once live. The Sharpe lift (~+3 to +5 ann) is real OOS data,
not pre-registration aspiration. The infrastructure for shipping
this is unchanged from the prior recipe — same broker (Tradier
recommended), same friction monitor, same dry-run gate. The only
difference is the scalar on the legs.

The broader methodological lesson is the one the user surfaced:
**when the deterministic system extracts real alpha, the failure of
a learner to match it is a framing failure of the learner, not a
no-signal verdict on the data.** The internal-features arc's
`confirmed-null` was a *learner-objective* null, not a data-
predictability null. Mean-variance over the joint return stream
trivially solves the problem the forecaster framing made impossible.

## Master walk-forward log

| date | row pointer | verdict |
|---|---|---|
| 2026-05-28 | `apps/docs/docs/leaderboard.md` row (this finding) | [`confirmed-OOS`](../leaderboard.md#verdict-labels) — learned beats deterministic on Sharpe and max-DD across all 4 splits |

## Related findings

- [`meta-allocator-internal-features`](meta-allocator-internal-features.md) — the forecasting-framed arc that this finding methodologically corrects.
- [`post-2020-arc-ranking`](post-2020-arc-ranking.md) — established the deterministic (1, 2) recipe as the canonical baseline beaten here.
- [`vol-v3-sleeve-sizing`](vol-v3-sleeve-sizing.md) — established `vega_scale=2.0` at `c_options_bps≤200`; this finding shows the data supports a larger scale (`~2.24`) at the same friction assumption.
- [`vol-v3-dolthub-oos`](vol-v3-dolthub-oos.md) — the vol_v3 substrate whose alpha drives both the deterministic and learned recipes.

## Driver + outputs

- Driver: `apps/docs/scripts/learned_ensemble_vs_deterministic.py`.
- Per-split results: `Output/learned_ensemble_vs_deterministic.json`.
