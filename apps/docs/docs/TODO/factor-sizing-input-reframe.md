# Factor sizing-input reframe — re-purpose apps/factor as a meta-gate input

**v0 resolved 2026-05-12: `confirmed-null` on the loss-axis hypothesis.**
`mse_alpha` training calibrates score magnitudes (val MSE-alpha 52×
smaller than rank_ic) but adds zero information for the rank-based
signal-quality emission — both arms hit Spearman ρ = +0.486 and lag-1
autocorr +0.82-+0.91. **v1 proceeds with the rank_ic head as the
upstream signal-quality source** (no benefit to switching). Closing
finding: [`factor-sizing-input-v0`](../findings/factor-sizing-input-v0.md).
Leaderboard row: 2026-05-12 sizing-input v0 head-to-head,
[`confirmed-null`](../leaderboard.md#verdict-labels). Implementation
landed in `factor.{train,train_walkforward}` (`loss_kind='mse_alpha'`,
`alpha_target_rb`, `signal_quality_per_val_bar`, `val_start_date`);
drivers `apps/factor/scripts/sizing_input_eval.py` (local) +
`apps/factor/scripts/modal/sizing_input_eval.py` (T4).

The original test design and pre-registration are kept below for
audit. **v1 (deployment layer) is the next sub-section that fires** —
the calibration-layer null doesn't block it, the signal-quality
artifact has the downstream-useful properties either way.

---

Motivating findings:
[`factor-rankic-long-only-mismatch`](../findings/factor-rankic-long-only-mismatch.md),
[`factor-loss-pivot`](../findings/factor-loss-pivot.md),
[`passive-ew-benchmark`](../findings/passive-ew-benchmark.md),
[`prediction-problem-pivot-arc`](../findings/prediction-problem-pivot-arc.md).

The four findings converge on the same diagnosis for cross-sectional
return prediction on broad equity at this signal magnitude
(`mean val IC ≈ +0.005`):

- **No portfolio constructor recovers a positive-expectancy tilt** off
  the rank-IC head — long-only top-N loses to EW; long-short delivers
  val Sharpe `−0.067` after friction.
- **No tested loss recovers it either** — block_sharpe and ir_vs_ew
  both made val Sharpe materially worse than rank_ic at this IC scale
  via softmax-temperature collapse.
- **The remaining structural argument** is that `pearson_rank_ic` is
  scale-invariant on scores: it accepts any monotone re-scaling of the
  head's output as equally good. The scores are not calibrated to the
  *units* of forward alpha, so they cannot be consumed as a per-ticker
  alpha forecast by a downstream sizing layer that decides "is this
  signal large enough to be worth deploying?"

This TODO closes the loop by reframing apps/factor away from being a
*tilting model* (whose output is portfolio weights) and toward being a
*sizing-input model* (whose output is a per-ticker calibrated alpha
forecast consumed by a meta-gate). Aligned with the
[prediction-problem-pivot-arc](../findings/prediction-problem-pivot-arc.md)
operational rule: **schedule the trigger, not the trade.**

## Hypothesis

At factor-narrow's IC scale (mean val IC `≈ +0.005-0.012`), training the
indicator-linear head with **`masked_mse` on per-bar cross-sectionally
demeaned forward log-returns (alpha targets)** produces scores whose
*magnitude* tracks expected per-ticker alpha. The per-bar
top-decile-minus-bottom-decile predicted-alpha dispersion is then a
candidate **sizing input** for the macro meta-gate
([`macro-regime-diagnostic`](../findings/macro-regime-diagnostic.md) v1b).

This is falsifiable on two layers:

- **Calibration layer** (within factor walk-forward): does training on
  `mse_alpha` instead of `pearson_ic` change the head's behaviour in
  a way that matters for sizing-input use? Specifically, does the
  resulting per-val-bar dispersion signal have temporal stability and
  realized-alpha correlation that rank-IC training does not produce?
- **Deployment layer** (cross-app meta-gate): does adding factor
  signal-quality as a *second* gate feature alongside VIX-state
  materially lift pooled per-app-z-scored alpha on the n=17
  pivot-arc windows beyond what the binary VIX-median gate alone
  delivered (+0.215 z)?

## Test design — v0 (calibration layer)

Universe: factor-narrow (297 tickers, `min_history_bars=6500`,
matching [`factor-indicator-baseline`](../findings/factor-indicator-baseline.md)
and [`factor-loss-pivot`](../findings/factor-loss-pivot.md)).

Walk-forward: existing 6-window setup, train=63 / val=39 / step=39
blocks at `rebal_days=20`. Linear head, `IndicatorGridConfig()`
defaults (74 channels), `n_steps=200 lr=1e-2 wd=1e-3 commission_bps=10`.

Two arms, same head architecture, same windows:

- **`pearson_ic`** (baseline, reproduction of factor-loss-pivot's
  rank_ic arm) — current `pearson_rank_ic` loss. Score magnitude
  uncalibrated.
- **`mse_alpha`** (new) — `masked_mse` against
  `fwd_log_return − cross_sectional_mean(fwd_log_return)` per bar
  (alpha target). Score magnitude calibrated to alpha units.

Per-window metrics tracked on both arms:

| Metric | Where it comes from |
|---|---|
| `train_ic`, `val_ic` | `pearson_rank_ic` against `fwd_ret_rb` — comparison metric, identical computation for both arms |
| `val_sharpe`, `val_sharpe_long_short`, `val_ir_vs_ew` | existing constructors via `block_sharpe` family — diagnostic, not pre-registered |
| `train_mse_alpha`, `val_mse_alpha` | `masked_mse(scores, alpha_target)` — new arm's training-aligned metric |
| `signal_quality_per_val_bar: np.ndarray` | per-val-bar top-decile-mean − bottom-decile-mean predicted alpha — the **artifact this TODO ships** |
| `signal_quality_mean: float`, `signal_quality_std: float` | summary stats over val bars |

Per-arm artifact: `Output/sizing-input-{arm}-windows.npz` with the
full per-val-bar dispersion time series (shape `(n_windows,
val_window_blocks)`), plus per-window val_start date. Driver:
`apps/factor/scripts/sizing_input_eval.py`.

## Pre-registered pass / fail — v0

The verdict is about whether the `mse_alpha` arm produces a sizing
signal **distinguishable from noise at the IC scale we have**. It is
*not* about whether the training change lifts val Sharpe directly —
the
[loss-pivot](../findings/factor-loss-pivot.md) finding already showed
loss-level changes don't move that needle, and we'd be re-running the
same null.

| Verdict | Criterion | Next move |
|---|---|---|
| PASS  | Both true: (a) `signal_quality_per_val_bar` autocorrelation at lag 1 ≥ +0.20 on the `mse_alpha` arm pooled across windows (vs the `pearson_ic` arm; comparison is the headline); (b) per-window mean signal-quality is rank-correlated with per-window val Sharpe with Spearman ρ ≥ +0.40. | Promote signal-quality to a feature input on the macro meta-gate (v1). |
| FAIL  | The `mse_alpha` arm's signal-quality looks indistinguishable from `pearson_ic`'s on both criteria. | The training-objective change didn't extract a calibrated sizing signal; the head's outputs are still essentially rank information. Pivot: try a non-rank-IC training target like `forward_target_kind='vol_innovation'` (already wired) or shut down the sizing-input direction entirely. |
| INCONCLUSIVE | One criterion clears, one fails. | Stratify per-window. If one or two windows carry the signal-quality variance, that's already the regime-gate evidence — proceed to v1 but with realistic priors. |

## Test design — v1 (deployment layer)

Conditional on v0 PASS or INCONCLUSIVE.

Extend `apps/gate/scripts/macro_meta_gate_eval.py` to consume factor
signal-quality at each pivot-arc window's `val_start` as a second
gate input alongside VIX-state. Re-train factor heads aligned to
gate/pairs/vol val_start dates (different windowing than v0).

Three meta-gate arms vs no-gate baseline:

- VIX-only (v1b baseline, +0.215 z-score lift already on the books).
- Factor-signal-only (continuous percentile rank of signal-quality
  at val_start vs trailing-3y rolling distribution; deploy when
  percentile ≥ 0.5).
- VIX-AND-factor (both gates must agree).

Pre-registered pass: any factor arm or composite delivers pooled
per-app-z-scored lift ≥ +0.30 (≈ +0.10 absolute over the VIX-only
baseline). Fail floor: within ±0.05 of VIX-only.

## Implementation plan

1. **objectives.py** — confirm `masked_mse` is reusable for the alpha
   target (it is — same `(n_bars, n_tickers)` masked layout). No new
   loss function needed.
2. **train.py** — extend `precompute_inputs` with optional
   `alpha_target_rb` (per-bar demeaned `fwd_ret_rb` using `mask_rb`).
3. **train_walkforward.py** — add `'mse_alpha'` to the `valid_losses`
   set; wire the train-loop branch and signal-quality emission.
4. **objectives.py** — add a small helper `top_minus_bottom_decile`
   for the per-bar dispersion stat (numpy, eval-only).
5. **train_walkforward.py** — extend `WalkForwardWindow` with
   `signal_quality_per_val_bar`, `signal_quality_mean`,
   `train_mse_alpha`, `val_mse_alpha`, `val_start_date`.
6. **scripts/sizing_input_eval.py** — new driver mirroring
   `loss_pivot_eval.py`'s shape (two-arm comparison, JSON + npz
   artifacts, verdict logic).
7. **Smoke test locally** (`--max-tickers 30 --n-steps 50`), then
   Modal run for the verdict.

The macro meta-gate v1 wiring is a follow-up TODO that this one
**spawns** rather than includes — keep v0 narrow so the verdict is
clean.
