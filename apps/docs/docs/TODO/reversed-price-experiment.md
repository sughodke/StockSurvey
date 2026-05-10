# Reversed-price training — falsify the time-symmetry hypothesis

The diagnostic analysis is in
[`findings/time-reversal-symmetry.md`](../findings/time-reversal-symmetry.md).
This page is the experiment that turns it into a leaderboard row.

## Hypothesis (gated by [factor-ssl-walkforward](../findings/factor-ssl-walkforward.md))

The supervised-`cnn` encoder + rank-IC factor head landed at mean
val IC = +0.0031 on stooq_us_long. Two readings of that floor are
consistent with the data:

- **A — encoder is symmetric-feature-bound.** It's mostly reading
  time-symmetric chart shapes; the asymmetric pieces of the input
  (causal CWT wedge, EMA-driven indicator targets, crash-vs-rally
  shape) aren't carrying the +0.0031 alpha.
- **B — encoder is asymmetric-feature-bound.** It is using
  asymmetric information, but only a thin slice of that survives
  into 20-day cross-sectional return prediction.

These imply different next moves. Under A, the lever is supervision
or feature selection (the encoder *can* see more, the objective
pulled it toward the wrong slice — see masked-AE open question).
Under B, the encoder is doing what it's supposed to do; the
+0.0031 ceiling is structural to the universe / horizon, and the
research focus should pivot to a
[different prediction problem](different-prediction-problem.md).

## Test design

Train and evaluate two new pipelines side-by-side with the existing
forward-time pipeline:

1. **Forward (control)** — existing supervised-`cnn` backbone on
   forward prices, factor walk-forward, mean val IC = +0.0031.
2. **Reversed-input** — reverse every price series in time before
   any pretrain or factor step. Replay pretrain runs on reversed
   prices (causal CWT eats the reversed series; reconstruction
   targets are RSI / MACD / vol / CCI computed on reversed prices).
   Factor walk-forward runs against `forward_log_returns` of the
   *reversed* series (which equals `-backward_log_returns` of the
   original, in the original frame).
3. **Reversed-input + flipped target sign** — same as (2) but the
   factor head's target is negated, so the rank-IC gradient pushes
   toward predicting *original-frame backward returns*. This is
   the cleanest mathematical inverse of the forward pipeline.

## Decision rule

- **Clean inversion (verdict = `confirmed-OOS` for symmetric-feature
  hypothesis A).** Run (3) lands within ±0.002 val IC of the
  forward pipeline's +0.0031 (i.e. between +0.0011 and +0.0051,
  same sign). Equivalent: run (2) lands at −0.0031 ± 0.002.
- **No inversion (verdict = `reversed-OOS` — implies hypothesis B
  is closer to the truth).** Run (3) lands materially below
  +0.0011 (e.g. near zero or negative). The encoder is
  asymmetry-load-bearing; the +0.0031 ceiling is structural.
- **Partial inversion (`partial-OOS`).** Run (3) lands at some
  intermediate value, say +0.001 to +0.002 (~30–60% of forward
  IC). Some asymmetry, some symmetry. Stratify per-window to find
  which windows preserve the inversion and which don't.

## Cost estimate

- Reversed-price replay pretrain: ~same wall-time as the forward
  pretrain (one Modal-T4 run, see
  [`apps/replay/scripts/modal/train_cnn_multihead.py`](https://github.com/sughodke/StockSurvey/tree/master/apps/replay/scripts/modal)).
  The CWT and indicator computations are unchanged in code — they
  just see reversed inputs.
- Factor walk-forward against reversed targets: another ~hour at
  192 GB Modal (after `3451900`'s memory fix).
- Total: ~one day end-to-end including artifact handoff.

## Implementation knobs

- **Reversal point.** Reverse the price DataFrame index *before*
  it enters `ss_features.load_ticker` — so the entire downstream
  pipeline (CWT, indicators, returns) consumes reversed bars
  consistently. A single one-line reversal at ingest is cleaner
  than reversing per-feature.
- **Universe consistency.** Same 297-ticker stooq_us_long pool,
  same `min_history_bars=6500` filter — eligibility is computed on
  history length, which is invariant to reversal.
- **Date range.** Same `2000-01-03 → 2026-04-01`, but be careful
  about which end of the reversed series is "train" and which is
  "val" in the walk-forward. Either reverse the train/val window
  ordering too (so train still trails val in original-frame time)
  or hold it fixed (so train *leads* val in original-frame time).
  The cleaner choice is the latter — it tests "if the encoder were
  symmetric, does walk-forward in either direction produce the
  same val IC magnitude". Document the choice on the leaderboard
  row.
- **Target sign for run (3).** Negate `forward_log_returns` once at
  the loss boundary; do not touch the rest of the pipeline.

## Out of scope for this experiment

- The masked-AE encoder. Run reversed-price training on the
  supervised-`cnn` encoder first; if the result is interesting,
  re-run on `--decoder masked-ae` (per
  [`replay-decoders.md`](../findings/replay-decoders.md))
  to fill in the 2×2.
- Other strategies (regime, relational). The diagnostic is
  factor-specific. Apply to other heads only if the forward-vs-
  reversed val IC gap is large enough on factor to be worth the
  generalization test.

## Where this lands

- **Leaderboard row** added on completion, verdict label per the
  decision rule above. Reference back to
  [`findings/time-reversal-symmetry.md`](../findings/time-reversal-symmetry.md)
  in the notes column.
- **Follow-up findings page** if the result is clean enough to
  warrant prose beyond a leaderboard row — most likely yes, since
  whichever way it lands changes the
  [supervision-is-binding](../notes.md#what-we-already-know-about-supervision-being-the-binding-constraint)
  framing.
- **Update to the operational rules in `CLAUDE.md`** if the
  `confirmed-OOS` arm fires (then "the supervised-`cnn` encoder is
  reading time-symmetric chart shapes; pretext is the lever, not
  encoder capacity" goes onto the rules list).
