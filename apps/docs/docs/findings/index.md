# Findings

Historical results and the decision rationale they produced. The
operational *rules* extracted from these findings live in `CLAUDE.md`
under "Important implementation notes" — this section preserves the
underlying numbers so future-us can re-evaluate. The
[Leaderboard](../leaderboard.md) carries the same results in
verdict-per-row form ([verdict-label vocabulary](../leaderboard.md#verdict-labels)).

Default eval window unless otherwise stated: 2013-01-29 → 2025-12-11,
10bps commission, 20-day rebal.

## Regime app

- [Regime baselines, Optuna instability, and the JAX-Adam scale-weight collapse](regime-baselines.md)
- [Log-returns CWT input degrades Sharpe](log-returns-vs-raw-close.md)

## Factor app

- [Deterministic-indicator val-IC baseline (the bar the supervised-`cnn` backbone must beat)](factor-indicator-baseline.md)
- [Supervised-`cnn` walk-forward on the polar Morlet bundle does not clear the indicator baseline](factor-ssl-walkforward.md)
- [Time-reversal symmetry diagnostic — what reversed-price training would tell us about the encoder](time-reversal-symmetry.md)
- [f32 forward log returns silently regressed val IC by 6× — Pearson cancellation in the +0.003 signal regime](factor-f32-precision-cancellation.md)
- [Multi-task aux head regularizes the MLP arm (+0.012 vs mlp) but does not clear the indicator baseline](factor-multitask-aux-head.md)
- [`aux_weight` sweep falsifies cross-sectional magnitude extraction at H=20 — fourth-branch outcome (train fits, val anti-correlates)](factor-multitask-aux-weight-sweep.md)
- [Loss-pivot eval — Sharpe and IR-vs-EW losses underperform rank-IC by 0.37 Sharpe (rank-IC's spread-thin behavior was inadvertent risk control)](factor-loss-pivot.md)

## Replay app

- [Decoder options — what `--decoder` actually selects (linear / mlp / cnn / masked-ae)](replay-decoders.md)
- [2D DWT keep-LL compression preserves indicator-reconstruction signal at ~4× input reduction](replay-dwt-compression.md)
- [Length-axis (K) sufficiency — the K=96 default was over-provisioned for indicator reconstruction](replay-length-axis-compression.md)

## Relational app

- [Analog cross_ticker val edge collapses off mega-caps](relational-universe-shift.md)
- [DWT-L1 fingerprint compression fails OOS across all four distance scorers](relational-dwt-failure.md)
- [Polar Morlet input bundle fails the Phase-2 OOS gate for analog k-NN](relational-morlet-failure.md)

## Cross-app

- [Passive equal-weight benchmark — every "shippable" relational row was alpha-zero or alpha-negative](passive-ew-benchmark.md)
- [Rank-IC trains a signed signal that long-only top-N can only half-execute](factor-rankic-long-only-mismatch.md)
