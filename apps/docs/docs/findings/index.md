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
- [Sizing-input v0 — MSE-on-alpha calibrates score magnitude but adds zero information for the rank-invariant signal-quality emission](factor-sizing-input-v0.md)
- [Sizing-input v1 — factor signal-quality at 6-window resolution is too lagged to clear the VIX meta-gate](factor-sizing-input-v1.md)

## Replay app

- [Decoder options — what `--decoder` actually selects (linear / mlp / cnn / masked-ae)](replay-decoders.md)
- [2D DWT keep-LL compression preserves indicator-reconstruction signal at ~4× input reduction](replay-dwt-compression.md)
- [Length-axis (K) sufficiency — the K=96 default was over-provisioned for indicator reconstruction](replay-length-axis-compression.md)

## Relational app

- [Analog cross_ticker val edge collapses off mega-caps](relational-universe-shift.md)
- [DWT-L1 fingerprint compression fails OOS across all four distance scorers](relational-dwt-failure.md)
- [Polar Morlet input bundle fails the Phase-2 OOS gate for analog k-NN](relational-morlet-failure.md)
- [Relational arc synthesis — 12-phase research arc (lifted from `NO_OPTIONS.md`): what's shippable, what's falsified, the "fingerprint embedding for selection / not for hedging" rule](relational-arc-synthesis.md)

## Cross-app

- [Passive equal-weight benchmark — every "shippable" relational row was alpha-zero or alpha-negative](passive-ew-benchmark.md)
- [Rank-IC trains a signed signal that long-only top-N can only half-execute](factor-rankic-long-only-mismatch.md)

## Gate app

- [Drawdown gate v0 — real Pearson signal (+0.26), marginal Sharpe lift (+0.07 alpha)](gate-drawdown-v0.md)

## Pairs app

- [Pairs classical v0 — confirmed-null per pre-reg, regime-conditional partial signal](pairs-classical-v0.md)

## Vol app

- [Vol surface v0 — multivariate prediction works (val r +0.12), per-cell alpha +0.089 just below threshold (inconclusive, 5/5 positive)](vol-surface-v0.md)

## CFR app

- [CFR Phase 1 — tabular CFR clears trailing-best-greedy by +0.609 (6/6 wins) but ties naive uniform mix; menu is the binding constraint](cfr-phase1.md)

## Arc syntheses

- [Prediction-problem-pivot arc — three orthogonal v0 tests, three regime-conditional partial signals, one operational rule (regime filter > richer predictor)](prediction-problem-pivot-arc.md)
- [Relational arc synthesis — see Relational app section above](relational-arc-synthesis.md)

## Macro / regime

- [Macro regime diagnostic — 5 of 6 macro features predict pivot-arc window outcomes; VIX-above-median = 6× win-rate lift](macro-regime-diagnostic.md)
