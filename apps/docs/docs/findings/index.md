# Findings

Historical results and the decision rationale they produced. The
operational *rules* extracted from these findings live in `CLAUDE.md`
under "Important implementation notes" — this section preserves the
underlying numbers so future-us can re-evaluate.

Default eval window unless otherwise stated: 2013-01-29 → 2025-12-11,
10bps commission, 20-day rebal.

## Regime app

- [Regime baselines, Optuna instability, and the JAX-Adam scale-weight collapse](regime-baselines.md)
- [Log-returns CWT input degrades Sharpe](log-returns-vs-raw-close.md)

## Factor app

- [Deterministic-indicator val-IC baseline (the bar SSL must beat)](factor-indicator-baseline.md)

## Replay app

- [2D DWT keep-LL compression preserves SSL signal at ~4× input reduction](replay-dwt-compression.md)

## Relational app

- [Analog cross_ticker val edge collapses off mega-caps](relational-universe-shift.md)
- [DWT-L1 fingerprint compression fails OOS across all four distance scorers](relational-dwt-failure.md)
