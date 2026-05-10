---
tags:
  - regime-2010-24
  - confirmed-OOS
  - hypothesis-user
---

# Log-returns CWT input degrades Sharpe

**Operational rule:** default to raw close for the regime trainer's
CWT input. `--use-log-returns` flag preserved for non-ranking research
(vol forecasting, regime-break detection).

## Setup

Controlled walk-forward eval, Stooq 2010-2024, 20 trials per window,
kernel half-extent 3 fixed in both arms.

## Result

Raw close beats log-returns on val Sharpe in every window:

| Stat        | Raw close   | Log-returns   |
|-------------|-------------|---------------|
| Median val Sharpe | +0.15 | +0.03         |
| Mean val Sharpe   | +0.07 | -0.29         |
| Worst window      | -0.41 | -1.06         |

Per-window: log-returns has **higher train but lower val** Sharpe — overfitting
signature.

## Mechanism

Raw close bleeds price-level trend into long-scale wavelet power,
embedding an implicit cross-sectional momentum factor. Log-returns
purifies trend out, leaving only "vol regime shift" — which is not a
known cross-sectional return predictor.

## Artifacts

`Output/regime-eval-{rawclose-kernel3,logreturns}.{log,json}`

## Source

[`031df265`](https://github.com/sughodke/StockSurvey/commit/031df265) — recorded in `CLAUDE.md` under "Key findings" (2026-04-25).

Master walk-forward log: [Leaderboard](../leaderboard.md) (the
*Log-returns CWT input vs raw close* row —
[`confirmed-OOS`](../leaderboard.md#verdict-labels), 3/3 windows raw
close > log-returns).
