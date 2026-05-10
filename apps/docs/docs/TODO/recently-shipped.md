# Recently shipped (and dropped from this list)

- **Long-short market-neutral constructor for the rank-IC heads**
  (2026-05-10) — `factor.objectives.{long_short_weights,
  block_sharpe_long_short}` + per-window
  `WalkForwardWindow.{train,val}_sharpe_long_short` column. Driver
  at `apps/factor/scripts/long_short_eval.py`. Resolved
  [`long-short-constructor`](long-short-constructor.md) TODO with
  verdict `confirmed-null`: discarded-short-signal hypothesis
  falsified, line of work pivots to
  [`different-prediction-problem`](different-prediction-problem.md).
  Closing finding:
  [`factor-rankic-long-only-mismatch`](../findings/factor-rankic-long-only-mismatch.md).
- `ss-portfolio.broker` (Alpaca adapter shared between regime + relational) +
  `ss-relational live` paper-trade stack with the four risk rails.
- `apps/relational/scripts/build_canonical_checkpoints.py` — six canonical
  RelationalCheckpoint JSONs covering the scoreboard winners.
- Three critical live-trading code-review fixes: `apply_position_cap` no
  longer re-introduces zero-weight names; live bar fetch covers full CWT
  kernel support; `submit_orders` surfaces per-symbol rejections to
  `LiveRunResult.rejected_orders`. (commits
  [`a1beead`](https://github.com/sughodke/StockSurvey/commit/a1beead) /
  [`8c21c9b`](https://github.com/sughodke/StockSurvey/commit/8c21c9b) /
  [`4ee4d0d`](https://github.com/sughodke/StockSurvey/commit/4ee4d0d))
- `ss_cli` (shared CLI flag groups) + `ss_portfolio.bt_helpers` (shared
  bt.Strategy template) + `packages/tg_ops/` (shared `_conv1d` permute) +
  block-windows generator → `ss_features.walkforward`. (commit
  [`ddfadce`](https://github.com/sughodke/StockSurvey/commit/ddfadce))
- Polar Morlet + Gaussian + log-L2 amplitude input bundle (commit
  [`954a88a`](https://github.com/sughodke/StockSurvey/commit/954a88a)
  + [streaming-predict refactor](https://github.com/sughodke/StockSurvey/commit/12ea630))
  replaces the
  `--include-zscore-stats / --include-returns / --include-return-sign`
  optional channels. Canonical channel layout per scale: `(|c|, |c|^2,
  cos(arg), sin(arg), g, g^2, log_L2_amp)` = `7 * n_scales` per lag,
  exposed as `ss_features.CHANNELS_PER_SCALE`. CSCO zero-shot R² at the
  canonical 295-ticker pool: vol 0.48 → 0.64 (+0.16), cci 0.85 → 0.89
  (+0.04), rsi 0.90 → 0.87 (-0.03). MACD head pathology unchanged
  (pre-existing; see "MACD head pathology" below).
