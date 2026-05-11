# Recently shipped (and dropped from this list)

- **`packages/iv` (`ss_iv`)** (2026-05-10) — promoted
  `iv_data.py` + `short_vol.py` from
  `apps/relational/src/relational/` to `packages/iv/` when
  `apps/vol` became a second consumer (clean lift: pure numpy +
  pandas + `ss_features`, no relational-internal imports). Updated
  consumers: relational `research/diagnostic_dislocation_vs_vol.py`
  and `diagnostic_short_vol_pnl.py`. Pyproject + workspace sources
  updated. Tests 131/131 pass.
- **Prediction-problem-pivot arc** (2026-05-10) — three new apps
  scaffolded + v0 walk-forwards + closing arc-level synthesis.
  Three orthogonal prediction problems all show non-zero
  multivariate signal but partial-OOS at consistent magnitudes
  (mean alpha +0.07 to +0.10, regime-conditional). Closing
  finding:
  [`prediction-problem-pivot-arc`](../findings/prediction-problem-pivot-arc.md).
  Resolved TODOs:
  [`different-prediction-problem`](different-prediction-problem.md),
  [`apps-pairs`](apps-pairs.md), [`apps-vol`](apps-vol.md). New
  apps: `apps/gate`, `apps/pairs`, `apps/vol` (each ~600-700
  LoC).
- **Relational arc synthesis** (2026-05-10) — lifted the
  `apps/relational/NO_OPTIONS.md` 12-phase research arc into
  [`relational-arc-synthesis`](../findings/relational-arc-synthesis.md)
  and deleted the source. Captures: shippable strategies (Phase-2
  long-only equal-weight, transition-triggered rebal,
  velocity-magnitude scorer), falsified strategy classes (sizing
  overlays, pair-trades / market-neutral / cluster-pair, NN-pair
  hedge), and the operational rule "fingerprint embedding for
  selection and timing; not for hedging" (now in CLAUDE.md).
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
