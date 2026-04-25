"""regime.research: pre-trainer experiments and bt-library backtests.

These scripts predate the JAX-Adam trainer in `regime.trainer` and test
the regime signal under different lookback / top-N / divergence choices.
Useful as a sanity check on the trained checkpoint and for exploring
where the differentiable optimizer might be over-fitting.

  * `backtest_bt`       — bt-library portfolio backtest (rsi, scalogram,
                          regime, equal); produces equity curves + stats.
  * `backtest_ranking`  — plain-numpy long-short walk-forward eval.
  * `optimize_regime`   — Optuna walk-forward hyperparameter search.

Run as scripts: `python -m regime.research.backtest_bt --data-dir ...`
"""
