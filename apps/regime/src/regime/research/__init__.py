"""regime.research: optimization + backtest workbench.

Two optimizers and two backtest harnesses, all sharing the same
`ss_indicators` / `ss_wavelets` / `ss_portfolio` primitives:

  * `optimize_adam`     — JAX-Adam gradient descent over 14 continuous
                          model parameters (scale weights + temperature)
                          for *fixed* hyperparameters. Used by
                          `regime train`.
  * `optimize_regime`   — Optuna TPE walk-forward search over 7 discrete
                          hyperparameters (lookback, n_tail, top_n,
                          divergence, scale subsets). Slow but robust.
  * `backtest_bt`       — bt-library portfolio backtest (rsi, scalogram,
                          regime, equal); produces equity curves + stats.
                          No optimization — runs given parameters.
  * `backtest_ranking`  — plain-numpy long-short walk-forward eval.
                          No optimization — runs given parameters.

Run as scripts: `python -m regime.research.optimize_regime --data-dir ...`
The `regime train` CLI is a thin wrapper around `optimize_adam.train()`.
"""
