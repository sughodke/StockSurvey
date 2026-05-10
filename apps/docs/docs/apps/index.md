# Apps

Each app under `apps/` is a runnable workspace member with its own CLI
or scripts.

| App | Status | Summary |
| --- | --- | --- |
| [`regime`](regime.md) | Active | CWT-regime portfolio strategy with Optuna+vectorbt walk-forward search and Alpaca live runner. |
| [`relational`](relational.md) | Active | Sector-relative + fingerprint-space CWT scorers; six scoreboard winners + paper trading. |
| [`factor`](factor.md) | Active | Cross-sectional rank-IC scorer (tinygrad). Walk-forward eval over CWT-backbone or indicator inputs. |
| [`replay`](replay.md) | Active | Multi-head CNN trainer reconstructing technical indicators from causal CWT slices. |
| [`gate`](gate.md) | Active (v0 partial-OOS) | Aggregate drawdown forecaster — EW-exposure regime gate. Numpy OLS predictor. First test of the prediction-problem pivot. |
| [`pairs`](pairs.md) | Active (walkforward pending) | Pair-spread mean reversion. Engle-Granger screening + classical z-score trade rules. Numpy + statsmodels. |
| `lie` | Active | Shape-feature research arc (cross-sectional, manifold experiments). |
| [`notebook`](notebook.md) | Active | Jupyter playground + scalogram visualizer CLIs (`ss-scalogram`, `ss-scalogram-video`). |
| `v1` | Parked | Legacy single-ticker workflow + aiohttp web service. |
| `docs` | Active | This Material for MkDocs site. |
