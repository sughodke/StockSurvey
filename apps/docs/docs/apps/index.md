# Apps

Each app under `apps/` is a runnable workspace member with its own CLI or scripts.

| App | Status | Summary |
| --- | --- | --- |
| `regime` | Active | CWT-regime portfolio strategy with Optuna+vectorbt walk-forward search and Alpaca live runner. |
| `relational` | Active | Sector-relative + fingerprint-space CWT scorers; six scoreboard winners + paper trading. |
| `factor` | Active | Cross-sectional rank-IC scorer (tinygrad). Walk-forward eval over CWT-backbone or indicator inputs. |
| `replay` | Active | Multi-head CNN trainer reconstructing technical indicators from causal CWT slices. |
| `lie` | Active | Shape-feature research arc (cross-sectional, manifold experiments). |
| `notebook` | Active | Jupyter playground + scalogram visualizer CLIs (`ss-scalogram`, `ss-scalogram-video`). |
| `v1` | Parked | Legacy single-ticker workflow + aiohttp web service. |
| `docs` | Active | This Material for MkDocs site. |
