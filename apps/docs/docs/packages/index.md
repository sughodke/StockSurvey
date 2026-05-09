# Packages

Shared library code under `packages/`. Distribution names are hyphenated
(`ss-indicators`); import names are underscored (`ss_indicators`).

| Package | Import | Summary |
| --- | --- | --- |
| `loaders` | `ss_loaders` | Kaggle CSV matrix, Stooq archive, Yahoo, CryptoCompare, symbol lists. |
| `indicators` | `ss_indicators` | Numpy matrix-form RSI/MACD/BBands/SMA/EMA + CCI, Corwin-Schultz spread, divergences, rolling Pearson. |
| `wavelets` | `ss_wavelets` | Strictly-causal Ricker CWT + windowed power means. |
| `portfolio` | `ss_portfolio` | Numpy block-Sharpe with costs, CAGR/drawdown/Sortino/Calmar, water-fill cap, Alpaca broker. |
| `features` | `ss_features` | `TickerData`, `load_prices`, `Backbone` + `load_backbone` (numpy npz I/O). |
| `plotting` | `ss_plotting` | Training curves, equity comparison, scalogram heatmap helpers. |
| `stream` | `ss_stream` | Point-in-time universe iterator over the Stooq archive. |
