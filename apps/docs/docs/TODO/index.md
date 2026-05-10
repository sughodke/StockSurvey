# TODO

Active backlog for the StockSurvey monorepo. Each section from the original
`TODO.md` lives as a separate page below.

## Migrations and ports

- [Migrate non-research apps to the polar Morlet input bundle](polar-morlet-migration.md)
- [Port `ss_portfolio.sharpe.block_sharpe_with_costs` to tinygrad](port-sharpe-tinygrad.md)

## Research threads

- [Different prediction problem — pair-spread / drawdown / IV-vs-realized](different-prediction-problem.md)
- [Ablation — disentangle why long-period RSI underperforms](rsi-long-period-ablation.md)
- [Diagnose why w=1 row underperforms in the FiLM (w, n) head](film-w1-diagnostic.md)
- [Backbone architecture — broader window/indicator coverage](backbone-architecture.md)
- [DWT-compression follow-ups](dwt-compression-followups.md)
- [Rebal-days sweep](rebal-days-sweep.md)
- [Reversed-price training — falsify the time-symmetry hypothesis](reversed-price-experiment.md)

## Operations

- [Review follow-ups — paper-trade can proceed without these](review-followups.md)
- [Modal-cron live deployment for ss-relational](modal-cron-live-deployment.md)
- [Memory and wall-time audit follow-ups](memory-walltime-followups.md)

## Done

- [Recently shipped (and dropped from this list)](recently-shipped.md)
- [Long-short constructor — resolved 2026-05-10, `confirmed-null`](long-short-constructor.md)

## Superseded

- [Streaming feature pipeline](streaming-feature-pipeline.md) — pre-Modal JAX/Colab-era OOM design; the problem was solved a different way (Modal-T4 + the [memory + wall-time audit](memory-walltime-followups.md) trio shipped 2026-05-09). Kept as design archaeology.
