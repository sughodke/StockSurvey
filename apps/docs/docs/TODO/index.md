# TODO

Active backlog for the StockSurvey monorepo. Each section from the original
`TODO.md` lives as a separate page below.

## Migrations and ports

- [Migrate non-research apps to the polar Morlet input bundle](polar-morlet-migration.md)
- [Port `ss_portfolio.sharpe.block_sharpe_with_costs` to tinygrad](port-sharpe-tinygrad.md)

## Research threads

- [`apps/cfr` — Deep CFR meta-allocator over existing scorers](apps-cfr.md)
- [Factor sizing-input reframe — re-purpose apps/factor as a meta-gate input](factor-sizing-input-reframe.md)
- [Ablation — disentangle why long-period RSI underperforms](rsi-long-period-ablation.md)
- [Diagnose why w=1 row underperforms in the FiLM (w, n) head](film-w1-diagnostic.md)
- [Backbone architecture — broader window/indicator coverage](backbone-architecture.md)
- [DWT-compression follow-ups](dwt-compression-followups.md)
- [Rebal-days sweep](rebal-days-sweep.md)
- [Reversed-price training — falsify the time-symmetry hypothesis](reversed-price-experiment.md)
- [EW + rank-IC overlay test (parked — option 2 fallback)](ew-overlay-test.md)

## Operations

- [Review follow-ups — paper-trade can proceed without these](review-followups.md)
- [Modal-cron live deployment for ss-relational](modal-cron-live-deployment.md)
- [Memory and wall-time audit follow-ups](memory-walltime-followups.md)

## Done

- [Recently shipped (and dropped from this list)](recently-shipped.md)
- [Long-short constructor — resolved 2026-05-10, `confirmed-null`](long-short-constructor.md)
- [Different prediction problem — resolved 2026-05-10, three v0 partial-OOS rows; superseded by [`prediction-problem-pivot-arc`](../findings/prediction-problem-pivot-arc.md)](different-prediction-problem.md)
- [`apps/pairs` v0 — resolved 2026-05-10, `confirmed-null` per pre-reg](apps-pairs.md)
- [`apps/vol` v0 — resolved 2026-05-10, `inconclusive` (5/5 pos)](apps-vol.md)
- [Factor endogenous-horizon entropy-weight sweep — resolved 2026-05-14, `confirmed-null` on rescue hypothesis; see extended [`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)](factor-horizon-entropy-reg.md)
- [`apps/critic` Φ value function — resolved 2026-05-15, `confirmed-null` on both v0 (cross-app window-level) and v0.1 (pair-level rescue); see [`critic-phi-quality-v0`](../findings/critic-phi-quality-v0.md)](critic-phi-value-function.md)
- [Factor bilevel horizon objective — resolved 2026-05-15, `confirmed-null` on deployment-return supervision rescue; see extended [`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md) under "Bilevel objective"](factor-bilevel-horizon-objective.md)
- [Factor horizon-aligned IndicatorGridConfig — resolved 2026-05-15, `confirmed-null` on input-side rescue; see extended [`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md) under "Horizon-aligned feature grid"](factor-horizon-aligned-grid.md)

## Superseded

- [Streaming feature pipeline](streaming-feature-pipeline.md) — pre-Modal JAX/Colab-era OOM design; the problem was solved a different way (Modal-T4 + the [memory + wall-time audit](memory-walltime-followups.md) trio shipped 2026-05-09). Kept as design archaeology.
