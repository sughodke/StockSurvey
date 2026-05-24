# TODO

Active backlog for the StockSurvey monorepo. Each section from the original
`TODO.md` lives as a separate page below.

## Migrations and ports

- [Migrate non-research apps to the polar Morlet input bundle](polar-morlet-migration.md)
- [Port `ss_portfolio.sharpe.block_sharpe_with_costs` to tinygrad](port-sharpe-tinygrad.md)

## Research threads

- [Joint v0 — illiquid-options VRP × securities-lending stress — small-capacity re-frame of the discarded `confirmed-OOS` vol signal; Stage-0 PASS (DoltHub `option_chain` has free per-contract NBBO), runnable free with spread-defined cohort](vol-borrow-illiquid-vrp.md)
- [Higher-EV — borrow-stress conditioning on the *liquid* vol universe — the un-reached novel-data leg of the illiquid-VRP arc, re-pointed off the quote-availability wall onto the v3-locked liquid recipe; Stage-0 borrow-data feasibility gate first, then gauss314 (locked v3 + borrow split) → DoltHub OOS contingent on Phase-A H3 ≥ +0.10](vol-borrow-liquid-universe.md)
- [Tradier forward structural coverage probe — pre-registered $0 sandbox-MCP probe of the v1 microcap pick cohort against the full OPRA tape; resolves whether the falsified arc's DoltHub-coverage gap is *also* an OPRA-coverage gap and gates the entire paid-vendor spend question](vol-tradier-forward-coverage.md)
- [Crypto venue port — factor indicator-grid walk-forward on CryptoCompare top-50 — pre-registered single-test from `.research-venue-fit.md` (mean val IC > +0.025, 4/5 positive, DSR t > +1.5); scaffold built (commit `04bf48d`), local smoke-test at n_steps=50 looks encouraging (val IC +0.0504, 4/5 positive) but is NOT a verdict; Modal T4 eval (~$0.10) not yet kicked off](factor-crypto-venue-test.md)
- [DCA basket Optuna search — pre-registered bucket search over 3,600 basket combinations with N=200 trial budget; train 2005-2018, val 2019-2025 (incl. 2020 COVID + 2022 Fed-pivot); falsification bar locked: delta-t > +1.0 over canonical 13-ETF on same-method val = `confirmed-OOS`; lower = `confirmed-null`. Pre-reg page committed before eval to prevent retroactive bar-edit.](dca-basket-optuna.md)
- [DCA × vol overlay joint Optuna search — pre-registered joint search over 16,800 (basket × vega_scale) combinations, N=200 trials, train rebal 0-19 / val 20-32 of vol-v3-DoltHub 33-obs sample; falsification bar locked: delta-t > +1.0 vs canonical-13-ETF + vol×3 under identical method = `confirmed-OOS`. Pre-reg committed before eval; sister arc to `dca-basket-optuna`.](dca-vol-ensemble-optuna.md)
- [`ss_loaders.load_cryptocompare` is broken (hits retired v1 endpoint, `AttributeError: 'DataFrame' object has no attribute 'time'`) — the crypto venue prep script inlines a v2-compatible fetcher; the library function needs a small fix in a separate PR (move to `Data.Data` nesting, repaginate via `toTs`). Out of scope for the venue test, but a one-day pickup before the next caller needs it.](review-followups.md#ss_loaders-load_cryptocompare-v2-endpoint-fix)
- [`apps/cfr` — Deep CFR meta-allocator over existing scorers](apps-cfr.md)
- [Factor sizing-input reframe — re-purpose apps/factor as a meta-gate input](factor-sizing-input-reframe.md)
- [Ablation — disentangle why long-period RSI underperforms](rsi-long-period-ablation.md)
- [Diagnose why w=1 row underperforms in the FiLM (w, n) head](film-w1-diagnostic.md)
- [Backbone architecture — broader window/indicator coverage](backbone-architecture.md)
- [DWT-compression follow-ups](dwt-compression-followups.md)
- [Done — factor short-horizon × fixed `(C,L)` representation — resolved 2026-05-18: representation `confirmed-null` (2 non-CWT (C,L) encoders × 3 horizons), horizon `confirmed-OOS` (indicator grid @5d +0.0212 IC, 6/6); the +0.012 ceiling was a `rebal_days=20` artifact. See finding](factor-shorthorizon-representation.md)
- [Done — factor short-horizon edge microstructure/cost stress — resolved 2026-05-19 `partial-OOS`: a 1-day implementation lag halves the 5d edge (+0.0212→+0.0114, 5/6); ≈46% was bid-ask bounce, the rest is a modest cost-controlled ~+0.011 IC. See finding](factor-shorthorizon-microstructure.md)
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
- [Factor target-side REINFORCE for π — resolved 2026-05-15, `partial-OOS` (β=32 Δ-fix +0.108 clears literal cut but 0/6 per-window wins → not promoted; arc is benchmark-artifact-bound, w0-gate moot); see extended [`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md) under "Target-side REINFORCE — higher-β"](factor-reinforce-target-side.md)
- [Factor return-coupled recurrent CWT embedding — resolved 2026-05-17, `confirmed-null` (every k ≤ +0.0120 indicator baseline, no low-k plateau); closes the cwt-recursive-compression arc + the CWT-as-predictor question arc-wide; see extended [`cwt-recursive-compression`](../findings/cwt-recursive-compression.md#return-coupled-embedding-the-arc-closure)](factor-cwt-return-coupled.md)

## Superseded

- [Streaming feature pipeline](streaming-feature-pipeline.md) — pre-Modal JAX/Colab-era OOM design; the problem was solved a different way (Modal-T4 + the [memory + wall-time audit](memory-walltime-followups.md) trio shipped 2026-05-09). Kept as design archaeology.
