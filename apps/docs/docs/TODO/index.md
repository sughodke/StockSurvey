# TODO

Active backlog for the StockSurvey monorepo. Each section from the original
`TODO.md` lives as a separate page below.

## Migrations and ports

- [Migrate non-research apps to the polar Morlet input bundle](polar-morlet-migration.md)
- [Port `ss_portfolio.sharpe.block_sharpe_with_costs` to tinygrad](port-sharpe-tinygrad.md)

## Research threads

- [Done — Meta-allocator regime forecasting across the 6 strategy arcs (`confirmed-OOS` for B3 inverse-arc-vol vs B2 1/N on 6-arc panel; **FALSIFIED 2026-05-24 on 5-arc no-vol_v3 panel** — ΔSR collapsed +0.367 → +0.039, CI now includes 0, `confirmed-null`; inverse-vol's edge was vol_v3-carried)](meta-allocator-regime-forecasting.md)
- [Vol_v3 sleeve sizing (pre-reg) — falsifiable validation across (vega × c_bps friction) grid; spawned by the 2026-05-24 no-vol_v3 falsification which redirected the deployment recipe from "inverse-vol over arc bundle" to "DCA + sized vol_v3 sleeve"](vol-v3-sleeve-sizing.md)
- [Joint v0 — illiquid-options VRP × securities-lending stress — small-capacity re-frame of the discarded `confirmed-OOS` vol signal; Stage-0 PASS (DoltHub `option_chain` has free per-contract NBBO), runnable free with spread-defined cohort](vol-borrow-illiquid-vrp.md)
- [Higher-EV — borrow-stress conditioning on the *liquid* vol universe — the un-reached novel-data leg of the illiquid-VRP arc, re-pointed off the quote-availability wall onto the v3-locked liquid recipe; Stage-0 borrow-data feasibility gate first, then gauss314 (locked v3 + borrow split) → DoltHub OOS contingent on Phase-A H3 ≥ +0.10](vol-borrow-liquid-universe.md)
- [Tradier forward structural coverage probe — pre-registered $0 sandbox-MCP probe of the v1 microcap pick cohort against the full OPRA tape; resolves whether the falsified arc's DoltHub-coverage gap is *also* an OPRA-coverage gap and gates the entire paid-vendor spend question](vol-tradier-forward-coverage.md)
- [Crypto venue port — factor indicator-grid walk-forward on CryptoCompare top-50 — pre-registered single-test from `.research-venue-fit.md` (mean val IC > +0.025, 4/5 positive, DSR t > +1.5); scaffold built (commit `04bf48d`), local smoke-test at n_steps=50 looks encouraging (val IC +0.0504, 4/5 positive) but is NOT a verdict; Modal T4 eval (~$0.10) not yet kicked off](factor-crypto-venue-test.md)
- [DCA basket Optuna search — pre-registered bucket search over 3,600 basket combinations with N=200 trial budget; train 2005-2018, val 2019-2025 (incl. 2020 COVID + 2022 Fed-pivot); falsification bar locked: delta-t > +1.0 over canonical 13-ETF on same-method val = `confirmed-OOS`; lower = `confirmed-null`. Pre-reg page committed before eval to prevent retroactive bar-edit.](dca-basket-optuna.md)
- [Done — DCA × vol overlay joint Optuna search (`partial-OOS`; Δ val deflated-t = +0.612 over canonical-13-ETF + vol×3, below the +1.0 confirmed-OOS bar; every top-10 trial picks vega_scale=3.0 — sizing robust, basket below noise; winner SPY+GLD+vol×3 is the joint-search version of the basket-only arc's simplification finding)](dca-vol-ensemble-optuna.md)
- [Factor head trained directly against studentized Sharpe-diff vs EW — first user of the differentiable Ledoit-Wolf analogue (`block_studentized_sharpe_diff_vs_ew`, commit `cb8f84e`); pre-registered head-to-head against `ir_vs_ew` + `rank_ic` on factor-narrow 5d windows; honest expectation = `confirmed-null` (the cross-sectional null is robust to optimization method).](factor-studentized-sharpe-diff-loss.md)
- [Vol-hyperparam × DCA ensemble joint Optuna search — **pre-registered (commit `e3b2b8b`), eval scaffold at `apps/vol/scripts/optuna_vol_hyperparam_ensemble.py`, eval NOT yet run**. Joint search over 2,100 (top_k × gate_lookback × rebal_trading_days × gate_quantile × vega_scale) combinations, N=200 trials, date-based split 2023-08→2024-12 train / 2025-01→2026-03 val of vol-v3-DoltHub substrate, DCA basket fixed at canonical-13; falsification bar locked: delta-t > +1.0 vs canonical v3 recipe + vol×3 incumbent = `confirmed-OOS`. Sister arcs `dca-basket-optuna` (basket-only) and `dca-vol-ensemble-optuna` (basket × overlay-sizing).](vol-hyperparam-ensemble-optuna.md)
- [`ss_loaders.load_cryptocompare` is broken (hits retired v1 endpoint, `AttributeError: 'DataFrame' object has no attribute 'time'`) — the crypto venue prep script inlines a v2-compatible fetcher; the library function needs a small fix in a separate PR (move to `Data.Data` nesting, repaginate via `toTs`). Out of scope for the venue test, but a one-day pickup before the next caller needs it.](review-followups.md#ss_loaders-load_cryptocompare-v2-endpoint-fix)
- [CNC followups — basis-tracking-error stress + funding-regime gate landed 2026-05-28; both falsified deployability on Hyperliquid (`friction-fragile` at break-even ~4.07 bps/d; gate falsified at all thresholds {0.5, 1.0, 2.0, 5.0} bps/d with 24-25 mean Sh loss ≥ 2.08 vs ≤1.0 ceiling); arc closes substrate-confirmed but deployment-falsified; only remaining adjacent path is venue port to OKX / paid Binance / Bybit deep-history feed](cnc-followups.md)
- ~~[Follow-the-leader v1 — cross-sectional consensus arm](follow-consensus-arm.md)~~ — **DONE 2026-05-28** `confirmed-OOS` vs locked TODO bar (`partial-OOS` under CI-based reading); fold-3 (UNSEEN 2025-01→2025-10, n=198d) Sh +0.86 / α +5.13pp/yr / pos-Q 100%; pooled Sh +1.0 / α +5.41pp/yr; standalone defl-t +2.58 clears, edge-vs-SPY defl-t +1.37 below +2.0; ship as ensemble constituent with SPY β-hedge. See [`findings/follow-consensus-arm`](../findings/follow-consensus-arm.md).
- [Follow-consensus β-hedged ensemble weight](follow-consensus-ensemble-weight.md) — pre-reg locks 7-point `follow_w` grid as 3rd leg of canonical (DCA + 2 × vol-v3) stack; verdict bar `confirmed-OOS` requires OOS ΔSharpe ≥ +0.10 vs 2-leg baseline AND CI excludes 0 AND max-DD ≤ 1.2× baseline; **pending Quiver xlsx refresh** to extend fold-3 from n=198 to ≥400 days before eval runs (compound-deflation against an already-wide CI would otherwise consume the sample). Sister arc to [`follow-consensus-arm`](../findings/follow-consensus-arm.md) and consumer of [`vol-v3-sleeve-sizing`](vol-v3-sleeve-sizing.md).
- [HRP asymmetric two-sided modulator](hrp-asymmetric-modulator.md) — pre-reg locks 96-cell grid (open_rule × lookback × threshold × floor) over the 6-window walk-forward; rescues the symmetric `eff_rank/n_active` gate whose w5 +0.586 lift was a Sharpe-arithmetic artifact (sub-period lifts sum to +0.043; 2022 detection alpha real, 2023 recovery alpha *destroyed*); verdict bar requires w5 OOS α-lift ≥ +0.10 AND sub-period lifts sum ≥ +0.08 AND 2023-recovery sub-period α ≥ 0; bootstrap CI > ±0.40 auto-downgrades one tier. Closes the HRP-modulator arc one way or the other. Parent: [`lie-hrp-baseline`](../findings/lie-hrp-baseline.md).
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

- [NPZ explicit-dates backfill — plumb `rebal_dates`/`dates` through 8 return-stream producers so cross-arc Ledoit-Wolf CI and per-window meta-allocator probes stop falling back to tail-alignment; ~25 LOC across 8 scripts, half-day; the H1 inflation +1.43 honest → +2.21 tail-aligned in the meta-allocator brief is the operational cost already paid](npz-explicit-dates-backfill.md)
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
