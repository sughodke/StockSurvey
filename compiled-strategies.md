# Code-truth strategy census (2026-05-25)

Scope: every strategy implementation found in `apps/` and `packages/` Python
code, excluding `apps/docs` and `apps/v1`. Sourced from grep + code-reading,
NOT from findings/leaderboard. "Strategy" = anything producing target weights,
per-ticker ranking scores, sized positions, or signal streams.

## Summary

- Total strategies catalogued: ~55 (including CFR `score_kind` variants
  and CFR composed menus as distinct entries; collapsing variants gives ~40
  unique implementations).
- Live-wired (has a `*.live` module path): 9 — regime (3 strategies share
  one `live.py`), relational (6 strategies share one `live.py`), DCA, Vol
  (`vol/live.py`, options strangles).
- Research-only: the remainder (factor heads, CFR action-menu modes,
  pairs, gate, lie HRP, critic, all `apps/<app>/scripts/*`).

## By group

### 1. CWT / wavelet ranking

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| CWT | `weights_regime` (canonical) | `apps/regime/src/regime/trainer.py` (called via `inference.py:57`) | `--strategy regime` | prices, scales, lookback | per-ticker scores → top-N normalized weights | yes via `regime.live` | Causal Ricker CWT power; rank by recent-vs-historical divergence (KL/JS/cosine/L2) over short vs long scales | Canonical version lives in `packages/portfolio/src/ss_portfolio/strategies.py:47`; trainer-side `weights_scalogram` and `weights_rsi` are the alternate heads selectable via `STRATEGIES` tuple |
| CWT | `weights_scalogram` | `apps/regime/src/regime/trainer.py:215` | `--strategy scalogram` | prices, CWT cube | per-ticker scores → weights | yes via `regime.live` | Direction − momentum × coherence over CWT scale axis | |
| CWT | `weights_regime_parameterized` | `apps/regime/src/regime/research/optimize_regime.py:41` | N/A | prices + Optuna hyperparams | weights | research-only | Optuna-search variant of `weights_regime` | Optimize harness |
| CWT | `weights_scalogram` (bt research) | `apps/regime/src/regime/research/backtest_bt.py:63` | bt-runner only | prices | weights | research-only | bt-library variant of scalogram | |
| CWT | `weights_regime` (bt research) | `apps/regime/src/regime/research/backtest_bt.py:97` | bt-runner only | prices | weights | research-only | bt variant | |
| CWT | `weights_excess_regime` (relational base) | `apps/relational/src/relational/scoring.py:110` | called from empirical/gmm dispatch | prices, sector membership, CWT | weights | yes via relational | Sector-relative divergence; subtract sector mean fingerprint before scoring | Building block for `empirical` + `gmm` |
| CWT | `weights_excess_regime_empirical` | `apps/relational/src/relational/empirical_sectors.py:313` | `RelationalCheckpoint.strategy='empirical'` | prices, sector map, CWT | weights | yes via `relational.live` | Empirical sector centroids in fingerprint space | |
| CWT | `weights_excess_regime_gmm` | `apps/relational/src/relational/empirical_sectors_gmm.py:383` | `strategy='gmm'` | prices, CWT, fitted GMM | weights | yes via `relational.live` | GMM-clustered sector centroids over fingerprint space | |
| CWT | `weights_regime_analog` | `apps/relational/src/relational/analog_knn.py:471` | `strategy='analog'` | prices, CWT, history index | weights | yes via `relational.live` | kNN over fingerprint embeddings; forward-return weighted | Only strategy currently plumbed for `wavelet='morlet'` |
| CWT | `weights_regime_farthest` | `apps/relational/src/relational/farthest.py:85` | `strategy='farthest'` | prices, CWT | weights | yes via `relational.live` | Anti-mode: pick names farthest from sector centroid | |
| CWT | `weights_regime_diversified` | `apps/relational/src/relational/diversify.py:79` | `strategy='diversified'` | prices, CWT | weights | yes via `relational.live` | Coverage-maximizing selection across fingerprint clusters | |
| CWT | `weights_velocity_magnitude` | `apps/relational/src/relational/regime_velocity.py:376` | `strategy='velocity'` (sub-kind 'magnitude') | prices, CWT trajectories | weights | yes via `relational.live` | Rank by fingerprint-trajectory speed | |
| CWT | `weights_axis_alignment` | `apps/relational/src/relational/regime_velocity.py:396` | `strategy='velocity'` (sub-kind 'axis') | prices, CWT trajectories | weights | yes via `relational.live` | Project trajectory onto historical winner axis | |
| CWT | `train_scorer_walkforward` (CWT-input variant) | `apps/factor/src/factor/cwt_gru_walkforward.py` | factor walk-forward harness | CWT cube | weights/scores | research-only | tinygrad scorer over wavelet-input encoder | |
| CWT | `smoke_cwt_gru` / `cwt_bundle_walkforward` | `apps/factor/scripts/{smoke_cwt_gru,cwt_bundle_walkforward}.py` | script | CWT bundles | scores | research-only | CWT-bundle benchmark | |

### 2. Cross-sectional indicator-based

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Indicator | `weights_rsi` (canonical) | `apps/regime/src/regime/trainer.py:293` | `--strategy rsi` | prices | weights | yes via `regime.live` | Rank by trailing RSI mean / tail | |
| Indicator | `weights_rsi` (bt research) | `apps/regime/src/regime/research/backtest_bt.py:44` | bt-runner only | prices | weights | research-only | bt variant | |
| Indicator | `weights_equal` | `apps/regime/src/regime/research/backtest_bt.py:123` | bt-runner only | prices | weights | research-only | EW baseline with spread gate | Comparison baseline |
| Indicator | `train_scorer_indicators` | `apps/factor/src/factor/indicator_features.py:260` | factor train harness | 74-channel indicator stack | per-ticker scores | research-only | Linear/MLP head over RSI/MACD/CCI/vol/coherence panel | Default 820-bar warmup |
| Indicator | `train_scorer_indicators_walkforward` | `apps/factor/src/factor/indicator_features.py:278` | factor WF harness | indicator stack | scores per window | research-only | Walk-forward variant of above | |
| Indicator | shape-kNN long-short | `apps/lie/scripts/shape_knn_longshort.py` | script | per-ticker shape features | sized long-short | research-only | Cross-sectional shape kNN, H=21 horizon | Canonical "lie" cross-sectional path per memory |
| Indicator | `train_scorer_spectral` / `_walkforward` | `apps/factor/src/factor/cl_encoders.py:366,386` | factor harness | Spectral grid features | scores | research-only | C-L spectral encoder benchmark | |
| Indicator | `train_scorer_minirocket` / `_walkforward` | `apps/factor/src/factor/cl_encoders.py:376,396` | factor harness | MiniRocket convolutions | scores | research-only | MiniRocket kernel features benchmark | |
| Indicator | `feature_aug_walkforward` | `apps/factor/scripts/feature_aug_walkforward.py` | script | augmented indicator stack | scores | research-only | Feature-augmented WF | |
| Indicator | `forecast_probe_walkforward` | `apps/factor/scripts/forecast_probe_walkforward.py` | script | indicator stack | scores | research-only | Forecast-probe WF | |
| Indicator | `no_backbone_baseline` / `_matched` | `apps/factor/scripts/no_backbone_baseline*.py` | script | raw indicator stack | scores | research-only | No-backbone benchmark | |
| Indicator | `universe_pivot_walkforward` | `apps/factor/scripts/universe_pivot_walkforward.py` | script | indicators × wider universe | scores | research-only | Universe-pivot WF | |
| Indicator | `vol_overlay_walkforward` | `apps/factor/scripts/vol_overlay_walkforward.py` | script | indicators + vol overlay | scores | research-only | Vol-overlay WF | |
| Indicator | `horizon_pivot_walkforward` / `horizon_mixture_walkforward` / `horizon_regime_gated` | `apps/factor/scripts/horizon_*.py` | script | indicators + horizon switch | scores | research-only | Horizon-mixture WF heads | |

### 3. Multi-asset basket / DCA-style

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Basket | DCA canonical (`run_live`) | `apps/dca/src/dca/live.py:125`; weights at `_compute_current_weights` (l.60); cadence at `_evaluate_cadence_gate` (l.85) | `ss-dca live` | prices, `DCACheckpoint` (Phase 4d 13-ETF universe @ 1/13 each) | target weights | yes (canonical live) | Fixed-target EW; rebal on cadence (≥80td floor) OR per-name 5% drift | Canonical live strategy per CLAUDE.md |
| Basket | `simulate_basket` (Optuna search) | `apps/dca/scripts/optuna_basket_search.py:155` | script | prices, basket config | returns | research-only | Drift-rebal target basket simulator over Optuna sample | |
| Basket | `dca_block_returns` / `simulate_basket_daily` | `apps/dca/scripts/optuna_dca_vol_ensemble.py:155,214` | script | prices + vol stream | returns | research-only | DCA ensemble with vol overlay | |
| Basket | `regime_scaled_dca` (`vol_target_exposure`, `dd_gate_exposure`) | `apps/dca/scripts/regime_scaled_dca.py:62,75` | script | DCA returns | exposure scalar series | research-only | DCA × vol-target and DCA × DD-gate variants | Composable with #6 |
| Basket | `vol_sleeve_friction_grid` | `apps/dca/scripts/vol_sleeve_friction_grid.py` | script | DCA + vol sleeve | returns | research-only | DCA + vol sleeve friction sweep | |

### 4. Volatility / IV harvesting

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Vol | `evaluate_short_vol` / `short_vol_pnl_panel` | `packages/iv/src/ss_iv/short_vol.py:32,41` | called from `apps/vol` | IV + realized vol panels | PnL stream | yes via `vol.live` | Short straddle/strangle PnL accounting at the panel level | Primitive |
| Vol | `evaluate_universe_short_vol` | `packages/iv/src/ss_iv/short_vol.py:123` | called from `apps/vol` | universe IV panel | aggregated PnL | yes via vol pipeline | Universe-level short-vol aggregator | |
| Vol | `evaluate_gated_short_vol` | `apps/vol/src/vol/backtest.py:32` | research-only entrypoint | IV-RV gap predictions, regime gate | gated PnL | research (used to back live ckpt) | Apply top-K + regime gate on top of short-vol PnL | |
| Vol | `evaluate_portfolio_short_vol` | `apps/vol/src/vol/portfolio.py:75` | research entrypoint | IV-RV features, rebal dates | per-rebal Sharpe | research | Per-rebal portfolio Sharpe simulator | |
| Vol | `predict_iv_rv_gap` | `apps/vol/src/vol/inference.py:26` | called by `vol.live` | feature panel, `VolCheckpoint` | per-ticker scores | yes via `vol.live` | Numpy OLS over 10 surface-shape features → forward 20d IV/RV gap | v3 deployment recipe |
| Vol | `select_top_k` + `gate_fires` | `apps/vol/src/vol/inference.py:60,74` | called by `vol.live` | predicted gap, rebal date | selection mask, gate firing bool | yes | Top-K by predicted gap; 126d-VIX-rolling-median per-rebal gate | |
| Vol | `build_short_strangle` | `apps/vol/src/vol/strangle.py:63` | called by `vol.live` | option chain + predictor selection | sized options legs | yes (`vol.live`, requires options broker) | Build short strangle legs for selected names | Live infra noted as unbuilt in CLAUDE.md but code exists |
| Vol | `submit_short_strangle` | `apps/vol/src/vol/alpaca_chain.py:164` | called by `vol.live` | Strangle objs + Alpaca client | filled orders | yes | Alpaca options submission | |
| Vol | `run_live` (vol) | `apps/vol/src/vol/live.py:88` | `ss-vol live` | checkpoint + Alpaca | live orders | yes | End-to-end gated short-strangle deployment | |
| Vol | `run_ensemble` | `apps/vol/src/vol/ensemble.py:51` | `ss-vol ensemble` | multiple vol checkpoints | aggregated weights | yes | Ensemble across hyperparam variants | |
| Vol | `borrow_*` scripts | `apps/vol/scripts/{borrow_stage0_probe,run_borrow_phaseB_oos,run_walkforward_v3_borrow}.py` | script | borrow-cost + IV panel | sized positions | research-only | Illiquid-microcap VRP arm (per CLAUDE.md: reversed-OOS, do not redeploy) | |

### 5. Pairs / mean-reversion

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Pairs | `trade_signals` | `apps/pairs/src/pairs/predictor.py:25` | called by backtest | spread z-score series | position {-1, 0, +1} stream | research-only | Gatev-Goetzmann-Rouwenhorst classical z-cross rule (entry=2.0, exit=0.5) | |
| Pairs | `backtest_pair` | `apps/pairs/src/pairs/backtest.py:51` | research entrypoint | pair prices + signal | PnL | research-only | Dollar-neutral pair backtest | |
| Pairs | `aggregate_pair_pnl` | `apps/pairs/src/pairs/backtest.py:120` | research entrypoint | per-pair PnLs | aggregate stream | research-only | Aggregate across selected pairs | |
| Pairs | `engle_granger_test` + `screen_pairs` | `apps/pairs/src/pairs/cointegration.py:41`, `pair_universe.py:77` | screening | universe | qualifying pairs | research-only | Engle-Granger cointegration screen (correlation prefilter + EG p-value) | |
| Pairs | `run_pair_predictor_walkforward` | `apps/pairs/scripts/run_pair_predictor_walkforward.py` | script | pair features | sized signal | research-only | ML-augmented pair predictor variant | |
| Pairs | `eg_gate_eval` / `run_oracle_walkforward` | `apps/pairs/scripts/{eg_gate_eval,run_oracle_walkforward}.py` | script | pairs + gate | gated PnL | research-only | EG-gate ablations | |

### 6. Drawdown / risk gates

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Gate | `train_predictor` + `predict` + `apply_gate` | `apps/gate/src/gate/predictor.py:50,88,108` | research entrypoint | aggregate features (vol, return, trailing DD, breadth) | exposure scalar | research-only | Numpy OLS forecaster of forward max DD → EW exposure gate | gate v0 |
| Gate | `gated_returns` / `evaluate_gated_arm` | `apps/gate/src/gate/backtest.py:30,47` | research entrypoint | base returns + gate stream | gated returns | research-only | Apply predicted-DD gate to upstream return stream | |
| Gate | `build_ew_aggregate` + features | `apps/gate/src/gate/aggregate.py:25,111` | called by gate | universe | aggregate series + features | research-only | EW aggregate + 6 trailing features | |
| Gate | `macro_meta_gate_eval` / `_continuous` | `apps/gate/scripts/macro_meta_gate_{eval,continuous}.py` | script | macro features | gate firing | research-only | Macro-feature variant of the gate | |
| Gate | `gross_exposure_modulator` (effective-rank gate) | `apps/lie/src/lie/symmetry_rank.py:85` | called from `lie.inference` when ckpt opts in | rolling correlation | gross-exposure scalar | yes via `lie.cli weights` | Effective-rank-of-correlation modulator (low erank → de-risk) | "Symmetry breaking" gate |
| Gate | `_evaluate_cadence_gate` | `apps/dca/src/dca/live.py:85` | DCA live | drift + cadence | rebal trigger bool | yes | 5% drift OR ≥80td cadence-floor rebal trigger | DCA's own gate |
| Gate | `gate_fires` | `apps/vol/src/vol/inference.py:74` | vol live | VIX history | bool | yes | 126d-VIX-rolling-median per-rebal regime gate | Vol's regime gate |
| Gate | `dynamic_rebal_cadence_oracle` | `apps/gate/scripts/dynamic_rebal_cadence_oracle.py` | script | universe + cadence options | oracle decision | research-only | Oracle-bound study for adaptive rebal cadence | |

### 7. Meta-allocator

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Meta | (no source-tree meta-allocator package; meta-allocator experiments live in repo-root research scripts referenced in git status — not in `apps/`/`packages/`) | n/a | n/a | sub-strategy return streams | weights | research-only | Inverse-arc-vol / persistence / 1/N tested via untracked root-level scripts | The only in-tree meta-allocator surface is CFR (group 10) — true "select across already-deployed sub-strategies" code is the untracked `per_window_meta_allocator.py` / `ensemble_discovery.py` in the dirty working tree, NOT under `apps/` |

### 8. Tinygrad / autograd-trained scorers

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Tinygrad | `init_linear` / `apply_linear` | `apps/factor/src/factor/scorers.py:27,41` | `get_scorer('linear')` | hidden feature vec | per-ticker score | research-only | Linear head on backbone output | |
| Tinygrad | `init_mlp` / `apply_mlp` | `apps/factor/src/factor/scorers.py:47,67` | `get_scorer('mlp')` | hidden feature vec | scores | research-only | 2-3 layer MLP head | |
| Tinygrad | `init_mlp_multitask` / `apply_mlp_multitask` | `apps/factor/src/factor/scorers.py:82,113` | scorer name | hidden feature vec | per-task scores | research-only | Shared trunk, per-task heads | |
| Tinygrad | `init_mlp_horizon` / `apply_mlp_horizon[_full]` | `apps/factor/src/factor/scorers.py:131,182,216` | scorer name | feature vec + horizon emb | per-horizon scores | research-only | Horizon-mixture head | |
| Tinygrad | `train_scorer` (Stage 1 + optional Stage 2) | `apps/factor/src/factor/train.py` | factor harness | inputs + targets | trained scorer params | research-only | tinygrad rank-IC loss; optional backbone fine-tune | |
| Tinygrad | `train_scorer_walkforward` | `apps/factor/src/factor/train_walkforward.py:243` | factor WF harness | aligned tickers | per-window scorer + scores | research-only | Walk-forward over `WalkForwardWindow`s | |
| Tinygrad | `RegretNet` (Deep CFR) | `apps/cfr/src/cfr/deep.py:61` | `cfr` deep-mode | state vec | per-action predicted regret | research-only | tinygrad MLP for advantage prediction | |
| Tinygrad | `policy_from_predicted_regret` | `apps/cfr/src/cfr/deep.py:214` | called from deep CFR | predicted regret | policy distribution | research-only | Regret-matching over predicted regret | |
| Tinygrad | `train_policy` (Φ-imitation) | `apps/critic/src/critic/policy.py:71` | `critic` harness | (state, action) features + pretrained Φ | policy params | research-only | tinygrad policy trained to maximize learned Φ; vanilla + CQL variants | |
| Tinygrad | `apply_policy` / `policy_score` / `policy_inclusion` | `apps/critic/src/critic/policy.py:59,163,169` | critic deploy path | policy params + features | per-pair inclusion score | research-only | Sigmoid policy over pair-feature vec | |
| Tinygrad | replay reconstruction heads (RSI/MACD/vol/CCI/price + FiLM) | `apps/replay/src/replay/decoders.py` (+ `train_cnn_multihead.py` modal) | replay trainer | causal CWT slice | reconstructed indicator series | research-only | Multi-head CNN reconstruction; produces backbone npz that `factor` consumes — not directly a portfolio strategy but is the SSL backbone source | Counted because backbone feeds scorers |

### 9. Reference / canonical anomaly factors

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| Reference | `momentum_12_1.py` | `apps/factor/scripts/momentum_12_1.py` | script | prices | per-ticker scores | research-only | Jegadeesh-Titman 12-1 momentum benchmark | |
| Reference | `low_vol_bab.py` | `apps/factor/scripts/low_vol_bab.py` | script | prices | scores | research-only | Frazzini-Pedersen low-vol / BAB benchmark | |
| Reference | `long_short_eval` | `apps/factor/scripts/long_short_eval.py` | script | scorer output | long-short PnL | research-only | Generic long-short evaluator over an upstream scorer | |
| Reference | `loss_pivot_eval` / `verdict_studentized_sharpe_diff` | `apps/factor/scripts/{loss_pivot_eval,verdict_studentized_sharpe_diff}.py` | script | scorer pair | studentized Sharpe diff | research-only | Pre-reg comparison of two scorers via studentized Sharpe-diff loss | |
| Reference | `weights_hrp` | `apps/lie/src/lie/hrp.py:88` | `lie.inference` (strategy='hrp') | prices, lookback | weights | research live via `ss-lie weights` CLI (no Alpaca path) | Lopez-de-Prado Hierarchical Risk Parity (tree-cluster → quasi-diag → recursive bisection) | Optionally composed with erank gate (group 6) |
| Reference | `TimelessPredictor` (kNN) | `apps/lie/src/lie/predictor.py:27` | called from `lie` scripts | embedding panel | predicted forward returns | research-only | kNN with hard temporal-gap exclusion (default 60d) | Per memory: ticker-only cross-sectional path passed t=+3.75 |
| Reference | `Top13FConsensusMode` | `apps/cfr/src/cfr/modes_13f.py:28` | CFR menu mode | 13F consensus panel | EW weights over top-K consensus names | research-only | EW over top-K most-broadly-held names per 13F filers; 45d filing lag | |

### 10. CFR / RL allocators

| group | strategy name | implemented at | dispatch | inputs | output | live-wired? | mechanism | notes |
|---|---|---|---|---|---|---|---|---|
| CFR | `TabularCFR` | `apps/cfr/src/cfr/tabular.py:33` | `cfr.cli` walkforward | menu + state index | regret table + action policy | research-only | Tabular CFR over discrete state × action menu | |
| CFR | `CFRWalkForward` | `apps/cfr/src/cfr/walkforward.py:92` | `cfr.cli` | tabular CFR + menu + prices | per-window policy → returns | research-only | Walk-forward CFR allocator | |
| CFR | `regret_matching` + `sample_action` | `apps/cfr/src/cfr/regret.py:95,120` | called by CFRs | regret vec | action prob | research-only | Vanilla regret-matching | |
| CFR | `EqualWeightMode` (CFR menu) | `apps/cfr/src/cfr/menu.py:133` | menu mode `ew` | universe + mask | EW weights | research-only | EW over liquid set | |
| CFR | `CashMode` | `apps/cfr/src/cfr/menu.py:124` | menu mode `cash` | n/a | zero weights | research-only | Cash | |
| CFR | `TopKMode(score_kind='momentum')` | `apps/cfr/src/cfr/menu.py:285` (`score_kind='momentum'`) | menu mode `mom` | prices, window | top-K weights | research-only | Top-K by trailing log return | |
| CFR | `TopKMode(score_kind='reversal')` | same | menu mode `rev` | prices | top-K weights | research-only | Top-K by *lowest* trailing log return | |
| CFR | `TopKMode(score_kind='low_vol')` | same | menu mode `lowv` / `lowv252` | prices | top-K weights | research-only | Top-K by lowest trailing realized vol | |
| CFR | `TopKMode(score_kind='high_vol')` | same | menu mode `highv` | prices | top-K weights | research-only | Top-K by highest trailing realized vol | |
| CFR | `TopKMode(score_kind='mom_12_1')` | same | menu mode `mom121` | prices | top-K weights | research-only | Jegadeesh-Titman 12-1 inside the CFR menu | |
| CFR | `TopKMode(score_kind='sharpe_top')` | same | menu mode `shtop` | prices | top-K weights | research-only | Top-K by trailing 252d Sharpe | |
| CFR | `TopKMode(score_kind='trend_str')` | same | menu mode `trend` | prices | top-K weights | research-only | Top-K by trailing log return / max DD (window-Calmar) | |
| CFR | `default_phase1_menu` | `apps/cfr/src/cfr/menu.py:434` | named menu | universe + top_k | composed action menu (16 actions) | research-only | 5 modes × 4 gross levels | |
| CFR | `default_phase2a_menu` | `apps/cfr/src/cfr/menu.py:463` | named menu | universe + top_k | composed action menu (28 actions) | research-only | 9 modes × 4 gross levels (adds documented-alpha modes) | |
| CFR | `PassiveEW` baseline | `apps/cfr/src/cfr/baselines.py:115` | baseline | universe | EW returns | research-only | Buy-and-hold EW baseline for CFR comparison | |
| CFR | `TrailingBestGreedy` | `apps/cfr/src/cfr/baselines.py:141` | baseline | menu + history | greedy best-arm-so-far | research-only | "Pick yesterday's winner" naive bandit | |
| CFR | `NaiveUniform` | `apps/cfr/src/cfr/baselines.py:201` | baseline | menu | uniform mix | research-only | 1/N over actions | |
| CFR | Deep CFR walkforward | `apps/cfr/src/cfr/deep_walkforward.py` | `cfr.cli deep` | menu, state vec, RegretNet | per-window policy | research-only | Deep CFR variant (group 8 net + this harness) | |

## Orphans / hard-to-classify

| name | location | why orphan |
|---|---|---|
| `softmax_weights`, `select_top_n_matrix`, `apply_position_cap` | `packages/portfolio/src/ss_portfolio/weights.py:9,25,57` | Portfolio primitives — turn scores into weights with caps. Not strategies themselves; consumed by every group above. |
| `apply_spread_mask`, `apply_nan_mask` | `packages/portfolio/src/ss_portfolio/screening.py:18,34` | Liquidity / NaN gates applied universally before weighting. |
| `vbt_backtest`, `build_strategy`, `block_sharpe_with_costs`, `studentized_sharpe_diff` | `packages/portfolio/src/ss_portfolio/{backtest.py,bt_helpers.py,sharpe.py,sharpe_diff_smooth.py}` | Eval/scoring infrastructure, not strategies. |
| `_load_vix_series`, `build_state_features`, `build_action_features` | `apps/critic/src/critic/features.py` | Feature builders for the critic stack — produce inputs to group-8 trainers. |
| `Triple` + dataset loaders | `apps/critic/src/critic/dataset.py` | Cross-app (state, action, reward) data assembly for the critic — pulls from factor / vol / gate / pairs outputs into a unified training panel. Operationally a meta layer over groups 5/6/8, but no weight output of its own yet. |
| `simulate_irregular_daily_pnl`, `simulate_fixed_horizon_daily_pnl`, `simulate_oracle_daily_pnl` | `apps/factor/src/factor/horizon.py:73,259,357` | Trade-cadence / horizon simulators around an upstream scorer; not strategies themselves. |
| `long_short_net_returns` | `apps/lie/src/lie/longshort.py:15` | Long-short return-stream accounting primitive. |
| `forward_max_drawdown` | `apps/gate/src/gate/target.py:22` | Target builder for the gate predictor. |
| `forward_iv_rv_gap` | `apps/vol/src/vol/target.py:18` | Target builder for the vol predictor. |
| `ManifoldMapper` | `apps/lie/src/lie/manifold.py:32` | Embedding builder consumed by `TimelessPredictor`. |
| `kalman_cwt.py`, `rnn_cwt.py` | `apps/notebook/src/ss_notebook/` | Scratch CWT-encoder explorations; no portfolio output wired in. |
| `metric_correlation_diagnostic`, `smoke_forward_skip`, `prep_crypto_universe` | `apps/factor/scripts/` | Diagnostics / data prep, not strategies. |

## Methodology notes

- Grep patterns used (in order): `def weights_|def target_weights|def signal_|def score_`, `class .*Strategy|.*Scorer|.*Signal|.*Allocator`, `STRATEGIES\s*=|--strategy|.strategy ==`, then per-app `^def |^class ` over `src/*.py`, then `apps/<app>/scripts/` enumeration via `find`.
- Dispatch identification:
  - **regime**: `argparse choices=['regime','scalogram','rsi']` (`cli.py:41`) + `STRATEGIES` tuple (`trainer.py:340`) + `inference.py` if/elif.
  - **relational**: `SUPPORTED_STRATEGIES` tuple (`persist.py:37`) + `inference.py` if-chain (l.117-172).
  - **lie**: `SUPPORTED_STRATEGIES=('hrp',)` + `inference.py:63`.
  - **vol** / **dca**: single canonical checkpoint type, no strategy enum — `run_live` dispatches directly.
  - **cfr**: action menu is a runtime-composed object (`ActionMenu`); strategies are the menu's `BaseMode` implementations.
  - **factor / critic / gate / pairs**: research-only — no `live.py`. Selection is by import + script entrypoint.
- Ambiguity resolution:
  - CFR `TopKMode` is parametrized by a `score_kind` string; I list each canonical `score_kind` as a separate strategy because they encode distinct ranking rules (momentum vs reversal vs low_vol etc.). This inflates the CFR row count but reflects the code's intent.
  - Effective-rank gate (`gross_exposure_modulator`) is plumbed through `lie.inference` as an *opt-in* modulator on HRP — counted under group 6 (gates) with a cross-reference to group 9 (HRP).
  - Replay reconstruction heads aren't portfolio strategies, but they produce the SSL backbone that scorers consume — included under group 8 with that caveat.
  - Several `apps/factor/scripts/modal/*.py` files are remote shims for in-tree `factor/src/factor/*.py` strategies (e.g. `modal/train_indicator.py` → `indicator_features.py`); not double-counted.
  - The "meta-allocator" bucket is largely empty in the committed tree — the active meta-allocator experiments live in untracked root-level Python files (`.research-*`, `apps/docs/scripts/per_window_meta_allocator.py`, `ensemble_discovery.py`) per the dirty git status, which are out of scope for a "committed code-truth" census.
- Truncation: no group exceeded 30 entries. CFR (group 10) at 18 entries is the largest after collapsing baselines + variants.
