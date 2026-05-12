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
| [`pairs`](pairs.md) | Active (v0 confirmed-null) | Pair-spread mean reversion. Engle-Granger screening + classical z-score trade rules. Numpy + statsmodels. |
| [`vol`](vol.md) | Active (v0 inconclusive, 5/5 pos) | Implied vol surface predictor. Skew/smile/IV-HV/OI/VIX-spread features → forward IV-RV gap. Numpy OLS. |
| `lie` | Active | Lie-group / hierarchical-network arc — three sub-arcs: (v1) HRP + effective-rank regime indicator on correlation spectrum; (v2) timeless 26-D market-state fingerprint + manifold + kNN forward-return predictor; (v3) per-ticker shape-feature cross-sectional kNN (`+3.75 t-stat ticker-only`, strongest result). The market-state-vector path is the **market-internal regime classifier** sibling to the macro classifier `packages/macro` would feed. |
| [`cfr`](cfr.md) | Active (Phase 1 partial-OOS 2026-05-12) | Deep CFR meta-allocator across existing strategy menu. Tabular CFR over `(infoset, action)` at Phase 1; deep CFR over a multi-modal encoder at Phase 2+. **Phase 1 (6 windows, full stooq_us_long, 25y): CFR beats trailing-best-greedy by +0.609 Sharpe in 6/6 windows (PASS), but ties naive uniform mix and undershoots passive EW by 0.09. Menu is the binding constraint — Phase 2 = add 13F-imitation modes.** |
| [`notebook`](notebook.md) | Active | Jupyter playground + scalogram visualizer CLIs (`ss-scalogram`, `ss-scalogram-video`). |
| `v1` | Parked | Legacy single-ticker workflow + aiohttp web service. |
| `docs` | Active | This Material for MkDocs site. |

## Input bundle per app

What each app actually consumes as its model input — the contract
between the data layer and the head. Documented per-app on each
linked page; this table is the cross-app index.

| App | Input bundle | Source |
| --- | --- | --- |
| [`regime`](regime.md) | Causal Ricker CWT of **raw close** (not log-returns) over `LONG_SCALES`, windowed power means. | `ss_wavelets.causal_cwt` + `regime/trainer.py` |
| [`relational`](relational.md) | CWT fingerprints — **Ricker** default, **polar Morlet** for wider universes — via `extract_fingerprints(scales, channels_per_scale)`. Default scales `[5,7,10,12,21,26,50,90]`. | `relational/fingerprints.py`; see [polar Morlet finding](../findings/relational-morlet-failure.md) |
| [`factor`](factor.md) | Two parallel paths: (1) supervised-`cnn` backbone trained on the **polar Morlet bundle** (`K=96, F=105` → flat 5632); (2) `IndicatorGridConfig` 74-channel deterministic stack (30 RSI + 16 CCI + 6 vol + 18 MACD + 4 coherence). | `ss_features.load_backbone` / `factor.indicator_features` |
| [`replay`](replay.md) | **Polar Morlet + Gaussian companion + log-L2 amplitude** bundle (F=105 = 7 channels × 15 scales). Reconstruction targets: RSI / MACD / vol / CCI. | `ss_features.causal_polar_morlet_matrix`; see [decoder options](../findings/replay-decoders.md) |
| [`gate`](gate.md) | **10 trailing aggregate features** over the EW universe (vol, return, trailing drawdown, breadth, etc.). Numpy-only, no CWT. | `gate/aggregate.py::build_aggregate_features` |
| [`pairs`](pairs.md) | **Log-prices only** (close → `engle_granger_test(log_p_a, log_p_b)` → spread → z-score). No CWT, no indicators. | `pairs/cointegration.py` |
| [`vol`](vol.md) | **10 implied-vol surface features** (ATM / OTM / DOTM IV + skew + smile + multi-horizon IV-HV ratio + OI imbalance + VIX-spread + strike-spread). | `ss_iv` + `vol/features.py::build_vol_features` |
| `lie` | Two parallel paths: (a) **26-D market-state fingerprint** per bar — top-8 correlation eigenvalues (as fraction of trace) + participation ratio + cross-sectional mean/std/skew/kurt of vol-normalized returns at 5d/21d/63d + aggregate skew/kurt/tail-fraction + spectral gap. PCA-projected via `ManifoldMapper`, fed to `TimelessPredictor` (kNN with 60-day temporal-gap exclusion). (b) **Per-ticker shape features** for cross-sectional kNN; 168-D CWT bundle was head-to-head-tested and parked (`lie v4 shape > cwt`). | (a) `lie/state_builder.py::build_market_state` + `lie/manifold.py` + `lie/predictor.py`; (b) `lie/ticker_features.py::build_ticker_features` + `lie/cross_sectional.py` |
| [`cfr`](cfr.md) | **Tabular CFR over `(infoset, action)`**: infoset is `(trailing-vol-bucket, dispersion-bucket)` — 3×3 = 9 cells; action is `(mode, gross-bucket)` over universe-agnostic modes (EW, top-K momentum / reversal / low-vol / high-vol) × {0, 0.5, 1.0, 2.0} gross. 16 actions × 9 infosets at Phase 1. Phase 2+ swaps the table for a regret-net + policy-net over a multi-modal encoder. | `cfr/menu.py` + `cfr/state.py` + `cfr/tabular.py` |
| [`notebook`](notebook.md) | Visualization only — renders the same CWT primitives the other apps consume. | n/a |
| `v1` | Legacy single-ticker indicators in `v1/util/indicators.py` (RSI n=14 etc., pre-`ss_indicators`). Parked. | n/a |

## Prediction target per app

What each app is trying to *predict* (or, for the search-based apps,
optimize). The split between supervised-target apps and
search/heuristic apps is a real architectural divide — only the
former have a single number we can call an IC.

| App | Prediction target | Resolution | How it's scored |
| --- | --- | --- | --- |
| [`factor`](factor.md) | Cross-sectional forward 20d log-return rank | Per ticker, per bar | Pearson rank-IC (train); block-Sharpe (eval) |
| [`gate`](gate.md) | Forward 20d max drawdown of EW universe aggregate | Single time series | OLS R² / Pearson r; downstream gated-arm Sharpe alpha |
| [`pairs`](pairs.md) | Pair-spread mean reversion (z-score crossing trigger, not a regression target) | Per pair, per bar | State-machine PnL (entry ±2σ, exit ±0.5σ) |
| [`vol`](vol.md) | Forward 20d IV-vs-realized-vol gap | Per surface cell, per bar | OLS Pearson r per cell; quantile-gated short-vol Sharpe |
| [`replay`](replay.md) | Reconstructed technical indicators (RSI / MACD / vol / CCI / price) from causal CWT slices | Per ticker, per bar, per indicator | MSE per indicator (SSL reconstruction loss) |
| `lie` | (v2) forward universe-aggregate return via kNN over 26-D market-state fingerprint; (v3) forward per-ticker return via kNN over per-ticker shape features | (v2) per bar; (v3) per ticker per bar | Information coefficient (Spearman-style rank correlation) |
| [`cfr`](cfr.md) | *Counterfactual regret per `(infoset, action)`* over a forward block. Trained policy = regret matching on cumulative regret. | Per rebal, per action | Mean per-window val Sharpe minus trailing-best-greedy Sharpe |
| [`regime`](regime.md) | *No point-prediction target.* Optuna search over CWT-power-divergence weight functions; objective is walk-forward portfolio Sharpe | Portfolio | vectorbt walk-forward Sharpe |
| [`relational`](relational.md) | *No point-prediction target.* Heuristic scorers in CWT fingerprint space (empirical, gmm, analog, farthest, diversified, velocity) | Portfolio | Walk-forward Sharpe |

A few observations the table makes visible:

- **Two distinct architectural classes.** `factor` / `gate` / `pairs` /
  `vol` / `replay` / `lie` train against an explicit target with a
  loss function (or kNN over a labeled embedding). `regime` and
  `relational` are search-based / heuristic — they construct
  portfolios directly with no learnable target. The "rank-IC ceiling"
  debate only applies to the former.
- **Aggregation level changes the IC ceiling.** Per-ticker targets
  (`factor`, per-ticker `lie` v3, per-cell `vol`) cap at +0.005 to
  +0.012 in the cross-sectional-return case; aggregate / scalar
  targets (`gate` forward-DD, universe-wide `lie` v2) carry more
  signal because they integrate noise across names — `gate` hit val
  Pearson r +0.26 on a single time series, vs +0.012 spread across
  297.
- **Two regime-classifier types coexist.** `apps/lie`'s
  `build_market_state` builds a **market-internal** regime
  fingerprint from correlation-spectrum + cross-sectional dispersion;
  `packages/macro`'s `load_macro_panel` is the **macro-economic**
  regime panel from FRED (+ Stooq gold). Both belong in the regime
  state vector for any meta-allocator (e.g. the
  [`apps/cfr` TODO](../TODO/apps-cfr.md)) — they tap orthogonal
  axes of "what regime are we in" (intra-universe collapse vs
  policy/credit/vol cycle position).
- **The pivot-arc widening.** Until 2026-05-09 the only supervised
  prediction target outside `replay`'s SSL reconstruction was
  `factor`'s cross-sectional return rank. The arc that landed
  `gate` / `pairs` / `vol` added three orthogonal targets;
  `apps/lie`'s v2 (universe-aggregate via market-state-kNN) and v3
  (per-ticker via shape-kNN) round out the shortlist. See
  [`prediction-problem-pivot-arc`](../findings/prediction-problem-pivot-arc.md).
- **What no app currently predicts:** absolute return level,
  individual-stock volatility, individual-stock drawdown, sector
  rotation, regime label as a categorical output, forward-event
  probability (earnings / M&A / Fed surprise), or order-flow
  imbalance. The macro-classifier-as-categorical-head sketched in
  the [`apps/cfr` TODO](../TODO/apps-cfr.md) and the
  [macro v2 plan](../findings/macro-regime-diagnostic.md) would be
  the most natural addition.
