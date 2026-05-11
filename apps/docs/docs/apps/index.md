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
| `lie` | Active | Shape-feature research arc (cross-sectional, manifold experiments). |
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
| `lie` | **Shape features** per ticker (cross-sectional H=21); 168D CWT bundle parked as `lie v4 shape > cwt`. | `lie/shape.py` |
| [`notebook`](notebook.md) | Visualization only — renders the same CWT primitives the other apps consume. | n/a |
| `v1` | Legacy single-ticker indicators in `v1/util/indicators.py` (RSI n=14 etc., pre-`ss_indicators`). Parked. | n/a |
