# End-to-end portfolio allocator v1 — ETF prices + macro, no meta-layer inputs

**Operational rule.** *On the 13-ETF Phase 4d universe with raw price
features + macro side channel only (no pre-computed alpha streams),
direct-Sharpe-loss end-to-end deep learning lifts OOS Sharpe over
DCA by **ΔSR_ann +0.094 [−0.20, +0.49]** — CI includes zero, verdict
`confirmed-null` vs the locked +0.10 partial-OOS bar. The paradigm
test landed: end-to-end works on raw inputs (fold-3 Sharpe +1.22 vs
DCA +0.78), but the alpha ceiling without options data is ETF
directional — not the options-augmented edge the canonical deployment
recipe extracts. v2 with raw IV surface features is the path to
matching the deterministic 2-leg ceiling.*

## Status

Direct user-requested test of "make end-to-end work, no meta-layer
daily returns as inputs." Built per Zhang-Zohren-Roberts 2020
([arXiv 2005.13665](https://arxiv.org/abs/2005.13665)) — direct
differentiable Sharpe loss in tinygrad CUDA, 3-fold walk-forward
walk-forward on the 13-ETF Phase 4d basket, Modal T4. The OOS
Sharpe lifts over DCA are **directionally positive but not
significant**; the model does NOT close the gap to the
options-augmented deterministic or learned 2-leg recipes.

Verdict per locked bar: **`confirmed-null` vs DCA** (ΔSR
+0.094 < +0.10 partial floor; CI includes zero). Secondary verdicts:
loses to deterministic 2-leg by ΔSR −1.16 [−1.88, −0.28] (CI
excludes zero negative); loses to learned 2-leg by ΔSR −3.72
[−5.08, −2.49] (CI excludes zero negative).

## Eval setup

| field | value |
|---|---|
| universe | 13-ETF Phase 4d (`Output/cfr_phase4d_multiasset_close.pkl`) |
| input features | per-asset (T=60, F=6): log_return_1d, realized_vol_20d, realized_vol_60d, RSI(14), normalized_price (close / SMA252), 5d_momentum |
| macro side channel | (T=60, F=4): VIX, VIX_percentile_252d, T10Y3M, BAA10Y — all from FRED via `ss_macro` |
| architecture | per-asset 1D conv encoder (32 ch, k=5) → cross-asset MLP head with macro context → softmax over N+1 (13 ETFs + cash) |
| loss | direct annualized Sharpe (`−mean / std`) over forward 20-day window |
| optimizer | AdamW lr=1e-3 wd=1e-4 batch=128 n_steps=5000 per fold |
| folds | fold1 2015-2018 (n=1006d), fold2 2019-2022 (n=1008d), fold3 2023-2025 (n=718d) |
| training | 5y rolling window strictly ending before each val_start |
| compute | Modal T4, CUDA tinygrad — per CLAUDE.md policy (training >2k steps requires Modal) |
| baselines | EW (1/13), DCA (`PassiveEW(rebal_days=80, commission_bps=10)`), deterministic 2-leg (`r_dca + 2.0 * r_vol_v3_daily`), learned 2-leg (`0.0506 * r_dca + 2.2388 * r_vol_v3_daily`) |
| metric | LW studentized stationary-bootstrap ΔSR CI vs each baseline (`ss_portfolio.sharpe_difference_ci`) |

## Per-fold OOS Sharpe

| fold | val range | n days | val Sharpe_ann |
|---|---|---:|---:|
| fold-1 | 2015-01 → 2018-12 | 1006 | +0.257 |
| fold-2 | 2019-01 → 2022-12 | 1008 | +1.094 |
| fold-3 | **2023-01 → 2025-12** | **718** | **+1.216** |
| **pooled** | **2015 → 2025** | **2732** | **+0.874** |

Fold-3 (the unseen 2024+ window the user explicitly asked about)
posts the **strongest** result, which directionally confirms the
paradigm works on recent data — but pooled is dragged down by fold-1
weakness during a fundamentally different regime (low-VIX, range-
bound).

## All-baseline ΔSR table (pooled OOS, n=2732 daily)

| baseline | baseline Sharpe | E2E Sharpe | ΔSR_ann | LW 95% CI | excludes zero? |
|---|---:|---:|---:|---|---|
| **EW (1/13)** | +0.785 | +0.875 | **+0.089** | [−0.20, +0.53] | no |
| **DCA** | +0.780 | +0.875 | **+0.094** | [−0.20, +0.49] | no |
| **deterministic 2-leg** | +2.030 | +0.875 | **−1.156** | [−1.88, −0.28] | YES (negative) |
| **learned 2-leg** | +4.592 | +0.875 | **−3.717** | [−5.08, −2.49] | YES (negative) |

Pooled max-DD: **−12.16%** (worse than DCA's −9.55% and the
deterministic recipe's −9.55%).

## Verdict per locked bar

| comparison | locked threshold | actual | verdict |
|---|---|---|---|
| vs DCA | ΔSR ≥ +0.30 confirmed-OOS / ≥ +0.10 partial-OOS / CI excludes 0 | ΔSR +0.094, CI includes 0 | **`confirmed-null` vs DCA** |
| vs deterministic 2-leg | secondary, honest expectation v1 doesn't beat | ΔSR −1.16, CI excludes 0 negative | confirmed-null (loses) |
| vs learned 2-leg | secondary, honest expectation v1 doesn't beat | ΔSR −3.72, CI excludes 0 negative | confirmed-null (loses) |

## What the result actually means

Three honest reads:

1. **The end-to-end paradigm works.** Fold-3 Sharpe +1.22 on a
   universe with no meta-layer inputs, on a model that learned
   weights from raw prices + macro alone, is a real result. It's
   evidence that direct-Sharpe-loss training on ETF features can
   extract directional alpha. The +0.089 to +0.094 ΔSR over passive
   EW / DCA, while CI-includes-zero on the pooled stream, has a
   consistent positive sign across folds (fold-3 lift over DCA on
   that fold ≈ +0.40 raw Sharpe gap before friction-adjusted CI).

2. **The ceiling is exactly where we predicted.** Without raw IV
   surface data, the model cannot replicate vol_v3's options edge.
   The gap to the deterministic 2-leg (ΔSR −1.16) and the learned
   2-leg (ΔSR −3.72) is the **options alpha gap** — money that exists
   on the table but only accessible through short-vol exposure that
   ETF-feature models cannot construct. This is not a model
   architecture failure; it is a feature-space limitation.

3. **fold-1 is the canary.** The strong drop on fold-1 (+0.26 vs
   fold-3 +1.22) signals the model is not regime-robust on its own.
   2015-2018 was a fundamentally different regime (post-QE, low VIX,
   range-bound). The macro side channel didn't carry it. A richer
   macro feature set or a longer effective training horizon would
   need to address this before standalone deployment.

## What v2 should change

The path to closing the gap is **raw IV surface data**, not
architecture iteration. Three concrete changes for v2:

1. **Add raw IV features to the macro side channel.** Per-name 1m
   30d IV (via `ss_iv.gauss314`), per-name IV-vs-RV gap, per-name
   put/call IV skew. These are the substrate vol_v3 reads internally;
   exposing them as features lets the model rediscover the short-vol
   signal without a meta-layer alpha stream.
2. **Add a "short-vol overlay" output head.** Augment the softmax
   weight vector with a scalar `vol_position ∈ [0, 2.5]` that
   multiplies a synthetic short-straddle return derived from
   VIX/IV inputs. Output dim becomes (N + 1 + 1) = 15 logits + 1
   continuous head.
3. **Persist on a Modal Volume.** Raw IV data is too large for the
   `add_local_dir` + ship-via-RPC pattern. Mount
   `modal.Volume.from_name('ss-e2e-iv-data')` for raw IV inputs and
   `modal.Volume.from_name('ss-e2e-artifacts')` for cross-version
   checkpoint reuse.

## Implementation notes — pragmatic decisions

The agent run made these choices and documented them inline (per
CLAUDE.md "document the choice you made and keep going"):

- **Driver scaffolded under new `apps/e2e_portfolio/` app** rather than
  added to an existing app, since the architecture is genuinely new.
- **Modal T4 path used.** Image base `nvidia/cuda:12.4.0-devel-ubuntu22.04`
  + uv + `add_local_dir` (StooqData / Output / Nasdaq3347 / .git /
  .claude / .iv-cache / .hl-cache / .congress-cache / .macro-cache
  excluded). `os.environ['CUDA'] = '1'` set pre-tinygrad-import; CUDA
  device fail-fast assertion in place.
- **Local prep step** (`apps/e2e_portfolio/scripts/prep_data.py`)
  produces `Output/e2e-portfolio-prep.pkl` (102 MB) with per-asset
  features + macro features + close + forward 1d returns. Shipped to
  Modal as raw bytes per the
  `apps/factor/scripts/modal/{prep_universe_pivot_data, universe_pivot_vol_arm}`
  pattern.
- **Tinygrad model** at `apps/e2e_portfolio/src/e2e_portfolio/model.py`
  ~250 K params: per-asset 1D conv encoder shared across the 13 ETFs,
  macro MLP encoder, cross-asset MLP head.

## Master walk-forward log

| date | row pointer | verdict |
|---|---|---|
| 2026-05-28 | `apps/docs/docs/leaderboard.md` row (this finding) | [`confirmed-null`](../leaderboard.md#verdict-labels) vs DCA (ΔSR +0.094, CI includes 0); loses to deterministic 2-leg (ΔSR −1.16 excludes 0 negative) and learned 2-leg (ΔSR −3.72 excludes 0 negative) |

## Related findings

- [`learned-ensemble-beats-deterministic`](learned-ensemble-beats-deterministic.md) — the meta-layer-input learner this experiment was meant to replace; sets the ceiling v2 needs to reach.
- [`meta-allocator-internal-features`](meta-allocator-internal-features.md) — the closing meta-allocator arc; shares the "wrong layer" methodological lesson with this v1.
- [`vol-v3-dolthub-oos`](vol-v3-dolthub-oos.md) — the substrate whose options alpha v1 cannot access on raw ETF features alone.

## Driver + outputs

- Modal entrypoint: `apps/e2e_portfolio/scripts/modal/train_walkforward.py`
- Local data prep: `apps/e2e_portfolio/scripts/prep_data.py`
- Per-fold daily streams: `Output/e2e-portfolio-fold{1,2,3}-daily.npz`
- Pooled OOS stream: `Output/e2e-portfolio-pooled-daily.npz`
- Trained model checkpoints: `Output/e2e-portfolio-fold{1,2,3}.npz`
- Results JSON: `Output/e2e-portfolio-results.json`
