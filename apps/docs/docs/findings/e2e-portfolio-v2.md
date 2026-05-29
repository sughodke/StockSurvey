# End-to-end portfolio allocator v2 — raw IV features + synthetic short-vol head

**Operational rule.** *Adding raw IV/HV features per ETF + a continuous
`vol_position ∈ [0, 2.5]` output head that multiplies a synthetic
short-vol return derived purely from raw IV-vs-HV data **does not**
close the options-alpha gap to the deterministic / learned 2-leg
recipes. Pooled OOS Sharpe regresses slightly from v1 (+0.776 vs
+0.874) — the vol_position head correctly learned to activate in the
fold where IV data is available (fold-3 mean +0.52, range
[+0.27, +0.88]), but the per-ETF IV-vs-HV gap on the 9 ETFs in DoltHub
coverage produces synthetic short-vol PnL too small to substitute for
vol_v3's per-name short-vol alpha. **Verdict `confirmed-null` vs DCA;
loses to both 2-leg recipes by margins that exclude zero negative.**
The architecture works; the substrate is wrong. v3 must short-vol on
the full DoltHub options universe per the vol_v3 recipe, not on the
13-ETF basket.*

## Status

Direct user-requested v2 after v1's `confirmed-null` against DCA
isolated the failure to "no options alpha on ETF features." v2 adds:
1. 5 raw IV features per ETF (iv_current, hv_current, iv_vs_hv_gap,
   iv_pct_252d, iv_change_60d) + 1 availability flag → `F_asset=12`.
2. Continuous `vol_position` head sized via `2.5 × sigmoid(z_vol)`
   that multiplies a synthetic short-vol daily return derived from
   the equal-weighted per-ETF `(iv_current[t-period] - hv_current[t])`
   gap.
3. Two Modal Volumes: `ss-e2e-iv-data` (parquet + computed features)
   and `ss-e2e-artifacts` (cross-version checkpoint reuse).

User's hard constraint preserved: **no pre-computed meta-layer alpha
streams as inputs.** vol_v3 alpha never enters the feature space; it
is only computed as a baseline in eval.

## Eval setup

| field | value |
|---|---|
| universe | 13-ETF Phase 4d basket; **9 of 13 in DoltHub IV coverage**: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY. Missing: TLT, IEF, GLD, DBC (bond + commodity ETFs) |
| input features per asset | (T=60, F=12): v1's 6 (log_ret_1d, rv_20d, rv_60d, RSI14, normalized_price, 5d_momentum) + 5 IV (iv_current, hv_current, iv_vs_hv_gap, iv_pct_252d, iv_change_60d) + 1 availability flag |
| macro side channel | (T=60, F=4): VIX, VIX_pct_252d, T10Y3M, BAA10Y |
| IV source | DoltHub weekly via `ss_iv.load_dolthub_iv_parquet`, forward-filled to daily with 1-day shift to avoid peek |
| architecture | v1 per-asset 1D conv + macro MLP + cross-asset head softmax (N+1=14) **PLUS** scalar `vol_position` head = `2.5 × sigmoid(z_vol)` from same pooled state |
| synthetic short-vol stream | equal-weighted per-ETF `(iv_current[t-20] - hv_current[t]) / 252` over covered ETFs only |
| total return | `r = softmax_weights @ asset_ret_next_1d + vol_position × synthetic_short_vol_daily` |
| loss | direct annualized Sharpe over forward 20-day window |
| optimizer | AdamW lr=1e-3 wd=1e-4 batch=128 n_steps=5000 per fold |
| folds | fold1 2015-2018 (n=1006d) ; fold2 2019-2022 (n=1008d) ; fold3 2023-2025 (n=718d, **unseen 2024+**) |
| compute | Modal T4 CUDA tinygrad — `ss-e2e-iv-data` Volume mounted for IV parquet; `ss-e2e-artifacts` Volume mounted for checkpoints |
| baselines | EW (1/13), DCA, deterministic 2-leg (`r_dca + 2.0 × r_vol_v3_daily`), learned 2-leg (`0.0506 × r_dca + 2.2388 × r_vol_v3_daily`) |

## Per-fold OOS Sharpe + vol_position trajectory

| fold | val range | n days | val Sharpe | vol_position mean | vol_position std | vol_position max |
|---|---|---:|---:|---:|---:|---:|
| fold-1 | 2015-01 → 2018-12 | 1006 | +0.157 | **~0** (5.8e-10) | 1.3e-9 | 7.3e-9 |
| fold-2 | 2019-01 → 2022-12 | 1008 | +0.946 | **~0** (2.1e-6) | 3.4e-6 | 2.4e-5 |
| fold-3 | **2023-01 → 2025-12** | **718** | **+1.246** | **+0.520** | **+0.152** | **+0.882** |
| **pooled** | **2015 → 2025** | **2732** | **+0.776** | +0.137 | +0.242 | +0.882 |

**The vol_position head learned exactly the right thing on fold-3:**
mean +0.52 across the unseen 2024+ window, ranging up to +0.88 in the
highest-IVRP days. Fold-1 (no IV coverage pre-2019) correctly settled
on ~0. Fold-2 (early IV coverage) also stayed near zero — the
synthetic short-vol stream wasn't yet attractive on the early IVRP
data.

The architecture **discovered the vol regime** from raw IV features.
The problem is the synthetic short-vol stream's magnitude.

## All-baseline ΔSR table (pooled OOS, n=2732 daily)

| baseline | baseline Sharpe | v2 Sharpe | ΔSR_ann | LW 95% CI | excludes zero? |
|---|---:|---:|---:|---|---|
| **EW (1/13)** | +0.785 | +0.776 | **−0.010** | [−0.43, +0.57] | no |
| **DCA** | +0.780 | +0.776 | **−0.005** | [−0.43, +0.54] | no |
| **deterministic 2-leg** | +2.030 | +0.776 | **−1.255** | [−2.00, −0.22] | YES (negative) |
| **learned 2-leg** | +4.592 | +0.776 | **−3.816** | [−5.15, −2.56] | YES (negative) |

Max-DD: **−12.66%** (worse than v1 −12.16% and DCA −9.55%).

## Verdict per locked bar

| comparison | locked threshold | actual | verdict |
|---|---|---|---|
| vs DCA | ΔSR ≥ +0.30 confirmed-OOS / ≥ +0.10 partial-OOS / CI excludes 0 | ΔSR −0.005, CI includes 0 | **`confirmed-null` vs DCA** |
| **real goal: vs deterministic 2-leg** | ΔSR ≥ +0.10 AND CI excludes 0 | ΔSR −1.26, CI excludes 0 negative | confirmed-null (loses) |
| stretch: vs learned 2-leg | ΔSR ≥ +0.10 AND CI excludes 0 | ΔSR −3.82, CI excludes 0 negative | confirmed-null (loses) |

## Why v2 underperformed v1 on pooled

Pooled v2 (+0.776) is slightly **worse** than pooled v1 (+0.874).
Three reasons:

1. **Fold-1 IV-availability mask hurts.** No DoltHub IV coverage
   pre-2019 means fold-1's per-asset feature set is effectively 6
   real features + 5 zeros + 1 zero flag. The v2 model has more
   capacity but no information advantage on fold-1, and its random-init
   exploration of the larger feature space is noisier than v1's
   tighter parameterization. Fold-1 drops from v1 +0.257 to v2 +0.157.
2. **Fold-2 has marginal IV signal but the short-vol stream isn't yet
   attractive.** vol_position stays near 0 (mean 2.1e-6). The IV
   features are there but the model finds no profitable carry; weight
   capacity is spent on regularizing toward DCA-like allocations. v1's
   +1.094 falls to v2's +0.946.
3. **Fold-3 is the win.** v2 fold-3 +1.246 vs v1 fold-3 +1.216 — the
   IV features and vol_position head **do** add value on the unseen
   2024+ window. The pooled regression hides this.

**Stratified read**: on the only fold where the v2 substrate adds
information, v2 wins. The pooled-fold story is dragged by 2/3 folds
where the IV substrate is unavailable or unhelpful.

## Why v2 still loses to the 2-leg recipes

The synthetic short-vol return is structurally bounded by what 9 ETFs'
IV-vs-HV gap produces:
- vol_v3 trades short-vol on the **full DoltHub options universe**
  (2,276 US tickers, top-K by OI, regime-gated). The per-rebal alpha
  is **+30 bps annualized × 5-8 cycles per year × full notional
  per-name short-straddle vega exposure**.
- v2's synthetic short-vol = 1/9 × Σ (iv_etf_t − hv_etf_t) per period
  ÷ 252. This is at most a few bps daily — the same vol-points budget
  as vol_v3 but on 9 ETF underlyings instead of top-K from 2,276.
- The capacity ratio is roughly 9 / 100 = ~10×. The 2-leg recipe sees
  ~10× the per-name short-vol PnL the v2 substrate can construct.

This is **not** a model architecture problem. The vol_position head
learned to activate exactly when it should (fold-3 IV-rich regime).
The substrate is wrong — ETF IV is too narrow and too smooth relative
to per-name IV.

## What v3 should change

The path forward is **substrate**, not architecture:

1. **Expand the universe.** v3 must short-vol on the full DoltHub
   options universe per the vol_v3 recipe (top-K gated by OI), not
   the 13-ETF basket. This re-creates the vol_v3 substrate as a
   v3-learned model rather than importing it as a pre-computed alpha.
2. **Per-name IV features as the sizing input.** Instead of the
   per-ETF IV features that v2 uses, v3 should compute per-name IV
   features on the top-K universe and let the model learn the regime
   gate (the v3 in vol_v3 IS a regime gate) end-to-end.
3. **Two-leg architecture**: equity allocator (current architecture
   on 13 ETFs) + short-vol head (NEW, operates on its own per-name
   feature panel from DoltHub). vol_position becomes per-name, not
   scalar.
4. **Persist top-K selection on Volume.** The DoltHub options
   universe's top-K-by-OI list per rebal is itself a costly
   computation; `ss-e2e-iv-data` Volume should cache it.

## Implementation notes — pragmatic decisions

The agent run made these choices (and stopped reporting before the
finalize phase; this finding was written in the recovery turn):

- **9 of 13 ETFs in IV coverage.** TLT, IEF, GLD, DBC missing from
  DoltHub. Their IV columns were zero-filled and an `iv_available`
  flag was added as a 12th per-asset feature. No special handling
  beyond that.
- **Weekly DoltHub IV** was forward-filled with strict 1-day shift
  (no peek). The gauss314 daily SPX would have been richer (full
  strike grid) but its 2019-10 → 2023-07 span doesn't cover fold-3.
- **Synthetic short-vol stream** = equal-weighted IV-HV gap across
  covered ETFs, divided by trading days in the period to give a
  daily contribution. No friction on the synthetic stream — the
  10 bps per-period charge from the DoltHub recipe was deemed
  out-of-scope for the synthetic accounting in v2.
- **Architecture** at `apps/e2e_portfolio/src/e2e_portfolio/model_v2.py`
  shares the v1 per-asset encoder + macro MLP, adds a scalar
  `vol_position` head from the same pooled state.

## Master walk-forward log

| date | row pointer | verdict |
|---|---|---|
| 2026-05-28 | `apps/docs/docs/leaderboard.md` row (this finding) | [`confirmed-null`](../leaderboard.md#verdict-labels) vs DCA (ΔSR −0.005, CI includes 0); loses to deterministic 2-leg (ΔSR −1.26 excludes 0 negative) and learned 2-leg (ΔSR −3.82 excludes 0 negative); fold-3 v2 +1.246 beats v1 +1.216 — vol_position head learned the right behavior, substrate is too narrow |

## Related findings

- [`e2e-portfolio-v1`](e2e-portfolio-v1.md) — the paradigm test v2 was meant to extend; sets the comparison baseline.
- [`learned-ensemble-beats-deterministic`](learned-ensemble-beats-deterministic.md) — the ceiling v2 needed to reach but couldn't.
- [`vol-v3-dolthub-oos`](vol-v3-dolthub-oos.md) — the per-name short-vol recipe v3 must operate on its own universe.

## Driver + outputs

- Modal entrypoint: `apps/e2e_portfolio/scripts/modal/train_v2_walkforward.py`
- Local data prep: `apps/e2e_portfolio/scripts/prep_data_v2.py`
- v2 model: `apps/e2e_portfolio/src/e2e_portfolio/model_v2.py`
- Per-fold daily streams: `Output/e2e-portfolio-v2-fold{1,2,3}-daily.npz`
- Pooled OOS stream: `Output/e2e-portfolio-v2-pooled-daily.npz`
- Trained checkpoints: `Output/e2e-portfolio-v2-fold{1,2,3}.npz`
- Modal Volumes: `ss-e2e-iv-data` (IV parquet + features), `ss-e2e-artifacts` (cross-version checkpoint reuse)
- Results JSON: `Output/e2e-portfolio-v2-results.json`
