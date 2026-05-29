# End-to-end portfolio v3 — one universe, two heads, no meta-layer inputs

**Operational rule.** *On the unified DoltHub options universe
(K=200 by data-coverage) with v3's two-head architecture (per-name
equity + per-name short-vol + scalar vol_scale) trained end-to-end
on direct-Sharpe loss, the pooled OOS Sharpe is **−0.68** —
catastrophically worse than v1 (+0.87) and v2 (+0.78). The
universe expansion from 13 ETFs to 200 DoltHub names introduces
single-name concentration risk on the equity head (fold-3 NVDA =
26.9% of vol weight; fold-2 equity head long FLR/OXY/APA/KSS junk
eats COVID drawdown) AND the vol head correctly turns off during
COVID (`vol_scale_mean = 0.13` on fold-2) but has no symmetric
long-vol hedge to actually profit from the regime shift. **Action-
space asymmetry is now the load-bearing failure mode** — v3.5 (add
long-vol head over free VIXY substrate) is the minimal isolated
test to fix this.*

## Status

`confirmed-null` per the locked pre-reg bar — every baseline CI
excludes 0 negative. v3 regressed materially from v1 and v2 on
pooled OOS and fold-3, AND on max-DD (−37.83% vs ~−12% in v1/v2).
The architecture works (vol head learned regime-correct behavior on
fold-2) but the action space and universe selection combine into a
structurally worse model than v1/v2 on the same walk-forward folds.

## Eval setup

| field | value |
|---|---|
| universe | DoltHub IV parquet, top-K=200 by IV+price history coverage; 9 of 13 Phase 4d ETFs in coverage + ~190 single names |
| per-name features | (T=60, F=11): v1's 6 price/return features + 5 IV/HV features (iv_current, hv_current, iv_vs_hv_gap, iv_pct_252d, iv_change_60d); availability mask |
| macro side channel | (T=60, F=4): VIX, VIX_pct_252d, T10Y3M, BAA10Y |
| architecture | per-name 1D conv encoder (32 ch, k=5) shared across K names; macro MLP encoder; cross-asset MLP head with three output heads: equity softmax over (K+cash), per-name short-vol logits with top-K_active=50 mask, scalar `vol_scale = 5·sigmoid(z_vol)` |
| loss | direct annualized Sharpe |
| optimizer | AdamW lr=1e-3 wd=1e-4 batch=128 n_steps=5000 per fold |
| folds | fold-1 2015-2018 (n=1033 daily); fold-2 2019-2022 (n=1034); fold-3 2023-2025-12 (n=778 unseen 2024+) |
| compute | Modal T4 CUDA tinygrad, detached run + `ss-e2e-iv-data` Volume mount for 1.08 GB prep pickle (1 GB exceeded RPC, used volume reuse pattern) |
| baselines | EW (1/13), DCA, vol_v3 standalone, deterministic 2-leg, learned 2-leg |
| metric | LW studentized stationary-bootstrap ΔSR CI vs each baseline |

## Per-fold OOS

| fold | val range | n days | val Sharpe (held-out daily) | vol_scale mean | top-10 vol names |
|---|---|---:|---:|---:|---|
| fold-1 | 2015-01 → 2018-12 | 1033 | +0.828 | 1.50 (std 0.47, range [0.74, 2.53]) | FCX, UAA, KO, NWL, UA, PEP, CPB, MAT, JNJ, EFX |
| fold-2 | 2019-01 → 2022-12 | 1034 | **−0.640** | **0.13** (correctly off during COVID) | FLR, OXY, APA, NVDA, KSS, KHC, ANET, PG, MAT, UA |
| fold-3 | **2023-01 → 2025-12** | **778** | **−1.119** | 2.77 (std 2.44, max 5.0) | **NVDA (26.9%)**, CPRI, KO, LMT, CME, INCY, APH, FLR, FTNT, FCX |
| **pooled** | **2015 → 2025** | **2845** | **−0.678** | 1.35 (std 1.68) | — |

Pooled max-DD: **−37.83%** (much worse than v1's −12.16% and v2's
−12.66%).

## All-baseline ΔSR table (pooled OOS, n=2753)

| baseline | baseline Sharpe | v3 Sharpe | ΔSR_ann | LW 95% CI | excludes zero? |
|---|---:|---:|---:|---|---|
| DCA | +0.781 | −0.861 | **−1.642** | [−3.80, −0.33] | YES (negative) |
| vol_v3 standalone | +4.606 | −0.861 | **−5.467** | [−8.03, −3.62] | YES (negative) |
| deterministic 2-leg | +1.978 | −0.861 | **−2.840** | [−5.21, −1.19] | YES (negative) |
| learned 2-leg | +4.662 | −0.861 | **−5.523** | [−8.12, −3.70] | YES (negative) |

Every baseline CI excludes zero on the negative side. v3
underperforms even passive DCA by a margin too large to be noise.

## Per-fold pathology — what the model actually did

### fold-1 (2015-2018) — the only positive fold
- Vol mass spread thin (~0.5% per name on top-10) — model didn't
  concentrate. Defensive posture, modest +0.828 Sharpe.
- DoltHub IV coverage starts mid-fold; vol_scale = 1.50 mean reflects
  short-vol baseline activity.

### fold-2 (2019-2022) — COVID disaster despite regime-correct vol head
- **vol_scale mean 0.13** — model correctly identified COVID regime
  and turned off short-vol. This is the *regime-detection behavior we
  wanted*. The vol head learned the right thing.
- BUT: the equity head was simultaneously long FLR (distressed EPC),
  OXY/APA (energy beta, collapsed in 2020 oil crash), KSS (dying
  retail), KHC, MAT (declining toy company). These positions ate the
  full COVID drawdown.
- The asymmetry: model could turn vol head OFF but couldn't FLIP to
  long-vol to actually profit from the spike. Equity head also can't
  short. Net result: −0.640 Sharpe through a regime the model
  *correctly identified* but couldn't *exploit*.

### fold-3 (2023-2025) — concentration risk on data-driven names
- vol_scale mean 2.77 — aggressive short-vol.
- **NVDA = 26.9% of vol weight** — model bet huge on short NVDA vol
  during the AI rally. NVDA's vol spikes (Q3 2023, Q1 2024 earnings)
  blew up the position.
- v1 fold-3 was +1.216; v2 was +1.246; v3 is **−1.119**. The K=200
  single-name universe replaced v1/v2's ETF-diversified equity
  exposure with concentrated single-name bets. The model's data-
  driven selection picked names that happen to dominate IV mass but
  also carry the largest idiosyncratic risk.

## Verdict per locked bar

| comparison | locked threshold | actual | verdict |
|---|---|---|---|
| vs DCA | ΔSR ≥ +0.30 confirmed-OOS / ≥ +0.10 partial-OOS / CI excludes 0 | ΔSR −1.642, CI excludes 0 (negative) | **`confirmed-null` vs DCA — much worse than null** |
| **real goal: vs deterministic 2-leg** | ΔSR ≥ +0.10 AND CI excludes 0 | ΔSR −2.840, CI excludes 0 (negative) | confirmed-null (loses badly) |
| stretch: vs learned 2-leg | ΔSR ≥ +0.10 AND CI excludes 0 | ΔSR −5.523, CI excludes 0 (negative) | confirmed-null (loses by 5.5σ) |

## What this finding establishes

Three operational reads, in increasing order of leverage:

1. **K=200 single-name universe is the wrong substrate** for an
   equity-head-of-the-allocator. The 13-ETF Phase 4d basket's
   structural diversification (sector ETFs are baskets of baskets)
   protects against idiosyncratic blowups in a way single-name
   data-driven selection cannot replicate. v3.5 and beyond should
   restrict the equity-head universe to ETFs even while keeping the
   vol-head universe at K=200 names with active short-vol PnL.
2. **The vol head learned the right regime-detection behavior on
   fold-2** (vol_scale dropped to 0.13 during COVID — the model
   *recognized* the regime). This validates the precondition that
   percentile IV features + direct-Sharpe loss can learn regime
   structure from data. The problem isn't the regime-detection; it's
   the **action space's asymmetry** — the model could fade short-vol
   but had no way to hedge or profit from the spike.
3. **Action-space asymmetry is now the load-bearing failure mode.**
   Every prior v1/v2/v3 finding pointed at the substrate or feature
   set as the culprit. v3 isolates this: even with the correct
   substrate (full DoltHub options universe = same as vol_v3) and
   correct feature set (raw IV/HV percentiles), the model can't
   match the deterministic recipe because it can't take long-vol
   positions during regime shocks.

The pre-locked v3.5 follow-up
([`TODO/e2e-portfolio-v3p5-long-vol-head`](../TODO/e2e-portfolio-v3p5-long-vol-head.md))
isolates the action-space fix: add a `long_vol_position ∈ [0, 5]`
head over free Stooq VIXY daily returns, with everything else
identical to v3. If v3.5 alone clears fold-2's COVID survival bar,
the action-space asymmetry diagnosis is correct. If not, v4's heavier
6-precondition design becomes the next test.

## Implementation notes

Bug fixes applied during the v3 run (vs the initial agent draft):

1. **`_dca_daily_phase4d()` signature** — removed vestigial `close`
   arg that crashed the pool/report step on first attempt; first run's
   per-fold checkpoints were lost when crash preceded the volume commit.
2. **NaN-mask `fwd_vol_pnl`** in daily eval — fold-1 had no DoltHub
   IV pre-2019 → vol_pnl was NaN → daily Sharpe was `+nan`. Now
   `np.nan_to_num(.., 0.0)` upstream.
3. **NaN-mask price returns + cumulative vol contribution** — defense
   in depth on the same path.
4. **Modal RPC payload limit** — 1.08 GB prep pickle exceeded RPC's
   practical limit; switched to `ss-e2e-iv-data` Volume upload via
   `modal volume put` + remote-side read from `/root/iv-data/`.
5. **Modal CLI cancellation during parallel runs** — ghost agent +
   local CLI both ran `uvx modal run`, killing one cascaded the
   other. Fixed by switching to `--detach` mode + tighter watcher
   polling Volume contents rather than tailing logs. Operational rule
   captured in CLAUDE.md (commit `2589c28`): before kicking off a
   Modal run, check `uvx modal app list` for parallel ephemeral
   containers; surgical kill via app stop `--yes`.

Outputs at `Output/e2e-portfolio-v3-{results.json,
pooled-daily.npz, fold{1,2,3}{,-daily}.npz}`, mirrored to
`ss-e2e-artifacts` Volume.

## Master walk-forward log

| date | row pointer | verdict |
|---|---|---|
| 2026-05-29 | `apps/docs/docs/leaderboard.md` row (this finding) | [`confirmed-null`](../leaderboard.md#verdict-labels) — loses to DCA by ΔSR −1.64, deterministic 2-leg by −2.84, learned 2-leg by −5.52; every CI excludes 0 negative; pooled OOS Sharpe −0.68 regresses materially from v1 +0.87 and v2 +0.78 |

## Related findings + next steps

- [`e2e-portfolio-v2`](e2e-portfolio-v2.md) — sister architecture
  on the 13-ETF universe; pooled +0.776 with fold-3 +1.246 (better
  than v3 on every metric except the vol head's regime-detection
  fidelity).
- [`e2e-portfolio-v1`](e2e-portfolio-v1.md) — paradigm test baseline.
- [`learned-ensemble-beats-deterministic`](learned-ensemble-beats-deterministic.md) — the ceiling v3 was meant to reach and didn't (loses by ΔSR −5.52).
- **Next:** [`TODO/e2e-portfolio-v3p5-long-vol-head`](../TODO/e2e-portfolio-v3p5-long-vol-head.md) (path 1, minimal action-space fix; in flight) + [`TODO/e2e-portfolio-v4-learned-regime-gate`](../TODO/e2e-portfolio-v4-learned-regime-gate.md) (path 2, full 6-precondition design).
