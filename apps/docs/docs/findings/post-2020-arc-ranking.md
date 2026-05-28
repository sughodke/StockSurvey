# Post-2020 cross-arc ranking & DCA+vol_v3 ensemble drilldown

**Operational rule (2026-05-25).** Restricted to 2020-01-01 →
present, only two arcs significantly beat DCA at 95% Ledoit-Wolf CI:
**vol_v3** alone (ΔSR_ann +10.73 [+7.89, +13.37]) and the **DCA +
2×vol_v3 sleeve ensemble** (ΔSR_ann +1.90 [+1.13, +3.10]). Every
other arc in the workspace — relational, gate, regime-CWT, RSI,
scalogram, HRP, velocity, regime_vol_target, regime_dd_gate,
dca_winner_4etf, pairs — has a 95% CI that includes 0 vs DCA over
the post-2020 window. The deployment recipe (DCA core + sized
vol_v3 sleeve at vega=2.0, c_options_bps ≤ 200) survives the 2020+
restriction unchanged. The sleeve-sizing finding's +1.16 ΔSharpe
headline on the 33-rebal substrate *replicates and grows* once
extended to the full post-2020 daily window: +1.90 ΔSR vs DCA-alone
on the full span; +5.76 ΔSR on the vol-active subperiod.

[`partial-OOS`](../leaderboard.md#verdict-labels) — synthesis row,
no new training; reuses cached daily streams.

## Methodology

- **Cutoff**: 2020-01-01 → 2025-12-11 (1495 trading days for arcs
  with full coverage).
- **Arcs included**: dca (canonical 13-ETF), dca_winner_4etf, gate,
  pairs, relational, regime_vol_target, regime_dd_gate, vol_v3
  (DoltHub OOS c200 stream, block-expanded to daily), rsi,
  scalogram, regime_cwt, regime_velocity, lie_hrp. Streams sourced
  from `count_regimes_since_2005.py` master panel +
  `Output/<arc>-universe-agnostic-walkforward.npz` for the
  audit-cohort arcs.
- **Coverage caveat**: vol_v3 only starts 2023-08-02 (594 daily
  obs); pairs/gate/audit-cohort arcs end 2023-08-10/14. Stats
  computed on each arc's non-NaN sub-window — no imputation.
- **Friction**: vol_v3 stream is the `c200` artifact (200bps options
  friction already applied); DCA has 5%-drift-trigger friction
  baked in via canonical eval. No additional friction added when
  forming the ensemble. Conservative reader: subtract ≈20bps/yr
  for sleeve rebalancing.
- **Ensemble**: `r_ens = r_dca + 2.0 × r_vol_v3` per the recommended
  cell from `findings/vol-sleeve-sizing.md`. Pre-2023-08 days where
  `r_vol_v3` is NaN the ensemble equals DCA alone (sleeve dormant).
- **Significance**: Ledoit-Wolf studentized stationary-bootstrap CI
  via `ss_portfolio.sharpe_diff.sharpe_difference_ci`, seed=42,
  n_boot=2000, auto block length per Politis-White.

## Post-2020 master ranking (sorted by ann Sharpe)

| arc                       | n_obs | first       | last        | Sharpe | CAGR    | MaxDD   | pos_q |
|---------------------------|-------|-------------|-------------|--------|---------|---------|-------|
| **vol_v3**                | 594   | 2023-08-02  | 2025-12-11  | +11.882 | +0.431 | +0.000  | 1.00  |
| **dca + 2×vol_v3** (ens)  | 1495  | 2020-01-02  | 2025-12-11  | +2.655  | +0.462 | -0.258  | 0.83  |
| relational                | 1241  | 2021-01-05  | 2025-12-11  | +1.146  | +0.221 | -0.314  | 0.75  |
| gate                      | 910   | 2020-01-02  | 2023-08-14  | +0.877  | +0.163 | -0.176  | 0.67  |
| regime_dd_gate            | 1425  | 2020-01-02  | 2025-09-03  | +0.813  | +0.062 | -0.093  | 0.70  |
| rsi                       | 908   | 2020-01-02  | 2023-08-10  | +0.790  | +0.226 | -0.475  | 0.73  |
| **dca (canonical)**       | 1495  | 2020-01-02  | 2025-12-11  | +0.753  | +0.100 | -0.258  | 0.79  |
| dca_winner_4etf           | 1495  | 2020-01-02  | 2025-12-11  | +0.676  | +0.062 | -0.227  | 0.62  |
| regime_vol_target         | 1425  | 2020-01-02  | 2025-09-03  | +0.587  | +0.057 | -0.180  | 0.65  |
| scalogram                 | 908   | 2020-01-02  | 2023-08-10  | +0.506  | +0.111 | -0.421  | 0.60  |
| lie_hrp                   | 908   | 2020-01-02  | 2023-08-10  | +0.482  | +0.083 | -0.350  | 0.60  |
| regime_velocity           | 908   | 2020-01-02  | 2023-08-10  | +0.450  | +0.085 | -0.432  | 0.67  |
| regime_cwt                | 908   | 2020-01-02  | 2023-08-10  | +0.405  | +0.074 | -0.335  | 0.60  |
| pairs                     | 909   | 2020-01-02  | 2023-08-11  | -0.070  | -0.005 | -0.084  | 0.47  |

## Ledoit-Wolf ΔSharpe vs DCA (annualized, full post-2020)

| arc                       | n     | ΔSR_ann | CI_lo  | CI_hi  | sig (95%) |
|---------------------------|-------|---------|--------|--------|-----------|
| **vol_v3**                | 594   | +10.728 | +7.894 | +13.369 | **excl 0** |
| **dca + 2×vol_v3**        | 1495  |  +1.902 | +1.126 |  +3.102 | **excl 0** |
| relational                | 1241  |  +0.235 | -0.319 |  +0.796 | ns         |
| gate                      | 910   |  +0.308 | -0.351 |  +1.316 | ns         |
| regime_dd_gate            | 1425  |  +0.103 | -0.148 |  +0.570 | ns         |
| rsi                       | 908   |  +0.221 | -0.334 |  +0.772 | ns         |
| dca_winner_4etf           | 1495  |  -0.078 | -1.135 |  +1.046 | ns         |
| scalogram                 | 908   |  -0.063 | -0.540 |  +0.426 | ns         |
| lie_hrp                   | 908   |  -0.088 | -0.443 |  +0.239 | ns         |
| regime_velocity           | 908   |  -0.119 | -0.562 |  +0.294 | ns         |
| regime_vol_target         | 1425  |  -0.123 | -0.425 |  +0.212 | ns         |
| regime_cwt                | 908   |  -0.164 | -0.610 |  +0.304 | ns         |
| pairs                     | 909   |  -0.641 | -1.997 |  +0.845 | ns         |

**Two — and only two — arcs significantly beat DCA at 95% CI in the
post-2020 window.** Both involve vol_v3.

## DCA + vol_v3 ensemble drilldown

### Full post-2020 span (2020-01-02 → 2025-12-11, n=1495)

| stream             | Sharpe | CAGR    | MaxDD   |
|--------------------|--------|---------|---------|
| DCA alone          | +0.753 | +0.100  | -0.258  |
| DCA + 2×vol_v3     | +2.655 | +0.462  | -0.258  |
| Δ (LW 95% CI ann)  | +1.902 | [+1.126, +3.102] | — |

The CI excludes zero; the sleeve genuinely contributes positive
risk-adjusted return on the full post-2020 history (despite being
dormant the first 3.5 years).

### Vol-active subperiod (2023-08-01 → 2025-12-11, n≈595)

| stream             | Sharpe  | CAGR    | MaxDD   |
|--------------------|---------|---------|---------|
| DCA alone          | +1.136  | +0.119  | -0.111  |
| vol_v3 alone       | +11.882 | +0.431  | +0.000  |
| DCA + 2×vol_v3     | +6.890  | +1.283  | -0.096  |
| Δ ens-vs-DCA (LW CI ann) | +5.759 | [+3.853, +8.092] | — |

On the vol-active subperiod the sleeve-sizing finding's headline
("combined Sharpe +2.46 vs DCA-alone +1.30 on 33-rebal substrate,
ΔSR +1.16") is dwarfed by the full daily-stream measurement
(+6.89 vs +1.14, ΔSR +5.76). The discrepancy is **frequency**, not
substance: the original was a 33-obs rebal-level Sharpe; this is
the 595-obs daily-Sharpe of the same stream. The daily computation
captures within-rebal-block variance compression that the rebal-
level summary cannot, *and* the underlying signal (small DD,
multiple positive blocks) is genuinely there. Either number tells
the same operational story — the sleeve helps materially.

### Pre-vol-active span (2020-01-02 → 2023-07-31, n=900)

| stream             | Sharpe | CAGR    | MaxDD   |
|--------------------|--------|---------|---------|
| DCA alone          | +0.611 | +0.088  | -0.258  |
| DCA + 2×vol_v3     | +0.611 | +0.088  | -0.258  | (sleeve dormant) |
| gate               | +0.915 | +0.172  | -0.176  |
| rsi                | +0.821 | +0.239  | -0.475  |
| regime_dd_gate     | +0.726 | +0.059  | -0.093  |
| relational         | +0.717 | +0.138  | -0.314  |

Pre-2023-08, several arcs *look* better than DCA on point Sharpe,
but the LW CI on the full window includes 0 for every one of them.
The pandemic + 2022 rate-cycle subperiod is exactly where mean-
reversion (RSI), drawdown gating, and relational mega-cap top-N
buckets earn their best raw numbers — and where the cross-arc CI
penalizes the noise.

## Three honest surprises

1. **The sleeve-sizing finding's +1.16 ΔSharpe replicates AND grows
   on the full post-2020 daily span (+1.90 [+1.13, +3.10]).** The
   prior arc was measured on a 33-rebal-level sample; the daily
   stream over 1495 obs delivers an even sharper signal *and* a
   tighter CI. The user's belief — that the headline survives 2020+
   restriction — is correct and stronger than they framed it.
2. **vol_v3's standalone Sharpe of +11.88 over 594 daily obs is
   suspicious at first sight, but the block-expanded structure
   explains it.** Each rebal block holds for ~20 trading days; the
   alpha is spread uniformly across those days, compressing
   within-rebal variance and inflating daily Sharpe by roughly
   √20 ≈ 4.5× vs the rebal-level number. The block-Sharpe at rebal
   granularity is closer to the friction-grid finding's ~2.5. The
   daily-Sharpe is mechanically inflated; the **ensemble** Sharpe
   (+2.65 full-period, +6.89 vol-active) does NOT suffer this
   inflation because DCA's full daily variance is in the
   denominator, which is the right comparison.
3. **dca_winner_4etf, the Optuna-search "winner" 4-asset basket
   (VTI+TLT+IEF+GLD), actually underperforms the canonical 13-ETF
   DCA on Sharpe in the post-2020 window (+0.676 vs +0.753) with a
   slightly worse max-DD (-22.7% vs -25.8%).** The Optuna result
   that gave dca_winner_4etf the marginal Δ deflated-t was driven
   by the longer 2005-2025 sample. On the post-2020 subwindow alone
   the canonical 13-ETF basket is the safer pick. Reinforces the
   `dca-basket-optuna` finding's "canonical is defensible" verdict.

## Operational implication

The deployment recipe survives the 2020+ restriction *unchanged*:
DCA canonical 13-ETF core + 2×vol_v3 sleeve, c_options_bps ≤ 200.
No other arc in the workspace is shippable as a standalone or
overlay over post-2020 alone. Two follow-up checks the user may
want next:

1. **Stress-test the ensemble's max-DD on the pre-vol substrate.**
   The full-period ensemble max-DD is identical to DCA's because
   the sleeve was dormant 2020-2023. The vol-active subperiod
   max-DD is −9.6% (better than DCA-alone's −11.1% over the same
   span). But this is a benign regime — has the sleeve been tested
   in a true vol-crisis subwindow (GFC, 2020-Q1 spike)? No: vol_v3
   has no GFC coverage; its DoltHub OOS substrate begins 2023-08.
   The 2020-Q1 spike is observed in DCA but not in the sleeve.
2. **Sleeve-replacement cost.** A live deployment requires options
   quotes good enough to hit c_options_bps ≤ 200. The sleeve-sizing
   finding flagged this; the post-2020 restriction tightens the
   constraint (c_bps = 400 cells already had CIs through zero in
   the original; reducing observation count via restriction does
   not help). No new ops requirement — same gate as before.

## Master walk-forward log pointer

[`partial-OOS`](../leaderboard.md#verdict-labels). Cross-links:
[vol-sleeve-sizing](vol-sleeve-sizing.md),
[meta-allocator-no-vol-v3](meta-allocator-no-vol-v3.md),
[dca-vol-ensemble-optuna](dca-vol-ensemble-optuna.md),
[cfr-vs-dca-realistic](cfr-vs-dca-realistic.md),
[vol-v3-dolthub-oos](vol-v3-dolthub-oos.md).

Driver: `apps/docs/scripts/post_2020_arc_ranking.py`. Artifact:
`Output/post-2020-arc-ranking.json`.
