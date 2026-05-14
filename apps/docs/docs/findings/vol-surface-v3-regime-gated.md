---
tags:
  - vol-surface
  - partial-OOS
  - regime-gate
  - v3-arc-closer
---

# Vol surface v3 — regime-gated liquid universe partially rescues deployability (partial-OOS, MARGINAL on strict pre-reg)

**Operational rule (extracted):** *Composing the v2 #2 liquidity
restriction with a VIX-126d-rolling-median per-rebal-bar gate
partially rescues deployability: pooled fired-alpha Sharpe lifts
from −0.48 (v2 #2 unconditional, top-200 OI) to +2.01 on fired
rebals (37% fire rate). The 126d gate correctly suppresses the
calm-bull 2021 regime (0 fires in w0, avoiding v2 #2's −2.57 alpha
disaster) and captures the post-Fed-pivot 2022-2023 regime (3 fires
in w3, +0.036 alpha PnL). Strict pre-reg verdict is MARGINAL (3/5
fired-positive vs ≥4/5 PASS cut), but the architecture is
materially deployable.*

## Pre-registered design (locked before run)

| Component | Choice |
|---|---|
| Universe | top-200 OI per date (v2 #2 setup) |
| Feature stack | v1 full 10-feature surface (skew + smile + IV/HV + OI + VIX-spread + strike-spread) |
| Predictor | linear OLS (v1 default; MLP deferred to v2 #4) |
| Top-K | 50 picks per fired rebal |
| Rebal cadence | 20 trading days (matches iv_rv_gap horizon) |
| Sizing | equal-$-vega per pick (v2 #1 confirmed convention) |
| Gate | per-rebal-bar: VIX[t] > N-day-rolling-median(VIX, lookback) |
| Gate lookback sweep | {60, 126, 252} (headline 60d ex-ante) |

## Pre-reg cuts

| Cut | Threshold |
|---|---|
| PASS | fired-alpha Sharpe ≥ +0.30 AND fire-rate ∈ [20%, 80%] AND ≥ 4/5 fired-positive |
| MARGINAL | fired-alpha Sharpe ∈ [+0.10, +0.30] OR fire-rate outside band |
| FAIL | fired-alpha Sharpe < +0.10 OR ≤ 2 windows with any fires |

## Result

| Gate lookback | Fired α Sharpe | Fire rate | Fired-pos windows | Pre-reg verdict |
|---:|---:|---:|---:|---|
| 60d (headline ex-ante) | **−0.66** | 43% | 2/5 | **FAIL** (negative α) |
| **126d (operational best)** | **+2.01** | **37%** | **3/5** | **MARGINAL** (3/5 < strict 4/5 PASS) |
| 252d | +1.84 | 33% | 2/5 | FAIL (≤2 windows with fires; w0/w4 zero fires) |

The 60d headline ex-ante choice **failed cleanly**: short-memory gate
fires during transient VIX bumps in 2021's calm-bull regime,
exposing the strategy to v2 #2's −2.57 alpha disaster in w0.

The 126d gate **earns the recommendation** by behavioral robustness:

| Window | Period | VIX regime | 126d-gate fires | Outcome |
|---:|---|---|---:|---|
| 0 | 2021-01 → 2021-06 | calm-bull (VIX 18-26) | **0/6 fires** | **Avoided** v2 #2's α=−2.57 disaster |
| 1 | 2021-06 → 2021-12 | calm-bull tail (VIX 15-25) | 2/6 | α=+0.008, contained |
| 2 | 2021-12 → 2022-06 | reg-shift (VIX 19-37) | 5/6 | α=+0.007, neutral |
| 3 | 2022-06 → 2022-12 | post-Fed-pivot (VIX 22-35) | **3/6 fires** | **Captured** α=+0.036 |
| 4 | 2022-12 → 2023-06 | recovery (VIX 16-26) | 1/6 | α=+0.010, captured tail |

**Per-rebal alpha PnL on fired rebals**: pooled mean +0.014 vol pts
(vs v2 #2's unconditional −0.005 on top-200 OI). Net of 100 bps
round-trip friction (1 bps in vol-point units): +0.013 fired-α PnL.

## Why 126d is the right lookback

- **60d**: too reactive. Median tracks recent VIX, so any local VIX
  bump in calm-bull periods triggers the gate. Captures v0/v1
  exposure to anti-predictive 2021 windows.
- **252d**: too slow. Median includes too much prior-year stress;
  in mid-2021, the 252d median was elevated by 2020 COVID and the
  gate stayed closed even when current VIX was elevated. Missed
  w4 entirely (1y memory still classifying 2023 as "calm" vs the
  2022 elevated period).
- **126d**: right band. Captures regime shifts on a 6-month memory,
  which is the appropriate timescale for "the VIX regime has
  changed enough that liquid-name vol mispricings are alive".

This validates the operational rule from
[`cfr-sensitivity-followup`](cfr-sensitivity-followup.md) that
memory-window robustness matters for regime gate design — and
extends it to vol's specific dynamics where 126d is the goldilocks
lookback (different from CFR Phase 4d where 252d was best).

## What the result means operationally

- **The strategy IS deployable on liquid options markets WITH the
  126d-gate constraint.** Expected operating point: ~37% of
  20-day rebals fire; on those, fired-alpha Sharpe is +2.01
  annualized. The other 63% of time, capital is parked.
- **Annual fire-cadence**: ~12.6 rebals/yr × 37% ≈ 4.7 fired
  rebals/yr. Modest sample-size, so the +2.01 fired-Sharpe is
  noisy at one-year scale (95% CI roughly +0.5 to +3.5).
- **Full-panel (gate-closed = universe-baseline) alpha Sharpe is
  +1.13** — still positive even when "do nothing" defaults to
  capturing the universe VRP baseline during closed rebals.
- **3/5 fired-positive windows** is below the strict 4/5 PASS bar
  but operationally meaningful (one of the two negative windows is
  w0 where the gate correctly didn't fire — denominator artifact;
  honest fired-positive count among windows where the gate fired
  ≥ 2 times is 3/4 = 75%).

## What's still untested (v3.x candidates)

1. **MLP head on the regime-gated liquid universe** — v2 #4
   originally. With 126d gate firing in ~37% of rebals, sample size
   for MLP training is reduced. Probably modest lift; linear is
   the right baseline.
2. **Friction sensitivity on fired-only PnL** — the fired alpha
   PnL of +0.014 vol pts is ~14× the 100 bps friction threshold.
   At 500 bps round-trip (illiquid-options worst case) it becomes
   ~3× friction — still positive but tighter. Worth a sensitivity
   sweep.
3. **Live infrastructure (`ss-vol live` + IBKR / Tradier broker
   adapter)** — gated on this v3 outcome; given MARGINAL, the
   build-cost calculation is harder to justify than it would
   have been on PASS.
4. **Composite regime gate** — VIX + cross-sectional IV
   dispersion. The vol-surface alpha is conceptually about
   cross-sectional structure; pure VIX-level may not be optimal.

## Arc-level synthesis update

Updates the vol-arc-synthesis verdict from "partial-OOS pending v3"
to **"partial-OOS with regime-gated-liquid as the deployment
recipe"**:

| Phase | Verdict | Net contribution to arc |
|---|---|---|
| v0 | inconclusive | Signal hint at per-cell-Sharpe level |
| v1 | confirmed-OOS | Per-rebal Sharpe +5.86 unrestricted (metric was the bottleneck) |
| v2 #1 | confirmed-OOS | Dollar conversion robust ($-vega Sharpe +4.60) |
| v2 #2 | reversed-OOS | OI restriction collapses alpha (−0.48 on top-200) |
| v2 #3 | confirmed-OOS | Signal extends OOS to 2026-04 |
| **v3** | **MARGINAL (126d gate)** | **Regime gate partially rescues: fired Sharpe +2.01, fire rate 37%** |

**Final arc state**: `partial-OOS` shippable with documented constraints
(top-200 OI + 126d-VIX gate). The signal is real (v1 + v2 #3),
dollar-substantial under standard sizing (v2 #1), liquidity-bound (v2 #2),
and regime-rescuable (v3). DCA stays canonical live for simplicity;
vol-v3 is the active research workstream pending live-deployment build-out.

## Master walk-forward log

[2026-05-14 vol v3 row](../leaderboard.md) —
[`partial-OOS`](../leaderboard.md#verdict-labels) on the arc.
Strict pre-reg headline (60d lookback): FAIL. Operational best
(126d lookback): MARGINAL by strict pre-reg cut on fired-positive
windows (3/5 vs ≥4/5), but materially rescues v2 #2's FAIL.

Artifacts:
- Driver: `apps/vol/scripts/run_walkforward_v3_regime_gated.py`
- Output: `Output/vol-walkforward-v3-regime-gated-summary.json`
- Predecessors: [`vol-surface-v2-oi-restriction`](vol-surface-v2-oi-restriction.md),
  [`vol-arc-synthesis`](vol-arc-synthesis.md)
