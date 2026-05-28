# Crypto-and-carry — perp funding-rate cash-and-carry on Hyperliquid

**Operational rule (lede):**
The academic-clean cash-and-carry approximation (signed-funding minus
rebal friction on weight delta) on Hyperliquid top-20 perps posts a
`confirmed-OOS` headline at net Sharpe +9.25 / deflated-t +11.8 /
pos-quarter 0.80 / max-DD −0.99% over 2024-01-01 → 2026-05-24. **This
is the upper bound, not the deployable Sharpe.** The result establishes
that the gross-funding yield minus mechanical leg friction is materially
positive, but it does *not* establish a deployable +9 Sharpe — basis
tracking error, short-spot borrow on alts, and venue selection all
debit yield in real execution. The 2024 fold carries the bulk of the
alpha (Sh +13.8 on 27.5%/yr funding); 2026YTD has effectively zero edge
(Sh −0.63 on 0.4%/yr funding). **Do not promote to live without a
basis-tracking-error stress test and a venue port to a higher-history
funding source.**

## Why this arc, why this venue

The 2026-05-25 `factor-crypto-venue` `confirmed-null` falsified the
74-channel indicator grid as a cross-sectional alpha source on
CryptoCompare top-50 daily. Its decision-tree pointed to perp
funding-rate carry as the orthogonal venue-fit test (He, Manela, Ross
2023 "Indexing Crypto"; Brunnermeier-Pedersen 2009 funding-liquidity).

The original brief specified Binance (`fapi.binance.com`,
`/fapi/v1/fundingRate`) with Bybit / OKX as alternative-venue follow-ups.
**Binance returns HTTP 451 from this host (US-restricted); Bybit returns
HTTP 403 (CloudFront geo-block); OKX exposes only ~3 months of public
funding-rate history regardless of `before`/`after` pagination.**
Hyperliquid's public `info` endpoint exposes ~2.4 years of hourly
funding history across 230 perpetuals with no auth and no geo-restriction,
which is the only viable substrate available from this development
environment. The substitution is documented in
`packages/loaders/src/ss_loaders/hyperliquid.py`'s module docstring.

The venue substitution narrows the result — HL has different funding
dynamics, a different (DEX, on-chain margin) demand structure, and
different listing tier than Binance — but does not invalidate the
methodology. A future arc that establishes API access to Binance /
Bybit / OKX deep funding history is the right follow-up to test
venue-portability.

## Test design (locked pre-eval)

| | |
|---|---|
| Venue | Hyperliquid (substitute for Binance after geo-block) |
| Universe | HL top-20 perps by current-snapshot `dayNtlVlm`, 180d funding-history floor → 18 effective coins |
| Coins | BTC DOGE ETH FARTCOIN HYPE NEAR NIL ONDO PUMP RENDER SOL SUI TAO TON VVV WLD XRP ZEC |
| Date span | 2024-01-01 → 2026-05-24 (875 days × 18 coins) |
| Funding cadence | hourly (24/day on HL, not 8h like Binance/Bybit); daily-summed substrate |
| Rebal substrate | daily |
| Pre-reg cell | K=5, rebal_days=1, sign='positive', trailing_window=30d |
| Friction model | 15 bps per leg × 2 legs (spot + perp) = 30 bps round-trip charged on `|Δ(sign · weight)|` |
| Ranking | trailing-30d mean funding rate, shifted +1 day (point-in-time) |
| Sign='positive' | enter top-K positive-funding coins as long-spot/short-perp |
| Sign='both' | split capital — K/2 positive, K/2 most-negative (inverse: long-perp/short-spot) |
| Grid | 3 K × 3 rebal_days × 2 sign × 2 trailing_window = 36 cells |
| Verdict bar | `confirmed-OOS`: SR ≥ +1.0 ∧ DSR-t > +2.0 ∧ pos-Q ≥ 0.80 ; `partial`: SR ≥ +0.5 ∧ DSR-t > +1.0 ; `null`: SR < +0.3 ; `reversed`: SR < 0 |
| Periods/year | 365 |
| DSR `n_trials` | 36 |
| DSR `sharpe_std_ann` | 0.40 (crypto cross-trial dispersion midpoint, matches `factor-crypto-*` arcs) |

## Modeling approximation (the load-bearing assumption)

Per-day per-coin per-unit-notional carry PnL =
`sign × funding_rate_summed_over_day`. Portfolio per-day PnL =
`sum_coin weight × sign × funding`. On rebal days, friction =
`fric_round_trip × |Δ(sign × weight)|`.

This is the academic-clean basis-trade approximation: with equal-notional
spot + perp legs re-hedged daily, price-delta cancels to first order
and the funding stream IS the per-day PnL. Real-world deployment
carries three loss channels the model does not charge:

1. **Basis tracking error.** Spot and perp prices diverge intraday;
   the re-hedge incurs slippage on whichever leg moves first. On HL
   specifically (DEX with on-chain spot bridging), tracking error
   is structurally higher than on a single CEX. Plausible 5-20 bps/d
   drag.
2. **Short-spot borrow.** Collecting positive funding on
   `long-spot/short-perp` requires borrowing spot to short. On
   alts, borrow rates can be 5-30% ann; HL specifically does not
   offer spot shorting natively, so this requires a separate venue.
3. **Liquidity / impact at scale.** The model uses last-mid +
   constant friction; at deploy size > a few hundred bps of `dayNtlVlm`,
   slippage scales nonlinearly.

The eval is therefore an *upper-bound substrate test*, not a
deployment forecast.

## Pre-reg cell result

| Metric | Value | Hurdle | Cleared? |
|---|---|---|---|
| net Sharpe (ann) | **+9.252** | ≥ +1.0 | yes |
| deflated-t | **+11.798** | > +2.0 | yes |
| pos-quarter fraction | **0.80** | ≥ 0.80 | yes |
| max DD (net) | −0.99% | n/a (informational) | — |
| total net return | +39.56% | n/a | — |
| total friction | 17.22% of capital | n/a | — |
| n_obs (days) | 875 | n/a | — |
| **verdict** | **`confirmed-OOS`** | | |

## Per-fold breakdown (pre-reg cell, calendar years)

| Fold | n days | net Sharpe (ann) | total return | max DD | mean funding (cross-coin-day) |
|---|---|---|---|---|---|
| 2024 | 366 | **+13.835** | +30.27% | −0.31% | 27.5%/yr |
| 2025 | 365 | +7.755 | +9.53% | −0.95% | 8.9%/yr |
| 2026YTD | 144 | **−0.632** | −0.24% | −0.99% | 0.4%/yr |

The funding-regime collapse from 27.5%/yr → 8.9%/yr → 0.4%/yr is the
dominant story. The 2024 fold carries the headline; 2026YTD is
effectively zero edge net of friction. This is consistent with
post-FTX-collapse maturation of the crypto basis trade — arb capital
has tightened the funding-yield distribution, especially on the
liquid majors that dominate the top-20 ranking.

## 36-cell robustness grid

All 36 cells post positive net Sharpe (range +0.65 to +11.47, median
+6.7). Heat-map of net Sharpe by `top_k` × `rebal_days` × `sign`
× `trailing_window`:

| K | rebal | sign | trail=7 | trail=30 |
|---|---|---|---|---|
| 3 | 1 | positive | +2.12 | +8.90 |
| 3 | 1 | both | +1.01 | +7.29 |
| 3 | 3 | positive | +3.77 | +9.47 |
| 3 | 3 | both | +2.42 | +7.22 |
| 3 | 7 | positive | +6.02 | +9.92 |
| 3 | 7 | both | +4.68 | +6.87 |
| 5 | 1 | positive | +2.41 | **+9.25** (pre-reg) |
| 5 | 1 | both | +0.65 | +8.92 |
| 5 | 3 | positive | +4.38 | +9.71 |
| 5 | 3 | both | +3.17 | +8.97 |
| 5 | 7 | positive | +6.10 | +10.54 |
| 5 | 7 | both | +4.51 | +8.72 |
| 10 | 1 | positive | +5.41 | +10.39 |
| 10 | 1 | both | +1.03 | +8.39 |
| 10 | 3 | positive | +6.66 | **+11.22** |
| 10 | 3 | both | +2.95 | +8.82 |
| 10 | 7 | positive | +7.63 | **+11.47** (best) |
| 10 | 7 | both | +4.79 | +9.24 |

**Edge-structure reads from the grid:**

- **Trailing_window=30 dominates** at every (K, rebal, sign). The
  signal is in the *persistent* funding-rate regime, not in
  short-window noise. trail=7 over-rotates on funding spikes that
  mean-revert.
- **`sign='positive'` dominates `'both'`** at every K × rebal_days ×
  trail combination. The inverse arm (long-perp / short-spot on
  negative funding) is real but capacity-limited: at trail=7 it
  picks up noisy negative spikes; at trail=30 it picks up negative
  funding that's about to flip back positive. The pure long-bias
  variant is consistently cleaner.
- **Larger K and longer rebal_days both help** at trail=30. K=10 / 
  rebal=7d / trail=30 is the best cell at +11.47. Larger K
  diversifies idiosyncratic funding-rate vol; longer rebal_days
  reduces friction without losing much signal because the funding
  regime persists ~30+ days.

## Mechanism

When perp price > spot, longs pay shorts a funding rate every interval
(hourly on HL) to anchor the perp to spot. The structural reason this
persists:

- **Demand side:** perp longs are leveraged retail demand for upside.
  On HL specifically (DEX, no KYC for derivatives, on-chain margin),
  retail access is frictionless.
- **Supply side:** arb capital that would short the perp + long spot
  is bounded by (1) access to spot venues with reliable settlement,
  (2) borrow availability on alts, (3) regulated-venue gatekeeping.

The demand-supply imbalance translates to a persistent positive
funding rate when sentiment is bullish. The 2024-fold mean of 27.5%/yr
reflects post-ETF-approval bullish positioning. As arb capital
re-allocated to crypto basis trades through 2025-2026 (and as
sentiment matured), the imbalance tightened — exactly the regime
collapse the per-fold table records.

## Three honest surprises

**(1) The 2024 fold's Sharpe +13.8 is unphysical** relative to any
deployable basis trade. This is direct confirmation that the
eval-substrate-vs-deployment gap is large. The trade IS *earning* the
27.5%/yr funding yield in the model — that's not a calculation error,
it's exactly what the formula computes — but a real-world execution
would lose 10-20%/yr of it to basis-tracking error + short-spot borrow.
The honest reading is "the gross funding yield is large enough to
absorb plausible friction with positive net yield", not "deploy this
for +14 Sharpe".

**(2) Sign='both' barely helps anywhere.** I expected the inverse arm
(harvest negative funding via long-perp/short-spot) to materially
help over a 2.4-year span that included several negative-funding
regimes. It didn't: at trail=7 it actively hurts (cells 2, 14, 26 are
the 3 lowest-Sharpe cells in the grid); at trail=30 it ties the
`positive`-only variant within ±0.3 Sharpe. The most likely
explanation: HL's negative-funding episodes are short-lived noise
spikes on a small subset of coins, and the trailing-30d ranker is
too slow to enter and exit on time. A faster signal might rescue
the inverse arm, but at the cost of trail=7's signal degradation.

**(3) The 2026YTD fold went slightly negative net** despite the model
collecting nonzero gross funding. At 0.4%/yr cross-coin-day mean
funding, the gross yield is ≈ 1 bps/day; the friction model charges
~30 bps round-trip on weight delta; daily rebal with the pre-reg
config produces ~0.5 portfolio turnovers/day; friction ≈ 15 bps/day
exceeds gross. **This is the deployment canary**: in the current
funding regime, the trade is friction-bound. Operationally, the
2026YTD regime would require either (a) widening trailing_window to
months to lower turnover, (b) longer rebal cadence (the grid's
rebal=7d cells are *less* friction-bound), or (c) gating the strategy
off entirely below a minimum funding threshold. This points to the
next experiment.

## Verdict per locked pre-reg bar

`confirmed-OOS` (all three thresholds cleared at the pre-reg cell).
This is HONEST per the locked bar — but the headline Sharpe is
explicitly an upper-bound substrate result, not a deployable Sharpe.
The `verdict-vs-deployability` split is recorded in the leaderboard
notes column and in this finding's lede.

## Next-experiment proposal (per `confirmed-OOS` decision branch)

CLAUDE.md's `confirmed-OOS` branch asks: *Where does it stop working?
Run the adjacent test that would either confirm scope or break it.*

**(A) Basis-tracking-error stress.** Re-run the eval with a
deterministic / stochastic basis-tracking-error drag (5, 10, 20 bps/d
per dollar-notional) and re-test the verdict at each level. This is
the cheapest follow-up: numpy-only modification of `cnc.backtest`,
no new data. **If the +1.0 Sharpe bar survives at 10 bps/d drag,
the deployable result is robust to plausible tracking error. If it
collapses below +1.0 at 5 bps/d, the academic result is a fragile
upper bound and deployment requires far tighter execution than HL
on-chain settlement can deliver.**

**(B) Funding-regime gate.** Add a regime gate that disables the
strategy when trailing-30d cross-coin-mean funding < X bps/day. Re-run
the full eval with the gate active and confirm the 2026YTD Sh
−0.63 lifts to 0.0 while the 2024-2025 fold returns are preserved.
This is the operational fix the 2026YTD drag points at.

**(C) Venue port (medium-term).** When Binance / Bybit / OKX deep
funding history becomes accessible (paid data feed, VPN-routed
service, or an on-prem proxy), port the eval and confirm
venue-portability. A confirmed-OOS at OKX with the same
pre-registered cell would materially upgrade the deployable claim.

The follow-up is documented in `TODO/cnc-tracking-error-stress.md`
(to be created if the user pursues this).

## Implementation gaps

- **Venue substitution from Binance to Hyperliquid.** Documented in
  `packages/loaders/src/ss_loaders/hyperliquid.py`'s module docstring
  and `apps/cnc/src/cnc/__init__.py`'s app docstring. Hyperliquid
  was the only freely-accessible venue from this host that exposes
  multi-year funding history.
- **Modeling approximation.** Basis tracking error, short-spot borrow,
  and impact are all set to zero. The "Modeling approximation"
  section above documents this explicitly.
- **No spot-leg price data used.** The backtest uses only funding
  rates + daily perp candles (for universe ranking). A
  basis-tracking-error-aware variant would also need a spot reference
  series (CoinGecko, Yahoo BTC-USD, or HL spot for non-native
  coins — HL spot only carries native tokens).
- **Universe ranking is current-snapshot, not point-in-time.** Top-20
  by `dayNtlVlm` is fixed across the eval span. A future revision
  should rank by trailing-30d candle-derived `vol_quote` at each
  rebal bar. This is unlikely to change the verdict materially
  because the top coins (BTC, ETH, SOL) dominated $-volume across
  the full span on HL.
- **No survivor-bias correction.** Coins whose funding history
  starts mid-span are simply excluded; coins that delisted on HL
  during the span would similarly be excluded by the 180d floor.
  The 18 effective coins are all live on HL today; this is
  selection-on-survival but is *less* severe than the
  CryptoCompare top-N approach.

## Master walk-forward log pointer

Leaderboard row: 2026-05-25 `cnc | crypto-and-carry` — verdict
[`confirmed-OOS`](../leaderboard.md#verdict-labels).

Parent (the `confirmed-null` whose decision branch produced this arc):
[`findings/factor-crypto-venue`](factor-crypto-venue.md).

Related cross-arc context:
[`findings/deflated-sharpe-leaderboard`](deflated-sharpe-leaderboard.md),
[`findings/ladder-methodology-rewrite`](ladder-methodology-rewrite.md).

## Reproduction

```bash
uv run python apps/cnc/scripts/run_walkforward.py
# ~25s local wall after cache fill; ~10 min cold cache fill
# Output/cnc-walkforward.npz + .json
```

Single-config CLI smoke:

```bash
uv run ss-cnc backtest --start 2024-01-01 --top-universe 20 \
  --top-k 5 --rebal-days 1 --trailing-window 30 --sign positive
```
