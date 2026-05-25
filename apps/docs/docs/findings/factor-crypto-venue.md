# Factor crypto venue port — confirmed-null

**Operational rule.** The 74-channel deterministic indicator grid that
produced the factor arc's `confirmed-OOS` equity result
(`findings/factor-shorthorizon-representation.md`, val IC +0.0212 6/6
at `rebal_days=5` on the factor-narrow universe) **does NOT transfer**
to crypto top-50 daily at the same horizon. Both linear and MLP heads
miss the pre-registered IC bar and miss the DSR-t hurdle. Per
`TODO/factor-crypto-venue-test.md`'s `confirmed-null` decision branch,
the next venue-fit arc is perp **funding-rate cash-and-carry**
(He-Manela-Ross 2023), not a refinement of cross-sectional indicator
features on crypto. Do not re-attempt the indicator-grid port without
a different feature class (e.g. on-chain flow, perp funding term
structure, dominance regime).

## Eval setup (locked pre-reg)

| field | value |
|---|---|
| universe | `cryptocompare_top50_v1` — 42 effective tickers (8 skipped at the 2000-bar floor: TON, SHIB, ICP, APT, ARB, OP, GRT, FLOW) |
| span | 2018-01 → 2026-05; periods_per_year = 73 (crypto trades 7d/wk) |
| horizon | `rebal_days = 5` (matches the equity port) |
| feature stack | 74-channel indicator grid (30 RSI + 16 CCI + 6 vol + 18 MACD + 4 coherence) |
| head | tinygrad linear and MLP, `n_steps=200`, `lr=0.01`, `weight_decay=1e-3` |
| walk-forward | 5 windows, train=110 blocks, val=55, step=55 |
| friction | `commission_bps = 10` (equity baseline; crypto-realistic 5–50 bps grid deferred) |
| pre-reg bar | `mean val IC > +0.025 ; 4/5 positive windows ; DSR t > +1.5` |
| Modal cost | < $0.10, two scorer runs wall-time 36s + 48s |

## Per-window results

| scorer | win | val_start | train_ic | val_ic | val_sh_LO | val_sh_LS |
|---|---:|---|---:|---:|---:|---:|
| linear | 0 | 2022-05-08 | +0.179 | +0.052 | −0.45 | +1.51 |
| linear | 1 | 2023-02-07 | +0.170 | −0.034 | −0.34 | −1.48 |
| linear | 2 | 2023-11-09 | +0.132 | +0.005 | −0.07 | −0.82 |
| linear | 3 | 2024-08-10 | +0.146 | +0.012 | +0.38 | −0.13 |
| linear | 4 | 2025-05-12 | +0.145 | +0.074 | −1.48 | +1.47 |
| **linear-mean** | — | — | **+0.155** | **+0.0217** | **−0.392** | **+0.110** |
| mlp | 0 | 2022-05-08 | +0.731 | +0.023 | −0.52 | +0.46 |
| mlp | 1 | 2023-02-07 | +0.640 | −0.009 | −0.26 | −0.60 |
| mlp | 2 | 2023-11-09 | +0.679 | +0.021 | −0.04 | −0.11 |
| mlp | 3 | 2024-08-10 | +0.610 | +0.007 | +0.36 | −0.24 |
| mlp | 4 | 2025-05-12 | +0.628 | +0.025 | −1.58 | −0.11 |
| **mlp-mean** | — | — | **+0.658** | **+0.0134** | **−0.408** | **−0.118** |

## Verdict vs pre-reg gates

| gate | linear | mlp | pass? |
|---|---:|---:|---|
| mean val IC > +0.025 | +0.0217 | +0.0134 | **FAIL** (both) |
| ≥4/5 positive val windows | 4/5 (0.80) | 4/5 (0.80) | pass (both) |
| DSR deflated-t > +1.5 | LO −1.47 / LS +0.06 | LO −1.55 / LS −0.87 | **FAIL** (all 4 cells) |

The 4/5 positive-window count is the only gate clearing — and at
n=5 windows that's binomial p ≈ 0.19 under a fair-coin null, well
above the conventional 5% bar for surprise. With both load-bearing
gates failing, we record this honestly as **`confirmed-null`** rather
than `partial-OOS`. The TODO's `partial-OOS` example explicitly
required *IC-bar yes, DSR-bar no* — here IC bar also fails.

## DSR ladder placement

After re-ranking with the new ArcSpecs in `apps/docs/scripts/compute_dsr.py`:

| arc | mode | trials | annSh | E[maxSR] | DSR | defl_t |
|---|---|---:|---:|---:|---:|---:|
| factor-crypto-linear-LS | standalone | 2 | +0.368 | 0.0397 | 0.522 | **+0.055** |
| factor-crypto-mlp-LS | standalone | 2 | −0.111 | 0.0397 | 0.192 | **−0.872** |
| factor-crypto-linear-LO | standalone | 2 | −0.411 | 0.0397 | 0.071 | **−1.472** |
| factor-crypto-mlp-LO | standalone | 2 | −0.448 | 0.0397 | 0.061 | **−1.548** |

All four sit deep in the sub-`t=+1.5` cluster. `sharpe_std_ann=0.40`
was used per the TODO close-out protocol's recommendation
(crypto cross-trial dispersion exceeds the 0.072 equity structural
residual; midpoint of the 0.35–0.50 band suggested).

## Mechanism — why didn't it transfer?

Three hypotheses, ordered by plausibility:

1. **The indicator grid is equity-microstructure-flavored.** RSI/CCI/MACD
   parameter ranges (`n` in {7, 14, 21, 30, ...}) were tuned for daily
   US equity bars where intraday auction dynamics shape close-to-close
   reversal. Crypto bars are continuous-time (no open/close auction)
   and the same RSI windowing extracts no equivalent structure.
2. **Crypto cross-section is driven by features the grid doesn't carry.**
   Liu-Tsyvinski-Wu 2022 documents a 3-factor cross-section
   (market / size / momentum) but the operational drivers — perp
   funding rate term structure, on-chain flow, dominance regime,
   listing tier — are absent from the grid. The grid measures
   short-horizon price-derived shapes; crypto's cross-section is
   conditioned on funding / flow state.
3. **MLP overfits train aggressively without OOS lift.** Train IC
   leaped from +0.155 (linear) to +0.658 (MLP) but val IC fell
   (+0.0217 → +0.0134). At n=110 train blocks and 74 features the
   MLP has enough DOF to memorize a noisy training rank; the linear
   head is the more honest probe and it also fails.

Hypothesis 3 is the cleanest evidence for hypothesis 2 — if the
features were carrying real cross-sectional signal, an MLP would have
extracted it; instead it extracted training noise.

## Caveats

- **Single horizon, single feature stack.** We tested rebal=5 only.
  Crypto trends operate at multi-week scales (Moskowitz-Ooi-Pedersen
  2012-style TS-momentum is the canonical positive result in crypto);
  this null does not falsify a longer-horizon momentum arc.
- **Spot-only, commission_bps=10.** The L/S construct is a conceptual
  market-neutral probe; spot shorting is uneconomic. A perp-funded
  L/S would *charge* funding rather than borrow — different stream
  entirely. Treat the L/S column as diagnostic, not deployable.
- **42 effective tickers.** Top-50 minus 8 sub-2000-bar names. Wider
  universes (top-100, top-200) might recover signal but the same
  feature-class concern applies.
- **Block-Sharpe at n=275 blocks.** Not thin; the DSR-t bar is
  defensible at this sample length.

## Master walk-forward log

- Leaderboard row 433: `2026-05-25 | factor | crypto venue port — confirmed-null` → [`confirmed-null`](../leaderboard.md#verdict-labels)
- Driver: `apps/factor/scripts/modal/train_indicator_crypto.py` ::walkforward
- Data prep: `apps/factor/scripts/prep_crypto_universe.py`
- Artifacts: `Output/walkforward-crypto-summary.json`,
  `Output/walkforward-crypto-linear-s200-wd0.001-windows.npz`,
  `Output/walkforward-crypto-mlp-s200-wd0.001-windows.npz`,
  `Output/walkforward-crypto-comparison.png`
- Modal run: `https://modal.com/apps/sid-ghodke/main/ap-eXYDZzA0envTyAIpQui8Ww`
- Pre-reg: [`TODO/factor-crypto-venue-test`](../TODO/factor-crypto-venue-test.md)
- Parent (equity result this ported): [`findings/factor-shorthorizon-representation`](factor-shorthorizon-representation.md)
- Methodology: [`findings/deflated-sharpe-leaderboard`](deflated-sharpe-leaderboard.md)

## Next experiment — funding-rate cash-and-carry

Per the TODO's `confirmed-null` decision branch and the
beyond-DCA research brief (`.research-beyond-dca.md` #2), the
highest-EV next venue-fit arc is **perp funding-rate cash-and-carry**
on crypto: long spot, short perp, harvest the funding-rate stream.
Mechanism is structural (perp shorts pay longs a funding payment to
keep perp price anchored to spot) and capacity-constrained
(funding-rate magnitude collapses as cash-and-carry inflows grow).
Free data via Binance/Bybit/OKX public funding-rate endpoints. He,
Manela, Ross 2023 is the canonical reference.

Build cost: ~1 person-day for the funding-rate loader, ~1 day for
the basis-computation module and backtest harness. Pre-reg should
lock: (a) spot-perp pair list, (b) funding-rate threshold for entry,
(c) execution friction (perp open/close + spot rebalance), (d) the
DSR-deflated Sharpe-difference bar versus a static BTC long.
