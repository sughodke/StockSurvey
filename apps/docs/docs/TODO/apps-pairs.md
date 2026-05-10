# `apps/pairs` — pair-spread mean reversion

**Status: in-flight (2026-05-10).** Scaffolding shipped — see
[`apps/pairs`](../apps/pairs.md). Classical baseline walk-forward
on factor-narrow is running; result lands as a leaderboard row +
finding when it completes. v2 ML head still parked pending v1
verdict.

## The prediction problem

For each cointegrated pair `(A, B)`, model the spread
`s_t = log(P_A,t / P_B,t)` as mean-reverting around a slowly-
moving fair value. Predict the *sign + magnitude* of the next
20-day spread change conditional on current `(s_t, s_{t-1}, …)`.
Trade signal: long-A / short-B (dollar-neutral) when spread is
high vs fair value and predicted to revert; flip when low.

This is structurally different from `apps/factor`:

- **Per-pair time-series**, not cross-sectional. The prediction
  problem is "this specific pair will revert," not "this name
  ranks above that name in tomorrow's return."
- **Long-short by construction.** A pair trade is one long leg +
  one short leg by definition; the EW-gate operational rule
  doesn't apply (the benchmark is zero, not market-EW).
- **Friction stack is roughly 2× equity-only** (each leg pays).
  Pair turnover is bursty (open + close at the spread crossing
  thresholds) rather than continuous like a 20-day rebal.
- **Liquidity matters more.** Pair trades are short-and-borrow on
  one side; we already use Corwin-Schultz spread as a liquidity
  proxy in the relational app. Reuse that gate.

## Test design

### Stage 0 — universe selection

- Source: `factor-narrow` (297 stooq_us_long names, the
  universe every recent factor row uses) or `stooq_us_long`
  (312). Start with factor-narrow for direct comparability.
- Pair candidates: all `(A, B)` with `A ≠ B`. Naive bound is
  C(297, 2) = ~44k pairs. Sector-restrict if needed (same
  sector ⇒ ~3k pairs).
- Cointegration screening on the *train* slice of each
  walk-forward window (no peeking):
  - Engle-Granger: regress `log(P_A) ~ β · log(P_B) + c`,
    ADF test on residuals at p < 0.05.
  - Johansen as a sanity backup for sector-bundle generalization
    (>2 names).
- Keep top-N pairs by ADF p-value per window; rebuild per
  window (cointegration is regime-specific — this is the same
  issue we hit with the relational analog scorer).

### Stage 1 — predictor

Two arms in parallel for the first run:

- **Classical baseline (no ML).** Trade z-score crossings: long
  spread when `z_t < −2σ`, short when `z_t > +2σ`, exit at
  `|z_t| < 0.5σ`. Half-life and σ estimated on train.
- **ML head.** Linear / MLP head trained on `(z_t, z_{t-1},
  ..., z_{t-20})` → predicts forward 20-day spread change.
  Train on cross-pair-pooled MSE *or* per-pair Pearson IC of
  prediction vs realized spread move.

Sharpe-aligned loss (`block_sharpe`) is *appropriate here*
because each pair's signal-to-noise is much higher than the
+0.005 cross-sectional return IC — the
[`factor-loss-pivot`](../findings/factor-loss-pivot.md)
operational rule doesn't apply when the underlying signal is
strong enough to justify concentration.

### Stage 2 — backtest harness

Per-pair PnL aggregator:
- For each (pair, t), maintain `position ∈ {long-spread,
  short-spread, flat}`.
- Daily PnL = `position · (s_{t+1} − s_t)` minus 10 bps × 2
  on each open / close (one for each leg).
- Aggregate across N pairs with `1/N` allocation per pair (or
  equal-Kelly if predictor confidence is quantified).
- Walk-forward: train cointegration + predictor on `train_window
  _blocks`, deploy on `val_window_blocks`, slide.

### Pre-registered cuts

| Outcome | Aggregate val Sharpe (N=top-50 pairs, 6 windows) | Verdict | Action |
|---|---|---|---|
| **Pass** | ≥ +0.50 mean, ≥ 4/6 positive windows | `confirmed-OOS` | Build the live-trading path: dollar-neutral pair trades on Alpaca, borrow check on the short leg. Ship the classical-baseline arm first if it dominates ML; ship ML if alpha > +0.20 over baseline. |
| **Marginal** | +0.20 to +0.50, ≥ 3/6 windows | `partial-OOS` | Stratify pairs by liquidity / sector. Most likely outcome: a few sector clusters work, others don't. Ship a sector-restricted version. |
| **Fail** | < +0.20 *or* ≤ 2/6 positive windows | `confirmed-null` | Pair-spread mean reversion not an alpha source on this universe at this horizon. Move to [`apps/vol`](apps-vol.md) (option 3 of 3). |

Sharpe at the strategy level — not Sharpe of the spread itself
(which is well-known to be high in-sample, low OOS for
overfit cointegration screening). The cut is on the *deployable
portfolio*, not on the predictor's R².

## Implementation scope

### App scaffolding (~250 LoC)

```
apps/pairs/
├── pyproject.toml                          # apps/factor as template
├── README.md
├── src/pairs/
│   ├── __init__.py
│   ├── cointegration.py                    # Engle-Granger + Johansen
│   ├── pair_universe.py                    # screening + persistence
│   ├── spread.py                           # z-score, half-life, σ
│   ├── predictor.py                        # classical + ML heads
│   ├── backtest.py                         # per-pair PnL harness
│   ├── walkforward.py                      # rolling train/val
│   ├── live.py                             # broker path (later)
│   └── cli.py                              # `ss-pairs` subcommands
├── scripts/
│   ├── screen_pairs.py
│   ├── run_baseline.py                     # classical only
│   └── run_walkforward.py                  # ML head + classical baseline
└── tests/
```

### What lives in shared `packages/`

- Cointegration screening could live in a new
  `packages/cointegration/` if `apps/relational` ever wants
  similar Engle-Granger machinery; **don't pre-design** —
  start in `apps/pairs` and lift later if a real second consumer
  emerges.
- Per-pair PnL harness: same call. `apps/pairs/backtest.py`
  for v1; lift to `packages/portfolio/` only if shared.

### Reuses (already exists)

- `ss_loaders.load_stooq_matrix` + ticker manifests.
- `ss_indicators.corwin_schultz_spread` for liquidity gating.
- `ss_features` walk-forward block generator.
- `ss_portfolio.metrics` for Sharpe / Sortino / drawdown.
- `ss_portfolio.broker` for live; per-pair order pairing wraps
  it (one buy + one sell per pair-trade signal).

Total new scope: ~600 LoC including tests. ~2 days of work.

## What this TODO is *not* a test of

- Not a test of triplets / Johansen >2 (could come later if
  `confirmed-OOS` for pairs).
- Not a test of intraday spread mean reversion (different
  data cadence; daily-bar only for v1).
- Not a test of ETF arbitrage (e.g. SPY vs constituent basket)
  — those have known structural pricing relationships and
  belong in their own app.
- Not a test of volatility-of-spread vs volatility-of-leg
  pricing (that's `apps/vol` territory).

## Implementation order

1. Scaffold `apps/pairs` workspace member, `pyproject.toml`,
   `__init__.py`, `cli.py` skeleton.
2. `cointegration.py`: Engle-Granger via numpy + scipy ADF.
   Smoke test on a known cointegrated pair (e.g. KO + PEP,
   MSFT + ORCL).
3. `pair_universe.py`: screen factor-narrow per train slice,
   persist top-N pair list per window.
4. `spread.py` + `predictor.py` (classical baseline first —
   no ML — to set the floor).
5. `backtest.py` + `walkforward.py`: per-pair PnL aggregator,
   walk-forward runner.
6. Run classical baseline. Land leaderboard row.
7. Add ML predictor (linear head with `block_sharpe` loss),
   compare to classical.
8. If pass: `live.py` + Alpaca pair-trading path.

## Why ranked second

`apps/gate` is cheaper — same data, same loaders, same `apps/
factor` infrastructure, only the target changes. `apps/pairs`
needs new infra (cointegration screening, per-pair PnL harness,
pair-trading broker path) but no new data source. `apps/vol`
needs all of that *plus* a new data source and a different
market structure.

If `apps/gate` confirms an alpha source (drawdown gating
over EW) we may not need pairs at all to clear shippability.
But if `apps/gate` nulls out, pairs is the natural next test
because the prediction problem is fundamentally different
(time-series mean reversion vs cross-sectional ordering vs
binary regime classification — three orthogonal information
hypotheses).
