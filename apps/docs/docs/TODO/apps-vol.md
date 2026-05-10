# `apps/vol` — IV-vs-realized vol mispricing

**Status: queued (rank 3 of 3 in the prediction-problem pivot).**
Highest cost in the pivot order — needs new data source,
new friction stack, new live-trading path. Tracking design
ahead of time so we know what we're committing to.

## The prediction problem

For each underlying ticker `T` at time `t`, predict whether
the implied volatility of next-week ATM options
`IV_t(T, expiry=t+5d)` over- or under-estimates the realized
volatility that will actually print
`RV_{t,t+5d}(T) = std(log_returns over (t, t+5d))`.

Trade signal:
- Long straddle (or variance swap) when `RV − IV > threshold`
  → realized exceeds implied → option premium underpriced.
- Short straddle when `IV − RV > threshold` → implied exceeds
  realized → option premium overpriced.

This is a *totally separate alpha source* from anything in the
repo today: not cross-sectional, not equity time-series, not
cointegration. It exploits the well-documented variance risk
premium (VRP) — short-vol tends to make money on average,
which means we'd run mostly short with the predictor steering
when to flip long.

### Why this is structurally different

| Property | `apps/factor` (rank-IC) | `apps/pairs` (cointegration) | `apps/vol` (IV/RV) |
|---|---|---|---|
| Data source | equity prices | equity prices | equity *and* options chain |
| Prediction target | cross-sectional return rank | spread mean reversion | next-period RV minus IV |
| Trade horizon | 20 day | days to weeks (cointegration half-life) | 5-30 day (option expiry) |
| Friction stack | 10 bps round-trip | 20 bps (2 legs) | **100-500 bps** (option bid/ask) |
| Liquidity gate | Corwin-Schultz spread | same | options OI + bid/ask spread |
| Deployment | long-only top-N or LS | dollar-neutral pair | options spreads on options-broker |
| Benchmark | passive EW (`alpha = model − EW`) | zero (LS by construction) | short-vol VRP carry (~+0.5 Sharpe baseline) |

The VRP baseline is the equivalent of "passive EW" for this
strategy class — short-straddle without any predictor earns
~+0.3 to +0.5 Sharpe historically (Coval & Shumway 2001 et seq).
Any predictor we ship has to add alpha *over that*, not merely
over zero.

## Test design

### Stage 0 — data

Source: DoltHub options chain dump
([`packages/loaders`](https://github.com/sughodke/StockSurvey/tree/master/packages/loaders)
already pulled this for the relational arc per CLAUDE.md "DoltHub
IV data is on hand"). Schema review needed:

- Daily snapshots of options chain per ticker.
- Required fields: `ticker, date, expiry, strike, type, bid,
  ask, volume, open_interest, iv` (or compute IV from prices).
- Universe restriction: liquid options only.
  - Open interest ≥ 100 contracts.
  - Bid/ask spread ≤ 5% of mid.
  - Time to expiry ≥ 3 trading days, ≤ 30.
  - ATM ± 0.5 strike (delta in [0.4, 0.6]).
- Initial universe: SPY + top 50-100 S&P names by options
  volume. Don't try to use the full equity universe — option
  liquidity is concentrated in mega-caps and ETFs.

### Stage 1 — IV / RV target

For each (ticker, date) where we have a clean ATM IV reading
for an `expiry ≈ date + H` (H ∈ {5, 10, 20} day horizons):

```
IV_t       = ATM IV reading at t for expiry t+H
RV_{t,t+H} = annualized stdev of log returns over (t, t+H]
gap_t      = RV_{t,t+H} − IV_t   # positive ⇒ realized exceeded implied
```

Predict `gap_t` from features observed at `t`:
- Lagged realized vol over multiple windows (5d, 20d, 60d).
- IV term-structure slope (3m IV minus 1m IV).
- Skew (25-delta put IV minus 25-delta call IV).
- Recent equity returns (mean reversion / momentum proxies).
- Cross-asset: SPY IV term-structure as macro vol regime.
- Calendar: earnings flag, FOMC week, holiday week.

Models:
- Classical baseline: `predict(gap_t) = c + α · (RV_{t-20d}
  − IV_t)` — vol of vols / mean reversion of basis.
- ML head: linear / MLP over the feature stack, trained on
  cross-sectional MSE pooled across (ticker, date).

### Stage 2 — backtest harness

Per-position PnL accounting is materially different from
equity:
- **Open straddle:** buy 1 call + 1 put at the ATM strike,
  pay 2 × ask.
- **Hold:** mark-to-market at mid daily.
- **Close at expiry:** value = `max(S_T − K, 0) + max(K −
  S_T, 0)` for call + put; sell back at bid if early-exit.
- **Cost stack:** entry (2 × ask − 2 × mid) + exit (2 × mid
  − 2 × bid). For 5% bid/ask spread, that's roughly 5% per
  side = 10% round-trip = 1000 bps. *Three orders of
  magnitude* more than equity.

This dominates the test — the predictor has to be *very*
right to clear options costs. Coval-Shumway-style short-vol
strategies typically run net of ~5-15% annualized friction.

### Pre-registered cuts

| Outcome | Walk-forward Sharpe vs short-vol baseline | Verdict | Action |
|---|---|---|---|
| **Pass** | ≥ baseline + 0.30, ≥ 4/6 windows positive alpha | `confirmed-OOS` | Ship: predictor-gated VRP harvester. Live trading needs options-broker integration (Tastytrade / IBKR — Alpaca options support is limited to a small ticker set). |
| **Marginal** | baseline + (0.10, 0.30) | `partial-OOS` | Stratify by ticker. Most likely outcome: a few high-liquidity names (SPY, QQQ) work, single-name doesn't. Ship index-only. |
| **Fail** | < baseline + 0.10 | `confirmed-null` | The VRP carry is the strategy; no predictor edge above it. Ship pure short-vol if its standalone Sharpe is acceptable, or shelve the workstream entirely. |

## Implementation scope

### App scaffolding (~400 LoC)

```
apps/vol/
├── pyproject.toml                          # apps/factor as template
├── README.md
├── src/vol/
│   ├── __init__.py
│   ├── data.py                             # DoltHub options loader
│   ├── iv_chain.py                         # ATM extraction, term-
│   │                                       # structure, skew
│   ├── target.py                           # IV/RV gap construction
│   ├── features.py                         # lagged RV, term-slope,
│   │                                       # skew, calendar flags
│   ├── predictor.py                        # classical + ML
│   ├── options_pnl.py                      # straddle / spread MTM
│   ├── backtest.py                         # walk-forward harness
│   ├── live.py                             # later
│   └── cli.py                              # `ss-vol` subcommands
├── scripts/
│   ├── prep_options_panel.py
│   ├── run_baseline.py                     # short-vol-no-predictor
│   └── run_walkforward.py                  # predictor-gated
└── tests/
```

### What lives in shared `packages/`

- `packages/loaders/` extension: DoltHub options loader. **The
  only piece that is clearly shared** — relational arc already
  imported it. Lift to `ss_loaders` if it isn't already.
- Options-PnL machinery stays in `apps/vol` for v1.

### Reuses (already exists)

- `ss_loaders` for equity ticker + price data (option backtest
  needs underlying for MTM).
- `ss_portfolio.metrics` for Sharpe / Sortino / drawdown.
- Walk-forward block generator from `ss_features`.

Everything else is new. Total new scope: ~1000 LoC including
options-PnL machinery and tests. ~5 days of work, plus
upfront data-pipeline work that's hard to estimate without
seeing the DoltHub schema.

## What this TODO is *not* a test of

- Not a test of *individual stock* options arbitrage with
  fundamental catalyst features (earnings / FDA / M&A) — that's
  event-driven trading, much wider scope.
- Not a test of variance swaps, gamma scalping, or other
  vol-trading variants. ATM straddle only — the simplest
  expression of "buy / sell vol."
- Not a test of intraday vol arbitrage (different data
  cadence, totally different infrastructure).
- Not a test of *cross-asset* vol relationships (equity vol
  vs treasury vol etc.). Single-asset vol only.

## Implementation order

1. Audit DoltHub options data on disk: schema, density per
   ticker, time coverage, IV cleanliness. If the data isn't
   there or is too noisy, this stops here — no point building
   a backtest on bad inputs.
2. Scaffold `apps/vol` workspace member.
3. `data.py` + `iv_chain.py`: load + clean + ATM extract.
4. `target.py`: IV/RV gap with multiple horizons.
5. `backtest.py` + `options_pnl.py`: short-straddle baseline
   harness with realistic options friction. Run unconditionally
   to establish the VRP-carry baseline Sharpe.
6. `predictor.py`: classical + ML; walk-forward eval gating
   the straddle.
7. Land 2-3 leaderboard rows (baseline VRP, classical-gated,
   ML-gated).
8. If pass: `live.py` + options-broker integration.

## Why ranked third (and what would change the order)

Three reasons it's last:
1. **Data risk.** DoltHub options data needs schema audit
   before we know if it supports the test at all.
2. **Friction-stack risk.** Options bid/ask is 100-1000 bps —
   the predictor has to be *much* more skillful than what we've
   shown ability to produce so far.
3. **Live-trading risk.** Alpaca's options support is thin;
   shipping requires a new broker integration (Tastytrade /
   IBKR) which is its own multi-week project.

What would promote this to first:
- If `apps/gate` and `apps/pairs` both null on equity strategies
  (i.e. equity alpha is fully exhausted at our budget), then
  vol becomes the only orthogonal alpha source on the table
  and is worth the upfront investment.
- If the DoltHub data turns out to be cleaner than expected and
  covers a long enough span (e.g. 10+ years on SPY + top
  50 single names), the data-risk halves.

But for now: ship the cheaper tests first, return here when
they've informed us what compute / infra investment is
justified.
