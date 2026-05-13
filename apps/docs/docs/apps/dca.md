---
tags:
  - dca
  - live
  - deployment
---

# `apps/dca` — Multi-asset DCA (canonical live strategy)

Status: **Live (canonical, 2026-05-13)** — fixed-target equal-weight
basket on the 13-asset Phase 4d universe (9 SPDR sector ETFs + TLT/IEF
+ GLD/DBC). Quarterly rebalance with a 5% per-name drift threshold
for off-cadence triggers. Built on `ss_portfolio.broker.AlpacaBroker`.

## Why DCA is the deployable strategy

[`cfr-vs-dca-realistic`](../findings/cfr-vs-dca-realistic.md) is the
load-bearing finding. The short version:

- Phase 4d's CFR active strategy posted **+0.056 raw alpha** vs EW on
  the same 13-asset universe.
- Apply realistic deployment friction (50 bps/yr for the bot —
  slippage, short-term tax inefficiency from 11× turnover,
  operational labor; 5 bps/yr for DCA — quarterly rebal, ETF tax
  efficiency) and the alpha collapses to **+0.015 net**.
- DCA's **worst-window Sharpe (+0.642)** beats CFR's worst window
  (+0.434) by **+0.21** — DCA wins on regime stress.
- On a $100k portfolio the realistic alpha is ~$200/yr, comfortably
  exceeded by the operational labor cost of running the bot.
- DCA is therefore the canonical live strategy; CFR Phase 4d is
  archived as a `confirmed-null` on realistic-alpha basis.

## What it does

```bash
uv run ss-dca live --params Output/dca-multiasset.json --dry-run     # default
uv run ss-dca live --params Output/dca-multiasset.json --live        # actually trade
```

Each invocation:

1. Loads the canonical 13-asset checkpoint
2. Pulls last 10 trading days from Alpaca (just enough for the
   latest close)
3. Evaluates the **cadence + drift gate** — skip rebalance unless
   cadence floor (80 trading days ≈ quarterly) is met OR any single
   name has drifted ≥ 5% from target
4. If gated through: build trades against current positions, apply
   the per-name cap, submit (or print, in dry-run)
5. State file (`~/.dca-state.json`) advances only on successful
   real submissions

## Risk rails

Five rails, each aborting with a clear reason rather than silently
continuing:

| # | Rail | Default | Override |
|---|---|---|---|
| 1 | Kill-switch file | `~/.dca-killswitch` | `--killswitch PATH` |
| 2 | Data freshness | 3 days | `--max-data-age-days N` |
| 3 | **Cadence + drift gate** (DCA-specific) | 80d cadence floor + 5% drift threshold | `--force-rebal` |
| 4 | Per-name cap (water-fill) | 0.15 (above 1/13 target — diagnostic-only) | `--max-position F` |
| 5 | Dry-run by default | `--dry-run` | `--live` |

The DCA-specific cadence/drift gate is the only thing this app does
that `regime` / `relational` don't — it lets you cron the script
weekly and have it no-op most weeks, fire on schedule, AND respond
to large drift events (a single ETF doubling).

## Canonical checkpoint

`Output/dca-multiasset.json` — written by
`apps/dca/scripts/build_checkpoint.py`:

| Field | Value |
|---|---|
| `name` | `multiasset-13etf-ew` |
| `universe` (13) | DBC, GLD, IEF, TLT, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY |
| `target_weights` | 1/13 = 7.6923% each |
| `min_rebal_days` | 80 (≈ quarterly) |
| `drift_threshold` | 0.05 (5% absolute weight) |
| `commission_bps` | 5.0 |
| `backtest_start` / `_end` | 2005-02-25 / 2025-12-11 |
| `backtest_sharpe` (full panel) | +0.673 |
| `backtest_cagr` | +8.4% |
| `backtest_max_drawdown` | −40.7% (GFC) |

Rebuild from scratch:

```bash
uv run python apps/dca/scripts/build_checkpoint.py
```

## Modules

- `dca.persist` — `DCACheckpoint` JSON I/O with sum-to-one + universe
  match validation
- `dca.state` — `~/.dca-state.json` last-rebal-date tracker
- `dca.live` — orchestrator with the 5 risk rails
- `dca.cli` — `ss-dca live` subcommand
- `apps/dca/scripts/build_checkpoint.py` — canonical checkpoint
  builder

## Operator playbook

```bash
# One-time: build the canonical checkpoint
uv run python apps/dca/scripts/build_checkpoint.py

# Cron (weekly): dry-run preview to a log
uv run ss-dca live --params Output/dca-multiasset.json
# → "gate held: 12d since last rebal, max drift 1.34% < 5%" most weeks
# → "cadence floor met (84d ≥ 80d)" once a quarter; lists trades

# When ready: flip to live
uv run ss-dca live --params Output/dca-multiasset.json --live

# Halt anytime
touch ~/.dca-killswitch

# Manual rebal override (rare)
uv run ss-dca live --params Output/dca-multiasset.json --live --force-rebal
```

## Caveats

- The basket is well-tested for the **post-Volcker monetary regime**
  (1985-2025). Pre-1985 stagflation / depression regimes are not in
  the data — the basket would behave very differently in 1973-74-style
  conditions where TLT/IEF specifically *hurt*.
- The bond sleeve (TLT, IEF) had a 40-year tailwind from falling
  rates that physically cannot repeat. Bias-corrected Sharpe is
  closer to **+0.55** than the historical +0.67.
- Max drawdown of **−40.7%** during GFC is the realistic worst case
  in the available data. A 1970s-style regime would likely be worse.

## What this app intentionally does NOT do

- No learned weights; no model training; no checkpoint refresh cycle
- No sector tilts, no momentum overlay, no regime classifier
- No tactical hedging, no options overlay
- No leverage

These are all things that the broader research stack
([`apps/cfr`](cfr.md), [`apps/factor`](factor.md),
[`apps/relational`](relational.md), [`apps/regime`](regime.md))
explored extensively and that
[`cfr-vs-dca-realistic`](../findings/cfr-vs-dca-realistic.md)
established do not survive realistic deployment friction at the
scale and signal regime we have. Adding any of them to this app
would re-introduce the operational tax that DCA exists to avoid.
