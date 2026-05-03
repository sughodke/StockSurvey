# `apps/relational/`

Research scaffolding for **relational-CWT alpha** — scoring families
that operate on CWT differences across tickers (NOTES.md "Where the
real alpha lives — relational CWTs"), instead of CWT divergences in
isolation.

Four ideas tracked, in priority order:

| # | Name | Status |
|---|---|---|
| **1** | **Stock minus sector** (excess CWT divergence) | **week-1 — wired** |
| 2 | CWT of cross-sectional dispersion | TODO |
| 3 | CWT of cross-sectional correlation | TODO |
| 4 | Cross-sector coherence (sector-pair CWT) | TODO |

## Why a separate app

`apps/regime/` runs the per-stock CWT-divergence strategy that is becoming live-trading focused; `apps/notebook/` houses the multi-head CNN training stack (replay/, scoring/). This app sits between them as the **relational-scoring research scratchpad** — pure CWT arithmetic, no neural-network training, no live broker. If a recipe pans out, it can be promoted into its own production app or folded back into `regime`.

Dep-wise, this app intentionally avoids the trainer-side dragons (no tinygrad, no JAX-Adam path) — only the numpy `ss_indicators` + `ss_wavelets` + `ss_portfolio` primitives, plus `bt`/`vectorbt` for backtesting.

## File map

```
apps/relational/
├── pyproject.toml
├── README.md
└── src/relational/
    ├── __init__.py            re-exports the public API
    ├── sectors.py             ticker → GICS sector mapping (Phase-2 universe);
                               canonical 11 SPDR sector ETFs
    ├── aggregates.py          sector_series(prices, mode='equal') —
                               equal-weighted constituent aggregate; ETF
                               and cap-weighted modes are TODO
    ├── scoring.py             excess_divergence_scores(...) +
                               weights_excess_regime(...) — drop-in for
                               any vectorbt/bt loop that already accepts
                               a (n_dates, n_tickers) weights df
    ├── cli.py                 ss-relational subcommands
    └── research/
        └── backtest_sector_excess.py
                               head-to-head bt backtest: weights_regime
                               (baseline) vs weights_excess_regime on
                               the Phase-2 universe + dates
```

## Math (idea #1)

For each (date, ticker):

```
score[stock, t] = divergence(stock_recent, stock_hist)
                − divergence(its_sector_recent, its_sector_hist)
```

Both divergences use the same primitives as `regime.trainer.weights_regime` (KL/JS/cosine/L2 over CWT power, uniform per-scale weighting). The sector aggregate's CWT and divergence are computed once per sector per date and broadcast back to constituents via `sectors.ticker_to_sector_idx`.

The interpretation: a stock's bare divergence is interesting; a stock's divergence *net of its sector's* is more interesting because it isolates the **idiosyncratic** regime shift — the stock doing something its sector isn't. The first might be "AAPL's regime shifted because all of XLK shifted." The second isolates "AAPL is shifting *more than tech is overall*."

## Sector aggregate source — current default

**Equal-weighted constituents from the existing universe.** Zero new data, zero new infra; validates the idea before investing in ETF data plumbing. Single-constituent sectors (e.g. Energy = {XOM} in the Phase-2 universe) produce a sector aggregate identical to the constituent, so excess_divergence is exactly 0 for those tickers — documented degenerate case, not a bug.

**Future swaps** (one-flag changes when ready):
- `sector_mode='etf'` — fetch XLK/XLF/etc. via ss_loaders, use the actual sector ETF series.
- `sector_mode='cap'` — market-cap-weighted; needs shares-outstanding panel.

## Running the head-to-head

```bash
ss-relational head-to-head --data-dir ./StooqData --top-n 10 \
    --start 2013-01-29 --end 2025-12-11
```

or invoke the script directly:

```bash
uv run python -m relational.research.backtest_sector_excess \
    --data-dir ./StooqData --top-n 10
```

Outputs:
- `Output/relational-sector-excess-equity.png` — equity curves side-by-side
- `Output/relational-sector-excess-stats.txt` — bt's full stats table

## Tests

```bash
uv run pytest apps/relational/tests/
```

Coverage:
- Sector mapping: every Phase-2 ticker is mapped; `sectors_for_universe`
  raises on unknown tickers by default; `ticker_to_sector_idx` returns
  the correct column indices.
- Scoring sanity: excess_divergence is exactly 0 when stock = sector
  (single-constituent case); positive when stock diverges from a
  sector trending the other way.

## Open questions / TODO

- Promote `weights_regime` baseline to ss_portfolio so this app and
  `apps/regime/` share one canonical impl (currently inlined in the
  research script to keep relational independent of regime).
- Ideas #2-4 from NOTES.md (cross-sectional dispersion, correlation,
  sector-pair coherence) — same scaffolding pattern: add a scoring
  function, register a `weights_<name>` builder, optional CLI subcommand.
- Walk-forward Optuna sweep — once the head-to-head shows a meaningful
  edge, add `research/optimize_excess_regime.py` mirroring
  `regime.research.optimize_regime`.
