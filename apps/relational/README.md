# `apps/relational/`

Research scaffolding for **relational-CWT alpha** — scoring families
that operate on CWT differences across tickers (apps/docs/docs/notes.md
"Multi-stock CWT framings — Where the real alpha may live"), instead of
CWT divergences in isolation.

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
├── ideas.md                   open recipe sketches
├── scripts/
│   └── build_canonical_checkpoints.py
│                              writes Output/relational-{strategy}.json for
│                              the six scoreboard winners
└── src/relational/
    ├── __init__.py            re-exports the public API
    ├── sectors.py             PHASE2_TICKERS + GICS mapping; SPDR ETF set
    ├── aggregates.py          sector_series(prices, mode='equal') aggregate
    ├── fingerprints.py        per-(ticker, date) CWT scalogram fingerprints
    ├── scalogram_cache.py     disk-cached CWT bundles (cache key = inputs hash)
    ├── scoring.py             original sector-excess (idea-1)
    ├── empirical_sectors.py   k-means cluster aggregate (idea-A)
    ├── empirical_sectors_gmm.py
    │                          soft-cluster GMM variant (Phase-10)
    ├── analog_knn.py          k-NN analog forecasting (idea-B)
    ├── farthest.py            centroid-distance scoring (idea-C)
    ├── diversify.py           greedy farthest-first thinning (idea-D)
    ├── regime_velocity.py     fingerprint-space directed motion (Phase-11)
    ├── transitions.py         transition-triggered rebal gate (Phase-9)
    ├── cluster_tracking.py    Hungarian-stabilized cluster IDs across refits
    ├── pairs.py               pair-trade construction (falsified, Phase-7)
    ├── nn_pairs.py            nearest-peer hedge (falsified, Phase-12)
    ├── short_vol.py           short-vol overlay (falsified, Phase-5)
    ├── iv_data.py             DoltHub IV loaders (Phase 1-3)
    ├── ot_stress.py           optimal-transport stress overlay
    ├── bocpd.py               Bayesian online change-point detector
    ├── scale_energy.py        scale-energy aggregation primitives
    ├── sizing.py              vol-target / risk-parity overlays
    ├── persist.py             RelationalCheckpoint JSON I/O
    ├── inference.py           target_weights(...) — strategy dispatch +
                               Corwin-Schultz spread gate + last-bar slice
    ├── live.py                run_live(...) sharing ss_portfolio.broker +
                               the four risk rails (kill-switch / freshness
                               / per-name cap / dry-run default)
    ├── cli.py                 ss-relational live + ss-relational head-to-head
    └── research/              full bt diagnostic suite (10+ scripts)
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

## Final scoreboard

Four scoring families have been wired up alongside the original
GICS sector-excess idea, all running on per-(ticker, date) **causal
CWT scalogram fingerprints** — flattened `(scales × w)`-dim slices
of the CWT bundle, computed once and disk-cached via
`relational.scalogram_cache`. Three are *scorers*; D is a *selector*
that operates on top of any other ranking.

Phase-2 universe (21 mega-caps), 2013-01-29 → 2025-12-11, top-N=10,
rebal=20d, commission=10bps. Baseline = `ss_portfolio.weights_regime`
(per-stock CWT-power divergence vs the same stock's history).

| Tag | Module | Mechanism | Sharpe | CAGR | Max DD | Calmar | vs baseline |
|---|---|---|---|---|---|---|---|
| baseline | `ss_portfolio.weights_regime` | self-divergence (own past) | 1.07 | 20.7% | -38.8% | 0.53 | — |
| **A** | `relational.empirical_sectors` | **peer-divergence** (k-means cluster aggregate vs static GICS) | **1.13** | **22.4%** | -38.0% | **0.59** | **win** |
| **C** | `relational.farthest` | **outlier** (L2 distance from cross-sectional fingerprint centroid) | **1.13** | 21.0% | **-32.2%** | **0.65** | **win** |
| B | `relational.analog_knn` | analog forecast (k-NN over historical fingerprints, score = mean forward return) | 1.07 | 20.9% | -37.1% | 0.56 | wash |
| D | `relational.diversify` | greedy farthest-first thinning of top-pool | 0.94 | 17.2% | -35.4% | 0.49 | loss |
| #1 | `relational.scoring` | static-GICS sector-excess (the original) | 0.99 | 19.1% | -39.5% | 0.49 | loss |

**Pattern:** both winners use a *cross-sectional* reference point —
empirical clusters (A) and the per-date market centroid (C). The
losers either pin the wrong reference (#1: GICS labels are stale)
or thin too aggressively on a small basket (D). B's k-NN forecast
on a 21-name universe rediscovers most of what `weights_regime`
already captures.

Idea C has the best Calmar (0.65) of any tested strategy — the
drawdown improvement is meaningful (-32.2% vs -38.8%) and is not
matched by a CAGR loss. Idea A has the best CAGR / yearly Sharpe
(1.41) and the best worst-year resilience (-2.1% vs -21.5% baseline).

### What the rankings mean

Three of the four scorers (baseline, A, C) are **direction-agnostic**
— a high score flags "stock dislocating right now," not "stock
predicted to rise." Only B explicitly forecasts forward returns.
Long-only baskets of dislocators outperform buy-and-hold because in
equities, idiosyncratic vol expansions and regime shifts are biased
toward positive resolution (drift premium + asymmetric news flow),
not because the signal calls direction.

Internal name for any composite blend: **dislocation score**. Avoid
"future winners" / "about to pop" framing — the same mechanism
will happily flag a stock crashing on bad earnings, and that's a
real failure mode of the strategy rather than a bug in the score.

## Paper trading via Alpaca

Six scoreboard winners are wired through `ss-relational live` for paper
trading, sharing `ss_portfolio.broker.AlpacaBroker` and the same four
risk rails as `regime live`.

```bash
# 1. Generate the canonical checkpoints (~30s; loads stooq_us_long for velocity)
uv run python apps/relational/scripts/build_canonical_checkpoints.py

# 2. Set Alpaca paper credentials (defaults to paper endpoint)
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...

# 3. Dry-run before submitting anything
uv run ss-relational live --params Output/relational-empirical.json --dry-run

# 4. Flip to live submit (still paper, still capped, still observable)
uv run ss-relational live --params Output/relational-empirical.json --live
```

Six checkpoints are produced:

| file | strategy | universe | top_n | docked Sharpe |
|---|---|---:|---:|---:|
| `relational-empirical.json` | k-means cluster (idea-A) | 21 (Phase-2) | 10 | +1.07 |
| `relational-gmm.json` | GMM soft-cluster (Phase-10) | 21 (Phase-2) | 10 | +1.10 |
| `relational-analog.json` | k-NN analog (idea-B) | 21 (Phase-2) | 10 | +1.10 |
| `relational-farthest.json` | centroid-distance (idea-C) | 21 (Phase-2) | 10 | +1.07 |
| `relational-diversified.json` | farthest-first thin (idea-D) | 21 (Phase-2) | 10 | +1.07 |
| `relational-velocity.json` | fingerprint velocity (Phase-11) | 312 (wide) | 20 | +0.60 |

**Phase-2 wins are mega-cap-specific.** Phase 8 of the relational arc
([`relational-arc-synthesis`](../docs/docs/findings/relational-arc-synthesis.md))
showed all four ideas (A/B/C/D) degrade from Sharpe ~1.1 to ~0.4 when
run on the wider 312-ticker `stooq_us_long` universe instead. The
checkpoint `train_sharpe`/`val_sharpe` fields record the actual
Phase-2 result; do not extrapolate to other universes without
retraining. Velocity is the only winner that was specifically
designed for and validated on the wide universe.

Risk rails (mirroring `regime live` exactly):
1. Kill-switch file (default `~/.relational-killswitch`).
2. Data freshness — abort if latest bar > `--max-data-age-days` (default 3).
3. Per-name cap — `--max-position` (default 0.25), water-fill via
   `ss_portfolio.apply_position_cap`. Cap distributes among nonzero
   names only, so spread-gated illiquid names cannot be re-introduced.
4. Dry-run by default — `--live` is opt-in.

`LiveRunResult.rejected_orders` surfaces per-symbol Alpaca rejections
(non-fractionable / sub-cent notional / etc.) so partial-submit days
don't go unnoticed.

## Open questions / TODO

- Ideas #2-4 from `apps/docs/docs/notes.md` (cross-sectional dispersion,
  correlation, sector-pair coherence) — same scaffolding pattern: add
  a scoring function, register a `weights_<name>` builder, optional
  CLI subcommand.
- Walk-forward Optuna sweep — once the head-to-head shows a meaningful
  edge, add `research/optimize_excess_regime.py` mirroring
  `regime.research.optimize_regime`.
- Transition-triggered cron mode (Phase-9) — wraps `ss-relational live`
  in a "should I trade today?" gate, only fires on detected fingerprint-
  cluster transitions. Reduces 325 scheduled rebals to ~25 over the
  same period; +0.21 Sharpe in the diagnostic.
