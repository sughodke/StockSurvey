# ss_stream

Point-in-time universe over a Stooq daily archive — the data layer
for backtests that don't pretend their universe is fixed.

## What it does

Reshapes Stooq's per-ticker `.txt` archive into a two-table parquet
event store, plus a `Universe` class that answers the only two
questions a survivorship-aware backtest actually asks each rebalance:

  * `active_at(date)` — which tickers were tradeable on this date?
  * `bars_between(start, end, tickers)` — give me their OHLCV.

Companion to `ss_loaders.load_stooq_matrix`, which produces the same
data as a fixed wide panel. Use the matrix for prototyping; use the
stream when universe churn matters.

## Why

A backtest that loads a `(T, N)` matrix of "all tickers that exist
today" silently conditions on survival. Tickers that delisted before
today aren't in the matrix at all, so your strategy never had a
chance to hold them and lose. Reported CAGR/Sharpe come out
optimistic by a known-but-uncomfortable amount.

The honest data model is an event stream:

```
{date, ticker, event_type, ...}
  event_type ∈ {listing, bar, delisting, ...}
```

`ss_stream` materializes that stream as parquet — listings encoded
implicitly via each ticker's first bar date, delistings via the last
bar date — and serves it through a query API instead of a matrix.

## Output schema

`ingest_stooq` writes two files under `<dst>/`:

```
instruments.parquet       one row per ticker
  ticker          str           e.g. "AAPL", "BRK-A"
  country         str           "us"
  exchange        str           "nasdaq" | "nyse" | "nysemkt"
  asset_class     str           "stocks" | "etfs"
  listing_date    datetime64    first bar in the source file
  last_seen_date  datetime64    last bar in the source file
  n_bars          int

bars/data.parquet         long-form OHLCV, sorted by (date, ticker)
  date            datetime64
  ticker          str
  open, high, low, close   float32
  volume          int64
```

At ~12K-ticker / ~13-year scale this is one ~500MB parquet for the
bars and a ~125KB parquet for the instruments. ZSTD-compressed,
float32 prices.

## Quick start

### CLI

```bash
# One-time: ingest the Stooq tree.
uv run ss-stream ingest --src ./StooqData --dst ./Output/stream

# Sanity-check the output.
uv run ss-stream info --path ./Output/stream --at 2015-06-01
```

The `--at` flag makes the survivorship effect visible:

```
instruments:       11,993
listing range:     1962-01-02 -> 2026-04-24
median bars/inst:  1,426

                              count
country exchange asset_class
us      nasdaq   etfs           938
                 stocks        4564
        nyse     etfs          2552
                 stocks        3634
        nysemkt  stocks         305

active on 2015-06-01: 4,098
```

### Python

```python
from ss_stream import Universe

u = Universe('Output/stream')

# Point-in-time universe membership.
u.active_at('2015-06-01')                      # set[str], ~4,098 tickers

# Long-form OHLCV slice.
u.bars_between('2024-01-02', '2024-01-31',
               tickers=['AAPL', 'MSFT', 'NVDA'])

# Wide panel for legacy matrix code.
u.panel('2024-01-02', '2024-01-31', field='close')
```

## Survivorship bias is now measurable

Counting tickers active each date directly exposes the bias the
matrix loader was hiding:

| Date       | `active_at` |
|------------|------------:|
| 2000-01-03 |         300 |
| 2008-09-15 |       2,733 |
| 2015-06-01 |       4,098 |
| 2020-03-23 |       5,845 |
| 2026-04-24 |      10,972 |

The 36× rise from 2000 → 2026 is **not** real listing growth —
NYSE+Nasdaq+NYSE-MKT had ~7K listings in 2000. It's the active-only
Stooq archive: every ticker that delisted between then and now was
purged, so the only "tickers active in 2000" are the ones that also
made it to 2026. Your backtest was implicitly trading a 300-name
survivor universe in 2000 and pretending that was the market.

## Limitations

  * **Survivorship-biased on the exit side.** Stooq's free per-exchange
    archive removes delisted tickers entirely, so the inferred
    listing dates are real (first bar of each file) but the
    delisting heuristic — `last_seen_date < today` — fires only for
    a handful of recently-stale names. Bear Stearns, Lehman, Enron,
    every dot-com casualty: not in the archive at all.
  * **No delisting reason or terminal return.** Even when a delisting
    fires, you don't know if it was an M&A premium, bankruptcy
    (-100%), or going-private. CRSP's `DLSTCD`/`DLRETX` aren't
    encoded.
  * **No historical index membership.** "S&P 500 on 2015-01-01"
    requires a separate constituents file; `ss_stream` only knows
    about exchange listings, not index inclusion.

## Forward path

The ingestion path is data-format-agnostic above the ticker-file
parser. Drop a survivorship-bias-free archive into the same `daily/`
tree (Norgate, Sharadar, etc.) and re-run `ss-stream ingest` — no
code changes. The "data stops → delisting" rule starts firing
correctly once the archive actually retains delisted history.

For honest research, **Norgate Data** (~$30–80/mo) is the de facto
retail-quant standard: native survivorship-bias-free, includes
delisted with reasons, point-in-time index membership. The CLAUDE.md
"Honest evaluation" section in `apps/regime/` discusses where this
ranks among the realism gaps still in the strategy.
