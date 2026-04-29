# TODO

## Extract shared CLI args into `packages/cli` (`ss_cli`)

CLI flags for "where's the data + when + save where" are duplicated across
~7 scripts in two apps. Two distinct flag groupings exist; both should
live in a single shared package so future scripts opt-in by one
function call instead of copying argparse blocks.

**Distribution name:** `ss-cli` &nbsp;**Import name:** `ss_cli`
**Layout:** `packages/cli/src/ss_cli/__init__.py` (mirrors the
`ss-loaders` / `ss-indicators` convention).

**API to export:**

```python
add_single_ticker_loader_args(parser)
    # --stooq-dir, --kaggle-dir, --start, --end
add_universe_loader_args(parser)
    # --data-dir (required), --start, --end
add_save_args(parser, *, default_output_dir='Output')
    # --save, --output-dir
```

The two loader helpers stay separate because the flag names actually
differ (single-ticker scripts can pick one of two sources;
universe scripts have a single `--data-dir`). `add_save_args` is
shared by both groupings.

**Call sites to migrate:**

Single-ticker (notebook app):
- `apps/notebook/src/ss_notebook/scalogram.py`
- `apps/notebook/src/ss_notebook/scalogram_video.py`
- `apps/notebook/src/ss_notebook/replay.py`

Universe (regime app):
- `apps/regime/src/regime/cli.py`
- `apps/regime/src/regime/research/backtest_bt.py`
- `apps/regime/src/regime/research/optimize_regime.py`
- `apps/regime/src/regime/research/backtest_ranking.py`

**Drift to normalize during migration:**
- `backtest_ranking.py` uses `--start-date` / `--end-date`; everything
  else uses `--start` / `--end`. Standardize on `--start` / `--end`
  and keep `--start-date` / `--end-date` as deprecated aliases for
  one release if any external scripts call it.

**Workspace wiring:**
- `packages/cli/pyproject.toml` with `[tool.hatch.build.targets.wheel]
  packages = ["src/ss_cli"]`.
- Root `pyproject.toml` already includes `packages/*` in the workspace
  members glob, so `uv sync --all-packages --inexact` picks it up.
- Each consumer adds `ss-cli` to its `dependencies` and
  `[tool.uv.sources] ss-cli = { workspace = true }`.

**Out of scope:**
- Legacy `apps/v1/scripts/*` `--save` flags. Parked workflow.
- The `regime live` arg block (`--params`, `--dry-run`, `--max-position`,
  `--killswitch`, `--max-data-age-days`) — single call site, no reuse.

## Add a realized-volatility (and/or autocorrelation) head to ss-replay

Once multi-head CNN is shaken out on rsi/macd/price, add another head
that tests a *higher-order* statistic — something the bundle doesn't
expose as a literal input feature. Right now every target is either
level-flavored (price) or a near-linear function of recent returns
(RSI, MACD).

**Why:** the encoder has CWT power per scale + raw returns + (mu, sigma).
RV is `std(returns_{t-19:t})` — variance of recent returns. CWT power
explicitly factorizes power by scale so summing per-scale power *should*
recover aggregate RV well. If multihead nails RV, the encoder genuinely
captures vol structure, not just first-order direction. If it underfits,
the bundle is missing variance-style information.

**ADX is not a viable substitute for RV here:** ADX measures trend
*consistency* (sign-agnostic but direction-aware), needs OHLC (high,
low, close) for `+DM/-DM/TR`. Yahoo close-only blocks it. RV measures
*magnitude* (sign-agnostic, direction-agnostic), needs only close.
Different signals, different dimensions of "market activity." A
close-only proxy for ADX-style trend strength would be the
autocorrelation of returns over a window (or Hurst exponent) — also
higher-order, also feasible.

**Implementation:**
- Add `realized_vol_n` and (optionally) `return_autocorr_n` to
  `TARGET_NAMES` in `apps/notebook/src/ss_notebook/replay/features.py`.
- Compute `realized_vol[t] = std(log_returns[t-n+1:t+1])` over n=20.
- Multi-head CNN picks them up automatically; no decoder-side change.

**Out of scope:**
- True ADX (would need OHLC source — change to Stooq daily archive
  which has high/low, breaks the Yahoo cross-source path).
- Williams %R, Stochastic, OBV — same OHLCV-needing blocker.
- Bollinger bands — trivially recoverable from `--include-zscore-stats`
  (BBands middle = mu, edges = mu ± k*sigma, both literal inputs).
  Sanity check, not a research lever.

## Ablation — disentangle why long-period RSI underperforms

The CSCO zero-shot RSI(n) sweep on the 30-ticker / `n_grid={5,7,9,13,21}`
/ K=64 run showed a sharp degradation at the long end:

| n  | in-grid | R²    |
|----|---------|-------|
| 9  | yes     | 0.964 |
| 13 | yes     | 0.902 |
| 18 | no      | 0.690 |
| 21 | yes     | 0.520 |

Two factors were proposed (see chat 2026-04-27):
1. **Grid spacing** — n=21 sits at the conditioning maximum with no
   right-neighbor; gap to its left-neighbor n=13 is 8 (vs spacings of
   2 below). The linear conditioning has fewer interpolation pairs
   here.
2. **Effective lookback** — Wilder RSI(n) has effective memory ~3×n
   bars. RSI(21) ≈ 63–84 bars; K=64 is at the edge. The model has
   long-horizon info via the rolling z-score stats and long CWT scales,
   but the *direct per-lag* path is window-bounded.

Three runs to disentangle (each is one CLI flag tweak from the
existing `ss-replay … --rsi-n-grid 5,7,9,13,21 …` cell):

| Run | `--rsi-n-grid`              | `--window-cols` | tests       |
|-----|-----------------------------|-----------------|-------------|
| A   | `5,7,9,13,17,21,25`         | 64              | factor 1    |
| B   | `5,7,9,13,21`               | 96              | factor 2    |
| C   | `5,7,9,13,17,21,25`         | 96              | combined    |

If A recovers RSI(21) R², factor 1 is dominant; spacing matters more
than lookback. If B recovers it, factor 2 is dominant; longer K is the
fix. C is the upper bound.

Beyond fixing one ticker's RSI numbers, this informs grid-design
heuristics for any future parameter-conditioned head — both grid
density and the input-bundle's effective lookback need to be
matched to the longest target parametrization.

**Out of scope** for the same diagnostic:
- Non-linear conditioning (sin/cos of n, or a small MLP on n). If A+B
  both fail, that's the next architectural lever.
- Re-running with the (w, n) 2D conditioning — that's a separate
  capability test, not a disentanglement of the existing failure.
