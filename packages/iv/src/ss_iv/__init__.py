"""ss_iv — implied-vol data + short-vol PnL primitives.

Promoted from `apps/relational/src/relational/{iv_data, short_vol}.py`
when `apps/vol` became a second consumer (2026-05-10). The original
files were pure numpy + pandas + `ss_features` — no `relational`-
internal imports — so the lift was clean.

Two logical halves:

  * **`loaders`** — fetch + cache implied-vol time series from two
    free sources:
      - `load_atm_iv()` reads gauss314/options-IV-SP500 (HF). Daily,
        SPX universe, 2019-10-14 → 2023-07-28. Single ~500 MB CSV.
        Rich schema: full strike grid (`DITM_IV` ... `DOTM_IV`), HV
        at multiple horizons (`hv_20` ... `hv_200`), VIX, OI + flow
        per side. **Use this for feature-richness experiments.**
      - `load_dolthub_iv_parquet()` reads the post-no-preference/
        options DoltHub export. Weekly, 2276 US tickers,
        2019-02-09 → 2026-04-30. Schema is just `iv_current` +
        `hv_current` per ticker per date. Use for long-span / wide-
        universe coverage; not for skew / smile experiments.
      - `load_dolthub_iv()` is the per-ticker HTTP-API fallback when
        the parquet export isn't on disk.
  * **`short_vol`** — short-straddle PnL accounting:
      - `short_vol_pnl_panel(iv, forward)` — per-`(date, ticker)`
        cycle P&L `iv - forward` in vol-points (annualized fraction).
      - `evaluate_short_vol(scores, iv, forward, prices, ...)` —
        pick top-N by score per rebalance; vol-point summary stats.
      - `evaluate_universe_short_vol(iv, forward, prices, ...)` —
        equal-weight every active ticker; the trivial VRP baseline.

  P&L is **vol-points**, not dollars. To approximate ATM straddle
  dollar P&L multiply by `vega × notional`. Vol-points are unit-free
  and the right comparison currency for cross-scorer / cross-feature
  evals.
"""
from ss_iv.loaders import (
    load_atm_iv, load_dolthub_iv, load_dolthub_iv_parquet,
)
from ss_iv.short_vol import (
    evaluate_short_vol, evaluate_universe_short_vol,
    short_vol_pnl_panel,
)


__all__ = [
    'evaluate_short_vol',
    'evaluate_universe_short_vol',
    'load_atm_iv',
    'load_dolthub_iv',
    'load_dolthub_iv_parquet',
    'short_vol_pnl_panel',
]
