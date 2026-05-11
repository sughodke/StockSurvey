"""ss_macro — FRED time-series loaders + canonical regime feature stack.

The "regime" question raised after the
[prediction-problem-pivot arc](../../../apps/docs/docs/findings/prediction-problem-pivot-arc.md):
all three v0 tests showed regime-conditional partial signals, with
"win" windows clustering in macro-stress eras (2008 GFC, 2022
hiking cycle) and "lose" windows in macro-calm eras (2014-2020
ZIRP, 2021 melt-up). `ss_macro` provides the data layer for the
regime classifier this points to.

Six canonical features (academic regime-classification standard):

  fed_funds       — `FEDFUNDS` policy stance
  slope_10y_3m    — `T10Y3M` yield curve slope (inversion proxy)
  credit_baa      — `BAA10Y` credit spread (BAA corporate − 10y)
  m2_yoy          — derived from `M2SL` (year-over-year % change)
  real_yield_10y  — `DFII10` 10-year TIPS real yield
  vix             — `VIXCLS` (CBOE Volatility Index)

All series fetched from FRED's no-auth CSV endpoint
(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`),
cached to `.macro-cache/` (analogous to `.iv-cache/`). Releases
have publishing lags (M2 weekly + ~2 week lag, CPI monthly + ~2
weeks); the loaders return data as-of-publish, callers responsible
for align-to-trading-bar with appropriate ffill discipline.

Public API:
  - `load_fred_series(series_id, ...)` — single FRED series.
  - `load_macro_panel(...)` — the canonical 6-feature stack
    aligned to a daily trading-bar index.
"""
from ss_macro.loaders import (
    DEFAULT_SERIES,
    fred_series_url,
    load_fred_series,
    load_macro_panel,
)


__all__ = [
    'DEFAULT_SERIES',
    'fred_series_url',
    'load_fred_series',
    'load_macro_panel',
]
