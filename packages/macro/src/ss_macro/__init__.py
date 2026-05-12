"""ss_macro — FRED time-series loaders + canonical regime feature stack.

The "regime" question raised after the
[prediction-problem-pivot arc](../../../apps/docs/docs/findings/prediction-problem-pivot-arc.md):
all three v0 tests showed regime-conditional partial signals, with
"win" windows clustering in macro-stress eras (2008 GFC, 2022
hiking cycle) and "lose" windows in macro-calm eras (2014-2020
ZIRP, 2021 melt-up). `ss_macro` provides the data layer for the
regime classifier this points to.

Canonical features (academic regime-classification standard):

  fed_funds       — `FEDFUNDS` policy stance
  slope_10y_3m    — `T10Y3M` yield curve slope (inversion proxy)
  credit_baa      — `BAA10Y` credit spread (BAA corporate − 10y)
  m2_level/_yoy   — `M2SL` raw + derived 12-month % change
  real_yield_10y  — `DFII10` 10-year TIPS real yield
  vix             — `VIXCLS` (CBOE Volatility Index)
  gold_vix        — `GVZCLS` CBOE Gold ETF Volatility (daily 2008-06+)
  gold_level/_yoy — Stooq `GLD.US` ETF close + 252d YoY % change
                    (FRED's no-auth CSV endpoint does NOT expose the
                    LBMA gold series, so we proxy via the SPDR Gold
                    Shares ETF). Gold-YoY is the regime-signal form
                    (rising = USD debasement / risk-off); raw level
                    is non-stationary across-sample. Loaded only
                    when `load_macro_panel(stooq_data_dir=...)` is
                    given.

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
    load_gold_features_from_stooq,
    load_macro_panel,
)


__all__ = [
    'DEFAULT_SERIES',
    'fred_series_url',
    'load_fred_series',
    'load_gold_features_from_stooq',
    'load_macro_panel',
]
