"""Load gauss314 full-schema IV CSV + derive vol-surface features.

`ss_iv.load_atm_iv()` returns only `ATM_IV` because that's all the
prior arc needed. The CSV at `.iv-cache/data_IV_USA.csv` actually
carries the full strike grid (DITM/ITM/sITM/ATM/sOTM/OTM/DOTM_IV) +
multi-horizon HV (`hv_20`...`hv_200`) + OI per side + VIX. We need
all of it for the skew / smile / IV-HV-ratio / OI-imbalance / VIX-
spread features.

Schema columns we load (per the head -1 audit on the live file):
  symbol, date, strikes_spread,
  calls_contracts_traded, puts_contracts_traded,
  calls_open_interest, puts_open_interest,
  expirations_number,
  DITM_IV, ITM_IV, sITM_IV, ATM_IV, sOTM_IV, OTM_IV, DOTM_IV,
  contracts_number,
  hv_20, hv_40, hv_60, hv_75, hv_90, hv_120, hv_180, hv_200,
  VIX

IV columns arrive in *percent* form (28.09 = 28.09%); we convert to
fraction (0.2809) at load time, matching `ss_iv.load_atm_iv`'s
convention. HV columns are ambiguous in the source file — they look
like vol-points (24.467 ≈ 24.5% vol). We treat them the same way
(divide by 100) for cross-comparison with IV.

Output is a long-form DataFrame indexed by `(date, symbol)`. Pivot
to wide per-feature in the predictor pipeline if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GAUSS314_CSV = Path('.iv-cache/data_IV_USA.csv')

PERCENT_COLS = (
    'DITM_IV', 'ITM_IV', 'sITM_IV', 'ATM_IV', 'sOTM_IV', 'OTM_IV', 'DOTM_IV',
    'hv_20', 'hv_40', 'hv_60', 'hv_75', 'hv_90', 'hv_120', 'hv_180', 'hv_200',
)

LOAD_COLS = (
    'symbol', 'date',
    'strikes_spread',
    'calls_open_interest', 'puts_open_interest',
    'DITM_IV', 'ITM_IV', 'ATM_IV', 'OTM_IV', 'DOTM_IV',
    'hv_20', 'hv_60', 'hv_120', 'hv_200',
    'VIX',
)

FEATURE_NAMES: list[str] = [
    'skew_otm',          # (DOTM_IV - ATM_IV) / ATM_IV
    'skew_itm',          # (DITM_IV - ATM_IV) / ATM_IV
    'smile_curvature',   # (DOTM_IV + DITM_IV - 2*ATM_IV) / ATM_IV
    'iv_over_hv20',      # ATM_IV / hv_20
    'iv_over_hv60',      # ATM_IV / hv_60
    'iv_over_hv120',     # ATM_IV / hv_120
    'hv_term',           # hv_20 / hv_200  (realized vol mean reversion)
    'oi_imbalance',      # puts_OI / (puts_OI + calls_OI)
    'vix_spread',        # ATM_IV - VIX
    'strike_spread_norm',# strikes_spread / ATM_IV
]


@dataclass(frozen=True)
class Gauss314Panel:
    """Long-form full-schema panel + per-(date, symbol) feature matrix."""
    raw:        pd.DataFrame   # long-form: cols incl ATM_IV, DOTM_IV, hv_20, ...
    features:   pd.DataFrame   # long-form: cols = FEATURE_NAMES
    dates:      pd.DatetimeIndex
    symbols:    list[str]


def load_gauss314_full(
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read the full-schema gauss314 CSV. Returns long-form DataFrame
    indexed by an integer range, with `date` as a datetime column,
    IV / HV / VIX columns converted to fractional form."""
    p = Path(csv_path) if csv_path is not None else GAUSS314_CSV
    if not p.exists():
        raise FileNotFoundError(
            f'gauss314 CSV not at {p} — `ss_iv.load_atm_iv()` will fetch '
            f'the file but downloads only the ATM_IV column. To get the '
            f'full schema, fetch manually:\n'
            f'  curl -L https://huggingface.co/datasets/gauss314/'
            f'options-IV-SP500/resolve/main/data_IV_USA.csv '
            f'-o {p}\n'
            f'(~500 MB; one-time)')
    df = pd.read_csv(p, usecols=list(LOAD_COLS), parse_dates=['date'])
    for c in PERCENT_COLS:
        if c in df.columns:
            df[c] = df[c] / 100.0
    if 'VIX' in df.columns:
        df['VIX'] = df['VIX'] / 100.0
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    return df


def build_vol_features(raw: pd.DataFrame) -> Gauss314Panel:
    """Compute the per-(date, symbol) feature stack.

    Each feature is `np.nan` if any required input is NaN or if a
    denominator is too small (we use `1e-8` clip rather than any
    masking — the predictor's downstream `dropna` handles it).
    """
    f = pd.DataFrame(index=raw.index)
    f['date']   = raw['date']
    f['symbol'] = raw['symbol']

    atm = raw['ATM_IV'].clip(lower=1e-8)

    f['skew_otm']         = (raw['DOTM_IV'] - raw['ATM_IV']) / atm
    f['skew_itm']         = (raw['DITM_IV'] - raw['ATM_IV']) / atm
    f['smile_curvature']  = (
        raw['DOTM_IV'] + raw['DITM_IV'] - 2.0 * raw['ATM_IV']) / atm

    f['iv_over_hv20']  = raw['ATM_IV'] / raw['hv_20'].clip(lower=1e-8)
    f['iv_over_hv60']  = raw['ATM_IV'] / raw['hv_60'].clip(lower=1e-8)
    f['iv_over_hv120'] = raw['ATM_IV'] / raw['hv_120'].clip(lower=1e-8)
    f['hv_term'] = raw['hv_20'] / raw['hv_200'].clip(lower=1e-8)

    oi_total = (raw['puts_open_interest'] +
                raw['calls_open_interest']).clip(lower=1.0)
    f['oi_imbalance'] = raw['puts_open_interest'] / oi_total

    f['vix_spread'] = raw['ATM_IV'] - raw['VIX']

    f['strike_spread_norm'] = raw['strikes_spread'] / atm

    dates = pd.DatetimeIndex(sorted(raw['date'].unique()))
    symbols = sorted(raw['symbol'].unique())
    return Gauss314Panel(
        raw=raw, features=f, dates=dates, symbols=symbols)


__all__ = [
    'FEATURE_NAMES',
    'GAUSS314_CSV',
    'Gauss314Panel',
    'build_vol_features',
    'load_gauss314_full',
]
