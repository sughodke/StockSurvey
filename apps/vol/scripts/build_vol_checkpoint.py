"""Build the `VolCheckpoint` JSON for `ss-vol live` from the
DoltHub-OOS predictor.

Re-runs the same train-window OLS fit as
`apps/vol/scripts/run_walkforward_v3_dolthub_oos.py` over 2019-10 →
2023-07 (the gauss314-overlap window) and freezes the coefficients +
z-score stats into `Output/vol-v3.json`. Universe = the symbols that
DoltHub carries in 2026 (current-deployment universe). VIX-gate +
strangle config use the v3 deployment recipe defaults.

Also bootstraps the local IV/HV cache from DoltHub so the first
`ss-vol live` run has the 4-week-change features available.

Run from repo root:
    uv run python apps/vol/scripts/build_vol_checkpoint.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from vol.iv_history import bootstrap_from_dolthub
from vol.persist import (
    LIVE_FEATURE_NAMES, StranglesConfig, VolCheckpoint, save_checkpoint,
    validate,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DOLTHUB_PARQUET = REPO_ROOT / '.iv-cache/volatility_history.parquet'
DEFAULT_OUT = REPO_ROOT / 'Output/vol-v3.json'

TRAIN_START = pd.Timestamp('2019-10-14')
TRAIN_END   = pd.Timestamp('2023-07-28')


def _build_features_and_target(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Same feature/target build as run_walkforward_v3_dolthub_oos.py."""
    iv_wide = panel.pivot(index='date', columns='symbol', values='iv_current')
    hv_wide = panel.pivot(index='date', columns='symbol', values='hv_current')

    log_p = np.log(prices.replace(0.0, np.nan)).dropna(how='all')
    log_r = log_p.diff()
    daily_idx = prices.index.sort_values()
    forward_rv = {}
    annualization = np.sqrt(252)
    for d in iv_wide.index:
        pos = daily_idx.searchsorted(d, side='left')
        if pos + 20 >= len(daily_idx):
            continue
        window = log_r.iloc[pos + 1: pos + 21]
        if window.shape[0] < 14:
            continue
        forward_rv[d] = window.std() * annualization
    forward_rv = pd.DataFrame(forward_rv).T.sort_index().reindex(index=iv_wide.index)
    target_wide = iv_wide - forward_rv

    iv_over_hv = (iv_wide / hv_wide.clip(lower=1e-6)).clip(-10, 10)
    iv_z = (iv_wide.sub(iv_wide.mean(axis=1), axis=0)
            .div(iv_wide.std(axis=1).clip(lower=1e-6), axis=0))
    iv_change_4w = iv_wide - iv_wide.shift(4)
    hv_change_4w = hv_wide - hv_wide.shift(4)

    def melt(wide, name):
        out = wide.stack().rename(name).reset_index()
        out.columns = ['date', 'symbol', name]
        return out

    out = melt(iv_over_hv, 'iv_over_hv')
    for w, n in [(iv_z, 'iv_z'), (iv_change_4w, 'iv_change_4w'),
                 (hv_change_4w, 'hv_change_4w'), (target_wide, 'iv_rv_gap')]:
        out = out.merge(melt(w, n), on=['date', 'symbol'], how='inner')
    return out.dropna()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', default=str(DOLTHUB_PARQUET))
    p.add_argument('--out', default=str(DEFAULT_OUT))
    p.add_argument('--universe-min-bars', type=int, default=200,
                   help='Only keep symbols with ≥ N rows in DoltHub')
    p.add_argument('--vega-budget', type=float, default=100.0,
                   help='Vega budget per name (USD)')
    p.add_argument('--top-k', type=int, default=100,
                   help='Number of top-K names to score per fired rebal')
    p.add_argument('--gate-lookback', type=int, default=126)
    p.add_argument('--skip-bootstrap', action='store_true',
                   help='Skip seeding the local IV-history cache from DoltHub')
    args = p.parse_args()

    print(f'Loading DoltHub parquet from {args.parquet}...', flush=True)
    t0 = time.perf_counter()
    df = pd.read_parquet(args.parquet)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'act_symbol': 'symbol'})
    df['iv_current'] = pd.to_numeric(df['iv_current'], errors='coerce')
    df['hv_current'] = pd.to_numeric(df['hv_current'], errors='coerce')
    df = df.dropna(subset=['iv_current', 'hv_current'])
    df = df[(df['iv_current'] > 0) & (df['hv_current'] > 0)]
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    print(f'  {len(df):,} rows in {time.perf_counter()-t0:.1f}s', flush=True)

    # Universe: symbols that DoltHub still carries in the last 90 days
    # of the parquet AND have enough train-window history.
    recent_cutoff = df['date'].max() - pd.Timedelta(days=90)
    recent_syms = set(df[df['date'] >= recent_cutoff]['symbol'].unique())
    train_counts = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)] \
        .groupby('symbol').size()
    well_covered = set(train_counts[train_counts >= args.universe_min_bars].index)
    universe = sorted(recent_syms & well_covered)
    print(f'  universe: {len(universe)} symbols (recent ∩ well-covered)', flush=True)
    if len(universe) < 10:
        raise SystemExit('universe too small; check filters')

    # Build features + target on the train window only
    panel = df[df['date'] <= TRAIN_END]
    syms = sorted(panel['symbol'].unique())
    print(f'Loading Stooq prices for forward-RV target...', flush=True)
    t0 = time.perf_counter()
    prices, _, _, _ = load_stooq_matrix(
        str(REPO_ROOT / 'StooqData'), min_history=100,
        start_date='2019-01-01', end_date='2026-05-31', tickers=syms)
    print(f'  prices {prices.shape} in {time.perf_counter()-t0:.1f}s', flush=True)

    print('Building train features + target...', flush=True)
    t0 = time.perf_counter()
    merged = _build_features_and_target(panel, prices)
    print(f'  {len(merged):,} usable rows in {time.perf_counter()-t0:.1f}s', flush=True)

    train = merged[(merged['date'] >= TRAIN_START) & (merged['date'] <= TRAIN_END)]
    X = train[LIVE_FEATURE_NAMES].values.astype(np.float64)
    y = train['iv_rv_gap'].values.astype(np.float64)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Xz = (X - mu) / sd
    Xa = np.concatenate([Xz, np.ones((len(Xz), 1))], axis=1)
    coefs, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    pred = Xa @ coefs
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f'  train R² = {r2:+.4f}', flush=True)
    for fn, c in zip(LIVE_FEATURE_NAMES, coefs[:-1]):
        print(f'    {fn:>20s}: {c:+.5f}', flush=True)
    print(f'    {"intercept":>20s}: {coefs[-1]:+.5f}', flush=True)

    cp = VolCheckpoint(
        feature_names=list(LIVE_FEATURE_NAMES),
        coefs=coefs.tolist(),
        feat_mean=mu.tolist(),
        feat_std=sd.tolist(),
        universe=universe,
        gate_fred_series='VIXCLS',
        gate_lookback_trading_days=int(args.gate_lookback),
        top_k=int(args.top_k),
        strangle=StranglesConfig(
            target_tenor_days=30, tenor_tolerance_days=7,
            target_delta_call=0.20, target_delta_put=0.20,
            vega_budget_per_name_usd=float(args.vega_budget),
            min_open_interest=100, min_bid_size=10,
            max_bid_ask_spread_pct=0.15,
        ),
        train_period=f'{TRAIN_START.date()} → {TRAIN_END.date()}',
        val_period='2023-08-01 → 2026-04-30',
        val_pearson_r=0.165,
        n_obs_oos=33,
        oos_ann_sharpe=2.822,
        oos_deflated_t=5.549,
        notes=(
            'Frozen predictor for ss-vol live; built from '
            'apps/vol/scripts/run_walkforward_v3_dolthub_oos.py train '
            'window (2019-10 → 2023-07). See '
            'apps/docs/docs/findings/vol-v3-dolthub-oos.md for the OOS '
            'verdict and three caveats (regime-tailwind, ρ with DCA, no '
            'options-broker friction).'),
    )
    validate(cp)
    out_path = Path(args.out)
    save_checkpoint(cp, out_path)
    print(f'\n→ wrote checkpoint to {out_path}', flush=True)

    if not args.skip_bootstrap:
        n = bootstrap_from_dolthub(universe)
        print(f'→ bootstrapped local IV/HV cache: {n} rows from DoltHub',
              flush=True)


if __name__ == '__main__':
    main()
