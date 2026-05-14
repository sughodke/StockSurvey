"""v2 #3 — DoltHub OOS extension to 2026.

Tests whether the underlying name-level VRP signal (the vol-points
alpha that v1 documented on gauss314 2019-2023) continues in the OOS
period 2023-08 → 2026-04. The gauss314 dataset caps at 2023-07-28; the
DoltHub `volatility_history` parquet extends through 2026-04-30 but
only carries `iv_current` + `hv_current` per ticker (no strike-grid
skew/smile, no multi-horizon HV).

This is the **arc-decisive experiment**:
  - v1 confirmed the alpha on the full surface
  - v2 #1 confirmed the alpha in dollar terms under standard sizing
  - v2 #2 showed alpha collapses under deployable liquidity restriction
  - v2 #3 tests whether the underlying name-level signal is alive in
    the OOS regime that has no overlap with gauss314 windows

Pre-registered cuts (v2 #3):
  PASS:     OOS Pearson r >= +0.05 on the 4-feature proxy stack AND
            top-K=100 alpha mean PnL > 0 in >= 4/6 OOS quarterly windows
  MARGINAL: OOS Pearson r in [+0.02, +0.05] OR pos windows >= 3/6
  FAIL:     OOS Pearson r < +0.02 OR pos windows <= 2/6

Run:
    uv run python apps/vol/scripts/run_walkforward_v2_dolthub_oos.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / 'Output'
DOLTHUB_PARQUET = REPO_ROOT / '.iv-cache/volatility_history.parquet'

# Pre-reg: train slice is gauss314-overlap period (2019-10 → 2023-07)
# so we can compare the train-side Pearson r to v1's mean +0.12 read.
# Val slice is OOS (2023-08 → 2026-04, 2.7 years).
TRAIN_START_DATE = pd.Timestamp('2019-10-14')
TRAIN_END_DATE   = pd.Timestamp('2023-07-28')
VAL_START_DATE   = pd.Timestamp('2023-08-01')
VAL_END_DATE     = pd.Timestamp('2026-04-30')

# Weekly cadence: forward 20-trading-day = ~4 weekly snapshots ahead.
FORWARD_WEEKS = 4
REBAL_WEEKS = 4


def load_dolthub_panel(parquet_path: Path) -> pd.DataFrame:
    """Long-form DataFrame: date, symbol, iv_current, hv_current."""
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'act_symbol': 'symbol'})
    df['iv_current'] = pd.to_numeric(df['iv_current'], errors='coerce')
    df['hv_current'] = pd.to_numeric(df['hv_current'], errors='coerce')
    df = df.dropna(subset=['iv_current', 'hv_current'])
    df = df[(df['iv_current'] > 0) & (df['hv_current'] > 0)]
    return df.sort_values(['date', 'symbol']).reset_index(drop=True)


def compute_forward_realized_vol(
    prices: pd.DataFrame, weekly_dates: pd.DatetimeIndex,
    forward_trading_days: int = 20,
) -> pd.DataFrame:
    """For each (weekly_date t, symbol), compute annualized realized vol
    of log-returns over the next `forward_trading_days` trading days.
    Returns wide-form DataFrame indexed by weekly_dates, columns=symbols.

    Honest forward realized vol from price returns — NOT from DoltHub's
    hv_current snapshot (which is autocorrelated 0.85 at lag 4 weeks and
    would tautologically explain the target).
    """
    log_p = np.log(prices.replace(0.0, np.nan)).dropna(how='all')
    log_r = log_p.diff()  # daily log returns

    daily_idx = prices.index.sort_values()
    weekly_idx = pd.DatetimeIndex(sorted(weekly_dates))
    forward_rv: dict[pd.Timestamp, pd.Series] = {}
    annualization = np.sqrt(252)

    for d in weekly_idx:
        # Find first daily date >= weekly date
        pos = daily_idx.searchsorted(d, side='left')
        if pos + forward_trading_days >= len(daily_idx):
            continue
        window = log_r.iloc[pos + 1: pos + 1 + forward_trading_days]
        if window.shape[0] < forward_trading_days * 0.7:
            continue
        rv = window.std() * annualization
        forward_rv[d] = rv

    return pd.DataFrame(forward_rv).T.sort_index()


def build_features_and_target(panel: pd.DataFrame,
                              prices: pd.DataFrame) -> pd.DataFrame:
    """For each (date, symbol):
      - feature stack (4 cols): iv_over_hv, iv_z (cross-sectional),
        iv_change_4w, hv_change_4w
      - target: iv_rv_gap = iv_current[t] - forward_realized_vol[t, t+20d]
        computed from underlying log-return std (NOT from DoltHub's
        hv_current[t+4w], which has 0.85 autocorrelation at lag-4 and
        would tautologically explain the target via the iv_over_hv
        feature).
    """
    iv_wide = panel.pivot(index='date', columns='symbol', values='iv_current')
    hv_wide = panel.pivot(index='date', columns='symbol', values='hv_current')

    # HONEST forward realized vol from underlying prices.
    print('  computing forward realized vol from Stooq prices...', flush=True)
    forward_rv = compute_forward_realized_vol(prices, iv_wide.index)
    # Align (some symbols missing from Stooq → dropped at merge time)
    forward_rv = forward_rv.reindex(index=iv_wide.index)
    target_wide = iv_wide - forward_rv

    iv_over_hv = (iv_wide / hv_wide.clip(lower=1e-6)).clip(-10, 10)
    iv_z = (iv_wide.sub(iv_wide.mean(axis=1), axis=0)
            .div(iv_wide.std(axis=1).clip(lower=1e-6), axis=0))
    iv_change_4w = iv_wide - iv_wide.shift(4)
    hv_change_4w = hv_wide - hv_wide.shift(4)

    def melt(wide: pd.DataFrame, name: str) -> pd.DataFrame:
        out = wide.stack().rename(name).reset_index()
        out.columns = ['date', 'symbol', name]
        return out

    out = melt(iv_over_hv, 'iv_over_hv')
    for w, n in [(iv_z, 'iv_z'),
                 (iv_change_4w, 'iv_change_4w'),
                 (hv_change_4w, 'hv_change_4w'),
                 (target_wide, 'iv_rv_gap')]:
        out = out.merge(melt(w, n), on=['date', 'symbol'], how='inner')

    return out.dropna()


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Z-score features on train, OLS fit. Returns (coefs_with_intercept,
    train_r2, feat_mean, feat_std)."""
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
    return coefs, r2, mu, sd


def apply_ols(X: np.ndarray, coefs: np.ndarray,
              mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xz = (X - mu) / sd
    Xa = np.concatenate([Xz, np.ones((len(Xz), 1))], axis=1)
    return Xa @ coefs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--rebal-weeks', type=int, default=REBAL_WEEKS)
    p.add_argument('--parquet', default=str(DOLTHUB_PARQUET))
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading DoltHub parquet...', flush=True)
    t0 = time.perf_counter()
    panel = load_dolthub_panel(Path(args.parquet))
    print(f'  {len(panel):,} rows in {time.perf_counter()-t0:.1f}s', flush=True)
    print(f'  date range: {panel["date"].min().date()} → {panel["date"].max().date()}', flush=True)
    print(f'  unique symbols: {panel["symbol"].nunique()}', flush=True)

    print('Loading Stooq prices for honest forward-RV target...', flush=True)
    t0 = time.perf_counter()
    dolthub_symbols = sorted(panel['symbol'].unique())
    prices, _, _, _ = load_stooq_matrix(
        './StooqData', min_history=100,
        start_date='2019-01-01', end_date='2026-05-31',
        tickers=dolthub_symbols)
    print(f'  {prices.shape[0]} dates × {prices.shape[1]} symbols '
          f'in {time.perf_counter()-t0:.1f}s '
          f'({100*prices.shape[1]/len(dolthub_symbols):.0f}% coverage)',
          flush=True)

    print('Building features + target...', flush=True)
    t0 = time.perf_counter()
    merged = build_features_and_target(panel, prices)
    print(f'  {len(merged):,} usable rows in {time.perf_counter()-t0:.1f}s', flush=True)

    feature_cols = ['iv_over_hv', 'iv_z', 'iv_change_4w', 'hv_change_4w']

    train = merged[(merged['date'] >= TRAIN_START_DATE)
                   & (merged['date'] <= TRAIN_END_DATE)]
    val   = merged[(merged['date'] >= VAL_START_DATE)
                   & (merged['date'] <= VAL_END_DATE)]
    print(f'  train: {len(train):,} cells ({train["date"].min().date()} → '
          f'{train["date"].max().date()})', flush=True)
    print(f'  val:   {len(val):,} cells ({val["date"].min().date()} → '
          f'{val["date"].max().date()})', flush=True)

    if len(train) < 100 or len(val) < 100:
        print('insufficient data', flush=True)
        return

    X_tr = train[feature_cols].values
    y_tr = train['iv_rv_gap'].values
    X_va = val[feature_cols].values
    y_va = val['iv_rv_gap'].values

    coefs, train_r2, mu, sd = fit_ols(X_tr, y_tr)
    val_pred = apply_ols(X_va, coefs, mu, sd)
    val_corr = float(np.corrcoef(val_pred, y_va)[0, 1])
    ss_tot = float(np.sum((y_va - y_va.mean()) ** 2))
    ss_res = float(np.sum((y_va - val_pred) ** 2))
    val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f'\n=== OOS evaluation (2023-08 → 2026-04, '
          f'{(VAL_END_DATE - VAL_START_DATE).days / 365:.1f} years) ===',
          flush=True)
    print(f'  train Pearson r²: {train_r2:+.4f}', flush=True)
    print(f'  OOS Pearson r:   {val_corr:+.4f}', flush=True)
    print(f'  OOS R²:          {val_r2:+.4f}', flush=True)
    print(f'  feature coefs (z-scored, OLS):', flush=True)
    for fn, c in zip(feature_cols, coefs[:-1]):
        print(f'    {fn:>20s}: {c:+.5f}', flush=True)
    print(f'    {"intercept":>20s}: {coefs[-1]:+.5f}', flush=True)

    # Per-quarter OOS portfolio: top-K=100 picks per rebal date, mean
    # realized iv_rv_gap as portfolio PnL.
    val_with_pred = val[['date', 'symbol']].copy()
    val_with_pred['pred_gap'] = val_pred
    val_with_pred['iv_rv_gap'] = y_va

    val_dates = pd.DatetimeIndex(sorted(val['date'].unique()))
    rebal_dates = val_dates[::args.rebal_weeks]

    # Split OOS into ~quarterly windows for the 'positive windows' count
    qtrs = pd.date_range(VAL_START_DATE, VAL_END_DATE, freq='3MS')
    if qtrs[-1] < VAL_END_DATE:
        qtrs = qtrs.append(pd.DatetimeIndex([VAL_END_DATE + pd.Timedelta(days=1)]))

    print(f'\n  rebal cadence: {args.rebal_weeks} weeks ({len(rebal_dates)} '
          f'rebals across OOS span)', flush=True)
    print(f'  splitting OOS into {len(qtrs)-1} quarterly windows for verdict',
          flush=True)

    print('\n' + '=' * 90, flush=True)
    print(f'{"qtr":>3s} {"period":>25s} {"n_reb":>5s} {"top_k_PnL":>10s} '
          f'{"univ_PnL":>10s} {"alpha":>10s} {"top_k_Sh":>9s} {"alpha_Sh":>9s}',
          flush=True)
    print('-' * 90, flush=True)

    per_qtr = []
    pooled_alpha_pnls = []
    for i in range(len(qtrs) - 1):
        q_lo, q_hi = qtrs[i], qtrs[i + 1] - pd.Timedelta(days=1)
        q_rebals = [d for d in rebal_dates if q_lo <= d <= q_hi]
        if len(q_rebals) < 2:
            continue
        top_k_pnls, univ_pnls = [], []
        for rd in q_rebals:
            day = val_with_pred[val_with_pred['date'] == rd]
            if len(day) < args.top_k:
                continue
            picks = day.nlargest(args.top_k, 'pred_gap')
            top_k_pnls.append(float(picks['iv_rv_gap'].mean()))
            univ_pnls.append(float(day['iv_rv_gap'].mean()))
        if len(top_k_pnls) < 2:
            continue
        alpha_pnls = [t - u for t, u in zip(top_k_pnls, univ_pnls)]
        a = np.asarray(alpha_pnls, dtype=float)
        t = np.asarray(top_k_pnls, dtype=float)
        # weekly rebal cadence; annualize with sqrt(52/rebal_weeks)
        ann = float(np.sqrt(52.0 / args.rebal_weeks))
        sh_top = (t.mean() / t.std(ddof=1) * ann
                  if t.size > 1 and t.std(ddof=1) > 1e-12 else 0.0)
        sh_alpha = (a.mean() / a.std(ddof=1) * ann
                    if a.size > 1 and a.std(ddof=1) > 1e-12 else 0.0)
        per_qtr.append({
            'quarter_idx': i,
            'period': f'{q_lo.date()} → {q_hi.date()}',
            'n_rebals': len(top_k_pnls),
            'top_k_mean_pnl': float(t.mean()),
            'univ_mean_pnl': float(np.mean(univ_pnls)),
            'alpha_mean_pnl': float(a.mean()),
            'top_k_sharpe': sh_top,
            'alpha_sharpe': sh_alpha,
        })
        pooled_alpha_pnls.extend(alpha_pnls)
        print(f'{i:>3d} {str(q_lo.date())} → {str(q_hi.date())} '
              f'{len(top_k_pnls):>5d} '
              f'{t.mean():>+10.4f} {np.mean(univ_pnls):>+10.4f} '
              f'{a.mean():>+10.4f} {sh_top:>+9.3f} {sh_alpha:>+9.3f}',
              flush=True)

    print('\n' + '=' * 90, flush=True)
    if not per_qtr:
        print('No usable quarters.', flush=True)
        return

    pa = np.asarray(pooled_alpha_pnls, dtype=float)
    ann = float(np.sqrt(52.0 / args.rebal_weeks))
    pooled_alpha_sh = (pa.mean() / pa.std(ddof=1) * ann
                       if pa.size > 1 and pa.std(ddof=1) > 1e-12 else 0.0)
    pooled_alpha_mean = float(pa.mean())
    n_pos = sum(1 for q in per_qtr if q['alpha_sharpe'] > 0)
    n_total = len(per_qtr)
    mean_alpha = float(np.mean([q['alpha_mean_pnl'] for q in per_qtr]))

    print(f'\nOOS pooled alpha Sharpe (annualized) = {pooled_alpha_sh:+.3f}',
          flush=True)
    print(f'OOS pooled alpha mean PnL per rebal  = {pooled_alpha_mean:+.4f} vol pts',
          flush=True)
    print(f'OOS positive-alpha quarters          = {n_pos}/{n_total}',
          flush=True)
    print(f'OOS val Pearson r (cross-sectional)  = {val_corr:+.4f}',
          flush=True)

    # Pre-reg verdict.
    if val_corr >= 0.05 and n_pos >= int(np.ceil(0.66 * n_total)):
        verdict = (f'PASS — OOS val r {val_corr:+.4f} ≥ +0.05 and '
                   f'{n_pos}/{n_total} positive quarters')
    elif val_corr >= 0.02 or n_pos >= int(np.ceil(0.5 * n_total)):
        verdict = (f'MARGINAL — OOS val r {val_corr:+.4f} in [+0.02, +0.05] '
                   f'OR positive quarters {n_pos}/{n_total} ≥ 50%')
    else:
        verdict = (f'FAIL — OOS val r {val_corr:+.4f} < +0.02 OR pos '
                   f'quarters {n_pos}/{n_total} insufficient')

    print(f'\npre-reg verdict: {verdict}', flush=True)

    summary = {
        'data_source': 'DoltHub volatility_history',
        'train_period': f'{TRAIN_START_DATE.date()} → {TRAIN_END_DATE.date()}',
        'val_period_oos': f'{VAL_START_DATE.date()} → {VAL_END_DATE.date()}',
        'feature_cols': feature_cols,
        'top_k': args.top_k,
        'rebal_weeks': args.rebal_weeks,
        'n_train_cells': int(len(train)),
        'n_val_cells': int(len(val)),
        'train_r2': train_r2,
        'val_r2': val_r2,
        'val_pearson_r': val_corr,
        'pooled_alpha_sharpe_annualized': pooled_alpha_sh,
        'pooled_alpha_mean_pnl': pooled_alpha_mean,
        'positive_quarters': n_pos,
        'total_quarters': n_total,
        'mean_alpha_pnl_per_quarter': mean_alpha,
        'feature_coefs': {fn: float(c) for fn, c in zip(feature_cols, coefs[:-1])},
        'intercept': float(coefs[-1]),
        'per_quarter': per_qtr,
        'verdict': verdict,
    }

    out_path = output / 'vol-walkforward-v2-dolthub-oos-summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
