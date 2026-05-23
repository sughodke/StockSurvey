"""v3 — regime-gated short-vol on DoltHub 2024-26 OOS.

Pre-registered extension of vol-v3-regime-gated (`vol-surface-v3-regime-gated.md`)
to never-seen 2024-26 data. The 30-rebal gauss314 stream that posted
ann Sharpe +1.15 / deflated t +1.32 is short — we want to know if the
**signal + the rho≈0 with DCA** both replicate on 2.7 more years of OOS
substrate before recommending the ~weeks-of-options-broker work to make
this deployable.

## Architecture

DoltHub carries `iv_current` + `hv_current` per (date, symbol) but NOT
the 10-feature gauss314 surface OR the OI columns v3 used. So this is
NOT a bit-identical v3 reproduction — it's the closest DoltHub-faithful
analogue:

1. **Predictor**: the v2-dolthub-oos 4-feature OLS
   (iv_over_hv, iv_z, iv_change_4w, hv_change_4w → iv_rv_gap), trained
   2019-10 → 2023-07 (gauss314 overlap) and frozen for the OOS span.
2. **Universe filter**: no OI panel on DoltHub. Skip — the entire DoltHub
   set is liquid optionable to begin with. (v3's OI-top-200 over
   gauss314 reduced 2073 → 200 names; DoltHub has ~3K daily but is
   already a curated optionable cohort.)
3. **Regime gate**: VIX > 126d rolling median, identical mechanism to
   v3. VIX pulled from FRED (`VIXCLS`) via ss_macro — gauss314's per-row
   VIX column doesn't exist on DoltHub.
4. **Top-K**: K=100 picks per weekly rebal date by predicted gap.
5. **Alpha**: per-rebal top-K mean iv_rv_gap minus universe mean.
   Gated stream zeros alpha on closed-gate rebals (deferred to passive
   universe baseline, contributes zero alpha).

## What we want to learn

- Does v3's pooled fired-alpha Sharpe (≥ +0.30 was the v3 PASS bar)
  reproduce on 120+ weekly rebals in 2024-26?
- Does the full-panel-alpha stream's correlation with DCA-block stay
  near zero? (The original ρ ≈ −0.002 was on 30 tail-aligned blocks.)
- Combined: does deflated-t lift over DCA-alone hold on a 5× larger
  vol sample, or was the +1.70 ensemble peak gauss314-era-specific?

## Falsification bar (pre-registered, locked before running)

- **PASS (replicates)**: pooled fired-alpha Sharpe ≥ +0.30 AND fire-rate
  ∈ [20%, 80%] AND ≥ 60% of OOS quarters positive AND |ρ(full_panel,
  DCA-block)| ≤ 0.15.
- **MARGINAL**: fired-alpha Sharpe ∈ [0, +0.30] OR fire-rate outside
  [20%, 80%] OR 40-60% positive quarters.
- **FAIL**: fired-alpha Sharpe < 0 OR < 40% positive quarters OR
  |ρ(full_panel, DCA-block)| > 0.30.

Output: `Output/vol-v3-dolthub-oos-{returns.npz,summary.json}`. The
returns NPZ carries dates (the missing-from-original-v3 metadata) so a
future ensemble script can date-align cleanly without tail-overlap
approximation.

Run from repo root:
    uv run python apps/vol/scripts/run_walkforward_v3_dolthub_oos.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_macro import load_fred_series

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / 'Output'
DOLTHUB_PARQUET = REPO_ROOT / '.iv-cache/volatility_history.parquet'

# Pre-reg split: 2019-10 → 2023-07 train (matches v3 gauss314 window) /
# 2023-08 → 2026-04 val (never-seen on DoltHub for the v3 architecture).
TRAIN_START = pd.Timestamp('2019-10-14')
TRAIN_END   = pd.Timestamp('2023-07-28')
VAL_START   = pd.Timestamp('2023-08-01')
VAL_END     = pd.Timestamp('2026-04-30')

REBAL_WEEKS = 4   # 20 trading days per v3
FORWARD_DAYS = 20
GATE_LOOKBACK_TRADING_DAYS = 126


# ---------------------------------------------------------------- loaders ---

def load_dolthub_panel(parquet_path: Path) -> pd.DataFrame:
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
    forward_trading_days: int = FORWARD_DAYS,
) -> pd.DataFrame:
    """Wide-form: index=weekly_dates, columns=symbols; forward 20d
    annualized realized vol from honest log-return std (not DoltHub's
    autocorrelated hv_current).
    """
    log_p = np.log(prices.replace(0.0, np.nan)).dropna(how='all')
    log_r = log_p.diff()
    daily_idx = prices.index.sort_values()
    weekly_idx = pd.DatetimeIndex(sorted(weekly_dates))
    forward_rv: dict[pd.Timestamp, pd.Series] = {}
    annualization = np.sqrt(252)
    for d in weekly_idx:
        pos = daily_idx.searchsorted(d, side='left')
        if pos + forward_trading_days >= len(daily_idx):
            continue
        window = log_r.iloc[pos + 1: pos + 1 + forward_trading_days]
        if window.shape[0] < forward_trading_days * 0.7:
            continue
        forward_rv[d] = window.std() * annualization
    return pd.DataFrame(forward_rv).T.sort_index()


def build_features_and_target(
    panel: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    iv_wide = panel.pivot(index='date', columns='symbol', values='iv_current')
    hv_wide = panel.pivot(index='date', columns='symbol', values='hv_current')
    print('  computing forward realized vol from Stooq prices...', flush=True)
    forward_rv = compute_forward_realized_vol(prices, iv_wide.index)
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
    for w, n in [(iv_z, 'iv_z'), (iv_change_4w, 'iv_change_4w'),
                 (hv_change_4w, 'hv_change_4w'),
                 (target_wide, 'iv_rv_gap')]:
        out = out.merge(melt(w, n), on=['date', 'symbol'], how='inner')
    return out.dropna()


# ---------------------------------------------------------------- predictor -

def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple:
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
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


# ----------------------------------------------------------- regime gate ----

def build_vix_gate(
    vix_daily: pd.Series, lookback_trading_days: int = GATE_LOOKBACK_TRADING_DAYS,
) -> pd.Series:
    """`fired[t]` = VIX[t] > rolling_median(VIX, lookback)[t]."""
    rm = vix_daily.rolling(window=lookback_trading_days,
                           min_periods=lookback_trading_days // 2).median()
    return (vix_daily > rm).rename('fired')


# ---------------------------------------------------------------- DCA loader

def load_dca_block_stream(rebal_dates: pd.DatetimeIndex) -> np.ndarray:
    """For each rebal date in `rebal_dates`, compute the DCA basket
    compounded return over [rebal_date, rebal_date + FORWARD_DAYS bars].
    Used for the rho(vol_alpha, DCA-block) correlation check.
    """
    import pickle
    from cfr.baselines import PassiveEW
    with open(REPO_ROOT / 'Output/cfr_phase4d_multiasset_close.pkl', 'rb') as f:
        close = pickle.load(f)
    daily = pd.Series(
        np.asarray(PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(close),
                   dtype=np.float64),
        index=close.index,
    )
    out = []
    for d in rebal_dates:
        pos = daily.index.searchsorted(d, side='left')
        if pos + FORWARD_DAYS >= len(daily):
            out.append(np.nan)
            continue
        window = daily.iloc[pos + 1: pos + 1 + FORWARD_DAYS]
        if window.size < FORWARD_DAYS * 0.7:
            out.append(np.nan)
        else:
            out.append((1.0 + window).prod() - 1.0)
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------- main ------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--rebal-weeks', type=int, default=REBAL_WEEKS)
    p.add_argument('--gate-lookback', type=int, default=GATE_LOOKBACK_TRADING_DAYS)
    p.add_argument('--parquet', default=str(DOLTHUB_PARQUET))
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading DoltHub parquet...', flush=True)
    t0 = time.perf_counter()
    panel = load_dolthub_panel(Path(args.parquet))
    print(f'  {len(panel):,} rows in {time.perf_counter()-t0:.1f}s', flush=True)
    print(f'  date range: {panel["date"].min().date()} → '
          f'{panel["date"].max().date()}', flush=True)

    print('Loading Stooq prices for honest forward-RV target...', flush=True)
    t0 = time.perf_counter()
    syms = sorted(panel['symbol'].unique())
    prices, _, _, _ = load_stooq_matrix(
        str(REPO_ROOT / 'StooqData'), min_history=100,
        start_date='2019-01-01', end_date='2026-05-31', tickers=syms)
    print(f'  {prices.shape[0]} dates × {prices.shape[1]} symbols in '
          f'{time.perf_counter()-t0:.1f}s '
          f'({100*prices.shape[1]/len(syms):.0f}% coverage)', flush=True)

    print('Building features + target...', flush=True)
    t0 = time.perf_counter()
    merged = build_features_and_target(panel, prices)
    print(f'  {len(merged):,} usable rows in {time.perf_counter()-t0:.1f}s', flush=True)

    feature_cols = ['iv_over_hv', 'iv_z', 'iv_change_4w', 'hv_change_4w']

    train = merged[(merged['date'] >= TRAIN_START) & (merged['date'] <= TRAIN_END)]
    val   = merged[(merged['date'] >= VAL_START)   & (merged['date'] <= VAL_END)]
    print(f'  train: {len(train):,} cells, val: {len(val):,} cells', flush=True)

    print('Loading VIX from FRED...', flush=True)
    vix = load_fred_series('VIXCLS').rename('VIX').dropna()
    gate = build_vix_gate(vix, args.gate_lookback)
    print(f'  VIX span: {vix.index[0].date()} → {vix.index[-1].date()}, '
          f'n={vix.size}', flush=True)

    # Train OLS once on the full gauss314-overlap train window.
    coefs, train_r2, mu, sd = fit_ols(
        train[feature_cols].values, train['iv_rv_gap'].values)
    val_pred = apply_ols(val[feature_cols].values, coefs, mu, sd)
    val_corr = float(np.corrcoef(val_pred, val['iv_rv_gap'].values)[0, 1])
    print(f'\n  train R² = {train_r2:+.4f}  val Pearson r = {val_corr:+.4f}',
          flush=True)
    for fn, c in zip(feature_cols, coefs[:-1]):
        print(f'    {fn:>20s}: {c:+.5f}', flush=True)

    val_aug = val[['date', 'symbol']].copy()
    val_aug['pred_gap'] = val_pred
    val_aug['iv_rv_gap'] = val['iv_rv_gap'].values

    # Per-rebal portfolio over weekly snapshots.
    val_dates = pd.DatetimeIndex(sorted(val['date'].unique()))
    rebal_dates = val_dates[::args.rebal_weeks]
    print(f'\n  rebal cadence: {args.rebal_weeks} weeks '
          f'({len(rebal_dates)} rebals in OOS span)', flush=True)

    # Score each rebal: top-K mean iv_rv_gap vs universe mean iv_rv_gap.
    full_panel_alpha = []      # gated × fire-flag
    fired_only_alpha = []
    rebal_kept = []
    full_top_k_pnl = []
    full_univ_pnl  = []
    fire_flags = []
    for rd in rebal_dates:
        day = val_aug[val_aug['date'] == rd]
        if len(day) < args.top_k:
            continue
        picks = day.nlargest(args.top_k, 'pred_gap')
        top_k = float(picks['iv_rv_gap'].mean())
        univ  = float(day['iv_rv_gap'].mean())
        alpha = top_k - univ
        # Gate on VIX as of the rebal date.
        gate_pos = gate.index.searchsorted(rd, side='right') - 1
        fires = bool(gate.iloc[gate_pos]) if gate_pos >= 0 else False
        full_top_k_pnl.append(top_k)
        full_univ_pnl.append(univ)
        fire_flags.append(fires)
        rebal_kept.append(rd)
        if fires:
            full_panel_alpha.append(alpha)
            fired_only_alpha.append(alpha)
        else:
            # closed gate: defer to passive universe baseline → alpha = 0
            full_panel_alpha.append(0.0)

    rebal_kept = pd.DatetimeIndex(rebal_kept)
    full_panel_alpha = np.asarray(full_panel_alpha, dtype=np.float64)
    fired_only_alpha = np.asarray(fired_only_alpha, dtype=np.float64)
    fire_flags = np.asarray(fire_flags, dtype=bool)

    n_total = full_panel_alpha.size
    n_fired = int(fire_flags.sum())
    fire_rate = n_fired / n_total if n_total else 0.0
    ann = float(np.sqrt(52.0 / args.rebal_weeks))

    full_pooled_sh = (full_panel_alpha.mean() / full_panel_alpha.std(ddof=1) * ann
                      if full_panel_alpha.size > 1 and full_panel_alpha.std(ddof=1) > 1e-12 else 0.0)
    fired_pooled_sh = (fired_only_alpha.mean() / fired_only_alpha.std(ddof=1) * ann
                       if fired_only_alpha.size > 1 and fired_only_alpha.std(ddof=1) > 1e-12 else 0.0)

    # Per-quarter positivity
    qtrs = pd.date_range(VAL_START, VAL_END, freq='3MS')
    if qtrs[-1] < VAL_END:
        qtrs = qtrs.append(pd.DatetimeIndex([VAL_END + pd.Timedelta(days=1)]))
    n_pos_q = 0
    n_total_q = 0
    per_qtr = []
    for i in range(len(qtrs) - 1):
        q_lo, q_hi = qtrs[i], qtrs[i + 1] - pd.Timedelta(days=1)
        mask = (rebal_kept >= q_lo) & (rebal_kept <= q_hi)
        if mask.sum() < 1:
            continue
        q_alpha = full_panel_alpha[mask].mean()
        n_total_q += 1
        if q_alpha > 0:
            n_pos_q += 1
        per_qtr.append({
            'period': f'{q_lo.date()} → {q_hi.date()}',
            'n_rebals': int(mask.sum()),
            'n_fired': int(fire_flags[mask].sum()),
            'mean_alpha': float(q_alpha),
        })

    # DCA-block correlation
    dca_block = load_dca_block_stream(rebal_kept)
    valid = ~np.isnan(dca_block)
    rho_full = (float(np.corrcoef(full_panel_alpha[valid], dca_block[valid])[0, 1])
                if valid.sum() > 2 else float('nan'))
    rho_fired = (float(np.corrcoef(
        full_panel_alpha[valid & fire_flags], dca_block[valid & fire_flags])[0, 1])
                 if (valid & fire_flags).sum() > 2 else float('nan'))

    # Headline
    print('\n' + '=' * 88, flush=True)
    print(f'VOL v3 — DoltHub OOS 2024-26', flush=True)
    print(f'  rebals: {n_total} total, {n_fired} fired (rate {fire_rate*100:.1f}%)',
          flush=True)
    print(f'  pooled fired-alpha Sharpe: {fired_pooled_sh:+.3f}', flush=True)
    print(f'  pooled full-panel  Sharpe: {full_pooled_sh:+.3f}', flush=True)
    print(f'  positive quarters: {n_pos_q}/{n_total_q} '
          f'({100*n_pos_q/max(n_total_q,1):.0f}%)', flush=True)
    print(f'  rho(full_panel, DCA-block):  {rho_full:+.3f}', flush=True)
    print(f'  rho(fired_only, DCA-block):  {rho_fired:+.3f}', flush=True)

    # Pre-reg verdict
    pass_sh    = fired_pooled_sh >= 0.30
    pass_fire  = 0.20 <= fire_rate <= 0.80
    pass_qtr   = n_total_q > 0 and (n_pos_q / n_total_q) >= 0.60
    pass_rho   = abs(rho_full) <= 0.15
    n_pass     = sum([pass_sh, pass_fire, pass_qtr, pass_rho])
    verdict = ('PASS' if n_pass == 4
               else 'MARGINAL' if n_pass >= 2
               else 'FAIL')
    print(f'\n  pre-reg gates: sh={pass_sh} fire={pass_fire} '
          f'qtr={pass_qtr} rho={pass_rho}  → {verdict}', flush=True)
    print('=' * 88, flush=True)

    # Dump stream NPZ with dates (the missing-from-original-v3 metadata).
    out_npz = output / 'vol-v3-dolthub-oos-returns.npz'
    np.savez(
        out_npz,
        full_panel_alpha=full_panel_alpha,
        fired_only_alpha=fired_only_alpha,
        rebal_dates=np.asarray([str(d.date()) for d in rebal_kept]),
        fire_flags=fire_flags,
        periods_per_year=np.float64(52.0 / args.rebal_weeks),
        gate_lookback=np.int32(args.gate_lookback),
        commission_bps=np.float64(0.0),  # short-vol PnL is accounted upstream
        pre_registered_bar=np.str_(
            'fired_pooled_sh>=+0.30 ; fire_rate in [0.20,0.80] ; '
            'positive_quarters>=60% ; |rho_full_DCA|<=0.15'),
        verdict=np.str_(verdict),
    )
    print(f'\n→ {out_npz}', flush=True)

    summary = {
        'data_source': 'DoltHub volatility_history',
        'train_period': f'{TRAIN_START.date()} → {TRAIN_END.date()}',
        'val_period_oos': f'{VAL_START.date()} → {VAL_END.date()}',
        'train_r2': train_r2,
        'val_pearson_r': val_corr,
        'rebal_weeks': args.rebal_weeks,
        'gate_lookback_days': args.gate_lookback,
        'top_k': args.top_k,
        'n_rebals_total': int(n_total),
        'n_rebals_fired': int(n_fired),
        'fire_rate': fire_rate,
        'pooled_fired_alpha_sharpe': fired_pooled_sh,
        'pooled_full_panel_alpha_sharpe': full_pooled_sh,
        'positive_quarters': f'{n_pos_q}/{n_total_q}',
        'rho_full_panel_vs_dca_block': rho_full,
        'rho_fired_only_vs_dca_block': rho_fired,
        'per_quarter': per_qtr,
        'pre_registered_gates': {
            'fired_sharpe_>=_+0.30': bool(pass_sh),
            'fire_rate_in_[0.20,0.80]': bool(pass_fire),
            'positive_quarters_>=_60%': bool(pass_qtr),
            'abs_rho_full_dca_<=_0.15': bool(pass_rho),
        },
        'verdict': verdict,
    }
    out_json = output / 'vol-v3-dolthub-oos-summary.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f'→ {out_json}', flush=True)


if __name__ == '__main__':
    main()
