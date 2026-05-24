"""Pre-registered vol-hyperparam × DCA ensemble joint Optuna search.

Per `apps/docs/docs/TODO/vol-hyperparam-ensemble-optuna.md` (committed
before this script ran in commit `e3b2b8b`). The pre-reg locks the
search space (2,100 combinations of vol-recipe knobs × vega_scale),
N_TRIALS=200, date-based split (train 2023-08 → 2024-12 / val 2025-01
→ 2026-03), DCA basket fixed at canonical-13, the DSR-deflated
objective, and the falsification bar.

Heavy lifting (DoltHub panel + Stooq prices + feature matrix +
forward-RV + OLS fit + VIX series + DCA daily stream) is done ONCE
before the Optuna loop; each trial only re-runs the per-rebal
accounting under its (top_k, gate_lookback, gate_quantile,
rebal_trading_days) parameters.

Run from repo root:
    uv run python apps/vol/scripts/optuna_vol_hyperparam_ensemble.py
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from cfr.baselines import PassiveEW
from ss_loaders import load_stooq_matrix
from ss_macro import load_fred_series
from ss_portfolio import standardize_oos


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_DIR = REPO_ROOT / 'StooqData'
DOLTHUB_PARQUET = REPO_ROOT / '.iv-cache/volatility_history.parquet'
DCA_PICKLE = REPO_ROOT / 'Output/cfr_phase4d_multiasset_close.pkl'

# ---- Locked pre-reg parameters --------------------------------------------

N_TRIALS = 200
SHARPE_STD_ANN = 0.25
COMMISSION_BPS_DCA = 10.0
FORWARD_DAYS = 20         # fixed; sweeping would force 2D non-overlap constraint

TRAIN_START_PREDICTOR = pd.Timestamp('2019-10-14')
TRAIN_END_PREDICTOR   = pd.Timestamp('2023-07-28')
SUBSTRATE_VAL_START   = pd.Timestamp('2023-08-01')   # full vol OOS substrate
SUBSTRATE_VAL_END     = pd.Timestamp('2026-04-30')

# Optuna train/val split (date-based, NOT count-based, so it's fair
# across trials with different rebal_trading_days)
OPT_TRAIN_START = pd.Timestamp('2023-08-01')
OPT_TRAIN_END   = pd.Timestamp('2024-12-31')
OPT_VAL_START   = pd.Timestamp('2025-01-01')
OPT_VAL_END     = pd.Timestamp('2026-03-15')

# Canonical incumbent under same method (v3-DoltHub recipe + vega=3.0)
CANON_TOP_K = 100
CANON_GATE_LOOKBACK = 126
CANON_REBAL = 20
CANON_GATE_QUANTILE = 0.50
CANON_VEGA = 3.0

# Search grids (locked)
TOP_K_GRID = [25, 50, 100, 200, 400]
GATE_LOOKBACK_GRID = [21, 63, 126, 252, 504]
REBAL_GRID = [20, 40, 60]
GATE_QUANTILE_GRID = [0.40, 0.50, 0.60, 0.70]
VEGA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


# ---- Heavy one-time setup -------------------------------------------------

def load_dolthub_panel(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'act_symbol': 'symbol'})
    df['iv_current'] = pd.to_numeric(df['iv_current'], errors='coerce')
    df['hv_current'] = pd.to_numeric(df['hv_current'], errors='coerce')
    df = df.dropna(subset=['iv_current', 'hv_current'])
    df = df[(df['iv_current'] > 0) & (df['hv_current'] > 0)]
    return df.sort_values(['date', 'symbol']).reset_index(drop=True)


def compute_forward_realized_vol(prices: pd.DataFrame,
                                  weekly_dates: pd.DatetimeIndex,
                                  forward_trading_days: int = FORWARD_DAYS
                                  ) -> pd.DataFrame:
    log_p = np.log(prices.replace(0.0, np.nan)).dropna(how='all')
    log_r = log_p.diff()
    daily_idx = prices.index.sort_values()
    weekly_idx = pd.DatetimeIndex(sorted(weekly_dates))
    forward_rv = {}
    ann = np.sqrt(252)
    for d in weekly_idx:
        pos = daily_idx.searchsorted(d, side='left')
        if pos + forward_trading_days >= len(daily_idx):
            continue
        window = log_r.iloc[pos + 1: pos + 1 + forward_trading_days]
        if window.shape[0] < forward_trading_days * 0.7:
            continue
        forward_rv[d] = window.std() * ann
    return pd.DataFrame(forward_rv).T.sort_index()


def build_features_and_target(panel: pd.DataFrame,
                              prices: pd.DataFrame) -> pd.DataFrame:
    iv_wide = panel.pivot(index='date', columns='symbol', values='iv_current')
    hv_wide = panel.pivot(index='date', columns='symbol', values='hv_current')
    forward_rv = compute_forward_realized_vol(prices, iv_wide.index)
    forward_rv = forward_rv.reindex(index=iv_wide.index)
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
                 (hv_change_4w, 'hv_change_4w'),
                 (target_wide, 'iv_rv_gap')]:
        out = out.merge(melt(w, n), on=['date', 'symbol'], how='inner')
    return out.dropna()


def fit_ols(X, y):
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


def apply_ols(X, coefs, mu, sd):
    Xz = (X - mu) / sd
    Xa = np.concatenate([Xz, np.ones((len(Xz), 1))], axis=1)
    return Xa @ coefs


# ---- Per-trial vol-stream construction ------------------------------------

def build_vol_alpha_stream(
    val_aug: pd.DataFrame,            # per-cell with date, symbol, pred_gap, iv_rv_gap
    vix: pd.Series,                   # daily VIX series indexed by date
    trading_idx: pd.DatetimeIndex,    # Stooq trading-day calendar
    top_k: int, gate_lookback: int, gate_quantile: float, rebal_trading_days: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Returns (full_panel_alpha, fire_flags, rebal_dates)."""
    # Build gate: VIX > rolling quantile-window
    rq = vix.rolling(window=gate_lookback,
                     min_periods=gate_lookback // 2).quantile(gate_quantile)
    fired_daily = (vix > rq)
    # Non-overlapping rebal sampling
    val_dates = pd.DatetimeIndex(sorted(val_aug['date'].unique()))
    rebals: list[pd.Timestamp] = []
    last_pos = -10**9
    for d in val_dates:
        pos = trading_idx.searchsorted(d, side='left')
        if pos >= len(trading_idx):
            break
        if (pos - last_pos) >= rebal_trading_days:
            rebals.append(d)
            last_pos = pos
    if not rebals:
        return (np.zeros(0), np.zeros(0, dtype=bool), pd.DatetimeIndex([]))
    full_alpha = []
    flags = []
    kept = []
    grp = val_aug.groupby('date', sort=False)
    for rd in rebals:
        if rd not in grp.groups:
            continue
        day = grp.get_group(rd)
        if len(day) < top_k:
            continue
        picks = day.nlargest(top_k, 'pred_gap')
        top_mean = float(picks['iv_rv_gap'].mean())
        univ_mean = float(day['iv_rv_gap'].mean())
        alpha = top_mean - univ_mean
        gate_pos = fired_daily.index.searchsorted(rd, side='right') - 1
        fires = bool(fired_daily.iloc[gate_pos]) if gate_pos >= 0 else False
        kept.append(rd)
        flags.append(fires)
        full_alpha.append(alpha if fires else 0.0)
    return (np.asarray(full_alpha, dtype=np.float64),
            np.asarray(flags, dtype=bool),
            pd.DatetimeIndex(kept))


def dca_block_returns_at(
    dca_daily: pd.Series, rebal_dates: pd.DatetimeIndex,
    forward_trading_days: int,
) -> np.ndarray:
    """Compound DCA daily returns over the forward `forward_trading_days`
    days starting AFTER each rebal_date."""
    idx = dca_daily.index
    out = np.zeros(len(rebal_dates), dtype=np.float64)
    for i, d in enumerate(rebal_dates):
        pos = idx.searchsorted(d, side='left')
        lo = pos + 1
        hi = min(pos + 1 + forward_trading_days, len(dca_daily))
        if lo >= len(dca_daily):
            out[i] = 0.0
            continue
        block = dca_daily.iloc[lo:hi].to_numpy()
        out[i] = float(np.prod(1.0 + block) - 1.0)
    return out


# ---- Trial evaluation -----------------------------------------------------

@dataclass
class TrialConfig:
    top_k: int
    gate_lookback: int
    gate_quantile: float
    rebal_trading_days: int
    vega_scale: float


def evaluate_config(
    cfg: TrialConfig, val_aug: pd.DataFrame, vix: pd.Series,
    trading_idx: pd.DatetimeIndex, dca_daily: pd.Series,
    train_start: pd.Timestamp, train_end: pd.Timestamp,
    val_start: pd.Timestamp, val_end: pd.Timestamp,
    n_trials: int = N_TRIALS, sharpe_std_ann: float = SHARPE_STD_ANN,
) -> dict:
    """Build the ensemble block-return stream for `cfg` and compute
    train/val MetricBlocks."""
    vol_alpha, fire_flags, rebal_dates = build_vol_alpha_stream(
        val_aug, vix, trading_idx,
        cfg.top_k, cfg.gate_lookback, cfg.gate_quantile,
        cfg.rebal_trading_days,
    )
    n = len(rebal_dates)
    if n < 6:
        return {'train_t': -1e6, 'val_t': -1e6, 'n_train': 0, 'n_val': 0,
                'reason': 'too_few_rebals', 'n_rebals_total': n}

    # Use rebal_trading_days as the forward DCA block length so the DCA
    # leg's block return matches the ensemble's annualization. The vol
    # stream's forward-RV target is fixed at 20d (pre-reg), but the
    # ensemble's per-period accounting follows the rebal cadence.
    fwd_days = cfg.rebal_trading_days
    dca_blocks = dca_block_returns_at(dca_daily, rebal_dates, fwd_days)
    ens = dca_blocks + cfg.vega_scale * vol_alpha

    # Split by date (NOT count) so the split is fair across cadences
    rebal_arr = rebal_dates.to_numpy()
    train_mask = (rebal_arr >= train_start.to_datetime64()) & (rebal_arr <= train_end.to_datetime64())
    val_mask   = (rebal_arr >= val_start.to_datetime64()) & (rebal_arr <= val_end.to_datetime64())
    ens_train = ens[train_mask]
    ens_val   = ens[val_mask]
    ens_train = ens_train[np.isfinite(ens_train)]
    ens_val   = ens_val[np.isfinite(ens_val)]

    ppy = 252.0 / cfg.rebal_trading_days
    sharpe_std_pp = sharpe_std_ann / math.sqrt(ppy)

    if ens_train.size < 5 or ens_val.size < 3:
        return {'train_t': -1e6, 'val_t': -1e6,
                'n_train': int(ens_train.size), 'n_val': int(ens_val.size),
                'reason': 'thin_split', 'n_rebals_total': n}

    mb_train = standardize_oos(ens_train, periods_per_year=ppy,
                               n_trials=n_trials, sharpe_std=sharpe_std_pp)
    mb_val   = standardize_oos(ens_val, periods_per_year=ppy,
                               n_trials=n_trials, sharpe_std=sharpe_std_pp)
    return {
        'train_t': mb_train.deflated_tstat,
        'train_ann_sharpe': mb_train.ann_sharpe,
        'train_max_dd': mb_train.max_dd,
        'val_t': mb_val.deflated_tstat,
        'val_ann_sharpe': mb_val.ann_sharpe,
        'val_max_dd': mb_val.max_dd,
        'val_skew': mb_val.skew,
        'val_kurt': mb_val.kurtosis,
        'n_train': int(ens_train.size),
        'n_val': int(ens_val.size),
        'n_rebals_total': n,
        'fire_rate': float(fire_flags.mean()) if fire_flags.size else 0.0,
        'ppy': ppy,
    }


def sample_config(trial: optuna.Trial) -> TrialConfig:
    return TrialConfig(
        top_k=trial.suggest_categorical('top_k', TOP_K_GRID),
        gate_lookback=trial.suggest_categorical('gate_lookback', GATE_LOOKBACK_GRID),
        gate_quantile=trial.suggest_categorical('gate_quantile', GATE_QUANTILE_GRID),
        rebal_trading_days=trial.suggest_categorical('rebal_trading_days', REBAL_GRID),
        vega_scale=trial.suggest_categorical('vega_scale', VEGA_GRID),
    )


def make_objective(val_aug, vix, trading_idx, dca_daily):
    def objective(trial: optuna.Trial) -> float:
        cfg = sample_config(trial)
        r = evaluate_config(cfg, val_aug, vix, trading_idx, dca_daily,
                            OPT_TRAIN_START, OPT_TRAIN_END,
                            OPT_VAL_START, OPT_VAL_END)
        trial.set_user_attr('train_ann_sharpe', r.get('train_ann_sharpe', 0.0))
        trial.set_user_attr('train_max_dd', r.get('train_max_dd', 0.0))
        trial.set_user_attr('n_train', r.get('n_train', 0))
        trial.set_user_attr('n_val', r.get('n_val', 0))
        trial.set_user_attr('fire_rate', r.get('fire_rate', 0.0))
        return r['train_t']
    return objective


# ---- Driver ---------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-trials', type=int, default=N_TRIALS)
    p.add_argument('--out', default=str(REPO_ROOT / 'Output/vol-hyperparam-ensemble-optuna.json'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    print(f'\n=== PRE-REGISTERED vol-hyperparam × DCA ensemble joint Optuna search ===')
    print(f'  pre-reg     = apps/docs/docs/TODO/vol-hyperparam-ensemble-optuna.md')
    print(f'  N_TRIALS    = {args.n_trials}')
    print(f'  sharpe_std  = {SHARPE_STD_ANN} ann (fixed across trial cadences)')
    print(f'  DCA basket  = canonical-13 (fixed; basket axis already adjudicated)')

    # --- Heavy one-time setup ---
    print('\n[1/5] Loading DoltHub parquet...', flush=True)
    t0 = time.perf_counter()
    panel = load_dolthub_panel(DOLTHUB_PARQUET)
    print(f'  {len(panel):,} rows in {time.perf_counter()-t0:.1f}s')

    print('[2/5] Loading Stooq prices...', flush=True)
    t0 = time.perf_counter()
    syms = sorted(panel['symbol'].unique())
    prices, _, _, _ = load_stooq_matrix(
        str(STOOQ_DIR), min_history=100,
        start_date='2019-01-01', end_date='2026-05-31', tickers=syms,
    )
    print(f'  {prices.shape[0]} dates × {prices.shape[1]} symbols '
          f'in {time.perf_counter()-t0:.1f}s')

    print('[3/5] Building features + target...', flush=True)
    t0 = time.perf_counter()
    merged = build_features_and_target(panel, prices)
    print(f'  {len(merged):,} usable rows in {time.perf_counter()-t0:.1f}s')

    print('[4/5] Fitting OLS predictor (frozen on 2019-10 → 2023-07)...', flush=True)
    fc = ['iv_over_hv', 'iv_z', 'iv_change_4w', 'hv_change_4w']
    train_pred = merged[(merged['date'] >= TRAIN_START_PREDICTOR)
                        & (merged['date'] <= TRAIN_END_PREDICTOR)]
    coefs, train_r2, mu, sd = fit_ols(train_pred[fc].values,
                                       train_pred['iv_rv_gap'].values)
    val_substrate = merged[(merged['date'] >= SUBSTRATE_VAL_START)
                            & (merged['date'] <= SUBSTRATE_VAL_END)].copy()
    val_substrate['pred_gap'] = apply_ols(val_substrate[fc].values, coefs, mu, sd)
    val_corr = float(np.corrcoef(val_substrate['pred_gap'],
                                  val_substrate['iv_rv_gap'])[0, 1])
    print(f'  train R² = {train_r2:+.4f}  val Pearson r = {val_corr:+.4f}')

    print('[5/5] Loading VIX + DCA cash stream...', flush=True)
    vix = load_fred_series('VIXCLS').rename('VIX').dropna()
    with open(DCA_PICKLE, 'rb') as f:
        dca_close = pickle.load(f)
    dca_daily = pd.Series(
        np.asarray(PassiveEW(rebal_days=80, commission_bps=COMMISSION_BPS_DCA)
                   .daily_returns(dca_close), dtype=np.float64),
        index=dca_close.index,
    )
    trading_idx = pd.DatetimeIndex(prices.index)
    print(f'  VIX n={vix.size}; DCA daily n={dca_daily.size}')

    val_aug = val_substrate[['date', 'symbol', 'pred_gap', 'iv_rv_gap']].copy()

    # --- Canonical reference under identical method ---
    print(f'\n--- Canonical reference: v3-DoltHub recipe + vol × {CANON_VEGA} ---')
    canon = TrialConfig(CANON_TOP_K, CANON_GATE_LOOKBACK, CANON_GATE_QUANTILE,
                        CANON_REBAL, CANON_VEGA)
    canon_r = evaluate_config(canon, val_aug, vix, trading_idx, dca_daily,
                              OPT_TRAIN_START, OPT_TRAIN_END,
                              OPT_VAL_START, OPT_VAL_END)
    print(f'  canon train: defl_t={canon_r["train_t"]:+.3f}  '
          f'annSh={canon_r["train_ann_sharpe"]:+.3f}  '
          f'maxDD={canon_r["train_max_dd"]:+.3f}  n={canon_r["n_train"]}')
    print(f'  canon val:   defl_t={canon_r["val_t"]:+.3f}  '
          f'annSh={canon_r["val_ann_sharpe"]:+.3f}  '
          f'maxDD={canon_r["val_max_dd"]:+.3f}  n={canon_r["n_val"]}')

    # --- Optuna search ---
    print(f'\n--- Optuna TPE search (n_trials={args.n_trials}) ---')
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    obj = make_objective(val_aug, vix, trading_idx, dca_daily)
    t0 = time.perf_counter()
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=False)
    print(f'  search done in {time.perf_counter()-t0:.1f}s')

    best = study.best_trial
    best_cfg = TrialConfig(**best.params)
    print(f'\n  best train deflated_t = {best.value:+.3f}')
    print(f'  best cfg = {best.params}')

    # --- Apply winner to val (already done inside evaluate_config when we re-run for full numbers) ---
    winner_full = evaluate_config(best_cfg, val_aug, vix, trading_idx, dca_daily,
                                  OPT_TRAIN_START, OPT_TRAIN_END,
                                  OPT_VAL_START, OPT_VAL_END)
    print(f'\n--- Winner on val ---')
    print(f'  winner val: defl_t={winner_full["val_t"]:+.3f}  '
          f'annSh={winner_full["val_ann_sharpe"]:+.3f}  '
          f'maxDD={winner_full["val_max_dd"]:+.3f}  n={winner_full["n_val"]}')

    # --- Verdict per pre-reg ---
    dt = winner_full['val_t'] - canon_r['val_t']
    dd_diff = winner_full['val_max_dd'] - canon_r['val_max_dd']
    vega_zero = (best_cfg.vega_scale == 0.0)
    if vega_zero:
        print(f'\n  NOTE: winner has vega_scale=0.0 (DCA-only) — pre-reg default '
              f'verdict = confirmed-null (vol recipe does not help)')
        verdict = 'confirmed-null'
    elif dt > 1.0 and dd_diff > -0.05:
        verdict = 'confirmed-OOS'
    elif dt >= 0.0 and dd_diff > -0.05:
        verdict = 'partial-OOS'
    else:
        verdict = 'confirmed-null'
    print(f'\n=== Verdict per pre-reg ===')
    print(f'  Δ deflated_t (winner_val − canon_val) = {dt:+.3f}')
    print(f'  Δ max_dd     (winner_val − canon_val) = {dd_diff:+.3f}')
    print(f'  winner vega_scale > 0?                 = {not vega_zero}')
    print(f'  -> {verdict}')

    # --- Persist ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'pre_reg_page': 'apps/docs/docs/TODO/vol-hyperparam-ensemble-optuna.md',
        'pre_reg_commit': 'e3b2b8b',
        'n_trials': args.n_trials,
        'sharpe_std_ann': SHARPE_STD_ANN,
        'forward_days': FORWARD_DAYS,
        'predictor_train_period': f'{TRAIN_START_PREDICTOR.date()} → {TRAIN_END_PREDICTOR.date()}',
        'predictor_train_r2': train_r2,
        'predictor_val_pearson_r': val_corr,
        'opt_train_period': f'{OPT_TRAIN_START.date()} → {OPT_TRAIN_END.date()}',
        'opt_val_period':   f'{OPT_VAL_START.date()} → {OPT_VAL_END.date()}',
        'canonical_cfg': {
            'top_k': CANON_TOP_K, 'gate_lookback': CANON_GATE_LOOKBACK,
            'gate_quantile': CANON_GATE_QUANTILE,
            'rebal_trading_days': CANON_REBAL, 'vega_scale': CANON_VEGA,
        },
        'canonical_result': canon_r,
        'winner_params': best.params,
        'winner_result':  winner_full,
        'delta_deflated_t': dt,
        'delta_max_dd':     dd_diff,
        'verdict': verdict,
        'top_10_trials_by_train_t': [
            {
                'params': t.params,
                'train_t': t.value,
                'train_ann_sharpe': t.user_attrs.get('train_ann_sharpe'),
                'train_max_dd': t.user_attrs.get('train_max_dd'),
                'n_train': t.user_attrs.get('n_train'),
                'n_val': t.user_attrs.get('n_val'),
                'fire_rate': t.user_attrs.get('fire_rate'),
            }
            for t in sorted(study.trials, key=lambda x: -(x.value or -1e18))[:10]
        ],
    }
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n→ {out}', flush=True)


if __name__ == '__main__':
    main()
