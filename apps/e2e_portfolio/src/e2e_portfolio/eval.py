"""Walk-forward eval driver: trains per-fold, scores val daily, persists artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from tinygrad import Tensor

from e2e_portfolio.data import (
    DEFAULT_CLOSE_PKL, K_FORWARD, PHASE4D_TICKERS, Panel,
    load_close, load_macro_panel, prepare_panel,
)
from e2e_portfolio.model import Allocator, Hparams, save_npz, load_npz
from e2e_portfolio.train import TrainConfig, train_one_fold

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'Output'


FOLDS = [
    {'name': 'fold1', 'val_start': '2015-01-01', 'val_end': '2018-12-31'},
    {'name': 'fold2', 'val_start': '2019-01-01', 'val_end': '2022-12-31'},
    {'name': 'fold3', 'val_start': '2023-01-01', 'val_end': '2025-12-31'},
]
TRAIN_WINDOW_YEARS = 5


def build_fold_panels(full_panel: Panel, val_start: str, val_end: str):
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)
    train_end_ts = val_start_ts - pd.Timedelta(days=1)
    train_start_ts = train_end_ts - pd.Timedelta(days=365 * TRAIN_WINDOW_YEARS)
    train = full_panel.slice_by_date(train_start_ts, train_end_ts)
    val = full_panel.slice_by_date(val_start_ts, val_end_ts)
    return train, val


def daily_returns_from_panel(model: Allocator, val_panel: Panel,
                             close: pd.DataFrame, k_forward: int) -> pd.Series:
    """Walk daily through the val window. Each day, run the model on
    the trailing-60d window to get target weights; mark to NEXT-DAY
    close-to-close return on each ETF. Cash leg returns 0.

    This is "rebal daily, hold one day" — the cleanest walk-forward
    reading of the K=20-trained model. The K-day forward objective
    teaches the model to size positions on a 20-day horizon, but for
    PnL accounting we mark daily so the stream is comparable to DCA.
    """
    n = len(val_panel.dates)
    if n == 0:
        return pd.Series([], dtype=np.float64)
    # Predict weights on all val samples in batches.
    Tensor.training = False
    bsz = 256
    all_w = []
    for i in range(0, n, bsz):
        j = min(n, i + bsz)
        xa = Tensor(val_panel.X_assets[i:j])
        xm = Tensor(val_panel.X_macro[i:j])
        w = model(xa, xm).numpy()
        all_w.append(w)
    W = np.concatenate(all_w, axis=0)  # (n_val, N+1)
    weights_assets = W[:, :-1]  # drop cash leg

    # Compute next-day arithmetic returns for each anchor date.
    close_arr = close[PHASE4D_TICKERS].values
    close_index = close.index
    idx_map = {d: i for i, d in enumerate(close_index)}

    daily_ret = []
    daily_dates = []
    for k, anchor in enumerate(val_panel.dates):
        t = idx_map.get(anchor)
        if t is None or t + 1 >= len(close_arr):
            continue
        entry = close_arr[t]
        exit_ = close_arr[t + 1]
        r_assets = exit_ / np.maximum(entry, 1e-9) - 1.0
        port_ret = float((weights_assets[k] * r_assets).sum())
        # cash leg contributes 0
        daily_ret.append(port_ret)
        daily_dates.append(close_index[t + 1])  # the realized return is on t+1
    return pd.Series(daily_ret, index=pd.DatetimeIndex(daily_dates), dtype=np.float64)


def run_fold(full_panel: Panel, close: pd.DataFrame, fold_cfg: dict,
             cfg: TrainConfig, hp: Hparams,
             save_prefix: str = 'e2e-portfolio') -> dict:
    name = fold_cfg['name']
    print(f'\n=== {name}: val {fold_cfg["val_start"]} -> {fold_cfg["val_end"]} ===')
    train_panel, val_panel = build_fold_panels(
        full_panel, fold_cfg['val_start'], fold_cfg['val_end'])
    print(f'  train: {len(train_panel.dates)} samples '
          f'({train_panel.dates[0].date() if len(train_panel.dates) else None} -> '
          f'{train_panel.dates[-1].date() if len(train_panel.dates) else None})')
    print(f'  val:   {len(val_panel.dates)} samples '
          f'({val_panel.dates[0].date() if len(val_panel.dates) else None} -> '
          f'{val_panel.dates[-1].date() if len(val_panel.dates) else None})')

    # split a tail of train as inner-val for early stopping
    n_train = len(train_panel.dates)
    inner_val_n = max(64, n_train // 10)
    inner_train = Panel(
        X_assets=train_panel.X_assets[:-inner_val_n],
        X_macro=train_panel.X_macro[:-inner_val_n],
        fwd_ret=train_panel.fwd_ret[:-inner_val_n],
        dates=train_panel.dates[:-inner_val_n],
        tickers=train_panel.tickers,
    )
    inner_val = Panel(
        X_assets=train_panel.X_assets[-inner_val_n:],
        X_macro=train_panel.X_macro[-inner_val_n:],
        fwd_ret=train_panel.fwd_ret[-inner_val_n:],
        dates=train_panel.dates[-inner_val_n:],
        tickers=train_panel.tickers,
    )

    model, history = train_one_fold(inner_train, inner_val, cfg, hp)

    # Persist checkpoint.
    ckpt_path = OUT_DIR / f'{save_prefix}-{name}.npz'
    save_npz(model, str(ckpt_path))
    print(f'  saved {ckpt_path}')

    # Walk-forward daily return on OOS val.
    daily = daily_returns_from_panel(model, val_panel, close, K_FORWARD)
    print(f'  val daily: n={len(daily)}  '
          f'sh_ann={daily.mean()/max(daily.std(),1e-9)*np.sqrt(252):+.3f}')

    # Persist daily stream.
    out_npz = OUT_DIR / f'{save_prefix}-{name}-daily.npz'
    np.savez(out_npz,
             dates=daily.index.astype('datetime64[ns]').astype(np.int64),
             daily_ret=daily.values)
    print(f'  saved {out_npz}')

    return {
        'name': name,
        'val_start': fold_cfg['val_start'],
        'val_end': fold_cfg['val_end'],
        'n_val_days': int(len(daily)),
        'val_sharpe_ann': float(daily.mean() / max(daily.std(), 1e-9) * np.sqrt(252)),
        'val_mean_ret': float(daily.mean()),
        'val_std_ret': float(daily.std()),
        'daily_path': str(out_npz),
        'ckpt_path': str(ckpt_path),
    }


def baseline_streams(close: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the four baseline daily return streams over the full close index."""
    import sys
    sys.path.insert(0, str(REPO / 'apps/cfr/src'))
    from cfr.baselines import PassiveEW  # noqa: E402

    dca = PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(close)
    dca = pd.Series(np.asarray(dca, dtype=np.float64), index=close.index)
    ew_zero = PassiveEW(rebal_days=1, commission_bps=0.0).daily_returns(close)
    ew_zero = pd.Series(np.asarray(ew_zero, dtype=np.float64), index=close.index)

    # vol_v3 daily
    d = np.load(OUT_DIR / 'vol-v3-dolthub-oos-c200-returns.npz', allow_pickle=True)
    vol_dates = pd.to_datetime(np.asarray(d['rebal_dates'], dtype=str))
    vol_alpha = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    vol_daily = pd.Series(0.0, index=close.index)
    for i in range(len(vol_dates) - 1):
        mask = (vol_daily.index >= vol_dates[i]) & (vol_daily.index < vol_dates[i + 1])
        n = int(mask.sum())
        if n:
            vol_daily.loc[mask] = vol_alpha[i] / n
    mask = vol_daily.index >= vol_dates[-1]
    n = int(mask.sum())
    if n:
        vol_daily.loc[mask] = vol_alpha[-1] / n

    deterministic_2leg = dca + 2.0 * vol_daily
    # Learned 2-leg recipe.
    learned_2leg = 0.0506 * dca + 2.2388 * vol_daily

    return {
        'ew': ew_zero,
        'dca': dca,
        'deterministic_2leg': deterministic_2leg,
        'learned_2leg': learned_2leg,
    }


def summarize_vs_baseline(e2e: pd.Series, baseline: pd.Series, name: str) -> dict:
    """Annualized Sharpe diff with Ledoit-Wolf CI (stationary bootstrap)."""
    from ss_portfolio import sharpe_difference_ci

    common = e2e.index.intersection(baseline.index)
    a = e2e.reindex(common).fillna(0.0).values
    b = baseline.reindex(common).fillna(0.0).values
    if len(a) < 30:
        return {'name': name, 'n': len(a), 'delta_sr_ann': float('nan'),
                'ci_lo_ann': float('nan'), 'ci_hi_ann': float('nan'),
                'excludes_zero': False}
    ci = sharpe_difference_ci(a, b)
    ann = float(np.sqrt(252))
    return {
        'name': name,
        'n': int(len(a)),
        'a_sharpe_ann': float(a.mean() / max(a.std(), 1e-9) * ann),
        'b_sharpe_ann': float(b.mean() / max(b.std(), 1e-9) * ann),
        'delta_sr_ann': float(ci.delta_sr) * ann,
        'ci_lo_ann': float(ci.ci_lo) * ann,
        'ci_hi_ann': float(ci.ci_hi) * ann,
        'excludes_zero': bool(not ci.includes_zero),
    }


def pool_and_report(per_fold: list[dict], close: pd.DataFrame,
                    save_prefix: str = 'e2e-portfolio') -> dict:
    """Concatenate per-fold daily streams to a pooled OOS series, then
    Sharpe-diff every baseline."""
    pooled_ret = []
    pooled_dates = []
    for fold in per_fold:
        d = np.load(fold['daily_path'])
        dates = pd.DatetimeIndex(d['dates'].astype('datetime64[ns]'))
        pooled_ret.append(d['daily_ret'])
        pooled_dates.append(dates)
    pooled_ret = np.concatenate(pooled_ret)
    pooled_idx = pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in pooled_dates]))
    pooled = pd.Series(pooled_ret, index=pooled_idx).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep='first')]

    np.savez(OUT_DIR / f'{save_prefix}-pooled-daily.npz',
             dates=pooled.index.astype('datetime64[ns]').astype(np.int64),
             daily_ret=pooled.values)

    baselines = baseline_streams(close)
    comps = {}
    for bname, bstream in baselines.items():
        comps[bname] = summarize_vs_baseline(pooled, bstream, bname)

    ann = float(np.sqrt(252))
    summary = {
        'pooled_n': int(len(pooled)),
        'pooled_sharpe_ann': float(pooled.mean() / max(pooled.std(), 1e-9) * ann),
        'pooled_mean_ret': float(pooled.mean()),
        'pooled_std_ret': float(pooled.std()),
        'pooled_max_dd': float(((1 + pooled).cumprod() / (1 + pooled).cumprod().cummax() - 1).min()),
        'per_fold': per_fold,
        'baseline_comparisons': comps,
    }
    return summary
