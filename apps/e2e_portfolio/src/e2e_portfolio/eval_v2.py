"""v2 walk-forward eval: trains per-fold, builds daily streams including
vol_position * synthetic short-vol leg."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from tinygrad import Tensor

from e2e_portfolio.data import PHASE4D_TICKERS
from e2e_portfolio.data_v2 import PanelV2
from e2e_portfolio.model_v2 import AllocatorV2, HparamsV2, save_npz
from e2e_portfolio.train_v2 import TrainConfigV2, train_one_fold_v2

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'Output'

FOLDS = [
    {'name': 'fold1', 'val_start': '2015-01-01', 'val_end': '2018-12-31'},
    {'name': 'fold2', 'val_start': '2019-01-01', 'val_end': '2022-12-31'},
    {'name': 'fold3', 'val_start': '2023-01-01', 'val_end': '2025-12-31'},
]
TRAIN_WINDOW_YEARS = 5
K_FORWARD = 20


def build_fold_panels(full_panel: PanelV2, val_start: str, val_end: str):
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)
    train_end_ts = val_start_ts - pd.Timedelta(days=1)
    train_start_ts = train_end_ts - pd.Timedelta(days=365 * TRAIN_WINDOW_YEARS)
    return (full_panel.slice_by_date(train_start_ts, train_end_ts),
            full_panel.slice_by_date(val_start_ts, val_end_ts))


def daily_returns_from_panel(model: AllocatorV2, val_panel: PanelV2,
                             close: pd.DataFrame,
                             vol_synth_daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Walk daily: model emits (weights, vol_position); mark to next-day
    asset returns AND to next-day synthetic short-vol contribution
    (already daily-scaled — see data_v2.fwd_vol_pnl_per_day).
    Returns (daily_total_ret, daily_vol_position_series).
    """
    n = len(val_panel.dates)
    if n == 0:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)

    Tensor.training = False
    bsz = 256
    all_w = []
    all_vp = []
    for i in range(0, n, bsz):
        j = min(n, i + bsz)
        xa = Tensor(val_panel.X_assets[i:j])
        xm = Tensor(val_panel.X_macro[i:j])
        w, vp = model(xa, xm)
        all_w.append(w.numpy())
        all_vp.append(vp.numpy())
    W = np.concatenate(all_w, axis=0)
    VP = np.concatenate(all_vp, axis=0)
    weights_assets = W[:, :-1]

    close_arr = close[PHASE4D_TICKERS].values
    close_index = close.index
    idx_map = {d: i for i, d in enumerate(close_index)}
    vol_daily_map = {d: float(v) for d, v in vol_synth_daily.items()}

    daily_ret, daily_dates, daily_vp = [], [], []
    for k, anchor in enumerate(val_panel.dates):
        t = idx_map.get(anchor)
        if t is None or t + 1 >= len(close_arr):
            continue
        entry = close_arr[t]
        exit_ = close_arr[t + 1]
        r_assets = exit_ / np.maximum(entry, 1e-9) - 1.0
        port_ret = float((weights_assets[k] * r_assets).sum())
        # vol leg: VP[k] * vol_daily at t+1.
        next_date = close_index[t + 1]
        r_vol = float(VP[k]) * vol_daily_map.get(next_date, 0.0)
        daily_ret.append(port_ret + r_vol)
        daily_dates.append(next_date)
        daily_vp.append(float(VP[k]))
    return (pd.Series(daily_ret, index=pd.DatetimeIndex(daily_dates),
                      dtype=np.float64),
            pd.Series(daily_vp, index=pd.DatetimeIndex(daily_dates),
                      dtype=np.float64))


def vol_synth_daily_stream(close: pd.DataFrame,
                           parquet_path: Path | None = None) -> pd.Series:
    """Per-day synthetic short-vol return contribution, computed from
    raw IV/HV. Used both at training-feature time (as fwd_vol_pnl) and
    at eval time (daily mark-to-market)."""
    from e2e_portfolio.data_v2 import _build_iv_panel
    iv_df, hv_df, covered = _build_iv_panel(close.index, parquet_path=parquet_path)
    iv = iv_df.values
    hv = hv_df.values
    T, N = iv.shape
    avail = np.array([1.0 if t_ in covered else 0.0 for t_ in PHASE4D_TICKERS])
    denom = max(int(avail.sum()), 1)
    # Daily contribution = (iv_today - hv_today) / 252, EW over covered.
    vp_per_day = ((iv - hv) * avail[None, :]).sum(axis=1) / denom
    # Annualized vol-points -> daily return contribution.
    # Friction: amortize 10 bps over 252 trading days when iv_available.
    daily = vp_per_day / 252.0 - (10e-4 / 252.0)
    return pd.Series(daily, index=close.index)


def run_fold(full_panel: PanelV2, close: pd.DataFrame,
             vol_synth_daily: pd.Series, fold_cfg: dict,
             cfg: TrainConfigV2, hp: HparamsV2,
             save_prefix: str = 'e2e-portfolio-v2') -> dict:
    name = fold_cfg['name']
    print(f'\n=== {name}: val {fold_cfg["val_start"]} -> {fold_cfg["val_end"]} ===',
          flush=True)
    train_panel, val_panel = build_fold_panels(
        full_panel, fold_cfg['val_start'], fold_cfg['val_end'])
    print(f'  train n={len(train_panel.dates)}  val n={len(val_panel.dates)}',
          flush=True)

    n_train = len(train_panel.dates)
    inner_val_n = max(64, n_train // 10)
    inner_train = PanelV2(
        X_assets=train_panel.X_assets[:-inner_val_n],
        X_macro=train_panel.X_macro[:-inner_val_n],
        fwd_ret=train_panel.fwd_ret[:-inner_val_n],
        fwd_vol_pnl=train_panel.fwd_vol_pnl[:-inner_val_n],
        dates=train_panel.dates[:-inner_val_n],
        tickers=train_panel.tickers,
        covered_tickers=train_panel.covered_tickers,
    )
    inner_val = PanelV2(
        X_assets=train_panel.X_assets[-inner_val_n:],
        X_macro=train_panel.X_macro[-inner_val_n:],
        fwd_ret=train_panel.fwd_ret[-inner_val_n:],
        fwd_vol_pnl=train_panel.fwd_vol_pnl[-inner_val_n:],
        dates=train_panel.dates[-inner_val_n:],
        tickers=train_panel.tickers,
        covered_tickers=train_panel.covered_tickers,
    )

    model, history = train_one_fold_v2(inner_train, inner_val, cfg, hp)

    ckpt_path = OUT_DIR / f'{save_prefix}-{name}.npz'
    save_npz(model, str(ckpt_path))
    print(f'  saved {ckpt_path}', flush=True)

    daily, daily_vp = daily_returns_from_panel(model, val_panel, close, vol_synth_daily)
    sh_ann = (daily.mean() / max(daily.std(), 1e-9) * np.sqrt(252)) if len(daily) else 0.0
    print(f'  val daily: n={len(daily)}  sh_ann={sh_ann:+.3f}  '
          f'vp mean={daily_vp.mean():.3f} std={daily_vp.std():.3f}', flush=True)

    out_npz = OUT_DIR / f'{save_prefix}-{name}-daily.npz'
    np.savez(out_npz,
             dates=daily.index.astype('datetime64[ns]').astype(np.int64),
             daily_ret=daily.values,
             vol_position=daily_vp.values)

    return {
        'name': name,
        'val_start': fold_cfg['val_start'],
        'val_end': fold_cfg['val_end'],
        'n_val_days': int(len(daily)),
        'val_sharpe_ann': float(sh_ann),
        'val_mean_ret': float(daily.mean()) if len(daily) else 0.0,
        'val_std_ret': float(daily.std()) if len(daily) else 0.0,
        'vol_position_mean': float(daily_vp.mean()) if len(daily_vp) else 0.0,
        'vol_position_std': float(daily_vp.std()) if len(daily_vp) else 0.0,
        'vol_position_min': float(daily_vp.min()) if len(daily_vp) else 0.0,
        'vol_position_max': float(daily_vp.max()) if len(daily_vp) else 0.0,
        'daily_path': str(out_npz),
        'ckpt_path': str(ckpt_path),
    }


def baseline_streams(close: pd.DataFrame) -> dict[str, pd.Series]:
    import sys
    sys.path.insert(0, str(REPO / 'apps/cfr/src'))
    from cfr.baselines import PassiveEW  # noqa: E402

    dca = PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(close)
    dca = pd.Series(np.asarray(dca, dtype=np.float64), index=close.index)
    ew_zero = PassiveEW(rebal_days=1, commission_bps=0.0).daily_returns(close)
    ew_zero = pd.Series(np.asarray(ew_zero, dtype=np.float64), index=close.index)

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

    return {
        'ew': ew_zero,
        'dca': dca,
        'deterministic_2leg': dca + 2.0 * vol_daily,
        'learned_2leg': 0.0506 * dca + 2.2388 * vol_daily,
    }


def summarize_vs_baseline(e2e: pd.Series, baseline: pd.Series, name: str) -> dict:
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
                    save_prefix: str = 'e2e-portfolio-v2') -> dict:
    pooled_ret, pooled_dates, pooled_vp = [], [], []
    for fold in per_fold:
        d = np.load(fold['daily_path'])
        dates = pd.DatetimeIndex(d['dates'].astype('datetime64[ns]'))
        pooled_ret.append(d['daily_ret'])
        pooled_dates.append(dates)
        if 'vol_position' in d.files:
            pooled_vp.append(d['vol_position'])
    pooled_ret = np.concatenate(pooled_ret)
    pooled_idx = pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in pooled_dates]))
    pooled = pd.Series(pooled_ret, index=pooled_idx).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep='first')]
    pooled_vp_arr = np.concatenate(pooled_vp) if pooled_vp else np.array([])

    np.savez(OUT_DIR / f'{save_prefix}-pooled-daily.npz',
             dates=pooled.index.astype('datetime64[ns]').astype(np.int64),
             daily_ret=pooled.values)

    baselines = baseline_streams(close)
    comps = {bname: summarize_vs_baseline(pooled, b, bname)
             for bname, b in baselines.items()}
    ann = float(np.sqrt(252))
    return {
        'pooled_n': int(len(pooled)),
        'pooled_sharpe_ann': float(pooled.mean() / max(pooled.std(), 1e-9) * ann),
        'pooled_mean_ret': float(pooled.mean()),
        'pooled_std_ret': float(pooled.std()),
        'pooled_max_dd': float(((1 + pooled).cumprod() /
                                (1 + pooled).cumprod().cummax() - 1).min()),
        'pooled_vol_position_mean': float(pooled_vp_arr.mean())
            if len(pooled_vp_arr) else 0.0,
        'pooled_vol_position_std': float(pooled_vp_arr.std())
            if len(pooled_vp_arr) else 0.0,
        'pooled_vol_position_min': float(pooled_vp_arr.min())
            if len(pooled_vp_arr) else 0.0,
        'pooled_vol_position_max': float(pooled_vp_arr.max())
            if len(pooled_vp_arr) else 0.0,
        'per_fold': per_fold,
        'baseline_comparisons': comps,
    }
