"""v3.5 walk-forward eval — extends v3 eval with long-vol head + v3 baseline.

Key additions vs v3:
  - daily_returns_from_panel uses 4-tuple model output (ew, vw, svs, lvp)
  - per-day total return adds long_vol_position[t] * vixy_daily_ret[t]
  - baseline_streams adds a 'v3' entry (v3 pooled daily stream from
    Output/e2e-portfolio-v3-pooled-daily.npz) so the verdict bar's
    "ΔSR vs v3" comparison is direct
  - per-fold results carry long_vol_position mean / std / 2020-Q1 mean
    for the mechanism check
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from tinygrad import Tensor

from e2e_portfolio.data_v3p5 import PanelV3p5, load_vixy_close
from e2e_portfolio.model_v3p5 import AllocatorV3p5, HparamsV3p5, save_npz
from e2e_portfolio.train_v3p5 import TrainConfigV3p5, train_one_fold_v3p5

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'Output'

FOLDS = [
    {'name': 'fold1', 'val_start': '2015-01-01', 'val_end': '2018-12-31'},
    {'name': 'fold2', 'val_start': '2019-01-01', 'val_end': '2022-12-31'},
    {'name': 'fold3', 'val_start': '2023-01-01', 'val_end': '2025-12-31'},
]
TRAIN_WINDOW_YEARS = 5
K_FWD = 20


def build_fold_panels(full_panel: PanelV3p5, val_start: str, val_end: str):
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)
    train_end_ts = val_start_ts - pd.Timedelta(days=1)
    train_start_ts = train_end_ts - pd.Timedelta(days=365 * TRAIN_WINDOW_YEARS)
    return (full_panel.slice_by_date(train_start_ts, train_end_ts),
            full_panel.slice_by_date(val_start_ts, val_end_ts))


def daily_returns_from_panel(model: AllocatorV3p5, val_panel: PanelV3p5,
                             close: pd.DataFrame, vixy_close: pd.Series,
                             ) -> tuple[pd.Series, pd.Series, pd.Series,
                                        np.ndarray, np.ndarray]:
    """Returns (daily_total_ret, daily_short_vol_scale, daily_long_vol_position,
    equity_weights_ts, vol_weights_ts)."""
    K = val_panel.X_assets.shape[1]
    n = len(val_panel.dates)
    if n == 0:
        empty = pd.Series(dtype=np.float64)
        return empty, empty, empty, np.zeros((0, K + 1)), np.zeros((0, K))

    Tensor.training = False
    bsz = 64
    all_ew, all_vw, all_svs, all_lvp = [], [], [], []
    for i in range(0, n, bsz):
        j = min(n, i + bsz)
        xa = Tensor(val_panel.X_assets[i:j])
        xm = Tensor(val_panel.X_macro[i:j])
        vm = Tensor(val_panel.valid_mask[i:j])
        ew, vw, svs, lvp = model(xa, xm, vm)
        all_ew.append(ew.numpy())
        all_vw.append(vw.numpy())
        all_svs.append(svs.numpy())
        all_lvp.append(lvp.numpy())
    EW = np.concatenate(all_ew, axis=0)
    VW = np.concatenate(all_vw, axis=0)
    SVS = np.concatenate(all_svs, axis=0)
    LVP = np.concatenate(all_lvp, axis=0)
    equity_weights_assets = EW[:, :K]
    cash_w = EW[:, K]

    tickers = val_panel.tickers
    close_sub = close[tickers]
    close_arr = close_sub.values
    close_index = close_sub.index
    idx_map = {d: i for i, d in enumerate(close_index)}

    # VIXY daily close aligned to close_index.
    vixy_aligned = vixy_close.reindex(close_index).ffill()
    vixy_arr = vixy_aligned.values
    vixy_log_ret = np.diff(np.log(np.maximum(vixy_arr, 1e-9)),
                           prepend=np.log(np.maximum(vixy_arr[:1], 1e-9)))
    vixy_daily_ret = np.exp(vixy_log_ret) - 1.0
    vixy_daily_ret = np.nan_to_num(vixy_daily_ret, nan=0.0,
                                    posinf=0.0, neginf=0.0)

    n_close = close_arr.shape[0]

    # Per-name short-vol contribution.
    per_name_daily_vol = np.zeros((n_close, K), dtype=np.float64)
    for k, anchor in enumerate(val_panel.dates):
        t = idx_map.get(anchor)
        if t is None:
            continue
        if k + 1 < len(val_panel.dates):
            next_t = idx_map.get(val_panel.dates[k + 1])
        else:
            next_t = None
        t_end = (min(t + K_FWD, next_t) if next_t is not None
                 else min(t + K_FWD, n_close - 1))
        per_day_vol_pnl = val_panel.fwd_vol_pnl[k] / float(K_FWD)
        per_day_vol_pnl = np.nan_to_num(per_day_vol_pnl, nan=0.0,
                                        posinf=0.0, neginf=0.0)
        contribution = SVS[k] * (VW[k] * per_day_vol_pnl)
        for tt in range(t + 1, t_end + 1):
            if tt < n_close:
                per_name_daily_vol[tt] += contribution

    # Equity daily weights timeline.
    daily_equity_w = np.zeros((n_close, K), dtype=np.float64)
    # NEW: long-vol position timeline (carry per-anchor lvp until next anchor).
    daily_long_vol_pos = np.zeros(n_close, dtype=np.float64)
    for k, anchor in enumerate(val_panel.dates):
        t = idx_map.get(anchor)
        if t is None:
            continue
        if k + 1 < len(val_panel.dates):
            next_t = idx_map.get(val_panel.dates[k + 1])
        else:
            next_t = None
        t_end = next_t - 1 if next_t is not None else min(t + K_FWD, n_close - 1)
        t_end = min(t_end, n_close - 1)
        for tt in range(t + 1, t_end + 1):
            daily_equity_w[tt] = equity_weights_assets[k]
            daily_long_vol_pos[tt] = LVP[k]

    val_start = val_panel.dates[0]
    val_end = val_panel.dates[-1] + pd.Timedelta(days=K_FWD * 2)
    mask = (close_index >= val_start) & (close_index <= val_end)

    log_close = np.log(np.maximum(close_arr, 1e-9))
    log_ret = np.diff(log_close, axis=0, prepend=log_close[:1])
    ret = np.exp(log_ret) - 1.0
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)

    daily_eq_ret = (daily_equity_w * ret).sum(axis=1)
    daily_short_vol_contrib = np.nan_to_num(
        per_name_daily_vol.sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    daily_long_vol_contrib = daily_long_vol_pos * vixy_daily_ret
    daily_total = daily_eq_ret + daily_short_vol_contrib + daily_long_vol_contrib

    sel = np.where(mask)[0]
    out_dates = close_index[sel]
    out_ret = daily_total[sel]
    out_svs = pd.Series(0.0, index=out_dates)
    out_lvp = pd.Series(0.0, index=out_dates)
    anchor_svs = {idx_map.get(d): SVS[k]
                  for k, d in enumerate(val_panel.dates)
                  if idx_map.get(d) is not None}
    anchor_lvp = {idx_map.get(d): LVP[k]
                  for k, d in enumerate(val_panel.dates)
                  if idx_map.get(d) is not None}
    last_svs, last_lvp = 0.0, 0.0
    for i, tt in enumerate(sel):
        if tt in anchor_svs:
            last_svs = float(anchor_svs[tt])
        if tt in anchor_lvp:
            last_lvp = float(anchor_lvp[tt])
        out_svs.iloc[i] = last_svs
        out_lvp.iloc[i] = last_lvp

    return (pd.Series(out_ret, index=out_dates, dtype=np.float64),
            out_svs, out_lvp, EW, VW)


def run_fold(full_panel: PanelV3p5, close: pd.DataFrame,
             vixy_close: pd.Series,
             fold_cfg: dict, cfg: TrainConfigV3p5, hp: HparamsV3p5,
             save_prefix: str = 'e2e-portfolio-v3p5') -> dict:
    name = fold_cfg['name']
    print(f'\n=== {name}: val {fold_cfg["val_start"]} -> {fold_cfg["val_end"]} ===',
          flush=True)
    train_panel, val_panel = build_fold_panels(
        full_panel, fold_cfg['val_start'], fold_cfg['val_end'])
    print(f'  train n={len(train_panel.dates)}  val n={len(val_panel.dates)}',
          flush=True)
    if len(train_panel.dates) < cfg.batch_size:
        print(f'  SKIP: train too small', flush=True)
        return {'name': name, 'skipped': True}

    n_train = len(train_panel.dates)
    inner_val_n = max(64, n_train // 10)
    inner_train = PanelV3p5(
        X_assets=train_panel.X_assets[:-inner_val_n],
        X_macro=train_panel.X_macro[:-inner_val_n],
        valid_mask=train_panel.valid_mask[:-inner_val_n],
        fwd_ret=train_panel.fwd_ret[:-inner_val_n],
        fwd_vol_pnl=train_panel.fwd_vol_pnl[:-inner_val_n],
        fwd_long_vol_ret=train_panel.fwd_long_vol_ret[:-inner_val_n],
        dates=train_panel.dates[:-inner_val_n],
        tickers=train_panel.tickers,
    )
    inner_val = PanelV3p5(
        X_assets=train_panel.X_assets[-inner_val_n:],
        X_macro=train_panel.X_macro[-inner_val_n:],
        valid_mask=train_panel.valid_mask[-inner_val_n:],
        fwd_ret=train_panel.fwd_ret[-inner_val_n:],
        fwd_vol_pnl=train_panel.fwd_vol_pnl[-inner_val_n:],
        fwd_long_vol_ret=train_panel.fwd_long_vol_ret[-inner_val_n:],
        dates=train_panel.dates[-inner_val_n:],
        tickers=train_panel.tickers,
    )

    model, history = train_one_fold_v3p5(inner_train, inner_val, cfg, hp)

    ckpt_path = OUT_DIR / f'{save_prefix}-{name}.npz'
    save_npz(model, str(ckpt_path))
    print(f'  saved {ckpt_path}', flush=True)

    daily, daily_svs, daily_lvp, EW, VW = daily_returns_from_panel(
        model, val_panel, close, vixy_close)
    sh_ann = (daily.mean() / max(daily.std(), 1e-9) * np.sqrt(252)) if len(daily) else 0.0

    if VW.size > 0:
        vw_sorted = np.sort(VW, axis=1)[:, ::-1]
        top_k_mass = float(vw_sorted[:, :50].sum(axis=1).mean())
        top10_idx = np.argsort(VW.mean(axis=0))[::-1][:10]
        top10_names = [val_panel.tickers[i] for i in top10_idx]
        top10_mass = [float(VW.mean(axis=0)[i]) for i in top10_idx]
    else:
        top_k_mass, top10_names, top10_mass = 0.0, [], []

    # Mechanism check on 2020-Q1: long_vol_position mean during
    # Feb 2020 -> Apr 2020. Only meaningful on fold-2.
    q1_2020_mask = (daily_lvp.index >= '2020-02-01') & (daily_lvp.index <= '2020-04-30')
    q1_2020_lvp_mean = (float(daily_lvp[q1_2020_mask].mean())
                        if q1_2020_mask.any() else float('nan'))

    print(f'  val daily: n={len(daily)}  sh_ann={sh_ann:+.3f}  '
          f'svs mean={daily_svs.mean():.3f}  lvp mean={daily_lvp.mean():.3f}',
          flush=True)
    print(f'  top10 vol names: {list(zip(top10_names, [f"{m:.4f}" for m in top10_mass]))}',
          flush=True)
    if not np.isnan(q1_2020_lvp_mean):
        print(f'  >>> 2020-Q1 long_vol_position mean: {q1_2020_lvp_mean:.3f} '
              f'(mechanism check: >= 0.5 for confirmed-OOS, >= 0.3 for partial, '
              f'< 0.2 forces downgrade)', flush=True)

    out_npz = OUT_DIR / f'{save_prefix}-{name}-daily.npz'
    np.savez(out_npz,
             dates=daily.index.astype('datetime64[ns]').astype(np.int64),
             daily_ret=daily.values,
             short_vol_scale=daily_svs.values,
             long_vol_position=daily_lvp.values)

    return {
        'name': name,
        'val_start': fold_cfg['val_start'],
        'val_end': fold_cfg['val_end'],
        'n_val_days': int(len(daily)),
        'val_sharpe_ann': float(sh_ann),
        'val_mean_ret': float(daily.mean()) if len(daily) else 0.0,
        'val_std_ret': float(daily.std()) if len(daily) else 0.0,
        'short_vol_scale_mean': float(daily_svs.mean()) if len(daily_svs) else 0.0,
        'short_vol_scale_std': float(daily_svs.std()) if len(daily_svs) else 0.0,
        'long_vol_position_mean': float(daily_lvp.mean()) if len(daily_lvp) else 0.0,
        'long_vol_position_std': float(daily_lvp.std()) if len(daily_lvp) else 0.0,
        'long_vol_position_min': float(daily_lvp.min()) if len(daily_lvp) else 0.0,
        'long_vol_position_max': float(daily_lvp.max()) if len(daily_lvp) else 0.0,
        'long_vol_position_2020q1_mean': q1_2020_lvp_mean,
        'vol_top50_mass': float(top_k_mass),
        'vol_top10_names': top10_names,
        'vol_top10_mass': top10_mass,
        'daily_path': str(out_npz),
        'ckpt_path': str(ckpt_path),
    }


def _dca_daily_phase4d() -> pd.Series:
    import sys
    sys.path.insert(0, str(REPO / 'apps/cfr/src'))
    from cfr.baselines import PassiveEW
    REPO_PKL = REPO / 'Output' / 'cfr_phase4d_multiasset_close.pkl'
    import pickle
    with open(REPO_PKL, 'rb') as f:
        phase4d = pickle.load(f)
    dca = PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(phase4d)
    return pd.Series(np.asarray(dca, dtype=np.float64), index=phase4d.index)


def _vol_v3_daily() -> pd.Series:
    d = np.load(OUT_DIR / 'vol-v3-dolthub-oos-c200-returns.npz', allow_pickle=True)
    vol_dates = pd.to_datetime(np.asarray(d['rebal_dates'], dtype=str))
    vol_alpha = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    daily_idx = pd.date_range(vol_dates[0], vol_dates[-1] + pd.Timedelta(days=60), freq='B')
    vol_daily = pd.Series(0.0, index=daily_idx)
    for i in range(len(vol_dates) - 1):
        mask = (vol_daily.index >= vol_dates[i]) & (vol_daily.index < vol_dates[i + 1])
        n = int(mask.sum())
        if n:
            vol_daily.loc[mask] = vol_alpha[i] / n
    mask = vol_daily.index >= vol_dates[-1]
    n = int(mask.sum())
    if n:
        vol_daily.loc[mask] = vol_alpha[-1] / n
    return vol_daily


def _v3_pooled_daily() -> pd.Series:
    """v3 pooled OOS daily stream. Returns empty Series if v3 not run."""
    path = OUT_DIR / 'e2e-portfolio-v3-pooled-daily.npz'
    if not path.exists():
        return pd.Series(dtype=np.float64)
    d = np.load(path)
    dates = pd.DatetimeIndex(d['dates'].astype('datetime64[ns]'))
    return pd.Series(d['daily_ret'], index=dates).sort_index()


def baseline_streams() -> dict[str, pd.Series]:
    dca = _dca_daily_phase4d()
    vol_daily = _vol_v3_daily()
    vol_aligned = vol_daily.reindex(dca.index).fillna(0.0)
    v3_pooled = _v3_pooled_daily()
    return {
        'dca': dca,
        'vol_v3': vol_aligned,
        'deterministic_2leg': dca + 2.0 * vol_aligned,
        'learned_2leg': 0.0506 * dca + 2.2388 * vol_aligned,
        'v3': v3_pooled,
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


def pool_and_report(per_fold: list[dict],
                    save_prefix: str = 'e2e-portfolio-v3p5') -> dict:
    pooled_ret, pooled_dates = [], []
    pooled_svs, pooled_lvp = [], []
    for fold in per_fold:
        if fold.get('skipped'):
            continue
        d = np.load(fold['daily_path'])
        dates = pd.DatetimeIndex(d['dates'].astype('datetime64[ns]'))
        pooled_ret.append(d['daily_ret'])
        pooled_dates.append(dates)
        if 'short_vol_scale' in d.files:
            pooled_svs.append(d['short_vol_scale'])
        if 'long_vol_position' in d.files:
            pooled_lvp.append(d['long_vol_position'])
    if not pooled_ret:
        return {'pooled_n': 0}
    pooled_ret = np.concatenate(pooled_ret)
    pooled_idx = pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in pooled_dates]))
    pooled = pd.Series(pooled_ret, index=pooled_idx).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep='first')]
    svs_arr = np.concatenate(pooled_svs) if pooled_svs else np.array([])
    lvp_arr = np.concatenate(pooled_lvp) if pooled_lvp else np.array([])

    np.savez(OUT_DIR / f'{save_prefix}-pooled-daily.npz',
             dates=pooled.index.astype('datetime64[ns]').astype(np.int64),
             daily_ret=pooled.values)

    baselines = baseline_streams()
    comps = {bname: summarize_vs_baseline(pooled, b, bname)
             for bname, b in baselines.items() if len(b) > 0}
    ann = float(np.sqrt(252))
    return {
        'pooled_n': int(len(pooled)),
        'pooled_sharpe_ann': float(pooled.mean() / max(pooled.std(), 1e-9) * ann),
        'pooled_mean_ret': float(pooled.mean()),
        'pooled_std_ret': float(pooled.std()),
        'pooled_max_dd': float(((1 + pooled).cumprod() /
                                (1 + pooled).cumprod().cummax() - 1).min()),
        'pooled_short_vol_scale_mean': float(svs_arr.mean()) if len(svs_arr) else 0.0,
        'pooled_short_vol_scale_std': float(svs_arr.std()) if len(svs_arr) else 0.0,
        'pooled_long_vol_position_mean': float(lvp_arr.mean()) if len(lvp_arr) else 0.0,
        'pooled_long_vol_position_std': float(lvp_arr.std()) if len(lvp_arr) else 0.0,
        'per_fold': per_fold,
        'baseline_comparisons': comps,
    }
