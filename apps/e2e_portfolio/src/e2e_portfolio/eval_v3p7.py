"""v3.7 walk-forward — copy of v3.6 but with the imitation-prior + floor
model. Reuses v3.6's continuous-walk pattern (ZZR cadence).

Differences from v3.6:
  - Imports AllocatorV3p7 + sharpe_loss_v3p7 + save_npz from model_v3p7
  - `_make_jit_step` and the model factory know about the (model, opt, tickers)
    triple — v3.7's model needs `tickers` to build the per-name equity bias
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from tinygrad import Tensor, TinyJit, nn

from e2e_portfolio.data_v3p5 import PanelV3p5
from e2e_portfolio.model_v3p7 import (
    AllocatorV3p7, HparamsV3p7, sharpe_loss_v3p7, save_npz,
)
from e2e_portfolio.train_v3p5 import (
    TrainConfigV3p5, _attach_cash_zero,
)
from e2e_portfolio.eval_v3p5 import (
    baseline_streams, summarize_vs_baseline, K_FWD,
)
from e2e_portfolio.eval_v3p6 import (
    FOLDS, INITIAL_TRAIN_YEARS, RETRAIN_EVERY, REFINE_STEPS, _slice_panel,
)

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'Output'


def _make_jit_step_v3p7(
    model: AllocatorV3p7,
    opt: nn.optim.AdamW,
    B: int, K: int, T: int, F: int, Fm: int,
) -> tuple[callable, list[Tensor]]:
    xa_buf = Tensor.zeros(B, K, T, F).contiguous().realize()
    xm_buf = Tensor.zeros(B, T, Fm).contiguous().realize()
    vmask_buf = Tensor.zeros(B, K).contiguous().realize()
    fwd_buf = Tensor.zeros(B, K + 1).contiguous().realize()
    vol_pnl_buf = Tensor.zeros(B, K).contiguous().realize()
    long_vol_buf = Tensor.zeros(B).contiguous().realize()

    @TinyJit
    def _step() -> Tensor:
        opt.zero_grad()
        ew, vw, svs, lvp = model(xa_buf, xm_buf, vmask_buf)
        loss = sharpe_loss_v3p7(ew, fwd_buf, vw, svs, vol_pnl_buf,
                                 lvp, long_vol_buf)
        loss.backward()
        opt.step()
        return loss.realize()

    return _step, [xa_buf, xm_buf, vmask_buf, fwd_buf, vol_pnl_buf, long_vol_buf]


def _train_on_window(panel_train, n_steps, cfg, step_fn, buffers, rng,
                     label=''):
    xa_buf, xm_buf, vmask_buf, fwd_buf, vol_pnl_buf, long_vol_buf = buffers
    n_train = len(panel_train.dates)
    if n_train < cfg.batch_size:
        return float('nan')
    Tensor.training = True
    last_loss = float('nan')
    t0 = time.time()
    for step in range(n_steps):
        idx = rng.integers(0, n_train, size=cfg.batch_size)
        xa_buf.assign(Tensor(panel_train.X_assets[idx])).realize()
        xm_buf.assign(Tensor(panel_train.X_macro[idx])).realize()
        vmask_buf.assign(Tensor(panel_train.valid_mask[idx])).realize()
        fwd_buf.assign(Tensor(_attach_cash_zero(panel_train.fwd_ret[idx]))).realize()
        vol_pnl_buf.assign(Tensor(
            panel_train.fwd_vol_pnl[idx].astype(np.float32))).realize()
        long_vol_buf.assign(Tensor(
            panel_train.fwd_long_vol_ret[idx].astype(np.float32))).realize()
        loss = step_fn()
        last_loss = float(loss.numpy())
    Tensor.training = False
    if label:
        print(f'  [{label}] {n_steps} steps in {time.time()-t0:.0f}s  '
              f'final_loss={last_loss:+.4f}', flush=True)
    return last_loss


def daily_returns_v3p7(model: AllocatorV3p7, val_panel: PanelV3p5,
                        close: pd.DataFrame, vixy_close: pd.Series,
                        ) -> tuple[pd.Series, pd.Series, pd.Series,
                                   np.ndarray, np.ndarray]:
    """Same accounting logic as v3p5's daily_returns_from_panel but the
    4-tuple from model() uses v3p7 outputs (short_vol_scale, long_vol_position
    with floor)."""
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

    tickers = val_panel.tickers
    close_sub = close[tickers]
    close_arr = close_sub.values
    close_index = close_sub.index
    idx_map = {d: i for i, d in enumerate(close_index)}

    vixy_aligned = vixy_close.reindex(close_index).ffill()
    vixy_arr = vixy_aligned.values
    vixy_log_ret = np.diff(np.log(np.maximum(vixy_arr, 1e-9)),
                           prepend=np.log(np.maximum(vixy_arr[:1], 1e-9)))
    vixy_daily_ret = np.nan_to_num(np.exp(vixy_log_ret) - 1.0,
                                    nan=0.0, posinf=0.0, neginf=0.0)

    n_close = close_arr.shape[0]
    per_name_daily_vol = np.zeros((n_close, K), dtype=np.float64)
    for k, anchor in enumerate(val_panel.dates):
        t = idx_map.get(anchor)
        if t is None:
            continue
        next_t = (idx_map.get(val_panel.dates[k + 1])
                  if k + 1 < len(val_panel.dates) else None)
        t_end = (min(t + K_FWD, next_t) if next_t is not None
                 else min(t + K_FWD, n_close - 1))
        per_day = val_panel.fwd_vol_pnl[k] / float(K_FWD)
        per_day = np.nan_to_num(per_day, nan=0.0, posinf=0.0, neginf=0.0)
        contribution = SVS[k] * (VW[k] * per_day)
        for tt in range(t + 1, t_end + 1):
            if tt < n_close:
                per_name_daily_vol[tt] += contribution

    daily_equity_w = np.zeros((n_close, K), dtype=np.float64)
    daily_long_vol_pos = np.zeros(n_close, dtype=np.float64)
    for k, anchor in enumerate(val_panel.dates):
        t = idx_map.get(anchor)
        if t is None:
            continue
        next_t = (idx_map.get(val_panel.dates[k + 1])
                  if k + 1 < len(val_panel.dates) else None)
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
    ret = np.nan_to_num(np.exp(log_ret) - 1.0,
                        nan=0.0, posinf=0.0, neginf=0.0)

    daily_eq_ret = (daily_equity_w * ret).sum(axis=1)
    daily_short_vol_contrib = np.nan_to_num(
        per_name_daily_vol.sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    daily_long_vol_contrib = daily_long_vol_pos * vixy_daily_ret
    daily_total = daily_eq_ret + daily_short_vol_contrib + daily_long_vol_contrib

    sel = np.where(mask)[0]
    out_dates = close_index[sel]
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

    return (pd.Series(daily_total[sel], index=out_dates, dtype=np.float64),
            out_svs, out_lvp, EW, VW)


def run_walkforward_continuous_v3p7(
    full_panel: PanelV3p5,
    close: pd.DataFrame,
    vixy_close: pd.Series,
    folds_cfg: list[dict],
    cfg: TrainConfigV3p5,
    hp: HparamsV3p7,
    initial_steps: int = 5000,
    refine_steps: int = REFINE_STEPS,
    retrain_every: int = RETRAIN_EVERY,
    train_window_years: int = INITIAL_TRAIN_YEARS,
    save_prefix: str = 'e2e-portfolio-v3p7',
) -> list[dict]:
    val_starts = [pd.Timestamp(f['val_start']) for f in folds_cfg]
    val_ends = [pd.Timestamp(f['val_end']) for f in folds_cfg]
    earliest_val_start = min(val_starts)
    latest_val_end = max(val_ends)
    print(f'\n=== v3.7 continuous walk-forward: '
          f'val {earliest_val_start.date()} -> {latest_val_end.date()} ===',
          flush=True)

    model = AllocatorV3p7(hp, tickers=full_panel.tickers, seed=cfg.seed)
    opt = nn.optim.AdamW(model.parameters(), lr=cfg.lr,
                         weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    B = cfg.batch_size
    K = hp.n_assets
    T = hp.t_lookback
    F = hp.f_asset
    Fm = hp.f_macro
    step_fn, buffers = _make_jit_step_v3p7(model, opt, B, K, T, F, Fm)

    init_train_end = earliest_val_start - pd.Timedelta(days=1)
    init_train_start = init_train_end - pd.Timedelta(days=365 * train_window_years)
    init_train = _slice_panel(full_panel, init_train_start, init_train_end)
    print(f'  initial train: {init_train_start.date()} -> {init_train_end.date()} '
          f'(n={len(init_train.dates)})', flush=True)
    _train_on_window(init_train, initial_steps, cfg, step_fn, buffers, rng,
                     label='init')

    val_dates_full = full_panel.dates[
        (full_panel.dates >= earliest_val_start)
        & (full_panel.dates <= latest_val_end)
    ]

    def _fold_of_date(d):
        for f, vs, ve in zip(folds_cfg, val_starts, val_ends):
            if vs <= d <= ve:
                return f['name']
        return None

    sub_results = {f['name']: [] for f in folds_cfg}
    pos = 0
    while pos < len(val_dates_full):
        sub_start = val_dates_full[pos]
        end_idx = min(pos + retrain_every - 1, len(val_dates_full) - 1)
        sub_end = val_dates_full[end_idx]
        fold_name = _fold_of_date(sub_start)
        sub_val = _slice_panel(full_panel, sub_start, sub_end)
        if len(sub_val.dates) > 0 and fold_name is not None:
            daily, svs, lvp, EW, VW = daily_returns_v3p7(
                model, sub_val, close, vixy_close)
            if len(daily) > 0:
                sub_results[fold_name].append({
                    'start': str(sub_start.date()),
                    'end': str(sub_end.date()),
                    'n': int(len(daily)),
                    'daily': daily, 'svs': svs, 'lvp': lvp,
                })
        pos = end_idx + 1
        if pos >= len(val_dates_full):
            break
        next_train_end = val_dates_full[pos] - pd.Timedelta(days=1)
        next_train_start = next_train_end - pd.Timedelta(
            days=365 * train_window_years)
        next_train = _slice_panel(full_panel, next_train_start, next_train_end)
        _train_on_window(next_train, refine_steps, cfg, step_fn, buffers, rng,
                         label=f'refine@{sub_end.date()}')

    per_fold = []
    for fcfg in folds_cfg:
        name = fcfg['name']
        subs = sub_results[name]
        if not subs:
            per_fold.append({'name': name, 'skipped': True})
            continue
        daily_concat = pd.concat([s['daily'] for s in subs]).sort_index()
        svs_concat = pd.concat([s['svs'] for s in subs]).sort_index()
        lvp_concat = pd.concat([s['lvp'] for s in subs]).sort_index()
        daily_concat = daily_concat[~daily_concat.index.duplicated()]
        svs_concat = svs_concat[~svs_concat.index.duplicated()]
        lvp_concat = lvp_concat[~lvp_concat.index.duplicated()]
        sh_ann = (float(daily_concat.mean() / max(daily_concat.std(), 1e-9)
                        * np.sqrt(252)) if len(daily_concat) else 0.0)
        q1_2020_mask = ((lvp_concat.index >= '2020-02-01')
                        & (lvp_concat.index <= '2020-04-30'))
        q1_2020_lvp_mean = (float(lvp_concat[q1_2020_mask].mean())
                            if q1_2020_mask.any() else float('nan'))
        out_npz = OUT_DIR / f'{save_prefix}-{name}-daily.npz'
        np.savez(out_npz,
                 dates=daily_concat.index.astype('datetime64[ns]').astype(np.int64),
                 daily_ret=daily_concat.values,
                 short_vol_scale=svs_concat.values,
                 long_vol_position=lvp_concat.values)
        print(f'  {name}: {len(subs)} subs n={len(daily_concat)} '
              f'sh_ann={sh_ann:+.3f} svs_mean={svs_concat.mean():.3f} '
              f'lvp_mean={lvp_concat.mean():.3f}', flush=True)
        if not np.isnan(q1_2020_lvp_mean):
            print(f'    >>> 2020-Q1 long_vol_position mean: {q1_2020_lvp_mean:.3f}',
                  flush=True)
        per_fold.append({
            'name': name,
            'val_start': fcfg['val_start'],
            'val_end': fcfg['val_end'],
            'n_val_days': int(len(daily_concat)),
            'n_subperiods': len(subs),
            'val_sharpe_ann': float(sh_ann),
            'val_mean_ret': float(daily_concat.mean()) if len(daily_concat) else 0.0,
            'val_std_ret': float(daily_concat.std()) if len(daily_concat) else 0.0,
            'short_vol_scale_mean': float(svs_concat.mean()) if len(svs_concat) else 0.0,
            'long_vol_position_mean': float(lvp_concat.mean()) if len(lvp_concat) else 0.0,
            'long_vol_position_2020q1_mean': q1_2020_lvp_mean,
            'daily_path': str(out_npz),
        })

    final_ckpt = OUT_DIR / f'{save_prefix}-final.npz'
    save_npz(model, str(final_ckpt))
    print(f'  saved final ckpt: {final_ckpt}', flush=True)
    for r in per_fold:
        if not r.get('skipped'):
            r['ckpt_path'] = str(final_ckpt)
    return per_fold


def pool_and_report(per_fold: list[dict],
                    save_prefix: str = 'e2e-portfolio-v3p7') -> dict:
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
    pooled = pooled[~pooled.index.duplicated()]
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
        'pooled_long_vol_position_mean': float(lvp_arr.mean()) if len(lvp_arr) else 0.0,
        'per_fold': per_fold,
        'baseline_comparisons': comps,
    }
