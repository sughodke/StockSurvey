"""v3.6 rolling-retrain walk-forward — same architecture as v3.5.

Where v3.5 trains once per fold and freezes for the entire val period
(5-year train → 4-year val), v3.6 mirrors the Zhang-Zohren-Roberts
2020 rolling cadence: train once on the initial 5y window, then
fine-tune every `retrain_every` trading days using the new data that
just rolled into the trailing-5y window.

Key differences from v3.5:
  - Initial train: 5000 steps on the first 5y window (same as v3.5).
  - Then walk forward N=63 trading days (~quarterly) at a time:
      - Eval for the next 63d period with current weights
      - Slide training window forward by 63d (drop oldest, add newest)
      - Fine-tune for `refine_steps` (default 500) steps from current
        weights — NOT a full retrain
  - Stitch all sub-period daily streams into the pooled fold result.

Rationale: at fixed-fold training, fold-2's 2014-2018 window had no
major vol spike, so the optimizer rationally pushed `long_vol_position`
to zero and couldn't recover at COVID OOS. Rolling cadence gives the
model a chance to see late-2019 micro-vol-bumps and adjust the
long-vol bias closer to a reasonable prior before COVID hits.
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from tinygrad import Tensor, TinyJit, nn

from e2e_portfolio.data_v3p5 import PanelV3p5
from e2e_portfolio.model_v3p5 import (
    AllocatorV3p5, HparamsV3p5, sharpe_loss_v3p5, save_npz,
)
from e2e_portfolio.train_v3p5 import (
    TrainConfigV3p5, _attach_cash_zero, evaluate_sharpe_v3p5,
)
from e2e_portfolio.eval_v3p5 import (
    daily_returns_from_panel, baseline_streams, summarize_vs_baseline,
    K_FWD,
)

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'Output'

FOLDS = [
    {'name': 'fold1', 'val_start': '2015-01-01', 'val_end': '2018-12-31'},
    {'name': 'fold2', 'val_start': '2019-01-01', 'val_end': '2022-12-31'},
    {'name': 'fold3', 'val_start': '2023-01-01', 'val_end': '2025-12-31'},
]
INITIAL_TRAIN_YEARS = 5
RETRAIN_EVERY = 63       # 1 quarter trading days
REFINE_STEPS = 500       # fine-tune steps per rolling step (vs 5000 initial)


def _slice_panel(panel: PanelV3p5, start, end) -> PanelV3p5:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    mask = (panel.dates >= s) & (panel.dates <= e)
    idx = np.where(mask)[0]
    return PanelV3p5(
        X_assets=panel.X_assets[idx],
        X_macro=panel.X_macro[idx],
        valid_mask=panel.valid_mask[idx],
        fwd_ret=panel.fwd_ret[idx],
        fwd_vol_pnl=panel.fwd_vol_pnl[idx],
        fwd_long_vol_ret=panel.fwd_long_vol_ret[idx],
        dates=panel.dates[idx],
        tickers=panel.tickers,
    )


def _make_jit_step(
    model: AllocatorV3p5,
    opt: nn.optim.AdamW,
    B: int, K: int, T: int, F: int, Fm: int,
) -> tuple[callable, list[Tensor]]:
    """Build the JIT'd step closure and persistent buffer tensors.
    Buffers are reused across all retrains within a single fold so
    the JIT records once and replays for every fine-tune step."""
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
        loss = sharpe_loss_v3p5(ew, fwd_buf, vw, svs, vol_pnl_buf,
                                 lvp, long_vol_buf)
        loss.backward()
        opt.step()
        return loss.realize()

    return _step, [xa_buf, xm_buf, vmask_buf, fwd_buf, vol_pnl_buf, long_vol_buf]


def _train_on_window(
    panel_train: PanelV3p5,
    n_steps: int,
    cfg: TrainConfigV3p5,
    step_fn: callable,
    buffers: list[Tensor],
    rng: np.random.Generator,
    label: str = '',
) -> float:
    """Run `n_steps` training updates on `panel_train` using the
    pre-built JIT step closure + buffer tensors. Returns final loss."""
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


def run_fold_rolling(
    full_panel: PanelV3p5,
    close: pd.DataFrame,
    vixy_close: pd.Series,
    fold_cfg: dict,
    cfg: TrainConfigV3p5,
    hp: HparamsV3p5,
    initial_steps: int = 5000,
    refine_steps: int = REFINE_STEPS,
    retrain_every: int = RETRAIN_EVERY,
    train_window_years: int = INITIAL_TRAIN_YEARS,
    save_prefix: str = 'e2e-portfolio-v3p6',
) -> dict:
    """Rolling-retrain walk-forward for one fold."""
    name = fold_cfg['name']
    val_start = pd.Timestamp(fold_cfg['val_start'])
    val_end = pd.Timestamp(fold_cfg['val_end'])
    print(f'\n=== {name}: rolling-retrain val {val_start.date()} -> {val_end.date()} '
          f'(retrain_every={retrain_every}d, refine_steps={refine_steps}) ===',
          flush=True)

    model = AllocatorV3p5(hp, seed=cfg.seed)
    opt = nn.optim.AdamW(model.parameters(), lr=cfg.lr,
                         weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    B, K, T, F, Fm = (cfg.batch_size, hp.n_assets, hp.t_lookback,
                      hp.f_asset, hp.f_macro)
    step_fn, buffers = _make_jit_step(model, opt, B, K, T, F, Fm)

    # Initial full training on the first 5y window ending just before val_start.
    init_train_end = val_start - pd.Timedelta(days=1)
    init_train_start = init_train_end - pd.Timedelta(days=365 * train_window_years)
    init_train = _slice_panel(full_panel, init_train_start, init_train_end)
    print(f'  initial train: {init_train_start.date()} -> {init_train_end.date()} '
          f'(n={len(init_train.dates)})', flush=True)
    _train_on_window(init_train, initial_steps, cfg, step_fn, buffers, rng,
                     label='init')

    # Walk val_start -> val_end in chunks of retrain_every days.
    val_dates_full = full_panel.dates[
        (full_panel.dates >= val_start) & (full_panel.dates <= val_end)
    ]
    if len(val_dates_full) == 0:
        return {'name': name, 'skipped': True,
                'reason': 'no val dates in panel'}

    sub_period_results = []
    pos = 0
    while pos < len(val_dates_full):
        sub_start = val_dates_full[pos]
        end_idx = min(pos + retrain_every - 1, len(val_dates_full) - 1)
        sub_end = val_dates_full[end_idx]

        # Eval for this sub-period with current weights.
        sub_val = _slice_panel(full_panel, sub_start, sub_end)
        if len(sub_val.dates) > 0:
            daily, daily_svs, daily_lvp, EW, VW = daily_returns_from_panel(
                model, sub_val, close, vixy_close)
            if len(daily) > 0:
                sub_period_results.append({
                    'start': str(sub_start.date()),
                    'end': str(sub_end.date()),
                    'n': int(len(daily)),
                    'daily': daily,
                    'svs': daily_svs,
                    'lvp': daily_lvp,
                })

        pos = end_idx + 1
        if pos >= len(val_dates_full):
            break

        # Slide training window forward to end just before next sub-period.
        next_train_end = val_dates_full[pos] - pd.Timedelta(days=1)
        next_train_start = next_train_end - pd.Timedelta(days=365 * train_window_years)
        next_train = _slice_panel(full_panel, next_train_start, next_train_end)
        _train_on_window(next_train, refine_steps, cfg, step_fn, buffers, rng,
                         label=f'refine@{sub_end.date()}')

    # Stitch sub-periods together.
    if not sub_period_results:
        return {'name': name, 'skipped': True, 'reason': 'no sub-periods'}
    daily_concat = pd.concat([r['daily'] for r in sub_period_results]).sort_index()
    svs_concat = pd.concat([r['svs'] for r in sub_period_results]).sort_index()
    lvp_concat = pd.concat([r['lvp'] for r in sub_period_results]).sort_index()
    daily_concat = daily_concat[~daily_concat.index.duplicated(keep='first')]
    svs_concat = svs_concat[~svs_concat.index.duplicated(keep='first')]
    lvp_concat = lvp_concat[~lvp_concat.index.duplicated(keep='first')]

    sh_ann = (float(daily_concat.mean() / max(daily_concat.std(), 1e-9)
                    * np.sqrt(252)) if len(daily_concat) else 0.0)

    # Mechanism check: 2020-Q1 long_vol_position mean.
    q1_2020_mask = ((lvp_concat.index >= '2020-02-01')
                    & (lvp_concat.index <= '2020-04-30'))
    q1_2020_lvp_mean = (float(lvp_concat[q1_2020_mask].mean())
                        if q1_2020_mask.any() else float('nan'))

    # Save final-state checkpoint for this fold.
    ckpt_path = OUT_DIR / f'{save_prefix}-{name}.npz'
    save_npz(model, str(ckpt_path))

    out_npz = OUT_DIR / f'{save_prefix}-{name}-daily.npz'
    np.savez(out_npz,
             dates=daily_concat.index.astype('datetime64[ns]').astype(np.int64),
             daily_ret=daily_concat.values,
             short_vol_scale=svs_concat.values,
             long_vol_position=lvp_concat.values)

    print(f'  {len(sub_period_results)} sub-periods; n={len(daily_concat)} '
          f'sh_ann={sh_ann:+.3f}  svs mean={svs_concat.mean():.3f}  '
          f'lvp mean={lvp_concat.mean():.3f}', flush=True)
    if not np.isnan(q1_2020_lvp_mean):
        print(f'  >>> 2020-Q1 long_vol_position mean: {q1_2020_lvp_mean:.3f}',
              flush=True)

    return {
        'name': name,
        'val_start': fold_cfg['val_start'],
        'val_end': fold_cfg['val_end'],
        'n_val_days': int(len(daily_concat)),
        'n_subperiods': len(sub_period_results),
        'val_sharpe_ann': float(sh_ann),
        'val_mean_ret': float(daily_concat.mean()) if len(daily_concat) else 0.0,
        'val_std_ret': float(daily_concat.std()) if len(daily_concat) else 0.0,
        'short_vol_scale_mean': float(svs_concat.mean()) if len(svs_concat) else 0.0,
        'short_vol_scale_std': float(svs_concat.std()) if len(svs_concat) else 0.0,
        'long_vol_position_mean': float(lvp_concat.mean()) if len(lvp_concat) else 0.0,
        'long_vol_position_std': float(lvp_concat.std()) if len(lvp_concat) else 0.0,
        'long_vol_position_min': float(lvp_concat.min()) if len(lvp_concat) else 0.0,
        'long_vol_position_max': float(lvp_concat.max()) if len(lvp_concat) else 0.0,
        'long_vol_position_2020q1_mean': q1_2020_lvp_mean,
        'daily_path': str(out_npz),
        'ckpt_path': str(ckpt_path),
    }


def run_walkforward_continuous(
    full_panel: PanelV3p5,
    close: pd.DataFrame,
    vixy_close: pd.Series,
    folds_cfg: list[dict],
    cfg: TrainConfigV3p5,
    hp: HparamsV3p5,
    initial_steps: int = 5000,
    refine_steps: int = REFINE_STEPS,
    retrain_every: int = RETRAIN_EVERY,
    train_window_years: int = INITIAL_TRAIN_YEARS,
    save_prefix: str = 'e2e-portfolio-v3p6-cont',
) -> list[dict]:
    """ZZR-style continuous walk-forward: ONE model walks through every
    fold without resetting. Initial train ends before the earliest
    val_start; thereafter the model is fine-tuned every `retrain_every`
    days as the rolling window slides forward through the full val span.

    Returns per-fold results in the same shape as `run_fold_rolling` so
    that `pool_and_report` works unchanged. Each sub-period is attributed
    to its fold by start date.
    """
    if not folds_cfg:
        return []
    val_starts = [pd.Timestamp(f['val_start']) for f in folds_cfg]
    val_ends = [pd.Timestamp(f['val_end']) for f in folds_cfg]
    earliest_val_start = min(val_starts)
    latest_val_end = max(val_ends)
    print(f'\n=== continuous walk-forward: '
          f'val {earliest_val_start.date()} -> {latest_val_end.date()} '
          f'(retrain_every={retrain_every}d, refine_steps={refine_steps}, '
          f'init_steps={initial_steps}) ===', flush=True)

    model = AllocatorV3p5(hp, seed=cfg.seed)
    opt = nn.optim.AdamW(model.parameters(), lr=cfg.lr,
                         weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    B, K, T, F, Fm = (cfg.batch_size, hp.n_assets, hp.t_lookback,
                      hp.f_asset, hp.f_macro)
    step_fn, buffers = _make_jit_step(model, opt, B, K, T, F, Fm)

    # Single initial training window ending just before the earliest val_start.
    init_train_end = earliest_val_start - pd.Timedelta(days=1)
    init_train_start = init_train_end - pd.Timedelta(days=365 * train_window_years)
    init_train = _slice_panel(full_panel, init_train_start, init_train_end)
    print(f'  initial train: {init_train_start.date()} -> {init_train_end.date()} '
          f'(n={len(init_train.dates)})', flush=True)
    _train_on_window(init_train, initial_steps, cfg, step_fn, buffers, rng,
                     label='init-continuous')

    # Single continuous walk through every val date across all folds.
    val_dates_full = full_panel.dates[
        (full_panel.dates >= earliest_val_start)
        & (full_panel.dates <= latest_val_end)
    ]
    if len(val_dates_full) == 0:
        return [{'name': f['name'], 'skipped': True} for f in folds_cfg]

    def _fold_of_date(d: pd.Timestamp) -> str | None:
        for f, vs, ve in zip(folds_cfg, val_starts, val_ends):
            if vs <= d <= ve:
                return f['name']
        return None

    sub_period_results: dict[str, list[dict]] = {f['name']: [] for f in folds_cfg}

    pos = 0
    while pos < len(val_dates_full):
        sub_start = val_dates_full[pos]
        end_idx = min(pos + retrain_every - 1, len(val_dates_full) - 1)
        sub_end = val_dates_full[end_idx]
        fold_name = _fold_of_date(sub_start)

        sub_val = _slice_panel(full_panel, sub_start, sub_end)
        if len(sub_val.dates) > 0 and fold_name is not None:
            daily, daily_svs, daily_lvp, EW, VW = daily_returns_from_panel(
                model, sub_val, close, vixy_close)
            if len(daily) > 0:
                sub_period_results[fold_name].append({
                    'start': str(sub_start.date()),
                    'end': str(sub_end.date()),
                    'n': int(len(daily)),
                    'daily': daily,
                    'svs': daily_svs,
                    'lvp': daily_lvp,
                })

        pos = end_idx + 1
        if pos >= len(val_dates_full):
            break

        # Slide training window forward to end just before next sub-period.
        next_train_end = val_dates_full[pos] - pd.Timedelta(days=1)
        next_train_start = next_train_end - pd.Timedelta(days=365 * train_window_years)
        next_train = _slice_panel(full_panel, next_train_start, next_train_end)
        _train_on_window(next_train, refine_steps, cfg, step_fn, buffers, rng,
                         label=f'refine@{sub_end.date()}')

    # Stitch per-fold results.
    per_fold = []
    for fcfg in folds_cfg:
        name = fcfg['name']
        subs = sub_period_results[name]
        if not subs:
            per_fold.append({'name': name, 'skipped': True,
                             'reason': 'no sub-periods'})
            continue
        daily_concat = pd.concat([s['daily'] for s in subs]).sort_index()
        svs_concat = pd.concat([s['svs'] for s in subs]).sort_index()
        lvp_concat = pd.concat([s['lvp'] for s in subs]).sort_index()
        daily_concat = daily_concat[~daily_concat.index.duplicated(keep='first')]
        svs_concat = svs_concat[~svs_concat.index.duplicated(keep='first')]
        lvp_concat = lvp_concat[~lvp_concat.index.duplicated(keep='first')]

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

        print(f'  {name}: {len(subs)} sub-periods n={len(daily_concat)} '
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
            'short_vol_scale_std': float(svs_concat.std()) if len(svs_concat) else 0.0,
            'long_vol_position_mean': float(lvp_concat.mean()) if len(lvp_concat) else 0.0,
            'long_vol_position_std': float(lvp_concat.std()) if len(lvp_concat) else 0.0,
            'long_vol_position_min': float(lvp_concat.min()) if len(lvp_concat) else 0.0,
            'long_vol_position_max': float(lvp_concat.max()) if len(lvp_concat) else 0.0,
            'long_vol_position_2020q1_mean': q1_2020_lvp_mean,
            'daily_path': str(out_npz),
            'ckpt_path': '',  # final-state shared across folds; saved separately below
        })

    # Save final continuous-walk checkpoint once.
    final_ckpt = OUT_DIR / f'{save_prefix}-final.npz'
    save_npz(model, str(final_ckpt))
    print(f'  saved final continuous-walk ckpt: {final_ckpt}', flush=True)
    for r in per_fold:
        if not r.get('skipped'):
            r['ckpt_path'] = str(final_ckpt)
    return per_fold


def pool_and_report(per_fold: list[dict],
                    save_prefix: str = 'e2e-portfolio-v3p6') -> dict:
    """Same as v3.5's pool_and_report but reads v3p6 artifacts."""
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
