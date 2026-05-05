"""Does the +0.47 val-IC vol forecast help return prediction as a feature?

Open question after the vol-overlay null (`vol_overlay_walkforward.py`):
the vol forecast doesn't transfer to portfolio Sharpe via *sizing*, but
maybe it carries information that helps *predict returns* directly when
fed in as a feature. This driver tests that.

Per walk-forward window:
  1. Train a vol head on the train slice (base 74 indicator channels →
     `vol_innovation` target). Linear head, AdamW, Pearson IC loss.
  2. Calibrate the vol head's raw score on the train slice via lstsq:
     `forecast_log_ratio ≈ a + b * raw_score`. The Pearson IC loss is
     scale-invariant so the head's output magnitude is unscaled —
     calibration recovers the absolute log-vol-ratio.
  3. Apply (a, b) to the entire date range to get a per-(bar, ticker)
     forecast channel. Concat to base features → 75-dim augmented
     representation.
  4. Train two return heads from scratch on the same train slice with
     the same seed:
       * `base`: 74 indicator channels.
       * `aug` : 75 channels (base + forecast).
  5. Eval val IC and val Sharpe per head.

Compare aggregate val IC: if `aug > base` cleanly, vol-forecast info
has return-prediction value. If `aug ≈ base`, the +0.47 IC vol signal
doesn't carry return-direction info that the indicator stack hasn't
already extracted. If `aug < base`, the forecast feature noises out
the return head (overfit on the vol-shaped channel).

Run from the repo root:
    uv run python apps/factor/scripts/feature_aug_walkforward.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'

TRADING_DAYS = 252


def _resolve_ticker_list(min_history_bars: int, max_tickers: int) -> list[str]:
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    if min_history_bars > 0:
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} tickers; {len(entries)} pass '
          f'min_history_bars={min_history_bars}')
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers} for smoke run')
    return names


def _build_one_ticker(args):
    ticker, stooq_dir, cfg, start, end = args
    try:
        from factor import build_indicator_features
        from ss_features import TickerData, load_prices
        series = load_prices(ticker, stooq_dir=stooq_dir, start=start, end=end)
        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        feats, valid = build_indicator_features(prices, cfg)
        if not valid.any():
            return ticker, None, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ), None
    except Exception as e:
        return ticker, None, f'({type(e).__name__}: {e})'


def _train_linear_head(
    repr_train_t, fwd_train_t, mask_train_t,
    repr_val_t, fwd_val_t, mask_val_t,
    *, seed: int, n_steps: int, learning_rate: float, weight_decay: float,
):
    """Train one linear head from scratch with Pearson IC loss; return
    (head_params_numpy, train_ic, val_ic, head_apply_fn).

    All inputs are tinygrad Tensors already on the chosen device.
    """
    from tinygrad.tensor import Tensor
    from tinygrad.nn.optim import AdamW
    from factor import init_linear, apply_linear, pearson_rank_ic

    rng = np.random.default_rng(seed)
    hidden_flat = repr_train_t.shape[-1]
    head = init_linear(rng, hidden_flat)
    opt = AdamW(list(head.values()), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(n_steps):
        Tensor.training = True
        opt.zero_grad()
        s = apply_linear(head, repr_train_t)
        loss = -pearson_rank_ic(s, fwd_train_t, mask_train_t)
        loss.backward()
        opt.step()
    Tensor.training = False
    s_train = apply_linear(head, repr_train_t)
    s_val   = apply_linear(head, repr_val_t)
    train_ic = float(pearson_rank_ic(s_train, fwd_train_t, mask_train_t).item())
    val_ic   = float(pearson_rank_ic(s_val,   fwd_val_t,   mask_val_t  ).item())
    return ({k: v.numpy() for k, v in head.items()},
            train_ic, val_ic, s_train.numpy(), s_val.numpy())


def _block_sharpe_softmax_numpy(
    scores: np.ndarray, mask: np.ndarray, block_log_ret: np.ndarray,
    rebal_days: int, commission_frac: float,
) -> float:
    """Same softmax-temperature top-N portfolio + cost convention as
    `factor.objectives.block_sharpe`. Plain numpy."""
    s = scores.copy().astype(np.float64)
    m = mask.astype(np.float64)
    blr = np.where(np.isfinite(block_log_ret), block_log_ret, 0.0)
    s = np.where(np.isfinite(s), s, 0.0)
    s = s + np.log(m + 1e-12)
    s = s - s.max(axis=1, keepdims=True)
    exp_s = np.exp(s) * m
    w = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)
    port = (w * blr).sum(axis=1)
    init_cost = np.abs(w[0]).sum()
    diff_cost = (0.5 * np.abs(w[1:] - w[:-1]).sum(axis=1)
                 if w.shape[0] > 1 else np.zeros(0))
    costs = np.concatenate([[init_cost], diff_cost]) * commission_frac
    port = port - costs
    mean = port.mean()
    std = port.std() + 1e-9
    return float(mean / std * np.sqrt(TRADING_DAYS / rebal_days))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',  default='2000-01-01')
    p.add_argument('--end',    default='2026-04-01')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--train-window-blocks', type=int, default=63)
    p.add_argument('--val-window-blocks',   type=int, default=39)
    p.add_argument('--step-window-blocks',  type=int, default=39)
    p.add_argument('--n-steps',       type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=1e-2)
    p.add_argument('--weight-decay',  type=float, default=1e-3)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers',      type=int, default=0)
    p.add_argument('--n-workers',     type=int, default=mp.cpu_count())
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    commission_frac = args.commission_bps / 1e4

    from tinygrad.tensor import Tensor
    from factor import (
        IndicatorGridConfig, make_indicator_backbone, precompute_inputs,
        forward_vol_innovation,
    )
    from factor.train_walkforward import _generate_window_slices

    cfg = IndicatorGridConfig()
    F_base = cfg.feature_width()
    print(f'IndicatorGridConfig.feature_width() = {F_base}')

    names = _resolve_ticker_list(args.min_history_bars, args.max_tickers)
    print(f'\nbuilding features over {len(names)} tickers '
          f'(workers={args.n_workers}) ...')
    t0 = time.perf_counter()
    pool_args = [(n, str(STOOQ_SUBSET), cfg, args.start, args.end) for n in names]
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            results = pool.map(_build_one_ticker, pool_args)
    else:
        results = [_build_one_ticker(a) for a in pool_args]
    ticker_data = []
    failed = []
    for name, td, err in results:
        if td is None:
            failed.append((name, err))
        else:
            ticker_data.append(td)
    print(f'built {len(ticker_data)} / {len(names)} tickers in '
          f'{time.perf_counter() - t0:.1f}s')
    if failed:
        print(f'  failed: {len(failed)}')

    backbone = make_indicator_backbone(ticker_data, cfg)
    pre_ret = precompute_inputs(
        ticker_data, backbone,
        rebal_days=args.rebal_days, forward_target_kind='log_return')
    pre_vol = precompute_inputs(
        ticker_data, backbone,
        rebal_days=args.rebal_days, forward_target_kind='vol_innovation')

    repr_rb = pre_ret['representation_rb']        # (n_blocks, N, F_base)
    fwd_ret_rb = pre_ret['fwd_ret_rb']            # (n_blocks, N) log_return target
    fwd_vol_rb = pre_vol['fwd_ret_rb']            # (n_blocks, N) vol_innovation target
    mask_rb = pre_ret['mask_rb']                  # (n_blocks, N)
    mask_vol_rb = pre_vol['mask_rb']
    blr_rb = pre_ret['block_log_ret_rb']
    aligned = pre_ret['aligned']
    rebal_idx = pre_ret['rebal_idx']

    # Ground-truth vol_innovation per rebal bar — used for calibration check.
    vol_innov_full = forward_vol_innovation(
        aligned.prices, rebal_days=args.rebal_days)
    vol_innov_rb = vol_innov_full[rebal_idx]      # (n_blocks, N)

    n_blocks, N, _ = repr_rb.shape
    print(f'\nrebal blocks={n_blocks}, tickers={N}, F_base={F_base}')

    slices = _generate_window_slices(
        n_blocks, args.train_window_blocks, args.val_window_blocks,
        args.step_window_blocks)
    print(f'walk-forward windows: {len(slices)}')

    per_window: list[dict] = []
    for w_idx, (train_slc, val_slc) in enumerate(slices):
        t1 = time.perf_counter()
        # Common Tensors for this window.
        repr_tr_t = Tensor(repr_rb[train_slc])
        repr_va_t = Tensor(repr_rb[val_slc])
        mask_tr_t = Tensor(mask_rb[train_slc])
        mask_va_t = Tensor(mask_rb[val_slc])
        mask_vol_tr_t = Tensor(mask_vol_rb[train_slc])
        mask_vol_va_t = Tensor(mask_vol_rb[val_slc])
        fwd_ret_tr_t = Tensor(fwd_ret_rb[train_slc])
        fwd_ret_va_t = Tensor(fwd_ret_rb[val_slc])
        fwd_vol_tr_t = Tensor(fwd_vol_rb[train_slc])
        fwd_vol_va_t = Tensor(fwd_vol_rb[val_slc])

        # ---------- step 1: vol head ----------
        vol_head, vol_train_ic, vol_val_ic, s_vol_tr, s_vol_va = _train_linear_head(
            repr_tr_t, fwd_vol_tr_t, mask_vol_tr_t,
            repr_va_t, fwd_vol_va_t, mask_vol_va_t,
            seed=args.seed + 100 + w_idx, n_steps=args.n_steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )

        # ---------- step 2: calibrate vol head on train slice ----------
        target_tr = vol_innov_rb[train_slc].reshape(-1)
        score_tr  = s_vol_tr.reshape(-1)
        valid_cal = (mask_vol_rb[train_slc].reshape(-1) > 0) \
            & np.isfinite(target_tr) & np.isfinite(score_tr)
        x = score_tr[valid_cal].astype(np.float64)
        y = target_tr[valid_cal].astype(np.float64)
        A = np.column_stack([np.ones_like(x), x])
        (a_cal, b_cal), *_ = np.linalg.lstsq(A, y, rcond=None)
        cal_corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0

        # ---------- step 3: build augmented repr ----------
        # Calibrated forecast channel for both train + val slices.
        forecast_tr = (a_cal + b_cal * s_vol_tr).astype(np.float32)
        forecast_va = (a_cal + b_cal * s_vol_va).astype(np.float32)
        # z-norm forecast channel using train-slice stats so the
        # augmented head sees it on a comparable scale to the already-
        # z-normed base features.
        f_mu = float(np.nanmean(forecast_tr))
        f_sd = float(np.nanstd(forecast_tr) + 1e-6)
        forecast_tr_z = (forecast_tr - f_mu) / f_sd
        forecast_va_z = (forecast_va - f_mu) / f_sd
        repr_tr_aug_np = np.concatenate(
            [repr_rb[train_slc], forecast_tr_z[:, :, None]], axis=-1)
        repr_va_aug_np = np.concatenate(
            [repr_rb[val_slc],   forecast_va_z[:, :, None]], axis=-1)
        repr_tr_aug_t = Tensor(repr_tr_aug_np.astype(np.float32))
        repr_va_aug_t = Tensor(repr_va_aug_np.astype(np.float32))

        # ---------- step 4a: base return head ----------
        base_head, base_train_ic, base_val_ic, _, s_ret_va_base = _train_linear_head(
            repr_tr_t, fwd_ret_tr_t, mask_tr_t,
            repr_va_t, fwd_ret_va_t, mask_va_t,
            seed=args.seed + w_idx, n_steps=args.n_steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )

        # ---------- step 4b: augmented return head ----------
        aug_head, aug_train_ic, aug_val_ic, _, s_ret_va_aug = _train_linear_head(
            repr_tr_aug_t, fwd_ret_tr_t, mask_tr_t,
            repr_va_aug_t, fwd_ret_va_t, mask_va_t,
            seed=args.seed + w_idx, n_steps=args.n_steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )

        # ---------- step 5: val Sharpe per head ----------
        val_blr = blr_rb[val_slc]
        val_mask_np = mask_rb[val_slc]
        sh_base = _block_sharpe_softmax_numpy(
            s_ret_va_base, val_mask_np, val_blr, args.rebal_days, commission_frac)
        sh_aug = _block_sharpe_softmax_numpy(
            s_ret_va_aug,  val_mask_np, val_blr, args.rebal_days, commission_frac)

        # Inspect the augmented head's weight on the new channel — diagnostic
        # for whether the head is using the forecast feature at all.
        aug_W = np.asarray(aug_head['W'])
        forecast_w = float(aug_W[-1])
        aug_W_base = aug_W[:-1]
        forecast_share = float(np.abs(forecast_w) /
                               (np.abs(aug_W_base).sum() + abs(forecast_w) + 1e-12))

        per_window.append({
            'window_idx': w_idx,
            'train_block_start': train_slc.start,
            'val_block_start':   val_slc.start,
            'val_block_end':     val_slc.stop,
            'cal_corr':       cal_corr,
            'cal_intercept':  float(a_cal),
            'cal_slope':      float(b_cal),
            'vol_train_ic':   vol_train_ic,
            'vol_val_ic':     vol_val_ic,
            'base_train_ic':  base_train_ic,
            'base_val_ic':    base_val_ic,
            'aug_train_ic':   aug_train_ic,
            'aug_val_ic':     aug_val_ic,
            'base_val_sharpe': sh_base,
            'aug_val_sharpe':  sh_aug,
            'forecast_weight':  forecast_w,
            'forecast_share':   forecast_share,
        })
        wall = time.perf_counter() - t1
        print(f'  win {w_idx}  '
              f'vol_val_ic={vol_val_ic:+.3f}  '
              f'base_val_ic={base_val_ic:+.4f} aug_val_ic={aug_val_ic:+.4f} '
              f'(Δ={aug_val_ic - base_val_ic:+.4f})  '
              f'base_sh={sh_base:+.3f} aug_sh={sh_aug:+.3f}  '
              f'fcst_share={forecast_share:.3f}  '
              f'wall={wall:.1f}s')

    # ---------- aggregate ----------
    base_ic = np.array([w['base_val_ic'] for w in per_window])
    aug_ic  = np.array([w['aug_val_ic']  for w in per_window])
    base_sh = np.array([w['base_val_sharpe'] for w in per_window])
    aug_sh  = np.array([w['aug_val_sharpe']  for w in per_window])
    fcst_share = np.array([w['forecast_share'] for w in per_window])
    cal_corr_arr = np.array([w['cal_corr'] for w in per_window])

    print('\n' + '=' * 88)
    print('Feature-augmentation leaderboard — does forecast-vol help return prediction?')
    print('=' * 88)
    print(f'{"metric":<32} {"base (74ch)":>14} {"aug (75ch)":>14} {"Δ":>10}')
    print(f'{"mean val IC":<32} {base_ic.mean():>+14.4f} {aug_ic.mean():>+14.4f} '
          f'{aug_ic.mean() - base_ic.mean():>+10.4f}')
    print(f'{"median val IC":<32} {np.median(base_ic):>+14.4f} '
          f'{np.median(aug_ic):>+14.4f} '
          f'{np.median(aug_ic) - np.median(base_ic):>+10.4f}')
    print(f'{"positive-val-IC fraction":<32} '
          f'{(base_ic > 0).mean():>14.2f} {(aug_ic > 0).mean():>14.2f} '
          f'{(aug_ic > 0).mean() - (base_ic > 0).mean():>+10.2f}')
    print(f'{"mean val Sharpe":<32} {base_sh.mean():>+14.3f} '
          f'{aug_sh.mean():>+14.3f} {aug_sh.mean() - base_sh.mean():>+10.3f}')
    print()
    print(f'mean cal corr (vol head score → log-vol-ratio): {cal_corr_arr.mean():+.3f}')
    print(f'mean |forecast weight| share of head L1: '
          f'{fcst_share.mean():.3f}  '
          f'(0 = ignored, 1/{F_base+1}≈{1.0/(F_base+1):.3f} = uniform)')

    summary_path = output / 'feature-aug-summary.json'
    summary_path.write_text(json.dumps({
        'universe_size': len(ticker_data),
        'F_base': F_base,
        'rebal_days': args.rebal_days,
        'train_window_blocks': args.train_window_blocks,
        'val_window_blocks': args.val_window_blocks,
        'step_window_blocks': args.step_window_blocks,
        'n_steps': args.n_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'commission_bps': args.commission_bps,
        'aggregate': {
            'mean_base_val_ic':    float(base_ic.mean()),
            'mean_aug_val_ic':     float(aug_ic.mean()),
            'median_base_val_ic':  float(np.median(base_ic)),
            'median_aug_val_ic':   float(np.median(aug_ic)),
            'mean_base_val_sharpe': float(base_sh.mean()),
            'mean_aug_val_sharpe':  float(aug_sh.mean()),
            'mean_forecast_share':  float(fcst_share.mean()),
            'mean_cal_corr':        float(cal_corr_arr.mean()),
        },
        'per_window': per_window,
    }, indent=2))
    print(f'\n-> {summary_path}')


if __name__ == '__main__':
    main()
