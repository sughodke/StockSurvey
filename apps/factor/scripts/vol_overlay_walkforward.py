"""Vol-target overlay test on the 297-ticker walk-forward universe.

Question: does the +0.47 val-IC vol forecast (`forward_target_kind=
'vol_innovation'`) carry actionable risk-targeting information beyond
what *trailing* realized vol already encodes? IC is suggestive; the
honest test is whether vol-targeted Sharpe rises when you swap trailing
vol for forecast vol in the position sizer.

Walk-forward harness, same 297-ticker / rebal=20d / linear head config
as `forecast_probe_walkforward.py`. Per window:

  1. Train a return scorer (`forward_target_kind='log_return'`) — picks
     top-N tickers by predicted forward return.
  2. Train a vol scorer (`forward_target_kind='vol_innovation'`) — its
     score is a raw projection that needs calibration.
  3. Calibrate the vol score on the *train* slice via linear regression
     against the true `log(σ_fwd / σ_trail)` target the head was
     trained on. The Pearson IC loss is scale-invariant, so the head's
     output magnitude is unscaled — calibration recovers the absolute
     log-vol-ratio. Apply the same `(intercept, slope)` on val.
  4. On val slice, build top-N basket from return-head scores. Three
     portfolio variants:
       * `EW`        — equal-weight 1/N (baseline; what `forecast_probe`
                       reported as the +0.012 / +0.44 Sharpe arm).
       * `trail-VT`  — scale by `target_vol / σ_p_trailing`. σ_p uses
                       a diagonal-cov estimate `√Σ (w_i σ_trail_i)²`.
       * `fcst-VT`   — same scaling but with `σ_p_forecast` from the
                       calibrated vol head.
     Each variant is clipped to `max_leverage=2.0`.
  5. Compute block Sharpe per variant with the same 10bps one-sided
     commission convention as `objectives.block_sharpe`.

Read of the leaderboard:

  EW < trail-VT  : any vol-targeting helps (well-known).
  trail-VT < fcst-VT : the forecast carries info beyond trailing — the
                       +0.47 IC is operationally useful.
  fcst-VT ≈ trail-VT : forecast is redundant with trailing for sizing.

Run from the repo root:
    uv run python apps/factor/scripts/vol_overlay_walkforward.py
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


def _trailing_realized_vol(prices: np.ndarray, window: int) -> np.ndarray:
    """Per-ticker causal trailing realized vol annualized (matches the
    `vol_n{window}` channel in IndicatorGridConfig — pre-z-norm).
    Returns `(D, N)` float64."""
    from ss_features.vol import realized_vol
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    for j in range(prices.shape[1]):
        out[:, j] = realized_vol(prices[:, j], window) * np.sqrt(TRADING_DAYS)
    return out


def _apply_linear_numpy(head: dict, X: np.ndarray) -> np.ndarray:
    """Re-do `apply_linear` in numpy from saved head_params.
    `X` shape `(..., hidden_flat)`, returns `(...)` scores."""
    W = np.asarray(head['W'], dtype=np.float32)
    b = np.asarray(head['b'], dtype=np.float32).reshape(())
    return (X @ W + b).astype(np.float32)


def _hard_top_n_weights(scores: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Per-bar top-N selection. Masked tickers get -inf scores so they
    can never enter the basket. If fewer than N tickers are valid, use
    all valid (basket size shrinks)."""
    n_bars, n_tickers = scores.shape
    s = np.where(mask > 0, scores, -np.inf)
    out = np.zeros_like(scores, dtype=np.float64)
    for t in range(n_bars):
        valid_idx = np.where(np.isfinite(s[t]))[0]
        if len(valid_idx) == 0:
            continue
        k = min(n, len(valid_idx))
        # argpartition for top-k descending
        top_idx = valid_idx[np.argpartition(-s[t, valid_idx], k - 1)[:k]]
        out[t, top_idx] = 1.0 / k
    return out


def _vol_target_overlay(
    weights: np.ndarray, sigma_per_ticker: np.ndarray,
    target_vol: float, max_leverage: float,
) -> np.ndarray:
    """Diagonal-cov vol-target overlay row-wise.
    `σ_p² ≈ Σ_i (w_i σ_i)²`. Scales each row by `target_vol / σ_p`,
    clipped to `[0, max_leverage]`. Rows with σ_p == 0 pass through."""
    out = np.zeros_like(weights, dtype=np.float64)
    for t in range(weights.shape[0]):
        sig_eff = np.where(np.isfinite(sigma_per_ticker[t]),
                           sigma_per_ticker[t], 0.0)
        wv = weights[t] * sig_eff
        sigma_p = float(np.sqrt(np.nansum(wv * wv)))
        if sigma_p > 0:
            lev = min(target_vol / sigma_p, max_leverage)
            out[t] = weights[t] * lev
        else:
            out[t] = weights[t]
    return out


def _block_sharpe_numpy(
    weights: np.ndarray, block_log_ret: np.ndarray,
    rebal_days: int, commission_frac: float,
) -> dict:
    """Mirror of `objectives.block_sharpe` in numpy. Returns dict with
    Sharpe, mean, std, gross-final, turnover-final."""
    blr = np.where(np.isfinite(block_log_ret), block_log_ret, 0.0)
    port_ret = (weights * blr).sum(axis=1)
    n = weights.shape[0]
    if n == 0:
        return dict(sharpe=0.0, mean=0.0, std=0.0,
                    mean_gross=0.0, mean_turnover=0.0)
    init_cost = np.abs(weights[0]).sum()
    diff_cost = 0.5 * np.abs(weights[1:] - weights[:-1]).sum(axis=1) \
        if n > 1 else np.zeros(0)
    costs = np.concatenate([[init_cost], diff_cost]) * commission_frac
    port_ret = port_ret - costs
    mean = port_ret.mean()
    std = port_ret.std() + 1e-9
    sharpe = mean / std * np.sqrt(TRADING_DAYS / rebal_days)
    mean_gross = np.abs(weights).sum(axis=1).mean()
    return dict(sharpe=float(sharpe), mean=float(mean), std=float(std),
                mean_gross=float(mean_gross),
                mean_turnover=float(np.mean(diff_cost) / commission_frac
                                    if len(diff_cost) else 0.0))


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
    p.add_argument('--scorer',        default='linear')
    p.add_argument('--top-n',         type=int, default=10)
    p.add_argument('--target-vol',    type=float, default=0.15)
    p.add_argument('--max-leverage',  type=float, default=2.0)
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

    from factor import (
        IndicatorGridConfig, train_scorer_indicators_walkforward,
        precompute_inputs, make_indicator_backbone, forward_vol_innovation,
    )

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'IndicatorGridConfig.feature_width() = {F}')

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

    common_kwargs = dict(
        cfg=cfg,
        rebal_days=args.rebal_days,
        train_window_blocks=args.train_window_blocks,
        val_window_blocks=args.val_window_blocks,
        step_window_blocks=args.step_window_blocks,
        scorer=args.scorer,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    # ---------- arm 1: return scorer ----------
    print('\n--- arm: return scorer (forward_target_kind=log_return) ---')
    t1 = time.perf_counter()
    wf_ret = train_scorer_indicators_walkforward(
        ticker_data, **common_kwargs, forward_target_kind='log_return')
    print(f'arm wall: {time.perf_counter() - t1:.1f}s')

    # ---------- arm 2: vol scorer ----------
    print('\n--- arm: vol scorer (forward_target_kind=vol_innovation) ---')
    t1 = time.perf_counter()
    wf_vol = train_scorer_indicators_walkforward(
        ticker_data, **common_kwargs, forward_target_kind='vol_innovation')
    print(f'arm wall: {time.perf_counter() - t1:.1f}s')

    # ---------- post-process: overlay evaluation ----------
    print('\n--- post-process: overlay Sharpe per window ---')
    backbone = make_indicator_backbone(ticker_data, cfg)
    pre = precompute_inputs(
        ticker_data, backbone,
        rebal_days=args.rebal_days, forward_target_kind='log_return')
    repr_rb = pre['representation_rb']        # (n_blocks, N, hidden_flat)
    blr_rb = pre['block_log_ret_rb']          # (n_blocks, N)
    mask_rb = pre['mask_rb']                  # (n_blocks, N) float
    aligned = pre['aligned']
    rebal_idx = pre['rebal_idx']

    # Trailing realized vol per (rebal_bar, ticker) — same window as rebal.
    trail_vol_full = _trailing_realized_vol(aligned.prices, args.rebal_days)
    trail_vol_rb = trail_vol_full[rebal_idx]  # (n_blocks, N)

    # Ground-truth vol_innovation target — needed per rebal bar to
    # calibrate the vol head's raw score on the train slice.
    vol_innov_full = forward_vol_innovation(
        aligned.prices, rebal_days=args.rebal_days)
    vol_innov_rb = vol_innov_full[rebal_idx]  # (n_blocks, N), NaN at edges

    n_blocks, N, _ = repr_rb.shape
    print(f'  rebal blocks={n_blocks}, tickers={N}, '
          f'top_n={args.top_n}, target_vol={args.target_vol}, '
          f'max_leverage={args.max_leverage}, commission_bps={args.commission_bps}')

    per_window: list[dict] = []
    for w_idx in range(len(wf_ret.windows)):
        wr = wf_ret.windows[w_idx]
        wv = wf_vol.windows[w_idx]
        train_slc = slice(wr.train_block_start, wr.train_block_end)
        val_slc = slice(wr.val_block_start, wr.val_block_end)

        # Score val slice with both heads.
        s_ret_val = _apply_linear_numpy(wr.head_params, repr_rb[val_slc])
        s_vol_train = _apply_linear_numpy(wv.head_params, repr_rb[train_slc])
        s_vol_val   = _apply_linear_numpy(wv.head_params, repr_rb[val_slc])

        # Calibrate vol head: linear regression on train slice
        # forecast_log_ratio = a + b * raw_score, fit against true target.
        # Fit only on cells where target is finite + mask is True.
        train_mask_flat  = mask_rb[train_slc].reshape(-1)
        train_target_flat = vol_innov_rb[train_slc].reshape(-1)
        train_score_flat  = s_vol_train.reshape(-1)
        train_good = (train_mask_flat > 0) & np.isfinite(train_target_flat) \
            & np.isfinite(train_score_flat)
        if train_good.sum() < 10:
            print(f'  window {w_idx}: too few good cells for calibration; skip')
            continue
        x = train_score_flat[train_good]
        y = train_target_flat[train_good]
        # Solve y = a + b * x  ->  least-squares with intercept column.
        A = np.column_stack([np.ones_like(x), x])
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        cal_train_corr = float(np.corrcoef(x, y)[0, 1])
        forecast_log_ratio_val = (a + b * s_vol_val).astype(np.float64)
        forecast_vol_val = trail_vol_rb[val_slc] \
            * np.exp(np.clip(forecast_log_ratio_val, -2.0, 2.0))

        # Build portfolios on val slice from return-head scores.
        val_mask = mask_rb[val_slc]
        val_blr  = blr_rb[val_slc]
        val_trail_vol = trail_vol_rb[val_slc]
        ew_w  = _hard_top_n_weights(s_ret_val, val_mask, args.top_n)
        # trail-VT: replace any NaN trailing vol with 0 (leaves weight unchanged
        # at 0-vol cells; vol-target overlay is then identity for that ticker).
        trail_w = _vol_target_overlay(
            ew_w, val_trail_vol, args.target_vol, args.max_leverage)
        fcst_w  = _vol_target_overlay(
            ew_w, forecast_vol_val, args.target_vol, args.max_leverage)

        sh_ew    = _block_sharpe_numpy(ew_w,    val_blr, args.rebal_days, commission_frac)
        sh_trail = _block_sharpe_numpy(trail_w, val_blr, args.rebal_days, commission_frac)
        sh_fcst  = _block_sharpe_numpy(fcst_w,  val_blr, args.rebal_days, commission_frac)

        per_window.append({
            'window_idx': w_idx,
            'train_block_start': wr.train_block_start,
            'val_block_start':   wr.val_block_start,
            'val_block_end':     wr.val_block_end,
            'cal_train_corr':    cal_train_corr,
            'cal_intercept':     float(a),
            'cal_slope':         float(b),
            'ew':    sh_ew,
            'trail_vt': sh_trail,
            'fcst_vt':  sh_fcst,
        })
        print(f'  win {w_idx}: cal r(train)={cal_train_corr:+.3f}  '
              f'EW Sh={sh_ew["sharpe"]:+.3f}  '
              f'trail-VT Sh={sh_trail["sharpe"]:+.3f}  '
              f'fcst-VT Sh={sh_fcst["sharpe"]:+.3f}  '
              f'(gross EW={sh_ew["mean_gross"]:.2f} '
              f'trail={sh_trail["mean_gross"]:.2f} '
              f'fcst={sh_fcst["mean_gross"]:.2f})')

    # ---------- aggregate ----------
    if per_window:
        ew_sh    = [w['ew']['sharpe']      for w in per_window]
        trail_sh = [w['trail_vt']['sharpe'] for w in per_window]
        fcst_sh  = [w['fcst_vt']['sharpe']  for w in per_window]
        ew_gross    = [w['ew']['mean_gross']      for w in per_window]
        trail_gross = [w['trail_vt']['mean_gross'] for w in per_window]
        fcst_gross  = [w['fcst_vt']['mean_gross']  for w in per_window]
        cal_corr = [w['cal_train_corr'] for w in per_window]

        print('\n' + '=' * 88)
        print('Vol-overlay leaderboard — per-window val Sharpe (top-N from return scorer)')
        print('=' * 88)
        print(f'{"win":>3} {"cal_corr":>9} {"EW_sh":>8} {"trail_sh":>9} '
              f'{"fcst_sh":>9} {"Δ trail":>9} {"Δ fcst":>9} '
              f'{"EW_gross":>9} {"trail_gr":>9} {"fcst_gr":>9}')
        for w in per_window:
            d_t = w['trail_vt']['sharpe'] - w['ew']['sharpe']
            d_f = w['fcst_vt']['sharpe']  - w['ew']['sharpe']
            print(f'{w["window_idx"]:>3} {w["cal_train_corr"]:>+9.3f} '
                  f'{w["ew"]["sharpe"]:>+8.3f} {w["trail_vt"]["sharpe"]:>+9.3f} '
                  f'{w["fcst_vt"]["sharpe"]:>+9.3f} {d_t:>+9.3f} {d_f:>+9.3f} '
                  f'{w["ew"]["mean_gross"]:>9.2f} '
                  f'{w["trail_vt"]["mean_gross"]:>9.2f} '
                  f'{w["fcst_vt"]["mean_gross"]:>9.2f}')

        print('\nAggregate across windows:')
        print(f'  mean cal r(train, vol-head score → log-vol-ratio): '
              f'{np.mean(cal_corr):+.3f}')
        print(f'  mean val Sharpe — EW       : {np.mean(ew_sh):+.3f}  '
              f'(median {np.median(ew_sh):+.3f}, '
              f'pos-frac {np.mean(np.array(ew_sh) > 0):.2f})')
        print(f'  mean val Sharpe — trail-VT : {np.mean(trail_sh):+.3f}  '
              f'(median {np.median(trail_sh):+.3f}, '
              f'pos-frac {np.mean(np.array(trail_sh) > 0):.2f})')
        print(f'  mean val Sharpe — fcst-VT  : {np.mean(fcst_sh):+.3f}  '
              f'(median {np.median(fcst_sh):+.3f}, '
              f'pos-frac {np.mean(np.array(fcst_sh) > 0):.2f})')
        print(f'  mean Δ Sharpe trail vs EW : '
              f'{np.mean(np.array(trail_sh) - np.array(ew_sh)):+.3f}')
        print(f'  mean Δ Sharpe fcst vs EW  : '
              f'{np.mean(np.array(fcst_sh)  - np.array(ew_sh)):+.3f}')
        print(f'  mean Δ Sharpe fcst vs trail: '
              f'{np.mean(np.array(fcst_sh)  - np.array(trail_sh)):+.3f}')
        print(f'  mean gross — EW {np.mean(ew_gross):.2f} | '
              f'trail {np.mean(trail_gross):.2f} | '
              f'fcst {np.mean(fcst_gross):.2f}')

        summary_path = output / 'vol-overlay-summary.json'
        summary_path.write_text(json.dumps({
            'universe_size': len(ticker_data),
            'feature_width': F,
            'rebal_days': args.rebal_days,
            'top_n': args.top_n,
            'target_vol': args.target_vol,
            'max_leverage': args.max_leverage,
            'commission_bps': args.commission_bps,
            'train_window_blocks': args.train_window_blocks,
            'val_window_blocks': args.val_window_blocks,
            'step_window_blocks': args.step_window_blocks,
            'scorer': args.scorer, 'n_steps': args.n_steps,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'aggregate': {
                'mean_cal_train_corr': float(np.mean(cal_corr)),
                'mean_ew_sharpe':      float(np.mean(ew_sh)),
                'mean_trail_sharpe':   float(np.mean(trail_sh)),
                'mean_fcst_sharpe':    float(np.mean(fcst_sh)),
                'mean_delta_trail_vs_ew':    float(np.mean(np.array(trail_sh) - np.array(ew_sh))),
                'mean_delta_fcst_vs_ew':     float(np.mean(np.array(fcst_sh)  - np.array(ew_sh))),
                'mean_delta_fcst_vs_trail':  float(np.mean(np.array(fcst_sh)  - np.array(trail_sh))),
            },
            'per_window': per_window,
        }, indent=2))
        print(f'\n-> {summary_path}')

        windows_npz = output / 'vol-overlay-windows.npz'
        np.savez(windows_npz,
            ew_sharpe    = np.array(ew_sh),
            trail_sharpe = np.array(trail_sh),
            fcst_sharpe  = np.array(fcst_sh),
            ew_gross     = np.array(ew_gross),
            trail_gross  = np.array(trail_gross),
            fcst_gross   = np.array(fcst_gross),
            cal_train_corr = np.array(cal_corr),
        )
        print(f'-> {windows_npz}')


if __name__ == '__main__':
    main()
