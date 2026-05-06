"""Regime-gate test on the 297-ticker walk-forward universe.

Question: does the +0.47 val-IC vol forecast carry actionable
*timing* information beyond what the trailing vol level already says?
Three earlier operational tests on the same forecast all nulled —
sign-of-demeaned (def3ac9), vol-target overlay (9cbf8bb), and feature
augmentation (75839f8). All three were *return-prediction-via-vol*
pathways. The remaining untested pathway is the **regime gate**:
flip the strategy ON/OFF based on aggregate forecast vol rather than
resize positions continuously.

Hypothesis: when aggregate forecast vol is high, the cross-sectional
return signal is noisier (returns dominated by macro / surprise events
not anticipated by the indicator stack). When forecast vol is low,
signals are cleaner. Gating to sit out high-forecast-vol windows
should lift Sharpe even if mean return drops, because fewer losing
periods → better Sharpe than always-on.

Walk-forward harness, same 297-ticker / rebal=20d / linear head config
as `vol_overlay_walkforward.py`. Per window:

  1. Train return scorer (`forward_target_kind='log_return'`).
  2. Train vol scorer (`forward_target_kind='vol_innovation'`).
  3. Calibrate the vol score on the *train* slice via lstsq against
     the true `log(σ_fwd / σ_trail)` target.
  4. Compute per-bar aggregate forecast vol on val:
        σ_fcst[t, i] = σ_trail[t, i] · exp(calibrated_log_ratio[t, i])
        agg[t]       = mean over active tickers of σ_fcst[t, i]
  5. Set thresholds from the **train slice's aggregate forecast vol**
     percentiles (70th / 80th / 90th).
  6. On val: weights[t] = top-N return basket if agg[t] < threshold,
     else 0 (sit out, hold cash).
  7. Block Sharpe with same 10bps commission convention as
     `objectives.block_sharpe`. Match `vol_overlay_walkforward.py`'s
     hard-top-10 framing (mean Sharpe ~+0.215 there) for comparability.

Read of the leaderboard:

  always-on ≈ gate-X        : forecast vol uninformative for *when*.
  always-on < gate-X        : forecast vol gates timing — real edge.
  always-on > gate-X        : gate fires on the wrong regimes — null
                              again, fourth and final pathway closed.

Run from the repo root:
    uv run python apps/factor/scripts/regime_gate_walkforward.py
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
GATE_PERCENTILES = (70.0, 80.0, 90.0)


def _resolve_ticker_list(min_history_bars: int, max_tickers: int) -> list[str]:
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    if min_history_bars > 0:
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} tickers; {len(entries)} pass '
          f'min_history_bars={min_history_bars}', flush=True)
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers} for smoke run', flush=True)
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
    """Per-ticker causal trailing realized vol annualized (matches
    the `vol_n{window}` channel pre-z-norm). `(D, N)` float64."""
    from ss_features.vol import realized_vol
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    for j in range(prices.shape[1]):
        out[:, j] = realized_vol(prices[:, j], window) * np.sqrt(TRADING_DAYS)
    return out


def _apply_linear_numpy(head: dict, X: np.ndarray) -> np.ndarray:
    W = np.asarray(head['W'], dtype=np.float32)
    b = np.asarray(head['b'], dtype=np.float32).reshape(())
    return (X @ W + b).astype(np.float32)


def _hard_top_n_weights(scores: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Per-bar top-N selection. Masked tickers get -inf."""
    n_bars, n_tickers = scores.shape
    s = np.where(mask > 0, scores, -np.inf)
    out = np.zeros_like(scores, dtype=np.float64)
    for t in range(n_bars):
        valid_idx = np.where(np.isfinite(s[t]))[0]
        if len(valid_idx) == 0:
            continue
        k = min(n, len(valid_idx))
        top_idx = valid_idx[np.argpartition(-s[t, valid_idx], k - 1)[:k]]
        out[t, top_idx] = 1.0 / k
    return out


def _block_sharpe_numpy(
    weights: np.ndarray, block_log_ret: np.ndarray,
    rebal_days: int, commission_frac: float,
) -> dict:
    """Mirror of `objectives.block_sharpe` in numpy."""
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


def _aggregate_forecast_vol(
    fcst_vol: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """Per-bar mean over active tickers of σ_fcst.
    `fcst_vol` `(n_blocks, N)`, `mask` `(n_blocks, N)` 0/1.
    Returns `(n_blocks,)` aggregate vol — NaN where no active ticker."""
    n_blocks = fcst_vol.shape[0]
    out = np.full(n_blocks, np.nan, dtype=np.float64)
    for t in range(n_blocks):
        m = (mask[t] > 0) & np.isfinite(fcst_vol[t])
        if m.any():
            out[t] = float(fcst_vol[t, m].mean())
    return out


def _apply_gate(
    weights: np.ndarray, agg_vol: np.ndarray, threshold: float,
) -> tuple[np.ndarray, float]:
    """Zero rows where `agg_vol[t] >= threshold` (sit out high-vol bars).
    NaN agg_vol rows pass through (no gate fire). Returns
    `(gated_weights, fraction_off)`."""
    out = weights.copy()
    fire = (np.isfinite(agg_vol)) & (agg_vol >= threshold)
    out[fire] = 0.0
    n = len(agg_vol)
    frac_off = float(fire.sum()) / float(n) if n else 0.0
    return out, frac_off


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
    print(f'IndicatorGridConfig.feature_width() = {F}', flush=True)

    names = _resolve_ticker_list(args.min_history_bars, args.max_tickers)
    print(f'\nbuilding features over {len(names)} tickers '
          f'(workers={args.n_workers}) ...', flush=True)
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
          f'{time.perf_counter() - t0:.1f}s', flush=True)
    if failed:
        print(f'  failed: {len(failed)}', flush=True)

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
    print('\n--- arm: return scorer (forward_target_kind=log_return) ---', flush=True)
    t1 = time.perf_counter()
    wf_ret = train_scorer_indicators_walkforward(
        ticker_data, **common_kwargs, forward_target_kind='log_return')
    print(f'arm wall: {time.perf_counter() - t1:.1f}s', flush=True)

    # ---------- arm 2: vol scorer ----------
    print('\n--- arm: vol scorer (forward_target_kind=vol_innovation) ---', flush=True)
    t1 = time.perf_counter()
    wf_vol = train_scorer_indicators_walkforward(
        ticker_data, **common_kwargs, forward_target_kind='vol_innovation')
    print(f'arm wall: {time.perf_counter() - t1:.1f}s', flush=True)

    # ---------- post-process: regime-gate evaluation ----------
    print('\n--- post-process: regime-gate Sharpe per window ---', flush=True)
    backbone = make_indicator_backbone(ticker_data, cfg)
    pre = precompute_inputs(
        ticker_data, backbone,
        rebal_days=args.rebal_days, forward_target_kind='log_return')
    repr_rb = pre['representation_rb']        # (n_blocks, N, hidden_flat)
    blr_rb = pre['block_log_ret_rb']          # (n_blocks, N)
    mask_rb = pre['mask_rb']                  # (n_blocks, N) float
    aligned = pre['aligned']
    rebal_idx = pre['rebal_idx']

    trail_vol_full = _trailing_realized_vol(aligned.prices, args.rebal_days)
    trail_vol_rb = trail_vol_full[rebal_idx]  # (n_blocks, N)

    vol_innov_full = forward_vol_innovation(
        aligned.prices, rebal_days=args.rebal_days)
    vol_innov_rb = vol_innov_full[rebal_idx]  # (n_blocks, N)

    n_blocks, N, _ = repr_rb.shape
    print(f'  rebal blocks={n_blocks}, tickers={N}, '
          f'top_n={args.top_n}, '
          f'commission_bps={args.commission_bps}', flush=True)
    print(f'  gate percentiles: {GATE_PERCENTILES}', flush=True)

    per_window: list[dict] = []
    for w_idx in range(len(wf_ret.windows)):
        wr = wf_ret.windows[w_idx]
        wv = wf_vol.windows[w_idx]
        train_slc = slice(wr.train_block_start, wr.train_block_end)
        val_slc = slice(wr.val_block_start, wr.val_block_end)

        # Score val slice with return head; train+val with vol head.
        s_ret_val   = _apply_linear_numpy(wr.head_params, repr_rb[val_slc])
        s_vol_train = _apply_linear_numpy(wv.head_params, repr_rb[train_slc])
        s_vol_val   = _apply_linear_numpy(wv.head_params, repr_rb[val_slc])

        # Calibrate vol head on train slice.
        train_mask_flat   = mask_rb[train_slc].reshape(-1)
        train_target_flat = vol_innov_rb[train_slc].reshape(-1)
        train_score_flat  = s_vol_train.reshape(-1)
        train_good = (train_mask_flat > 0) & np.isfinite(train_target_flat) \
            & np.isfinite(train_score_flat)
        if train_good.sum() < 10:
            print(f'  window {w_idx}: too few good cells for calibration; skip',
                  flush=True)
            continue
        x = train_score_flat[train_good]
        y = train_target_flat[train_good]
        A = np.column_stack([np.ones_like(x), x])
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        cal_train_corr = float(np.corrcoef(x, y)[0, 1])

        # Forecast vol per ticker on TRAIN (for threshold) and VAL.
        forecast_log_ratio_train = (a + b * s_vol_train).astype(np.float64)
        forecast_log_ratio_val   = (a + b * s_vol_val).astype(np.float64)
        forecast_vol_train = trail_vol_rb[train_slc] \
            * np.exp(np.clip(forecast_log_ratio_train, -2.0, 2.0))
        forecast_vol_val = trail_vol_rb[val_slc] \
            * np.exp(np.clip(forecast_log_ratio_val, -2.0, 2.0))

        # Aggregate per-bar forecast vol.
        agg_train = _aggregate_forecast_vol(forecast_vol_train,
                                            mask_rb[train_slc])
        agg_val = _aggregate_forecast_vol(forecast_vol_val,
                                          mask_rb[val_slc])

        # Build always-on basket from return head scores on val.
        val_mask = mask_rb[val_slc]
        val_blr  = blr_rb[val_slc]
        always_w = _hard_top_n_weights(s_ret_val, val_mask, args.top_n)

        sh_always = _block_sharpe_numpy(
            always_w, val_blr, args.rebal_days, commission_frac)

        gates: dict[str, dict] = {}
        for pct in GATE_PERCENTILES:
            agg_train_finite = agg_train[np.isfinite(agg_train)]
            if len(agg_train_finite) == 0:
                continue
            thr = float(np.percentile(agg_train_finite, pct))
            gated_w, frac_off = _apply_gate(always_w, agg_val, thr)
            sh = _block_sharpe_numpy(
                gated_w, val_blr, args.rebal_days, commission_frac)
            gates[f'gate_{int(pct)}'] = {
                'threshold': thr,
                'fraction_off': frac_off,
                **sh,
            }

        per_window.append({
            'window_idx': w_idx,
            'train_block_start': wr.train_block_start,
            'val_block_start':   wr.val_block_start,
            'val_block_end':     wr.val_block_end,
            'cal_train_corr':    cal_train_corr,
            'cal_intercept':     float(a),
            'cal_slope':         float(b),
            'agg_train_mean':    float(np.nanmean(agg_train)),
            'agg_train_std':     float(np.nanstd(agg_train)),
            'agg_val_mean':      float(np.nanmean(agg_val)),
            'agg_val_std':       float(np.nanstd(agg_val)),
            'always_on': sh_always,
            **gates,
        })
        msg = (f'  win {w_idx}: cal r(train)={cal_train_corr:+.3f}  '
               f'agg_train(μ={np.nanmean(agg_train):.3f}, σ={np.nanstd(agg_train):.3f})  '
               f'agg_val(μ={np.nanmean(agg_val):.3f})  '
               f'always Sh={sh_always["sharpe"]:+.3f}')
        for pct in GATE_PERCENTILES:
            key = f'gate_{int(pct)}'
            if key in gates:
                g = gates[key]
                msg += f'  {key}: Sh={g["sharpe"]:+.3f} off={g["fraction_off"]:.2f}'
        print(msg, flush=True)

    # ---------- aggregate ----------
    if per_window:
        always_sh = [w['always_on']['sharpe'] for w in per_window]
        always_gross = [w['always_on']['mean_gross'] for w in per_window]
        cal_corr = [w['cal_train_corr'] for w in per_window]
        gate_data: dict[str, dict] = {}
        for pct in GATE_PERCENTILES:
            key = f'gate_{int(pct)}'
            sh = [w[key]['sharpe'] for w in per_window if key in w]
            gr = [w[key]['mean_gross'] for w in per_window if key in w]
            off = [w[key]['fraction_off'] for w in per_window if key in w]
            thr = [w[key]['threshold'] for w in per_window if key in w]
            gate_data[key] = {
                'sh': sh, 'gross': gr, 'off': off, 'thr': thr,
            }

        print('\n' + '=' * 96, flush=True)
        print('Regime-gate leaderboard — per-window val Sharpe (top-10 from return scorer)',
              flush=True)
        print('=' * 96, flush=True)
        hdr = f'{"win":>3} {"cal_r":>7} {"always":>8}'
        for pct in GATE_PERCENTILES:
            p = int(pct)
            hdr += f' {f"g{p}_sh":>8} {f"g{p}_off":>9}'
        print(hdr, flush=True)
        for w in per_window:
            row = (f'{w["window_idx"]:>3} {w["cal_train_corr"]:>+7.3f} '
                   f'{w["always_on"]["sharpe"]:>+8.3f}')
            for pct in GATE_PERCENTILES:
                key = f'gate_{int(pct)}'
                if key in w:
                    row += f' {w[key]["sharpe"]:>+8.3f} {w[key]["fraction_off"]:>9.3f}'
                else:
                    row += f' {"-":>8} {"-":>9}'
            print(row, flush=True)

        print('\nAggregate across windows:', flush=True)
        print(f'  mean cal r(train, vol-head score → log-vol-ratio): '
              f'{np.mean(cal_corr):+.3f}', flush=True)
        print(f'  always-on  : mean Sh {np.mean(always_sh):+.3f}  '
              f'(median {np.median(always_sh):+.3f}, '
              f'pos-frac {np.mean(np.array(always_sh) > 0):.2f}, '
              f'gross {np.mean(always_gross):.2f})', flush=True)
        for pct in GATE_PERCENTILES:
            key = f'gate_{int(pct)}'
            d = gate_data[key]
            sh = np.array(d['sh'])
            print(f'  {key:<10}: mean Sh {sh.mean():+.3f}  '
                  f'(median {np.median(sh):+.3f}, '
                  f'pos-frac {np.mean(sh > 0):.2f}, '
                  f'mean off {np.mean(d["off"]):.2f}, '
                  f'gross {np.mean(d["gross"]):.2f}, '
                  f'Δ vs always {sh.mean() - np.mean(always_sh):+.3f})',
                  flush=True)

        summary_path = output / 'regime-gate-summary.json'
        agg_payload: dict = {
            'mean_cal_train_corr': float(np.mean(cal_corr)),
            'mean_always_sharpe': float(np.mean(always_sh)),
            'median_always_sharpe': float(np.median(always_sh)),
            'pos_frac_always': float(np.mean(np.array(always_sh) > 0)),
            'mean_always_gross': float(np.mean(always_gross)),
        }
        for pct in GATE_PERCENTILES:
            key = f'gate_{int(pct)}'
            d = gate_data[key]
            sh = np.array(d['sh'])
            agg_payload[key] = {
                'mean_sharpe': float(sh.mean()),
                'median_sharpe': float(np.median(sh)),
                'pos_frac': float(np.mean(sh > 0)),
                'mean_fraction_off': float(np.mean(d['off'])),
                'mean_gross': float(np.mean(d['gross'])),
                'mean_threshold': float(np.mean(d['thr'])),
                'mean_delta_vs_always': float(sh.mean() - np.mean(always_sh)),
            }
        summary_path.write_text(json.dumps({
            'universe_size': len(ticker_data),
            'feature_width': F,
            'rebal_days': args.rebal_days,
            'top_n': args.top_n,
            'commission_bps': args.commission_bps,
            'gate_percentiles': list(GATE_PERCENTILES),
            'train_window_blocks': args.train_window_blocks,
            'val_window_blocks': args.val_window_blocks,
            'step_window_blocks': args.step_window_blocks,
            'scorer': args.scorer, 'n_steps': args.n_steps,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'aggregate': agg_payload,
            'per_window': per_window,
        }, indent=2))
        print(f'\n-> {summary_path}', flush=True)

        windows_npz = output / 'regime-gate-windows.npz'
        save_dict = {
            'always_sharpe': np.array(always_sh),
            'always_gross':  np.array(always_gross),
            'cal_train_corr': np.array(cal_corr),
        }
        for pct in GATE_PERCENTILES:
            key = f'gate_{int(pct)}'
            d = gate_data[key]
            save_dict[f'{key}_sharpe'] = np.array(d['sh'])
            save_dict[f'{key}_gross']  = np.array(d['gross'])
            save_dict[f'{key}_off']    = np.array(d['off'])
            save_dict[f'{key}_thr']    = np.array(d['thr'])
        np.savez(windows_npz, **save_dict)
        print(f'-> {windows_npz}', flush=True)


if __name__ == '__main__':
    main()
