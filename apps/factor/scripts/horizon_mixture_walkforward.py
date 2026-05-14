"""Endogenous-horizon mixture walk-forward — does state-conditional
horizon selection beat the best fixed-horizon baseline?

Architecture
------------
Shared MLP trunk on the deterministic 74-channel indicator backbone
(`IndicatorGridConfig`). Two heads:
  - Score head: per-ticker scalar, same as `train_scorer_walkforward`.
  - Horizon head: cross-sectional pool over the liquid universe →
    softmax over K horizon bins. Default bins {5, 10, 20, 40, 60}.

Training loss: per-bar `Σ_k π_t[k] · IC_k_t` — state-conditional
mixture-of-horizons rank IC (negate for minimization). The score head
learns to rank-predict whichever horizon's returns π_t weighted; the
horizon head learns to weight whichever horizon's IC fires at this
state.

Eval: daily PnL stream under irregular cadence (`argmax(π_t)` horizon
choice, hold flat between rebals). Per window we record:
  - Endogenous net Sharpe — the load-bearing metric.
  - Fixed-h baselines at each `h ∈ horizons` — null N2/N3.
  - Random-π baseline — null N4.
  - π entropy + argmax bin counts — null N1 (π collapse).

Success criterion: mean endog Sharpe ≥ best fixed-h baseline + 0.10,
AND N1/N2/N4 all pass.

Run from the repo root:
    uv run python apps/factor/scripts/horizon_mixture_walkforward.py \\
        --max-tickers 50 --n-steps 50    # smoke
    uv run python apps/factor/scripts/horizon_mixture_walkforward.py
        # full Phase-4 stooq_us_long
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2026-04-01')
    p.add_argument('--horizons', default='5,10,20,40,60',
                   help='comma-separated horizon bins (days). Fine rebal '
                        'grid is min(horizons).')
    p.add_argument('--train-window-blocks', type=int, default=252,
                   help='in fine bars (h_min spacing). Default ~5y at h_min=5.')
    p.add_argument('--val-window-blocks',   type=int, default=156,
                   help='in fine bars. Default ~3y at h_min=5.')
    p.add_argument('--step-window-blocks',  type=int, default=156)
    p.add_argument('--n-steps',       type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay',  type=float, default=1e-3)
    p.add_argument('--entropy-weight', type=float, default=0.0,
                   help='regularization pushing π_t away from one-hot.')
    p.add_argument('--mlp-hidden', type=int, default=32)
    p.add_argument('--mlp-layers', type=int, default=1)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers',      type=int, default=0,
                   help='cap for smoke runs. 0 = use all.')
    p.add_argument('--n-workers',     type=int, default=mp.cpu_count())
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(h) for h in args.horizons.split(','))

    from factor import (
        IndicatorGridConfig, make_indicator_backbone,
        train_scorer_horizon_walkforward,
    )

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'IndicatorGridConfig.feature_width() = {F}', flush=True)
    print(f'horizons = {horizons}  (h_min={min(horizons)}, K={len(horizons)})',
          flush=True)
    print(f'walk-forward (fine bars): train={args.train_window_blocks}, '
          f'val={args.val_window_blocks}, step={args.step_window_blocks}',
          flush=True)

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

    backbone = make_indicator_backbone(ticker_data, cfg)
    print(f'\n--- training horizon-mixture head (n_steps={args.n_steps}, '
          f'lr={args.learning_rate}, wd={args.weight_decay}, '
          f'ent_w={args.entropy_weight}) ---', flush=True)
    t1 = time.perf_counter()
    res = train_scorer_horizon_walkforward(
        ticker_data, backbone,
        horizons=horizons,
        train_window_blocks=args.train_window_blocks,
        val_window_blocks=args.val_window_blocks,
        step_window_blocks=args.step_window_blocks,
        mlp_hidden=args.mlp_hidden,
        mlp_layers=args.mlp_layers,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        commission_bps=args.commission_bps,
        temperature=args.temperature,
    )
    print(f'wall: {time.perf_counter() - t1:.1f}s', flush=True)

    # Per-window table.
    print('\n=== per-window results ===', flush=True)
    header = f'{"win":>3}  {"endog":>7}  '
    for h in horizons:
        header += f'{"fix"+str(h):>7}  '
    header += f'{"rand":>7}  {"ent":>5}  ' + '  '.join(
        f'h{h}cnt' for h in horizons)
    print(header, flush=True)
    for w in res.windows:
        row = f'{w.window_idx:>3}  {w.val_endog_sharpe:+7.3f}  '
        for h in horizons:
            row += f'{w.val_fixed_sharpes[h]:+7.3f}  '
        row += f'{w.val_random_sharpe:+7.3f}  {w.val_pi_entropy_mean:>5.2f}  '
        row += '  '.join(f'{w.val_pi_argmax_counts[h]:>5d}' for h in horizons)
        print(row, flush=True)

    # Aggregates + null check.
    endog_mean = res.mean_val_endog_sharpe
    random_mean = res.mean_val_random_sharpe
    h_best, best_fixed_mean = res.best_fixed_horizon
    delta_best = endog_mean - best_fixed_mean
    delta_rand = endog_mean - random_mean

    print('\n=== summary ===', flush=True)
    print(f'  mean endog Sharpe       = {endog_mean:+.3f}', flush=True)
    for h in horizons:
        marker = '  <-- best fixed' if h == h_best else ''
        print(f'  mean fixed-h={h:<3} Sharpe = {res.mean_fixed_sharpe(h):+.3f}{marker}',
              flush=True)
    print(f'  mean random-π Sharpe    = {random_mean:+.3f}', flush=True)
    print(f'  delta vs best fixed     = {delta_best:+.3f}   '
          f'(success threshold ≥ +0.10)', flush=True)
    print(f'  delta vs random-π       = {delta_rand:+.3f}   '
          f'(success threshold ≥ 0)', flush=True)

    # Null N1: π collapse — argmax-bin entropy across windows.
    all_counts = np.zeros(len(horizons), dtype=np.int64)
    total_bars = 0
    for w in res.windows:
        for k, h in enumerate(horizons):
            all_counts[k] += w.val_pi_argmax_counts[h]
        total_bars += sum(w.val_pi_argmax_counts.values())
    pi_global = all_counts / max(total_bars, 1)
    print(f'  argmax bin shares       = '
          f'{ {h: f"{pi_global[k]:.2f}" for k, h in enumerate(horizons)} }',
          flush=True)
    collapse_max = float(np.max(pi_global))
    print(f'  worst-bin share         = {collapse_max:.2f}   '
          f'(null N1 fails if > 0.90)', flush=True)

    # Verdict suggestions.
    verdict_n1 = collapse_max <= 0.90
    verdict_n2 = endog_mean > res.mean_fixed_sharpe(max(horizons))
    verdict_n4 = delta_rand > 0.0
    verdict_pass = (delta_best >= 0.10) and verdict_n1 and verdict_n2 and verdict_n4

    print('\n=== null-rejection checks ===', flush=True)
    print(f'  N1 (no π collapse)     : {"PASS" if verdict_n1 else "FAIL"}',
          flush=True)
    print(f'  N2 (beats fixed h_max) : {"PASS" if verdict_n2 else "FAIL"}',
          flush=True)
    print(f'  N3 (beats best fixed)  : {"PASS" if delta_best >= 0.10 else "FAIL"} '
          f'(delta {delta_best:+.3f})', flush=True)
    print(f'  N4 (beats random-π)    : {"PASS" if verdict_n4 else "FAIL"}',
          flush=True)
    print(f'  Overall verdict        : '
          f'{"confirmed-OOS" if verdict_pass else "partial-OOS or null"}',
          flush=True)

    # Persist windows + summary.
    blob: dict[str, np.ndarray] = {
        'window_idx':           np.array([w.window_idx for w in res.windows]),
        'val_endog_sharpe':     np.array([w.val_endog_sharpe for w in res.windows]),
        'val_random_sharpe':    np.array([w.val_random_sharpe for w in res.windows]),
        'val_endog_mean_holding': np.array(
            [w.val_endog_mean_holding for w in res.windows]),
        'val_endog_n_rebals':   np.array(
            [w.val_endog_n_rebals for w in res.windows]),
        'val_pi_entropy_mean':  np.array(
            [w.val_pi_entropy_mean for w in res.windows]),
    }
    for h in horizons:
        blob[f'val_fixed_sharpe_h{h}'] = np.array(
            [w.val_fixed_sharpes[h] for w in res.windows])
        blob[f'val_argmax_count_h{h}'] = np.array(
            [w.val_pi_argmax_counts[h] for w in res.windows])
    blob['_meta'] = np.array(json.dumps({
        'horizons':             list(horizons),
        'mean_endog_sharpe':    endog_mean,
        'mean_random_sharpe':   random_mean,
        'best_fixed_horizon':   h_best,
        'best_fixed_sharpe':    best_fixed_mean,
        'delta_vs_best_fixed':  delta_best,
        'delta_vs_random':      delta_rand,
        'pi_argmax_global_shares': {
            h: float(pi_global[k]) for k, h in enumerate(horizons)},
        'n_windows':            res.n_windows,
        'universe_size':        len(ticker_data),
        'feature_width':        F,
        'mlp_hidden':           args.mlp_hidden,
        'mlp_layers':           args.mlp_layers,
        'n_steps':              args.n_steps,
        'learning_rate':        args.learning_rate,
        'weight_decay':         args.weight_decay,
        'entropy_weight':       args.entropy_weight,
        'commission_bps':       args.commission_bps,
        'verdict_n1':           verdict_n1,
        'verdict_n2':           verdict_n2,
        'verdict_n3_delta':     delta_best >= 0.10,
        'verdict_n4':           verdict_n4,
        'verdict_pass':         verdict_pass,
    }))
    out_path = output / 'horizon-mixture-windows.npz'
    np.savez(out_path, **blob)
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
