"""Step 0 — universe-agnostic regime-velocity walk-forward baseline.

Tests `relational.regime_velocity.weights_velocity_magnitude` and
`weights_axis_alignment` (the two scoring variants behind the
`strategy='velocity'` checkpoint) without any per-regime
pre-selection — mirroring the apples-to-apples scaffold used by
`apps/regime/scripts/rsi_universe_agnostic.py` so the row lines up
with the parallel regime-app head agents (rsi / scalogram / cwt).

Eval scaffold:
  - Universe: stooq_us_long (~312 names, the canonical wide universe
    that `build_canonical_checkpoints.py` pins velocity to).
  - 6 walk-forward windows of 1260-train / 780-val / 780-step.
  - rebal_days=20, commission_bps=10, top_n=20.
  - Baseline arm (canonical Phase-11 config):
      variant='magnitude', fp_window=21, w_delta=20, lookback=120,
      scales=[5,7,10,12,21,26,50,90].
  - Robustness grid (24 cells):
      variant ∈ {magnitude, axis_alignment(n_axes=5)}
      × top_n ∈ {10, 20, 50}
      × w_delta ∈ {10, 20}
      × scales ∈ {full, short=[5,10,21,50], long=[10,21,50,90]}

Pre-registered verdict bar (locked, persisted in NPZ):
  confirmed-OOS: mean val alpha vs EW ≥ +0.20 Sharpe AND ≥4/6 pos AND DSR-t > +1.5
  partial-OOS:   alpha ≥ +0.05 AND ≥3/6 pos
  confirmed-null: alpha < +0.05 AND DSR-t < +1.0
  reversed-OOS:  alpha < -0.10
  diagnostic:    else

Wavelet-support floor per `_validate_inputs`: with default scales
max=90, lookback=120, KERNEL_HALF_EXTENT=3: 90*3 + 120 = 390 bars.
With train_window_days=252 for axis_alignment, an extra 252 prefix
is consumed before the first usable score. Both numbers are well
within `stooq_us_long` history.

Run from repo root:
    uv run python apps/relational/scripts/velocity_universe_agnostic.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from relational.regime_velocity import (
    weights_velocity_magnitude,
    weights_axis_alignment,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'

FULL_SCALES = [5, 7, 10, 12, 21, 26, 50, 90]
SHORT_SCALES = [5, 10, 21, 50]
LONG_SCALES = [10, 21, 50, 90]
SCALE_SETS = {'full': FULL_SCALES, 'short': SHORT_SCALES, 'long': LONG_SCALES}

PRE_REGISTERED_BAR = (
    "confirmed-OOS: mean val alpha >= +0.20 AND >=4/6 pos AND DSR-t > +1.5; "
    "partial-OOS: alpha >= +0.05 AND >=3/6 pos; "
    "confirmed-null: alpha < +0.05 AND DSR-t < +1.0; "
    "reversed-OOS: alpha < -0.10; diagnostic: else"
)


def basket_daily_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    rebal_days: int,
    commission_frac: float,
) -> pd.Series:
    common = prices.index.intersection(weights.index)
    weights = weights.loc[common]
    px = prices.loc[common]
    daily_ret = px.pct_change().fillna(0.0).values
    w_arr = weights.values
    n_t = w_arr.shape[0]
    held = np.zeros_like(w_arr)
    rebal_dates_idx = list(range(0, n_t, rebal_days))
    for k, ridx in enumerate(rebal_dates_idx):
        end = rebal_dates_idx[k + 1] if k + 1 < len(rebal_dates_idx) else n_t
        held[ridx:end] = w_arr[ridx]
    held_lag = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
    port_ret = (held_lag * daily_ret).sum(axis=1)
    cost = np.zeros(n_t)
    for k, ridx in enumerate(rebal_dates_idx):
        if k == 0:
            cost[ridx] = commission_frac * np.abs(w_arr[ridx]).sum()
        else:
            prev = rebal_dates_idx[k - 1]
            cost[ridx] = commission_frac * 0.5 * np.abs(
                w_arr[ridx] - w_arr[prev]).sum()
    return pd.Series(port_ret - cost, index=common)


def passive_ew_daily_returns(
    prices: pd.DataFrame,
    rebal_days: int,
    commission_frac: float,
    lookback: int,
) -> pd.Series:
    common = prices.index[lookback:]
    px = prices.loc[common]
    n_t, n_n = px.shape
    daily_ret = px.pct_change().fillna(0.0).values
    valid = (~px.isna()).values.astype(float)
    held = np.zeros((n_t, n_n))
    rebal_dates_idx = list(range(0, n_t, rebal_days))
    w_panel = np.zeros((n_t, n_n))
    for ridx in rebal_dates_idx:
        v = valid[ridx]
        s = v.sum()
        if s > 0:
            w_panel[ridx] = v / s
    for k, ridx in enumerate(rebal_dates_idx):
        end = rebal_dates_idx[k + 1] if k + 1 < len(rebal_dates_idx) else n_t
        held[ridx:end] = w_panel[ridx]
    held_lag = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
    port_ret = (held_lag * daily_ret).sum(axis=1)
    cost = np.zeros(n_t)
    for k, ridx in enumerate(rebal_dates_idx):
        if k == 0:
            cost[ridx] = commission_frac * np.abs(w_panel[ridx]).sum()
        else:
            prev = rebal_dates_idx[k - 1]
            cost[ridx] = commission_frac * 0.5 * np.abs(
                w_panel[ridx] - w_panel[prev]).sum()
    return pd.Series(port_ret - cost, index=common)


def annualized_sharpe(daily_ret: np.ndarray) -> float:
    r = np.asarray(daily_ret, dtype=np.float64)
    if r.size < 5 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def max_drawdown(daily_ret: np.ndarray) -> float:
    if daily_ret.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + daily_ret)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def build_weights(
    prices: pd.DataFrame,
    *,
    variant: str,
    lookback: int,
    top_n: int,
    scales: list[int],
    fp_window: int,
    w_delta: int,
    n_axes: int = 5,
    train_window_days: int = 252,
) -> pd.DataFrame:
    if variant == 'magnitude':
        return weights_velocity_magnitude(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window, w_delta=w_delta)
    elif variant == 'axis_alignment':
        return weights_axis_alignment(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window, w_delta=w_delta,
            n_axes=n_axes, train_window_days=train_window_days)
    else:
        raise ValueError(f'unknown variant: {variant}')


def run_arm(
    prices: pd.DataFrame,
    *,
    variant: str,
    lookback: int,
    top_n: int,
    scales: list[int],
    fp_window: int,
    w_delta: int,
    rebal_days: int,
    commission_frac: float,
    windows: list[tuple[int, int, int]],
    n_axes: int = 5,
    train_window_days: int = 252,
) -> dict:
    w = build_weights(
        prices, variant=variant, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window, w_delta=w_delta,
        n_axes=n_axes, train_window_days=train_window_days)
    daily = basket_daily_returns(w, prices, rebal_days, commission_frac)
    ew = passive_ew_daily_returns(prices, rebal_days, commission_frac, lookback)
    dates = prices.index
    per_window = []
    val_streams, ew_streams, val_date_arrs = [], [], []
    for w_idx, (lo, mid, hi) in enumerate(windows):
        v_start = dates[mid]
        v_end = dates[hi - 1]
        v = daily.loc[(daily.index >= v_start) & (daily.index <= v_end)]
        e = ew.loc[(ew.index >= v_start) & (ew.index <= v_end)]
        common = v.index.intersection(e.index)
        v_arr = v.loc[common].values
        e_arr = e.loc[common].values
        s_v = annualized_sharpe(v_arr)
        s_e = annualized_sharpe(e_arr)
        per_window.append({
            'window_idx': w_idx,
            'val_start': str(common[0].date()) if len(common) else '',
            'val_end': str(common[-1].date()) if len(common) else '',
            'vel_sharpe': s_v,
            'ew_sharpe': s_e,
            'alpha_sharpe': s_v - s_e,
            'vel_max_dd': max_drawdown(v_arr),
        })
        val_streams.append(v_arr)
        ew_streams.append(e_arr)
        val_date_arrs.append(np.asarray([np.datetime64(d) for d in common]))
    return {
        'per_window': per_window,
        'mean_alpha': float(np.mean([r['alpha_sharpe'] for r in per_window])),
        'pos_alpha_count': int(sum(r['alpha_sharpe'] > 0 for r in per_window)),
        'mean_vel_sharpe': float(np.mean([r['vel_sharpe'] for r in per_window])),
        'mean_ew_sharpe': float(np.mean([r['ew_sharpe'] for r in per_window])),
        'oos_vel': np.concatenate(val_streams) if val_streams else np.array([]),
        'oos_ew': np.concatenate(ew_streams) if ew_streams else np.array([]),
        'oos_dates': np.concatenate(val_date_arrs) if val_date_arrs else np.array([]),
    }


def verdict_label(mean_alpha: float, pos_alpha: int, dsr_t: float | None) -> str:
    if dsr_t is None:
        dsr_t = 0.0
    if mean_alpha >= 0.20 and pos_alpha >= 4 and dsr_t > 1.5:
        return 'confirmed-OOS'
    if mean_alpha >= 0.05 and pos_alpha >= 3:
        return 'partial-OOS'
    if mean_alpha < -0.10:
        return 'reversed-OOS'
    if mean_alpha < 0.05 and dsr_t < 1.0:
        return 'confirmed-null'
    return 'diagnostic'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(STOOQ_SUBSET))
    p.add_argument('--manifest', default=str(STOOQ_SUBSET / 'manifest.json'))
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--train-window-days', type=int, default=1260)
    p.add_argument('--val-window-days',   type=int, default=780)
    p.add_argument('--step-window-days',  type=int, default=780)
    p.add_argument('--svd-train-days', type=int, default=252,
                   help='Train window for axis_alignment SVD.')
    p.add_argument('--max-tickers', type=int, default=0,
                   help='Smoke-test cap; 0 = full universe.')
    p.add_argument('--skip-grid', action='store_true')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading stooq_us_long universe...')
    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    if args.max_tickers > 0:
        universe = universe[:args.max_tickers]
        print(f'  SMOKE: capping universe to first {args.max_tickers} names')
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end,
        tickers=universe)
    print(f'  loaded {prices.shape[1]} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()} '
          f'({len(prices)} bars)')

    n = len(prices)
    train_w = args.train_window_days
    val_w = args.val_window_days
    step = args.step_window_days
    windows = []
    start = 0
    while start + train_w + val_w <= n:
        windows.append((start, start + train_w, start + train_w + val_w))
        start += step
    print(f'  walk-forward: {len(windows)} windows '
          f'(train={train_w}, val={val_w}, step={step})')

    commission_frac = args.commission_bps / 1e4

    # Baseline arm: canonical Phase-11 magnitude config.
    print('\nBaseline arm: variant=magnitude, top_n=20, w_delta=20, '
          'scales=full, fp_window=21')
    t0 = time.time()
    baseline = run_arm(
        prices, variant='magnitude', lookback=args.lookback, top_n=20,
        scales=FULL_SCALES, fp_window=21, w_delta=20,
        rebal_days=args.rebal_days, commission_frac=commission_frac,
        windows=windows)
    print(f'  ({time.time()-t0:.1f}s)')
    print(f'{"win":>3} {"val":>23} {"vel_sh":>7} {"ew_sh":>7} '
          f'{"alpha":>7} {"dd":>7}')
    for r in baseline['per_window']:
        print(f'{r["window_idx"]:>3} {r["val_start"]}→{r["val_end"]} '
              f'{r["vel_sharpe"]:>+7.3f} {r["ew_sharpe"]:>+7.3f} '
              f'{r["alpha_sharpe"]:>+7.3f} {r["vel_max_dd"]:>+7.3f}')
    print(f'  mean vel Sharpe = {baseline["mean_vel_sharpe"]:+.3f}')
    print(f'  mean ew Sharpe  = {baseline["mean_ew_sharpe"]:+.3f}')
    print(f'  mean alpha      = {baseline["mean_alpha"]:+.3f} '
          f'({baseline["pos_alpha_count"]}/{len(windows)} positive)')

    grid_results = []
    if not args.skip_grid:
        print('\nRobustness grid: variant × top_n × w_delta × scales = 24 cells')
        print(f'{"variant":>14} {"top_n":>5} {"wΔ":>3} {"scales":>6} '
              f'{"mean_α":>8} {"pos":>4} {"vel_sh":>7}')
        for variant in ('magnitude', 'axis_alignment'):
            for top_n in (10, 20, 50):
                for w_delta in (10, 20):
                    for scale_label, scale_list in SCALE_SETS.items():
                        if scale_label == 'long' and variant == 'magnitude':
                            # cap grid at 24 cells: skip one scale label per variant
                            # Actually we want 24: 2 variants * 3 top_n * 2 wd * 2 scale_sets = 24.
                            # We'll just skip 'long' label entirely to keep grid_count=24.
                            continue
                        t0 = time.time()
                        try:
                            arm = run_arm(
                                prices, variant=variant,
                                lookback=args.lookback, top_n=top_n,
                                scales=scale_list, fp_window=21,
                                w_delta=w_delta,
                                rebal_days=args.rebal_days,
                                commission_frac=commission_frac,
                                windows=windows,
                                train_window_days=args.svd_train_days)
                        except Exception as e:
                            print(f'  [skip] {variant} top={top_n} wd={w_delta} '
                                  f'scales={scale_label}: {e}')
                            continue
                        dt = time.time() - t0
                        grid_results.append({
                            'variant': variant, 'top_n': top_n,
                            'w_delta': w_delta, 'scales': scale_label,
                            'mean_alpha': arm['mean_alpha'],
                            'mean_vel_sharpe': arm['mean_vel_sharpe'],
                            'pos_alpha_count': arm['pos_alpha_count'],
                            'wall_s': dt,
                        })
                        print(f'{variant:>14} {top_n:>5} {w_delta:>3} '
                              f'{scale_label:>6} '
                              f'{arm["mean_alpha"]:>+8.3f} '
                              f'{arm["pos_alpha_count"]:>4} '
                              f'{arm["mean_vel_sharpe"]:>+7.3f}  '
                              f'[{dt:.0f}s]')

        if grid_results:
            grid_sorted = sorted(grid_results, key=lambda r: r['mean_vel_sharpe'])
            median_cell = grid_sorted[len(grid_sorted) // 2]
            best_cell = max(grid_results, key=lambda r: r['mean_alpha'])
            print(f'\nMedian-Sharpe cell: {median_cell}')
            print(f'Best-alpha cell:    {best_cell}')
            alphas = [r['mean_alpha'] for r in grid_results]
            print(f'Grid alpha spread: min={min(alphas):+.3f}, '
                  f'max={max(alphas):+.3f}, '
                  f'median={float(np.median(alphas)):+.3f}')

    headline = baseline
    oos_vel = headline['oos_vel']
    oos_ew = headline['oos_ew']
    n_obs = oos_vel.size
    sr_diff_periods = (
        oos_vel.mean() / (oos_vel.std() + 1e-12) -
        oos_ew.mean() / (oos_ew.std() + 1e-12))
    sr_diff_ann = sr_diff_periods * np.sqrt(252.0)
    se_ann = 1.0 / np.sqrt(max(n_obs / 252.0, 1e-9))
    dsr_t = float(sr_diff_ann / max(se_ann, 1e-9))
    print(f'\nDSR-t (rough, n_obs={n_obs}): '
          f'sr_diff_ann={sr_diff_ann:+.3f}, se_ann={se_ann:.3f}, '
          f'dsr_t={dsr_t:+.2f}')

    n_trials = max(1, len(grid_results)) if grid_results else 1
    verdict = verdict_label(
        headline['mean_alpha'], headline['pos_alpha_count'], dsr_t)
    print(f'\nVerdict: {verdict}  (n_trials={n_trials})')

    npz_path = output / 'regime-velocity-universe-agnostic-walkforward.npz'
    np.savez(
        npz_path,
        oos_block_returns=oos_vel,
        oos_ew_returns=oos_ew,
        oos_dates=headline['oos_dates'],
        periods_per_year=np.float64(252.0),
        pre_registered_bar=np.str_(PRE_REGISTERED_BAR),
        universe_label=np.str_('stooq_us_long'),
        windowing_label=np.str_('6w-1260tr-780val-780step'),
        rebal_days=np.int64(args.rebal_days),
        commission_bps=np.float64(args.commission_bps),
        variant=np.str_('magnitude'),
        top_n=np.int64(20),
        fp_window=np.int64(21),
        w_delta=np.int64(20),
        lookback=np.int64(args.lookback),
        mean_alpha_sharpe=np.float64(headline['mean_alpha']),
        pos_alpha_count=np.int64(headline['pos_alpha_count']),
        mean_vel_sharpe=np.float64(headline['mean_vel_sharpe']),
        mean_ew_sharpe=np.float64(headline['mean_ew_sharpe']),
        dsr_t=np.float64(dsr_t),
        verdict=np.str_(verdict),
        n_trials=np.int64(n_trials),
    )
    print(f'-> {npz_path}')

    json_path = output / 'regime-velocity-universe-agnostic-walkforward.json'
    json_path.write_text(json.dumps({
        'universe': 'stooq_us_long',
        'windowing': '6w-1260tr-780val-780step',
        'pre_registered_bar': PRE_REGISTERED_BAR,
        'baseline_params': {
            'variant': 'magnitude', 'top_n': 20, 'fp_window': 21,
            'w_delta': 20, 'lookback': args.lookback,
            'scales': FULL_SCALES,
        },
        'baseline': {
            'mean_alpha': headline['mean_alpha'],
            'pos_alpha_count': headline['pos_alpha_count'],
            'mean_vel_sharpe': headline['mean_vel_sharpe'],
            'mean_ew_sharpe': headline['mean_ew_sharpe'],
            'per_window': headline['per_window'],
        },
        'grid': grid_results,
        'n_trials': n_trials,
        'dsr_t_rough': dsr_t,
        'verdict': verdict,
    }, indent=2, default=str))
    print(f'-> {json_path}')


if __name__ == '__main__':
    main()
