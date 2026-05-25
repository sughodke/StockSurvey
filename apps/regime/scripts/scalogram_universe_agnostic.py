"""Step 0 — universe-agnostic scalogram walk-forward baseline for the regime app.

Tests `regime.trainer.weights_scalogram` (top-N lowest-score names by
direction − momentum × coherence over the causal CWT cube) without any
per-regime pre-selection. Mirrors `rsi_universe_agnostic.py` so the
verdicts are directly comparable.

Eval scaffold:
  - Universe: stooq_us_long (312 names).
  - 6 windows of 1260-train / 780-val / 780-step (daily bars).
  - rebal_days=20, commission_bps=10, top_n=20 baseline.
  - Baseline scales = [5, 21, 90] (DEFAULT_SCALES in trainer.py).
  - Hyperparameter grid (12 cells):
      scales subset ∈ {SHORT+MID [5,21], MID+LONG [21,90], DEFAULT [5,21,90], ALL [5,21,63,126]}
      top_n ∈ {10, 20, 50}
      (lookback=252, n_tail=21 fixed — n_tail < lookback//2 by Optuna constraint)
  - Baseline = passive EW on the same universe.

Pre-registered verdict bar (LOCKED before running, persisted in NPZ):
  confirmed-OOS: mean val alpha ≥ +0.20 Sharpe AND ≥4/6 positive AND DSR-t > +1.5
  partial-OOS:   mean val alpha ≥ +0.05 Sharpe AND ≥3/6 positive
  confirmed-null: alpha < +0.05 Sharpe AND DSR-t < +1.0
  reversed-OOS:  mean val alpha < -0.10 Sharpe
  diagnostic:    anything else

Run from repo root:
    uv run python apps/regime/scripts/scalogram_universe_agnostic.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from regime.trainer import weights_scalogram


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


PRE_REGISTERED_BAR = (
    "confirmed-OOS: mean val alpha ≥ +0.20 AND ≥4/6 pos AND DSR-t > +1.5; "
    "partial-OOS: mean val alpha ≥ +0.05 AND ≥3/6 pos; "
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


def run_arm(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    rebal_days: int,
    commission_frac: float,
    windows: list[tuple[int, int, int]],
) -> dict:
    """Build scalogram weights ONCE for the entire panel, then carve into
    per-window val slices. weights_scalogram uses causal_cwt (causal by
    construction — no look-ahead) so global build + per-window slice is
    safe and saves wall time vs re-running per window.
    """
    w = weights_scalogram(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n, scales=scales)
    daily = basket_daily_returns(w, prices, rebal_days, commission_frac)
    ew = passive_ew_daily_returns(prices, rebal_days, commission_frac, lookback)
    dates = prices.index

    per_window = []
    val_streams = []
    ew_streams = []
    val_date_arrs = []
    for w_idx, (lo, mid, hi) in enumerate(windows):
        v_start = dates[mid]
        v_end = dates[hi - 1]
        val_ret = daily.loc[(daily.index >= v_start) & (daily.index <= v_end)]
        ew_ret = ew.loc[(ew.index >= v_start) & (ew.index <= v_end)]
        common = val_ret.index.intersection(ew_ret.index)
        val_ret = val_ret.loc[common].values
        ew_ret = ew_ret.loc[common].values
        s_scal = annualized_sharpe(val_ret)
        s_ew = annualized_sharpe(ew_ret)
        per_window.append({
            'window_idx': w_idx,
            'val_start': str(common[0].date()) if len(common) else '',
            'val_end': str(common[-1].date()) if len(common) else '',
            'scal_sharpe': s_scal,
            'ew_sharpe': s_ew,
            'alpha_sharpe': s_scal - s_ew,
            'scal_max_dd': max_drawdown(val_ret),
        })
        val_streams.append(val_ret)
        ew_streams.append(ew_ret)
        val_date_arrs.append(
            np.asarray([np.datetime64(d) for d in common]))

    mean_alpha = float(np.mean([r['alpha_sharpe'] for r in per_window]))
    pos_alpha = int(sum(r['alpha_sharpe'] > 0 for r in per_window))
    mean_scal_sh = float(np.mean([r['scal_sharpe'] for r in per_window]))
    mean_ew_sh = float(np.mean([r['ew_sharpe'] for r in per_window]))
    return {
        'per_window': per_window,
        'mean_alpha': mean_alpha,
        'pos_alpha_count': pos_alpha,
        'mean_scal_sharpe': mean_scal_sh,
        'mean_ew_sharpe': mean_ew_sh,
        'oos_scal': np.concatenate(val_streams) if val_streams else np.array([]),
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


# Scale subsets searched in the robustness grid. DEFAULT_SCALES is the
# fallback when all 3 Optuna boolean flags are False (i.e. the canonical
# pre-Optuna default in trainer.py). The other three subsets sample
# distinct points in scale-space without combinatorially exploding.
SCALE_SUBSETS = {
    'short-mid':   [5, 21],
    'mid-long':    [21, 90],
    'default':     [5, 21, 90],
    'all-spread':  [5, 21, 63, 126],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default=str(STOOQ_SUBSET / 'manifest.json'))
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--lookback', type=int, default=252)
    p.add_argument('--n-tail', type=int, default=21)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--train-window-days', type=int, default=1260)
    p.add_argument('--val-window-days',   type=int, default=780)
    p.add_argument('--step-window-days',  type=int, default=780)
    p.add_argument('--max-tickers', type=int, default=None,
                   help='Smoke test: cap universe size (default uses all).')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke mode: --max-tickers 30 and only the baseline arm.')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.smoke and args.max_tickers is None:
        args.max_tickers = 30

    print('Loading stooq_us_long universe...')
    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    if args.max_tickers:
        universe = universe[: args.max_tickers]
        print(f'  (smoke) restricted to {len(universe)} tickers')
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

    # Baseline arm: DEFAULT_SCALES [5,21,90], top_n=20, n_tail=21
    baseline_scales = SCALE_SUBSETS['default']
    print(f'\nBaseline arm: scales={baseline_scales}, top_n=20, '
          f'n_tail={args.n_tail}')
    baseline = run_arm(
        prices, lookback=args.lookback, n_tail=args.n_tail, top_n=20,
        scales=baseline_scales,
        rebal_days=args.rebal_days, commission_frac=commission_frac,
        windows=windows)
    print(f'{"win":>3} {"val":>23} {"scal":>7} {"ew":>7} '
          f'{"alpha":>7} {"dd":>7}')
    for r in baseline['per_window']:
        print(f'{r["window_idx"]:>3} {r["val_start"]}→{r["val_end"]} '
              f'{r["scal_sharpe"]:>+7.3f} {r["ew_sharpe"]:>+7.3f} '
              f'{r["alpha_sharpe"]:>+7.3f} {r["scal_max_dd"]:>+7.3f}')
    print(f'  mean scal Sharpe = {baseline["mean_scal_sharpe"]:+.3f}')
    print(f'  mean ew Sharpe   = {baseline["mean_ew_sharpe"]:+.3f}')
    print(f'  mean alpha       = {baseline["mean_alpha"]:+.3f} '
          f'({baseline["pos_alpha_count"]}/{len(windows)} positive)')

    grid_results: list[dict] = []
    if args.smoke:
        print('\n(smoke mode: skipping robustness grid)')
    else:
        print('\nRobustness grid (4 scale subsets × top_n ∈ {10,20,50}):')
        print(f'{"scales":>14} {"top_n":>5} {"mean_α":>8} '
              f'{"pos":>4} {"scal_sh":>8}')
        for sname, scales in SCALE_SUBSETS.items():
            for top_n in (10, 20, 50):
                arm = run_arm(
                    prices, lookback=args.lookback, n_tail=args.n_tail,
                    top_n=top_n, scales=scales,
                    rebal_days=args.rebal_days,
                    commission_frac=commission_frac,
                    windows=windows)
                grid_results.append({
                    'scales_name': sname, 'scales': scales, 'top_n': top_n,
                    'mean_alpha': arm['mean_alpha'],
                    'mean_scal_sharpe': arm['mean_scal_sharpe'],
                    'pos_alpha_count': arm['pos_alpha_count'],
                })
                print(f'{sname:>14} {top_n:>5} '
                      f'{arm["mean_alpha"]:>+8.3f} '
                      f'{arm["pos_alpha_count"]:>4} '
                      f'{arm["mean_scal_sharpe"]:>+8.3f}')

        grid_results_sorted = sorted(
            grid_results, key=lambda r: r['mean_scal_sharpe'])
        median_cell = grid_results_sorted[len(grid_results_sorted) // 2]
        print(f'\nMedian-Sharpe cell: scales={median_cell["scales_name"]}, '
              f'top_n={median_cell["top_n"]} → '
              f'scal_sh={median_cell["mean_scal_sharpe"]:+.3f}, '
              f'alpha={median_cell["mean_alpha"]:+.3f}')
        alphas = [r['mean_alpha'] for r in grid_results]
        print(f'Grid alpha spread: min={min(alphas):+.3f}, '
              f'max={max(alphas):+.3f}, '
              f'median={float(np.median(alphas)):+.3f}')

    # Headline = baseline (pre-registered canonical config).
    headline = baseline

    oos_scal = headline['oos_scal']
    oos_ew = headline['oos_ew']
    n_obs = oos_scal.size
    sr_diff_periods = (
        oos_scal.mean() / (oos_scal.std() + 1e-12) -
        oos_ew.mean() / (oos_ew.std() + 1e-12))
    sr_diff_ann = sr_diff_periods * np.sqrt(252.0)
    se_ann = 1.0 / np.sqrt(max(n_obs / 252.0, 1e-9))
    dsr_t = float(sr_diff_ann / max(se_ann, 1e-9))
    print(f'\nDSR-t (rough, n_obs={n_obs}): '
          f'sr_diff_ann={sr_diff_ann:+.3f}, se_ann={se_ann:.3f}, '
          f'dsr_t={dsr_t:+.2f}')

    verdict = verdict_label(
        headline['mean_alpha'], headline['pos_alpha_count'], dsr_t)
    print(f'\nVerdict: {verdict}')

    n_trials = len(grid_results) if grid_results else 1

    npz_path = output / 'scalogram-universe-agnostic-walkforward.npz'
    np.savez(
        npz_path,
        oos_block_returns=oos_scal,
        oos_ew_returns=oos_ew,
        oos_dates=headline['oos_dates'],
        periods_per_year=np.float64(252.0),
        pre_registered_bar=np.str_(PRE_REGISTERED_BAR),
        universe_label=np.str_('stooq_us_long'),
        windowing_label=np.str_('6w-1260tr-780val-780step'),
        rebal_days=np.int64(args.rebal_days),
        commission_bps=np.float64(args.commission_bps),
        scales=np.asarray(baseline_scales, dtype=np.int64),
        n_tail=np.int64(args.n_tail),
        top_n=np.int64(20),
        lookback=np.int64(args.lookback),
        mean_alpha_sharpe=np.float64(headline['mean_alpha']),
        pos_alpha_count=np.int64(headline['pos_alpha_count']),
        mean_scal_sharpe=np.float64(headline['mean_scal_sharpe']),
        mean_ew_sharpe=np.float64(headline['mean_ew_sharpe']),
        dsr_t=np.float64(dsr_t),
        verdict=np.str_(verdict),
        n_trials=np.int64(n_trials),
    )
    print(f'-> {npz_path}')

    json_path = output / 'scalogram-universe-agnostic-walkforward.json'
    json_path.write_text(json.dumps({
        'universe': 'stooq_us_long',
        'windowing': '6w-1260tr-780val-780step',
        'pre_registered_bar': PRE_REGISTERED_BAR,
        'baseline_params': {
            'scales': baseline_scales, 'n_tail': args.n_tail,
            'top_n': 20, 'lookback': args.lookback,
        },
        'baseline': {
            'mean_alpha': headline['mean_alpha'],
            'pos_alpha_count': headline['pos_alpha_count'],
            'mean_scal_sharpe': headline['mean_scal_sharpe'],
            'mean_ew_sharpe': headline['mean_ew_sharpe'],
            'per_window': headline['per_window'],
        },
        'grid': grid_results,
        'dsr_t_rough': dsr_t,
        'verdict': verdict,
        'n_trials': n_trials,
    }, indent=2))
    print(f'-> {json_path}')


if __name__ == '__main__':
    main()
