"""Step 0 — universe-agnostic HRP walk-forward baseline.

First formal apples-to-apples eval of `lie.hrp.weights_hrp` (Lopez de
Prado 2016 Hierarchical Risk Parity) and the opt-in
`lie.symmetry_rank.gross_exposure_modulator` gate. Mirrors the scaffold
used by `apps/relational/scripts/velocity_universe_agnostic.py` and the
regime-app head agents so the row lines up.

Eval scaffold:
  - Universe: stooq_us_long (~312 names).
  - 6 walk-forward windows of 1260-train / 780-val / 780-step.
  - rebal_days=20, commission_bps=10.
  - HRP weights ALL names in the universe by design (no top_n) -- that
    is the deliberate diversification-ceiling test vs passive EW.
  - Baseline arm: linkage_method='single', lookback=120, no modulator.
  - Robustness grid (6 cells):
      linkage_method in {single, average, ward}
      x modulator in {off, on}.

Pre-registered verdict bar (locked, persisted in NPZ):
  confirmed-OOS: mean val alpha vs EW >= +0.20 Sharpe AND >=4/6 pos AND DSR-t > +1.5
  partial-OOS:   alpha >= +0.05 AND >=3/6 pos
  confirmed-null: alpha < +0.05 AND DSR-t < +1.0
  reversed-OOS:  alpha < -0.10
  diagnostic:    else

HRP is numpy + scipy linkage; no Modal needed.

Run from repo root:
    uv run python apps/lie/scripts/hrp_universe_agnostic.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from lie.hrp import weights_hrp
from lie.symmetry_rank import gross_exposure_modulator, trailing_effective_rank


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'

PRE_REGISTERED_BAR = (
    "confirmed-OOS: mean val alpha >= +0.20 AND >=4/6 pos AND DSR-t > +1.5; "
    "partial-OOS: alpha >= +0.05 AND >=3/6 pos; "
    "confirmed-null: alpha < +0.05 AND DSR-t < +1.0; "
    "reversed-OOS: alpha < -0.10; diagnostic: else"
)


def build_hrp_weights_panel(
    prices: pd.DataFrame,
    *,
    lookback: int,
    rebal_days: int,
    linkage_method: str,
    use_modulator: bool,
    modulator_floor: float = 0.25,
) -> pd.DataFrame:
    """Build a (T, N) DataFrame of HRP target weights.

    Weights are recomputed at every rebalance bar; non-rebal bars are
    held at the most-recent rebal weights (handled downstream by the
    `basket_daily_returns` step-and-hold pattern). For non-rebal bars
    we leave the row zero; the daily-returns helper expands rebal rows
    into the held panel.
    """
    n_t, n_n = prices.shape
    out = np.zeros((n_t, n_n), dtype=np.float64)
    px = prices.to_numpy(dtype=np.float64)
    for t in range(lookback, n_t):
        if (t - lookback) % rebal_days != 0:
            continue
        sub = px[t - lookback:t + 1]  # need lookback+1 bars
        try:
            w = weights_hrp(sub, lookback=lookback, linkage_method=linkage_method)
        except Exception:
            continue
        if use_modulator:
            eff = trailing_effective_rank(sub, lookback=lookback)
            n_active = int((w > 0).sum())
            if n_active > 0:
                scalar = gross_exposure_modulator(
                    eff, n_assets=n_active, floor=modulator_floor)
                w = w * scalar
        out[t] = w
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def basket_daily_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    rebal_days: int,
    commission_frac: float,
    lookback: int,
) -> pd.Series:
    """Daily returns from a sparse weights panel (non-zero rows are
    rebalance bars; weights held in between). Costs charged at each
    rebal row as 0.5 * L1(delta_w); first rebal pays full L1."""
    common = prices.index[lookback:]
    px = prices.loc[common]
    w_arr = weights.loc[common].to_numpy()
    daily_ret = px.pct_change().fillna(0.0).values
    n_t, n_n = w_arr.shape
    # rebal rows = rows with any non-zero weight
    rebal_idxs = [i for i in range(n_t) if np.any(w_arr[i] != 0)]
    held = np.zeros((n_t, n_n))
    for k, ridx in enumerate(rebal_idxs):
        end = rebal_idxs[k + 1] if k + 1 < len(rebal_idxs) else n_t
        held[ridx:end] = w_arr[ridx]
    held_lag = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
    port_ret = (held_lag * daily_ret).sum(axis=1)
    cost = np.zeros(n_t)
    for k, ridx in enumerate(rebal_idxs):
        if k == 0:
            cost[ridx] = commission_frac * np.abs(w_arr[ridx]).sum()
        else:
            prev = rebal_idxs[k - 1]
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
    w_panel = np.zeros((n_t, n_n))
    rebal_idxs = list(range(0, n_t, rebal_days))
    for ridx in rebal_idxs:
        v = valid[ridx]
        s = v.sum()
        if s > 0:
            w_panel[ridx] = v / s
    for k, ridx in enumerate(rebal_idxs):
        end = rebal_idxs[k + 1] if k + 1 < len(rebal_idxs) else n_t
        held[ridx:end] = w_panel[ridx]
    held_lag = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
    port_ret = (held_lag * daily_ret).sum(axis=1)
    cost = np.zeros(n_t)
    for k, ridx in enumerate(rebal_idxs):
        if k == 0:
            cost[ridx] = commission_frac * np.abs(w_panel[ridx]).sum()
        else:
            prev = rebal_idxs[k - 1]
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
    rebal_days: int,
    commission_frac: float,
    linkage_method: str,
    use_modulator: bool,
    windows: list[tuple[int, int, int]],
) -> dict:
    w = build_hrp_weights_panel(
        prices, lookback=lookback, rebal_days=rebal_days,
        linkage_method=linkage_method, use_modulator=use_modulator)
    daily = basket_daily_returns(
        w, prices, rebal_days, commission_frac, lookback=lookback)
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
            'hrp_sharpe': s_v,
            'ew_sharpe': s_e,
            'alpha_sharpe': s_v - s_e,
            'hrp_max_dd': max_drawdown(v_arr),
        })
        val_streams.append(v_arr)
        ew_streams.append(e_arr)
        val_date_arrs.append(np.asarray([np.datetime64(d) for d in common]))
    return {
        'per_window': per_window,
        'mean_alpha': float(np.mean([r['alpha_sharpe'] for r in per_window])),
        'pos_alpha_count': int(sum(r['alpha_sharpe'] > 0 for r in per_window)),
        'mean_hrp_sharpe': float(np.mean([r['hrp_sharpe'] for r in per_window])),
        'mean_ew_sharpe': float(np.mean([r['ew_sharpe'] for r in per_window])),
        'oos_hrp': np.concatenate(val_streams) if val_streams else np.array([]),
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
    p.add_argument('--max-tickers', type=int, default=0)
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
          f'{prices.index[0].date()} -> {prices.index[-1].date()} '
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

    print('\nBaseline arm: linkage=single, modulator=off')
    t0 = time.time()
    baseline = run_arm(
        prices, lookback=args.lookback, rebal_days=args.rebal_days,
        commission_frac=commission_frac,
        linkage_method='single', use_modulator=False, windows=windows)
    print(f'  ({time.time()-t0:.1f}s)')
    print(f'{"win":>3} {"val":>23} {"hrp_sh":>7} {"ew_sh":>7} '
          f'{"alpha":>7} {"dd":>7}')
    for r in baseline['per_window']:
        print(f'{r["window_idx"]:>3} {r["val_start"]}->{r["val_end"]} '
              f'{r["hrp_sharpe"]:>+7.3f} {r["ew_sharpe"]:>+7.3f} '
              f'{r["alpha_sharpe"]:>+7.3f} {r["hrp_max_dd"]:>+7.3f}')
    print(f'  mean hrp Sharpe = {baseline["mean_hrp_sharpe"]:+.3f}')
    print(f'  mean ew Sharpe  = {baseline["mean_ew_sharpe"]:+.3f}')
    print(f'  mean alpha      = {baseline["mean_alpha"]:+.3f} '
          f'({baseline["pos_alpha_count"]}/{len(windows)} positive)')

    grid_results = []
    if not args.skip_grid:
        print('\nRobustness grid: linkage x modulator = 6 cells')
        print(f'{"linkage":>10} {"mod":>5} {"mean_α":>8} {"pos":>4} {"hrp_sh":>7}')
        for linkage in ('single', 'average', 'ward'):
            for use_mod in (False, True):
                t0 = time.time()
                try:
                    arm = run_arm(
                        prices, lookback=args.lookback,
                        rebal_days=args.rebal_days,
                        commission_frac=commission_frac,
                        linkage_method=linkage, use_modulator=use_mod,
                        windows=windows)
                except Exception as e:
                    print(f'  [skip] {linkage} mod={use_mod}: {e}')
                    continue
                dt = time.time() - t0
                grid_results.append({
                    'linkage_method': linkage,
                    'use_modulator': use_mod,
                    'mean_alpha': arm['mean_alpha'],
                    'mean_hrp_sharpe': arm['mean_hrp_sharpe'],
                    'pos_alpha_count': arm['pos_alpha_count'],
                    'wall_s': dt,
                })
                print(f'{linkage:>10} {str(use_mod):>5} '
                      f'{arm["mean_alpha"]:>+8.3f} '
                      f'{arm["pos_alpha_count"]:>4} '
                      f'{arm["mean_hrp_sharpe"]:>+7.3f}  [{dt:.0f}s]')

        if grid_results:
            best_cell = max(grid_results, key=lambda r: r['mean_alpha'])
            print(f'\nBest-alpha cell: {best_cell}')
            alphas = [r['mean_alpha'] for r in grid_results]
            print(f'Grid alpha spread: min={min(alphas):+.3f}, '
                  f'max={max(alphas):+.3f}, '
                  f'median={float(np.median(alphas)):+.3f}')

    headline = baseline
    oos_hrp = headline['oos_hrp']
    oos_ew = headline['oos_ew']
    n_obs = oos_hrp.size
    sr_diff_periods = (
        oos_hrp.mean() / (oos_hrp.std() + 1e-12) -
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

    npz_path = output / 'lie-hrp-universe-agnostic-walkforward.npz'
    np.savez(
        npz_path,
        oos_block_returns=oos_hrp,
        oos_ew_returns=oos_ew,
        oos_dates=headline['oos_dates'],
        periods_per_year=np.float64(252.0),
        pre_registered_bar=np.str_(PRE_REGISTERED_BAR),
        universe_label=np.str_('stooq_us_long'),
        windowing_label=np.str_('6w-1260tr-780val-780step'),
        rebal_days=np.int64(args.rebal_days),
        commission_bps=np.float64(args.commission_bps),
        linkage_method=np.str_('single'),
        use_modulator=np.bool_(False),
        lookback=np.int64(args.lookback),
        mean_alpha_sharpe=np.float64(headline['mean_alpha']),
        pos_alpha_count=np.int64(headline['pos_alpha_count']),
        mean_hrp_sharpe=np.float64(headline['mean_hrp_sharpe']),
        mean_ew_sharpe=np.float64(headline['mean_ew_sharpe']),
        dsr_t=np.float64(dsr_t),
        verdict=np.str_(verdict),
        n_trials=np.int64(n_trials),
    )
    print(f'-> {npz_path}')

    json_path = output / 'lie-hrp-universe-agnostic-walkforward.json'
    json_path.write_text(json.dumps({
        'universe': 'stooq_us_long',
        'windowing': '6w-1260tr-780val-780step',
        'pre_registered_bar': PRE_REGISTERED_BAR,
        'baseline_params': {
            'linkage_method': 'single', 'use_modulator': False,
            'lookback': args.lookback,
        },
        'baseline': {
            'mean_alpha': headline['mean_alpha'],
            'pos_alpha_count': headline['pos_alpha_count'],
            'mean_hrp_sharpe': headline['mean_hrp_sharpe'],
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
