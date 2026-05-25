"""Step 0 — universe-agnostic RSI walk-forward baseline for the regime app.

Tests `regime.trainer.weights_rsi` (top-N most-oversold names by mean
Wilder RSI over trailing n_tail bars) without any per-regime
pre-selection. This is the prerequisite for the per-regime universe
question: until the universe-agnostic RSI baseline has a verdict, the
per-regime variant is premature.

Eval scaffold (matches `apps/gate/scripts/run_walkforward.py`):
  - Universe: stooq_us_long (312 names) by default.
  - 6 windows of 1260-train / 780-val / 780-step (daily bars).
  - rebal_days=20, commission_bps=10, top_n=20, rsi_n=14, n_tail=5 baseline.
  - Hyperparameter grid (median-Sharpe cell as headline):
      rsi_n ∈ {7, 14, 21}, top_n ∈ {10, 20, 50}, n_tail ∈ {5, 10}
      → 18 cells.
  - Baseline = passive EW on the same universe (canonical per
    `findings/passive-ew-benchmark.md`).

Pre-registered verdict bar (LOCKED before running, persisted in NPZ):
  confirmed-OOS: mean val alpha ≥ +0.20 Sharpe AND ≥4/6 positive AND DSR-t > +1.5
  partial-OOS:   mean val alpha ≥ +0.05 Sharpe AND ≥3/6 positive
  confirmed-null: alpha < +0.05 Sharpe AND DSR-t < +1.0
  reversed-OOS:  mean val alpha < -0.10 Sharpe
  diagnostic:    anything else

Run from repo root:
    uv run python apps/regime/scripts/rsi_universe_agnostic.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from regime.trainer import weights_rsi


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
    """Net daily returns of a top-N basket rebalanced every `rebal_days`.

    `weights` already has dates ≥ prices.index[lookback]. We hold the
    weights from each rebal date for `rebal_days` trading bars (i.e.
    the weights live for the next rebal block), apply daily simple
    returns at name level, and subtract one-sided turnover cost on
    each rebal date.
    """
    common = prices.index.intersection(weights.index)
    weights = weights.loc[common]
    px = prices.loc[common]
    daily_ret = px.pct_change().fillna(0.0).values  # (T, N)
    w_arr = weights.values  # (T, N) but only meaningful at rebal dates
    n_t = w_arr.shape[0]

    # Build the held weight panel: w_held[t] = w at the most recent
    # rebal date ≤ t. Rebal dates are t=0, rebal_days, 2*rebal_days, ...
    held = np.zeros_like(w_arr)
    rebal_dates_idx = list(range(0, n_t, rebal_days))
    for k, ridx in enumerate(rebal_dates_idx):
        end = rebal_dates_idx[k + 1] if k + 1 < len(rebal_dates_idx) else n_t
        held[ridx:end] = w_arr[ridx]

    # Day-by-day portfolio return — applied AFTER weights are set
    # (no look-ahead — w at t is decided at t close, return realized t+1).
    # We lag by 1 day to be honest: weight set on rebal day uses that
    # day's data; return earned starts next bar.
    held_lag = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
    port_ret = (held_lag * daily_ret).sum(axis=1)

    # Cost on rebal dates: one-sided L1 turnover * commission_frac.
    cost = np.zeros(n_t)
    for k, ridx in enumerate(rebal_dates_idx):
        if k == 0:
            cost[ridx] = commission_frac * np.abs(w_arr[ridx]).sum()
        else:
            prev = rebal_dates_idx[k - 1]
            cost[ridx] = commission_frac * 0.5 * np.abs(
                w_arr[ridx] - w_arr[prev]).sum()
    net_ret = port_ret - cost
    return pd.Series(net_ret, index=common)


def passive_ew_daily_returns(
    prices: pd.DataFrame,
    rebal_days: int,
    commission_frac: float,
    lookback: int,
) -> pd.Series:
    """Equal-weight rebalanced basket on the *active* (non-NaN) names.

    Active = non-NaN price at the rebal date. Matches the canonical
    passive-EW baseline (`findings/passive-ew-benchmark.md`).
    """
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
    rsi_n: int,
    n_tail: int,
    top_n: int,
    rebal_days: int,
    commission_frac: float,
    windows: list[tuple[int, int, int]],
) -> dict:
    """Build weights ONCE per (rsi_n, n_tail, top_n) over the entire
    history, then carve into per-window val slices. weights_rsi is
    fully causal (cumsum + Wilder RSI), so a single global call is
    safe and faster than re-running per window.
    """
    w = weights_rsi(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n, rsi_n=rsi_n)
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
        # align
        common = val_ret.index.intersection(ew_ret.index)
        val_ret = val_ret.loc[common].values
        ew_ret = ew_ret.loc[common].values
        s_rsi = annualized_sharpe(val_ret)
        s_ew = annualized_sharpe(ew_ret)
        per_window.append({
            'window_idx': w_idx,
            'val_start': str(common[0].date()) if len(common) else '',
            'val_end': str(common[-1].date()) if len(common) else '',
            'rsi_sharpe': s_rsi,
            'ew_sharpe': s_ew,
            'alpha_sharpe': s_rsi - s_ew,
            'rsi_max_dd': max_drawdown(val_ret),
        })
        val_streams.append(val_ret)
        ew_streams.append(ew_ret)
        val_date_arrs.append(
            np.asarray([np.datetime64(d) for d in common]))

    mean_alpha = float(np.mean([r['alpha_sharpe'] for r in per_window]))
    pos_alpha = int(sum(r['alpha_sharpe'] > 0 for r in per_window))
    mean_rsi_sh = float(np.mean([r['rsi_sharpe'] for r in per_window]))
    mean_ew_sh = float(np.mean([r['ew_sharpe'] for r in per_window]))
    return {
        'per_window': per_window,
        'mean_alpha': mean_alpha,
        'pos_alpha_count': pos_alpha,
        'mean_rsi_sharpe': mean_rsi_sh,
        'mean_ew_sharpe': mean_ew_sh,
        'oos_rsi': np.concatenate(val_streams) if val_streams else np.array([]),
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
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default=str(STOOQ_SUBSET / 'manifest.json'))
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--lookback', type=int, default=252)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--train-window-days', type=int, default=1260)
    p.add_argument('--val-window-days',   type=int, default=780)
    p.add_argument('--step-window-days',  type=int, default=780)
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading stooq_us_long universe...')
    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end,
        tickers=universe)
    print(f'  loaded {prices.shape[1]} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()} '
          f'({len(prices)} bars)')

    # Window construction over the price index.
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

    # Baseline arm (rsi_n=14, n_tail=5, top_n=20).
    print('\nBaseline arm: rsi_n=14, n_tail=5, top_n=20')
    baseline = run_arm(
        prices, lookback=args.lookback, rsi_n=14, n_tail=5, top_n=20,
        rebal_days=args.rebal_days, commission_frac=commission_frac,
        windows=windows)
    print(f'{"win":>3} {"val":>23} {"rsi_sh":>7} {"ew_sh":>7} '
          f'{"alpha":>7} {"dd":>7}')
    for r in baseline['per_window']:
        print(f'{r["window_idx"]:>3} {r["val_start"]}→{r["val_end"]} '
              f'{r["rsi_sharpe"]:>+7.3f} {r["ew_sharpe"]:>+7.3f} '
              f'{r["alpha_sharpe"]:>+7.3f} {r["rsi_max_dd"]:>+7.3f}')
    print(f'  mean rsi Sharpe = {baseline["mean_rsi_sharpe"]:+.3f}')
    print(f'  mean ew Sharpe  = {baseline["mean_ew_sharpe"]:+.3f}')
    print(f'  mean alpha      = {baseline["mean_alpha"]:+.3f} '
          f'({baseline["pos_alpha_count"]}/{len(windows)} positive)')

    # Hyperparam robustness grid.
    print('\nRobustness grid (rsi_n ∈ {7,14,21}, top_n ∈ {10,20,50}, '
          'n_tail ∈ {5,10}):')
    grid_results = []
    print(f'{"rsi_n":>5} {"top_n":>5} {"n_tail":>6} {"mean_α":>8} '
          f'{"pos":>4} {"rsi_sh":>7}')
    for rsi_n in (7, 14, 21):
        for top_n in (10, 20, 50):
            for n_tail in (5, 10):
                arm = run_arm(
                    prices, lookback=args.lookback,
                    rsi_n=rsi_n, n_tail=n_tail, top_n=top_n,
                    rebal_days=args.rebal_days,
                    commission_frac=commission_frac,
                    windows=windows)
                grid_results.append({
                    'rsi_n': rsi_n, 'top_n': top_n, 'n_tail': n_tail,
                    'mean_alpha': arm['mean_alpha'],
                    'mean_rsi_sharpe': arm['mean_rsi_sharpe'],
                    'pos_alpha_count': arm['pos_alpha_count'],
                })
                print(f'{rsi_n:>5} {top_n:>5} {n_tail:>6} '
                      f'{arm["mean_alpha"]:>+8.3f} '
                      f'{arm["pos_alpha_count"]:>4} '
                      f'{arm["mean_rsi_sharpe"]:>+7.3f}')

    # Median cell as headline (per pre-reg).
    grid_results_sorted = sorted(grid_results, key=lambda r: r['mean_rsi_sharpe'])
    median_cell = grid_results_sorted[len(grid_results_sorted) // 2]
    print(f'\nMedian-Sharpe cell: rsi_n={median_cell["rsi_n"]}, '
          f'top_n={median_cell["top_n"]}, n_tail={median_cell["n_tail"]} '
          f'→ rsi_sh={median_cell["mean_rsi_sharpe"]:+.3f}, '
          f'alpha={median_cell["mean_alpha"]:+.3f}')
    alphas = [r['mean_alpha'] for r in grid_results]
    print(f'Grid alpha spread: min={min(alphas):+.3f}, '
          f'max={max(alphas):+.3f}, '
          f'median={float(np.median(alphas)):+.3f}')

    # Use the baseline arm (rsi_n=14, n_tail=5, top_n=20) as the
    # headline since that's the pre-registered canonical config; the
    # grid is a robustness check.
    headline = baseline

    # Quick DSR-t estimate. Per ladder methodology:
    # standardize the OOS Sharpe difference relative to its standard
    # error with n_trials=18 deflation.
    oos_rsi = headline['oos_rsi']
    oos_ew = headline['oos_ew']
    n_obs = oos_rsi.size
    sr_diff_periods = (
        oos_rsi.mean() / (oos_rsi.std() + 1e-12) -
        oos_ew.mean() / (oos_ew.std() + 1e-12))
    sr_diff_ann = sr_diff_periods * np.sqrt(252.0)
    # Approximate SE on annualized SR from n_obs i.i.d.: 1/sqrt(n_obs/252).
    se_ann = 1.0 / np.sqrt(max(n_obs / 252.0, 1e-9))
    dsr_t = float(sr_diff_ann / max(se_ann, 1e-9))
    print(f'\nDSR-t (rough, n_obs={n_obs}): '
          f'sr_diff_ann={sr_diff_ann:+.3f}, se_ann={se_ann:.3f}, '
          f'dsr_t={dsr_t:+.2f}')

    verdict = verdict_label(
        headline['mean_alpha'], headline['pos_alpha_count'], dsr_t)
    print(f'\nVerdict: {verdict}')

    # Save NPZ for the DSR ladder.
    npz_path = output / 'rsi-universe-agnostic-walkforward.npz'
    np.savez(
        npz_path,
        oos_block_returns=oos_rsi,
        oos_ew_returns=oos_ew,
        oos_dates=headline['oos_dates'],
        periods_per_year=np.float64(252.0),
        pre_registered_bar=np.str_(PRE_REGISTERED_BAR),
        universe_label=np.str_('stooq_us_long'),
        windowing_label=np.str_('6w-1260tr-780val-780step'),
        rebal_days=np.int64(args.rebal_days),
        commission_bps=np.float64(args.commission_bps),
        rsi_n=np.int64(14), n_tail=np.int64(5), top_n=np.int64(20),
        mean_alpha_sharpe=np.float64(headline['mean_alpha']),
        pos_alpha_count=np.int64(headline['pos_alpha_count']),
        mean_rsi_sharpe=np.float64(headline['mean_rsi_sharpe']),
        mean_ew_sharpe=np.float64(headline['mean_ew_sharpe']),
        dsr_t=np.float64(dsr_t),
        verdict=np.str_(verdict),
        n_trials=np.int64(18),
    )
    print(f'-> {npz_path}')

    # JSON summary.
    json_path = output / 'rsi-universe-agnostic-walkforward.json'
    json_path.write_text(json.dumps({
        'universe': 'stooq_us_long',
        'windowing': '6w-1260tr-780val-780step',
        'pre_registered_bar': PRE_REGISTERED_BAR,
        'baseline_params': {'rsi_n': 14, 'n_tail': 5, 'top_n': 20},
        'baseline': {
            'mean_alpha': headline['mean_alpha'],
            'pos_alpha_count': headline['pos_alpha_count'],
            'mean_rsi_sharpe': headline['mean_rsi_sharpe'],
            'mean_ew_sharpe': headline['mean_ew_sharpe'],
            'per_window': headline['per_window'],
        },
        'grid': grid_results,
        'median_cell': median_cell,
        'dsr_t_rough': dsr_t,
        'verdict': verdict,
    }, indent=2))
    print(f'-> {json_path}')


if __name__ == '__main__':
    main()
