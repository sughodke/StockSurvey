"""optimize_regime: walk-forward Optuna search over regime hyperparameters.

Splits the backtest period into rolling train/validate windows, runs a
TPE search on the train window with transaction costs, and reports the
best params + their out-of-sample score on the validate window. Useful
for testing whether the regime signal is stable across regimes (it
generally isn't — see CLAUDE.md notes).

Usage:
    python -m regime.research.optimize_regime --data-dir ./Nasdaq3347
    python -m regime.research.optimize_regime --data-dir ./Nasdaq3347 \\
        --n-trials 200 --objective calmar
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections import Counter

import bt
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from regime.research.backtest_bt import make_commission_fn, select_top_n_matrix
from ss_indicators import corwin_schultz_spread
from ss_loaders import load_price_matrix
from ss_wavelets import ALL_SCALES, causal_cwt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def weights_regime_parameterized(
    prices, lookback, n_tail, top_n, scales,
    divergence='kl', spread_df=None, max_spread=0.02,
):
    """Regime strategy with tunable lookback / n_tail / top_n / divergence."""
    coeffs = causal_cwt(prices.values, scales, lookback)
    power = coeffs ** 2

    n_dates, n_tickers = prices.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)
    for i in range(lookback, n_dates):
        recent = np.mean(power[:, i - n_tail + 1:i + 1, :], axis=1)
        historical = np.mean(power[:, i - lookback:i - n_tail + 1, :], axis=1)

        rd = recent / (recent.sum(axis=0, keepdims=True) + 1e-9)
        hd = historical / (historical.sum(axis=0, keepdims=True) + 1e-9)

        if divergence == 'kl':
            div = 0.5 * np.sum(rd * np.log((rd + 1e-9) / (hd + 1e-9)), axis=0)
            div += 0.5 * np.sum(hd * np.log((hd + 1e-9) / (rd + 1e-9)), axis=0)
        elif divergence == 'js':
            m = 0.5 * (rd + hd)
            div = 0.5 * np.sum(rd * np.log((rd + 1e-9) / (m + 1e-9)), axis=0)
            div += 0.5 * np.sum(hd * np.log((hd + 1e-9) / (m + 1e-9)), axis=0)
        elif divergence == 'cosine':
            dot = np.sum(rd * hd, axis=0)
            norm_r = np.sqrt(np.sum(rd ** 2, axis=0))
            norm_h = np.sqrt(np.sum(hd ** 2, axis=0))
            div = 1.0 - dot / (norm_r * norm_h + 1e-9)
        elif divergence == 'l2':
            div = np.sqrt(np.sum((rd - hd) ** 2, axis=0))
        scores[i - lookback] = div

    price_arr = prices.values
    for i in range(lookback, n_dates):
        has_nan = np.any(np.isnan(price_arr[i - lookback:i + 1]), axis=0)
        scores[i - lookback, has_nan] = np.nan

    if spread_df is not None:
        spread_arr = spread_df.values
        for i in range(scores.shape[0]):
            scores[i, spread_arr[i + lookback] > max_spread] = np.nan

    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(weights, index=prices.index[lookback:], columns=prices.columns)


def run_backtest(prices, weight_df, rebalance_days, commission_bps=10):
    """Run a single bt backtest and return its result."""
    rebal_weights = weight_df.iloc[::rebalance_days]
    common = prices.columns.intersection(weight_df.columns)
    p = prices[common]
    rw = rebal_weights[common]

    strategy = bt.Strategy('regime', [
        bt.algos.RunOnDate(*rw.index),
        bt.algos.WeighTarget(rw),
        bt.algos.Rebalance(),
    ])
    test = bt.Backtest(strategy, p,
                       commissions=make_commission_fn(commission_bps),
                       integer_positions=False)
    return bt.run(test)


def extract_metric(result, metric):
    """Pull a scalar metric out of a bt result by name."""
    stats = result.stats
    key = {
        'sharpe': 'daily_sharpe',
        'calmar': 'calmar',
        'cagr': 'cagr',
        'sortino': 'daily_sortino',
    }[metric]
    return float(stats.loc[key, 'regime'])


def make_objective(prices_train, rebalance_days, metric, commission_bps=10,
                   spread_train=None, max_spread=0.02):
    """Build an Optuna objective closed over the train slice."""
    def objective(trial):
        lookback = trial.suggest_int('lookback', 40, 252)
        n_tail = trial.suggest_int('n_tail', 3, lookback // 2)
        top_n = trial.suggest_int('top_n', 5, 30)
        divergence = trial.suggest_categorical('divergence', ['kl', 'js', 'cosine', 'l2'])

        use_short = trial.suggest_categorical('use_short_scales', [True, False])
        use_mid = trial.suggest_categorical('use_mid_scales', [True, False])
        use_long = trial.suggest_categorical('use_long_scales', [True, False])
        scales: list[int] = []
        if use_short:
            scales += [3, 5, 7]
        if use_mid:
            scales += [10, 12, 15, 21, 26]
        if use_long:
            scales += [42, 50, 63, 90, 126]
        if not scales:
            scales = [5, 21, 90]

        try:
            weight_df = weights_regime_parameterized(
                prices_train, lookback, n_tail, top_n, scales,
                divergence=divergence,
                spread_df=spread_train, max_spread=max_spread)
            if weight_df.sum(axis=1).sum() == 0:
                return float('-inf')
            result = run_backtest(prices_train, weight_df, rebalance_days,
                                  commission_bps=commission_bps)
            score = extract_metric(result, metric)
            if np.isnan(score) or np.isinf(score):
                return float('-inf')
            return score
        except Exception:
            return float('-inf')
    return objective


def _resolve_scales(params):
    scales: list[int] = []
    if params.get('use_short_scales', False):
        scales += [3, 5, 7]
    if params.get('use_mid_scales', False):
        scales += [10, 12, 15, 21, 26]
    if params.get('use_long_scales', False):
        scales += [42, 50, 63, 90, 126]
    return scales or [5, 21, 90]


def walk_forward_optimize(prices, n_trials, rebalance_days, metric,
                          commission_bps=10, max_spread=0.02, spread_df=None,
                          train_years=5, val_years=3, step_years=2):
    """Rolling walk-forward Optuna optimization. Returns one dict per window."""
    start = prices.index[0]
    end = prices.index[-1]
    results: list[dict] = []
    window_start = start

    while True:
        train_end = window_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        if val_end > end:
            break

        prices_train = prices.loc[window_start:train_end]
        prices_val = prices.loc[train_end:val_end]
        spread_train = spread_df.loc[window_start:train_end] if spread_df is not None else None
        spread_val = spread_df.loc[train_end:val_end] if spread_df is not None else None

        if len(prices_train) < 252 or len(prices_val) < 126:
            window_start += pd.DateOffset(years=step_years)
            continue

        print(f'\n{"=" * 70}')
        print(f'Window: train {window_start.date()}-{train_end.date()}, '
              f'validate {train_end.date()}-{val_end.date()}')
        print(f'  Train: {len(prices_train)} days, Validate: {len(prices_val)} days')
        print(f'  Transaction costs: {commission_bps} bps, max spread: {max_spread:.1%}')

        study = optuna.create_study(direction='maximize')
        objective = make_objective(prices_train, rebalance_days, metric,
                                   commission_bps=commission_bps,
                                   spread_train=spread_train, max_spread=max_spread)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        train_score = study.best_value
        print(f'  Best train {metric}: {train_score:.4f}')
        print(f'  Params: {best}')

        try:
            weight_df = weights_regime_parameterized(
                prices_val, best['lookback'], best['n_tail'], best['top_n'],
                _resolve_scales(best), divergence=best['divergence'],
                spread_df=spread_val, max_spread=max_spread)
            val_result = run_backtest(prices_val, weight_df, rebalance_days,
                                      commission_bps=commission_bps)
            val_score = extract_metric(val_result, metric)
        except Exception:
            val_score = float('nan')
        print(f'  Validation {metric}: {val_score:.4f}')

        results.append({
            'train_start': window_start, 'train_end': train_end, 'val_end': val_end,
            'best_params': best, 'train_score': train_score, 'val_score': val_score,
        })
        window_start += pd.DateOffset(years=step_years)
    return results


def plot_optimization_results(wf_results, metric):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Walk-Forward Optimization (metric: {metric})',
                 fontsize=13, fontweight='bold')
    n = len(wf_results)
    x = np.arange(n)
    labels = [f"{r['train_start'].year}-{r['train_end'].year}" for r in wf_results]
    train_scores = [r['train_score'] for r in wf_results]
    val_scores = [r['val_score'] for r in wf_results]

    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2, train_scores, width, label='Train', color='steelblue')
    ax.bar(x + width / 2, val_scores, width, label='Validation', color='darkgoldenrod')
    ax.set_ylabel(metric)
    ax.set_title(f'Train vs Validation {metric} per window')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)

    ax = axes[1]
    for pname in ['lookback', 'n_tail', 'top_n']:
        vals = [r['best_params'].get(pname, 0) for r in wf_results]
        ax.plot(x, vals, 'o-', label=pname)
    ax.set_ylabel('Parameter value')
    ax.set_title('Best parameter values per window')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Walk-forward hyperparameter optimization for regime strategy')
    parser.add_argument('--data-dir', required=True,
                        help='Path to Kaggle NASDAQ daily CSV directory')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--objective', default='sharpe',
                        choices=['sharpe', 'calmar', 'cagr', 'sortino'])
    parser.add_argument('--rebalance', type=int, default=20)
    parser.add_argument('--train-years', type=int, default=5)
    parser.add_argument('--val-years', type=int, default=3)
    parser.add_argument('--step-years', type=int, default=2)
    parser.add_argument('--commission-bps', type=int, default=10)
    parser.add_argument('--max-spread', type=float, default=0.02)
    parser.add_argument('--start', default='2000-01-01')
    parser.add_argument('--end', default='2025-12-31')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args(argv)

    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=504,
        start_date=args.start, end_date=args.end)

    print('Computing Corwin-Schultz spread estimates...')
    spread_df = corwin_schultz_spread(highs, lows)
    liquid_pct = (spread_df.iloc[-1] <= args.max_spread).mean()
    print(f'Liquid tickers (spread <= {args.max_spread:.1%}): {liquid_pct:.1%} of universe')

    wf_results = walk_forward_optimize(
        prices, n_trials=args.n_trials, rebalance_days=args.rebalance,
        metric=args.objective, commission_bps=args.commission_bps,
        max_spread=args.max_spread, spread_df=spread_df,
        train_years=args.train_years, val_years=args.val_years,
        step_years=args.step_years)

    print(f'\n{"=" * 70}')
    print(f'Walk-Forward Summary  (commission: {args.commission_bps} bps, '
          f'max spread: {args.max_spread:.1%})')
    print(f'{"=" * 70}')
    print(f'{"Window":<25} {"Train":>10} {"Validate":>10} {"Divergence":>12} '
          f'{"Lookback":>10} {"N_tail":>8} {"Top_n":>7} {"Scales":>8}')
    print('-' * 92)
    for r in wf_results:
        p = r['best_params']
        scales_str = ''
        if p.get('use_short_scales'):
            scales_str += 'S'
        if p.get('use_mid_scales'):
            scales_str += 'M'
        if p.get('use_long_scales'):
            scales_str += 'L'
        scales_str = scales_str or 'def'
        label = f"{r['train_start'].year}-{r['val_end'].year}"
        print(f'{label:<25} {r["train_score"]:>10.4f} {r["val_score"]:>10.4f} '
              f'{p["divergence"]:>12} {p["lookback"]:>10} {p["n_tail"]:>8} '
              f'{p["top_n"]:>7} {scales_str:>8}')

    print(f'\nParameter stability:')
    for pname in ['lookback', 'n_tail', 'top_n', 'divergence']:
        vals = [r['best_params'][pname] for r in wf_results]
        if isinstance(vals[0], (int, float)):
            print(f'  {pname}: mean={np.mean(vals):.1f}, std={np.std(vals):.1f}, '
                  f'range=[{min(vals)}, {max(vals)}]')
        else:
            print(f'  {pname}: {dict(Counter(vals))}')

    fig = plot_optimization_results(wf_results, args.objective)
    if args.save:
        fig.savefig('Output/optimize-regime-walkforward.png', dpi=150)
        print(f'\nSaved Output/optimize-regime-walkforward.png')
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
