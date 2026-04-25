"""regime.trainer: Optuna + vectorbt walk-forward search.

Production trainer for the regime strategy. Combines:
  * Optuna TPE search over discrete hyperparameters (lookback, n_tail,
    top_n, divergence, scale subsets).
  * Hard top-N equal-weight allocation
    (`ss_portfolio.select_top_n_matrix`).
  * vectorbt backtest engine (`ss_portfolio.vbt_backtest`) — much faster
    than bt-library, JIT-compiled by numba.
  * Walk-forward train/val rolling windows.

This is the spiritual successor to `regime.research.optimize_regime`
(Optuna + bt-library) — same approach, faster engine, plus we
vectorize the per-date score computation through `precompute_windows`
+ JAX divergences instead of looping in Python.

For the gradient-descent alternative on continuous params, see
`regime.research.optimize_adam`. For the bt-library reference
implementation, see `regime.research.optimize_regime`.

Requires the nix devShell (provides numba/llvmlite for vectorbt). See
README.md.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd

from ss_indicators import corwin_schultz_spread, get_divergence
from ss_loaders import load_price_matrix
from ss_portfolio import (
    apply_nan_mask,
    apply_spread_mask,
    select_top_n_matrix,
    vbt_backtest,
)
from ss_wavelets import causal_cwt, precompute_windows


# Scale-subset groups Optuna picks among via 3 boolean flags.
SHORT_SCALES = [3, 5, 7]
MID_SCALES = [10, 12, 15, 21, 26]
LONG_SCALES = [42, 50, 63, 90, 126]
DEFAULT_SCALES = [5, 21, 90]  # fallback when all three subsets are off


@dataclass
class WindowResult:
    """Best params + scores for one walk-forward window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    best_params: dict
    train_score: float
    val_score: float


@dataclass
class TrainResult:
    """Output of `train()` — one entry per walk-forward window."""
    windows: list[WindowResult]
    metric: str
    rebalance_days: int
    commission_bps: float
    max_spread: float

    @property
    def best_window(self) -> WindowResult:
        """Window with the highest validation score."""
        return max(self.windows, key=lambda w: w.val_score)


def _resolve_scales(params: dict) -> list[int]:
    """Build the scale list from Optuna's three boolean subset flags."""
    scales: list[int] = []
    if params.get('use_short_scales', False):
        scales += SHORT_SCALES
    if params.get('use_mid_scales', False):
        scales += MID_SCALES
    if params.get('use_long_scales', False):
        scales += LONG_SCALES
    return scales or DEFAULT_SCALES


def regime_weights(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    divergence: str = 'kl',
    spread_df: pd.DataFrame | None = None,
    max_spread: float = 0.02,
) -> pd.DataFrame:
    """Compute hard-top-N regime weights for a price matrix.

    The score per (date, ticker) is the chosen divergence between the
    recent vs historical CWT-power distributions across `scales`. We
    compute scores for all valid dates in one vectorized pass via
    `precompute_windows` + JAX divergence (much faster than the Python
    date-loop in `regime.research.optimize_regime`).

    `scale_log_weights = zeros` makes the divergence's internal softmax
    uniform — Optuna chooses *which* scales to include, but each
    included scale contributes equally (matching the legacy behavior).
    """
    coeffs = causal_cwt(prices.values, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(power, lookback, n_tail)

    div_fn = get_divergence(divergence)
    scale_log_weights = jnp.zeros(len(scales), dtype=jnp.float32)
    # `np.array(jnp_array)` (not `asarray`) forces a writable host copy
    # so `apply_nan_mask`/`apply_spread_mask` can NaN cells in place.
    scores = np.array(div_fn(
        jnp.asarray(recent), jnp.asarray(historical), scale_log_weights))

    scores = apply_nan_mask(scores, prices.values, lookback)
    if spread_df is not None:
        scores = apply_spread_mask(scores, spread_df.values, lookback, max_spread)

    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)


def _make_objective(
    prices, *, rebalance_days, metric, commission_bps, spread_df, max_spread,
):
    """Build an Optuna objective closed over the train slice."""
    def objective(trial: optuna.Trial) -> float:
        lookback = trial.suggest_int('lookback', 40, 252)
        n_tail = trial.suggest_int('n_tail', 3, lookback // 2)
        top_n = trial.suggest_int('top_n', 5, 30)
        divergence = trial.suggest_categorical(
            'divergence', ['kl', 'js', 'cosine', 'l2'])
        use_short = trial.suggest_categorical('use_short_scales', [True, False])
        use_mid = trial.suggest_categorical('use_mid_scales', [True, False])
        use_long = trial.suggest_categorical('use_long_scales', [True, False])
        scales = _resolve_scales({
            'use_short_scales': use_short,
            'use_mid_scales': use_mid,
            'use_long_scales': use_long,
        })

        try:
            weight_df = regime_weights(
                prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
                scales=scales, divergence=divergence,
                spread_df=spread_df, max_spread=max_spread)
            if weight_df.values.sum() == 0:
                return float('-inf')
            metrics = vbt_backtest(
                prices, weight_df,
                rebalance_days=rebalance_days,
                commission_bps=commission_bps)
            score = metrics[metric]
            return score if np.isfinite(score) else float('-inf')
        except Exception:
            return float('-inf')

    return objective


def train(
    prices: pd.DataFrame,
    spread_df: pd.DataFrame | None = None,
    *,
    n_trials: int = 50,
    rebalance_days: int = 20,
    metric: str = 'sharpe',
    commission_bps: float = 10.0,
    max_spread: float = 0.02,
    train_years: int = 5,
    val_years: int = 3,
    step_years: int = 2,
) -> TrainResult:
    """Walk-forward Optuna+vectorbt search over regime hyperparameters.

    Rolls a `train_years`/`val_years` window forward by `step_years` at
    a time. For each window: TPE-search `n_trials` hyperparameter combos
    on the train slice, then evaluate the best params out-of-sample on
    val. Returns one `WindowResult` per window.

    `metric` is the key from `vbt_backtest`'s output dict to maximize:
    `sharpe`, `cagr`, or `max_drawdown` (note max_drawdown is negative,
    so maximizing it minimizes the worst drawdown).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start, end = prices.index[0], prices.index[-1]
    windows: list[WindowResult] = []
    window_start = start

    while True:
        train_end = window_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        if val_end > end:
            break

        prices_train = prices.loc[window_start:train_end]
        prices_val = prices.loc[train_end:val_end]
        spread_train = (spread_df.loc[window_start:train_end]
                        if spread_df is not None else None)
        spread_val = (spread_df.loc[train_end:val_end]
                      if spread_df is not None else None)

        if len(prices_train) < 252 or len(prices_val) < 126:
            window_start += pd.DateOffset(years=step_years)
            continue

        print(f'\nWindow: train {window_start.date()}-{train_end.date()}, '
              f'val {train_end.date()}-{val_end.date()}')

        study = optuna.create_study(direction='maximize')
        objective = _make_objective(
            prices_train,
            rebalance_days=rebalance_days, metric=metric,
            commission_bps=commission_bps,
            spread_df=spread_train, max_spread=max_spread)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        train_score = study.best_value
        print(f'  Train {metric}: {train_score:+.4f}, params: {best}')

        try:
            weight_df = regime_weights(
                prices_val,
                lookback=best['lookback'], n_tail=best['n_tail'],
                top_n=best['top_n'], scales=_resolve_scales(best),
                divergence=best['divergence'],
                spread_df=spread_val, max_spread=max_spread)
            metrics = vbt_backtest(
                prices_val, weight_df,
                rebalance_days=rebalance_days,
                commission_bps=commission_bps)
            val_score = metrics[metric]
        except Exception:
            val_score = float('nan')
        print(f'  Val   {metric}: {val_score:+.4f}')

        windows.append(WindowResult(
            train_start=window_start, train_end=train_end, val_end=val_end,
            best_params=best, train_score=train_score, val_score=val_score))
        window_start += pd.DateOffset(years=step_years)

    return TrainResult(
        windows=windows, metric=metric,
        rebalance_days=rebalance_days,
        commission_bps=commission_bps, max_spread=max_spread)


def print_summary(result: TrainResult) -> None:
    """Pretty-print the per-window summary + best-window highlight."""
    print(f'\n{"=" * 80}')
    print(f'Walk-Forward Summary  (metric: {result.metric}, '
          f'commission: {result.commission_bps} bps, '
          f'max spread: {result.max_spread:.1%})')
    print('=' * 80)
    print(f'{"Window":<22} {"Train":>9} {"Val":>9} {"Div":>7} '
          f'{"LB":>4} {"NT":>4} {"TopN":>5} {"Scales":>8}')
    print('-' * 80)
    for w in result.windows:
        p = w.best_params
        scales_str = ''.join(
            c for c, on in [
                ('S', p.get('use_short_scales')),
                ('M', p.get('use_mid_scales')),
                ('L', p.get('use_long_scales')),
            ] if on) or 'def'
        label = f'{w.train_start.year}-{w.val_end.year}'
        print(f'{label:<22} {w.train_score:>+9.4f} {w.val_score:>+9.4f} '
              f'{p["divergence"]:>7} {p["lookback"]:>4} {p["n_tail"]:>4} '
              f'{p["top_n"]:>5} {scales_str:>8}')
    if result.windows:
        bw = result.best_window
        print(f'\nBest window: {bw.train_start.year}-{bw.val_end.year}  '
              f'val={bw.val_score:+.4f}, train={bw.train_score:+.4f}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Optuna + vectorbt walk-forward search for the regime strategy.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory of per-ticker OHLC CSVs (e.g. ./Nasdaq3347).')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna trials per walk-forward window.')
    parser.add_argument('--metric', default='sharpe',
                        choices=['sharpe', 'cagr', 'max_drawdown', 'total_return'])
    parser.add_argument('--rebalance-days', type=int, default=20)
    parser.add_argument('--commission-bps', type=float, default=10.0)
    parser.add_argument('--max-spread', type=float, default=0.02)
    parser.add_argument('--train-years', type=int, default=5)
    parser.add_argument('--val-years', type=int, default=3)
    parser.add_argument('--step-years', type=int, default=2)
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2025-12-31')
    parser.add_argument('--min-history', type=int, default=504)
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')
    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=args.min_history,
        start_date=args.start, end_date=args.end)
    print('Computing Corwin-Schultz spreads...')
    spread_df = corwin_schultz_spread(highs, lows)

    result = train(
        prices, spread_df,
        n_trials=args.n_trials,
        rebalance_days=args.rebalance_days,
        metric=args.metric,
        commission_bps=args.commission_bps,
        max_spread=args.max_spread,
        train_years=args.train_years,
        val_years=args.val_years,
        step_years=args.step_years)

    print_summary(result)


if __name__ == '__main__':
    main()
