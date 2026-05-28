"""Command-line entry point.

Two subcommands:

    regime train --data-dir ./StooqData
        Run an Optuna walk-forward search using vectorbt backtests.
        Optimizes 7 discrete hyperparameters (lookback, n_tail, top_n,
        divergence, scale subsets) over rolling train/val windows.
        Reports per-window best params + Sharpes; prints summary stats
        across windows.

        --source stooq (default): bulk Stooq daily archive (split-/div-
        adjusted, includes delistings, has volume). Pass the directory
        that contains `daily/`.
        --source kaggle: legacy `svaningelgem/nasdaq-daily-stock-prices`
        per-ticker CSV layout (no volume, no adj_close).

    regime live --params model.json --dry-run
        Load a checkpoint, fetch recent OHLC from Alpaca, compute target
        weights, and submit (or print) the trades needed to reach them.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from regime.persist import save_checkpoint_from_window
from regime.trainer import DEFAULT_PER_WINDOW_MIN_HISTORY, print_summary, train
from ss_cli import add_universe_loader_args
from ss_indicators import corwin_schultz_spread
from ss_loaders import load_price_matrix, load_stooq_matrix

DEFAULT_KILLSWITCH: str = '~/.regime-killswitch'

warnings.filterwarnings('ignore')


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--strategy', choices=['regime', 'scalogram', 'rsi'],
                   default='regime',
                   help='Which ranking strategy to search/train. '
                        '`regime`: divergence between recent vs historical CWT '
                        'power distributions (momentum-of-volatility-shift). '
                        '`scalogram`: direction − momentum × coherence '
                        '(mean-reversion on incoherent timescales). '
                        '`rsi`: trailing-mean Wilder RSI, lowest = most '
                        'oversold (counter-trend mean-reversion, no CWT). '
                        '`rsi` ignores `--use-log-returns` since RSI is '
                        'computed directly from prices.')
    p.add_argument('--source', choices=['stooq', 'kaggle'], default='stooq',
                   help='Data source layout. `stooq`: bulk archive with '
                        '`daily/<country>/<exchange>/<bucket>/*.txt` (split/'
                        'dividend-adjusted, has volume, includes delistings). '
                        '`kaggle`: per-ticker CSVs (no volume, no adj_close).')
    add_universe_loader_args(
        p,
        default_start='2010-01-01',
        default_end='2025-12-31',
        data_dir_help='For stooq: the directory that contains `daily/`. '
                      'For kaggle: the directory of per-ticker CSVs.')
    p.add_argument('--cache-path',
                   help='Pickle cache for the loaded panel. Subsequent runs '
                        'with the same source skip the file scan. Defaults to '
                        '`<data-dir>/.cache.pkl` for stooq; unused for kaggle.')
    p.add_argument('--n-trials', type=int, default=50,
                   help='Optuna trials per walk-forward window.')
    p.add_argument('--jobs', type=int, default=1,
                   help='Parallel Optuna trials per window (joblib threads). '
                        'JAX/FFT/numba release the GIL so this scales close to '
                        'linearly with core count. Default 1 = sequential.')
    p.add_argument('--seed', type=int, default=42,
                   help='Optuna TPE sampler seed. Pinned for reproducibility; '
                        'set explicitly to compare two runs apples-to-apples.')
    p.add_argument('--metric', default='sharpe',
                   choices=['sharpe', 'cagr', 'max_drawdown', 'total_return'],
                   help='Metric maximized by the search. Default sharpe.')
    p.add_argument('--rebalance-days', type=int, default=20,
                   help='Rebalance every N trading days. Default 20.')
    p.add_argument('--commission-bps', type=float, default=10.0,
                   help='Per-side commission cost in basis points. Default 10.')
    p.add_argument('--max-spread', type=float, default=0.02,
                   help='Live-trading sanity gate (fraction). Recorded into the '
                        'checkpoint so `regime live` rejects names whose spread '
                        'exceeds this on the rebalance bar. Training itself does '
                        'not filter on spread — costs are charged via vbt fees.')
    p.add_argument('--train-years', type=int, default=5,
                   help='Training window length, in years. Default 5.')
    p.add_argument('--val-years', type=int, default=3,
                   help='Validation window length, in years. Default 3.')
    p.add_argument('--step-years', type=int, default=3,
                   help='Step forward between walk-forward windows. Default 3 '
                        '(= val-years, so val periods do not overlap successor '
                        'train periods). Setting < val-years leaks val data '
                        'into the next window\'s training.')
    p.add_argument('--min-history', type=int, default=252,
                   help='Lenient panel-wide ghost filter — drops tickers with '
                        'fewer than this many bars in the requested date range. '
                        'Default 252 (≈1 trading year). The strict survivorship '
                        'filter is per-walk-forward; see --per-window-min-history.')
    p.add_argument('--per-window-min-history', type=int,
                   default=DEFAULT_PER_WINDOW_MIN_HISTORY,
                   help=f'Per-walk-forward survivorship filter — a ticker must '
                        f'have at least this many valid bars (and a valid first '
                        f'bar) within each window to be eligible for that '
                        f'window. Default {DEFAULT_PER_WINDOW_MIN_HISTORY} '
                        f'(~3y = 4 × max wavelet scale + max lookback). '
                        f'Lower values widen the universe (include newer IPOs) '
                        f'but admit scored bars whose underlying CWT was '
                        f'computed on partial-window z-norms. This is the '
                        f'survivorship-bias fix: each window picks its own '
                        f'point-in-time universe rather than requiring '
                        f'tickers to exist for the whole date range.')
    p.add_argument('--include-etfs', action='store_true',
                   help='Stooq only: include `<exchange> etfs` in addition to '
                        'stocks. Default off (regime targets equities).')
    p.add_argument('--use-log-returns', action='store_true',
                   help='Run the CWT on log returns instead of raw close. '
                        'Off by default — controlled walk-forward eval '
                        '(Output/regime-eval-{rawclose-kernel3,logreturns}.log) '
                        'showed log-returns lowers val Sharpe across all '
                        'three windows tested (raw close wins by +0.31 / '
                        '+0.12 / +0.66 on the 3 walk-forward val periods). '
                        'Kept as a flag for non-ranking research objectives '
                        '(vol forecasting, regime-break detection) where '
                        'stationary input matters more than momentum-bleed.')
    p.add_argument('--save-params',
                   help='Write the highest-val-Sharpe window to this JSON path '
                        '(consumable by `regime live`).')


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to a checkpoint JSON written by `regime train --save-params`.')
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Compute and print trades without submitting (default).')
    p.add_argument('--live', dest='dry_run', action='store_false',
                   help='Actually submit orders. Default is dry-run.')
    p.add_argument('--max-position', type=float, default=0.25,
                   help='Per-name weight cap (0-1). Default 0.25.')
    p.add_argument('--max-data-age-days', type=int, default=3,
                   help='Abort if latest bar is older than this many days. Default 3.')
    p.add_argument('--killswitch', default=DEFAULT_KILLSWITCH,
                   help=f'Abort if this file exists. Default {DEFAULT_KILLSWITCH}.')
    p.add_argument('--allow-falsified', action='store_true', default=False,
                   help='Permit live dispatch of checkpoints whose strategy is '
                        'flagged as confirmed-null / reversed-OOS in the '
                        'leaderboard (rsi, scalogram, regime on stooq_us_long). '
                        "Default off — the apples-to-apples audit "
                        '(findings/regime-{rsi,scalogram,cwt}-baseline.md) '
                        'shows all three lose to passive EW on the wide '
                        'universe; this flag is explicit opt-in for '
                        'mega-cap-only research deployments.')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='regime',
        description='Regime-divergence strategy: walk-forward search + live trade.')
    sub = p.add_subparsers(dest='cmd', required=True)
    _add_train_args(sub.add_parser(
        'train', help='Walk-forward Optuna search via vectorbt.'))
    _add_live_args(sub.add_parser(
        'live', help='Score the universe and rebalance via Alpaca.'))
    return p


def _run_train(args: argparse.Namespace) -> None:
    if args.source == 'stooq':
        cache = args.cache_path or str(Path(args.data_dir) / '.cache.pkl')
        prices, highs, lows, volumes = load_stooq_matrix(
            args.data_dir, min_history=args.min_history,
            start_date=args.start, end_date=args.end,
            include_etfs=args.include_etfs, cache_path=cache)
        median_vol_dollars = (prices.iloc[-1] * volumes.iloc[-1]).median()
        print(f'Median latest-bar dollar volume: ${median_vol_dollars:,.0f}')
    else:
        prices, highs, lows = load_price_matrix(
            args.data_dir, min_history=args.min_history,
            start_date=args.start, end_date=args.end)

    print('Computing Corwin-Schultz spreads...')
    spread_df = corwin_schultz_spread(highs, lows)
    median_spread = spread_df.iloc[-1].median()
    print(f'Median latest-bar spread across universe: {median_spread:.2%} '
          f'(passed into vbt as per-side cost = commission + spread/2)')

    result = train(
        prices, spread_df,
        strategy=args.strategy,
        n_trials=args.n_trials,
        n_jobs=args.jobs,
        seed=args.seed,
        rebalance_days=args.rebalance_days,
        metric=args.metric,
        commission_bps=args.commission_bps,
        train_years=args.train_years,
        val_years=args.val_years,
        step_years=args.step_years,
        per_window_min_history=args.per_window_min_history,
        use_log_returns=args.use_log_returns,
    )
    print_summary(result)

    if args.save_params and result.windows:
        path = save_checkpoint_from_window(
            args.save_params, result.best_window,
            universe=list(prices.columns),
            rebal_days=args.rebalance_days,
            max_spread=args.max_spread,
            commission_bps=args.commission_bps,
        )
        print(f'\nSaved best-window checkpoint to {path}')


FALSIFIED_REGIME_STRATEGIES = frozenset({'rsi', 'scalogram', 'regime'})


def _check_falsified(checkpoint_path: str, allow: bool) -> None:
    """Gate confirmed-null / reversed-OOS regime strategies behind
    an explicit opt-in. Each of `rsi`, `scalogram`, `regime` lost to
    passive EW on stooq_us_long in the 2026-05-25 audit — see
    findings/regime-{rsi,scalogram,cwt}-baseline.md.
    """
    from regime.persist import load_checkpoint
    cp = load_checkpoint(checkpoint_path)
    if cp.strategy in FALSIFIED_REGIME_STRATEGIES and not allow:
        raise SystemExit(
            f"refusing to dispatch checkpoint with strategy='{cp.strategy}': "
            f"the 2026-05-25 universe-agnostic audit "
            f"(findings/regime-{{rsi,scalogram,cwt}}-baseline.md) recorded "
            f"this strategy as confirmed-null or reversed-OOS vs passive EW "
            f"on stooq_us_long. Pass --allow-falsified to override (e.g. for "
            f"mega-cap-only research deployments where the strategy may "
            f"still be defensible)."
        )


def _run_live(args: argparse.Namespace) -> None:
    from regime.live import format_run, run_live

    _check_falsified(args.params, args.allow_falsified)

    result = run_live(
        args.params,
        dry_run=args.dry_run,
        max_position=args.max_position,
        max_data_age_days=args.max_data_age_days,
        killswitch_path=args.killswitch,
    )
    print(format_run(result))


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == 'train':
        _run_train(args)
    elif args.cmd == 'live':
        _run_live(args)


if __name__ == '__main__':
    main()
