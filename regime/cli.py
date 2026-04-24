"""Command-line entry point: load -> train -> report."""

from __future__ import annotations

import argparse
import warnings

from regime.cwt import ALL_SCALES
from regime.data import corwin_schultz_spread, load_price_matrix
from regime.reporting import plot_training, print_results
from regime.trainer import train

warnings.filterwarnings('ignore')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='regime',
        description='Differentiable regime-divergence optimizer (JAX + optax Adam).')
    p.add_argument('--data-dir', required=True,
                   help='Directory of per-ticker OHLC CSVs (e.g. ./Nasdaq3347).')
    p.add_argument('--lookback', type=int, default=120,
                   help='Causal normalization + historical window length, in days.')
    p.add_argument('--n-tail', type=int, default=20,
                   help='Recent-window length (must be < --lookback).')
    p.add_argument('--rebal-days', type=int, default=20,
                   help='Rebalance every N trading days.')
    p.add_argument('--commission-bps', type=float, default=10,
                   help='Per-side commission cost in basis points.')
    p.add_argument('--max-spread', type=float, default=0.02,
                   help='Drop tickers whose Corwin-Schultz spread exceeds this fraction.')
    p.add_argument('--n-steps', type=int, default=500,
                   help='Number of Adam steps.')
    p.add_argument('--learning-rate', type=float, default=0.05)
    p.add_argument('--train-frac', type=float, default=0.7,
                   help='Fraction of blocks used for training; remainder is held-out val.')
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-12-31')
    p.add_argument('--save', action='store_true',
                   help='Write training plot to Output/regime-training.png.')
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=504,
        start_date=args.start, end_date=args.end)

    print('Computing Corwin-Schultz spreads...')
    spread_df = corwin_schultz_spread(highs, lows)
    liquid_pct = (spread_df.iloc[-1] <= args.max_spread).mean()
    print(f'Liquid tickers (spread <= {args.max_spread:.1%}): {liquid_pct:.1%}')

    result = train(
        prices, spread_df,
        scales=ALL_SCALES,
        lookback=args.lookback,
        n_tail=args.n_tail,
        rebal_days=args.rebal_days,
        commission_bps=args.commission_bps,
        max_spread=args.max_spread,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        train_frac=args.train_frac,
    )

    print_results(result)
    plot_training(
        result,
        save_path='Output/regime-training.png' if args.save else None,
    )


if __name__ == '__main__':
    main()
