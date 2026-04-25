"""
plot_rsi_fold_comparison : compare true RSI (computed on full series then folded)
vs approximate RSI (computed independently per cycle from folded price).

Shows how much information is lost when RSI is computed from folded price
without cross-cycle state.

Usage:
    uv run python plot_rsi_fold_comparison.py --offline AAPL
    uv run python plot_rsi_fold_comparison.py --offline --save NVDA
    uv run python plot_rsi_fold_comparison.py --offline --rsi-period 14 TSLA
"""

import argparse
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt

from v1.models.security import Security
from v1.util.indicators import relative_strength

warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)


def detect_period(x, top_k=1):
    """Find dominant period via FFT."""
    n = len(x)
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x)))[1:]
    freqs = np.fft.rfftfreq(n)[1:]
    idx = np.argsort(fft_vals)[::-1][0]
    period = int(np.round(1.0 / freqs[idx]))
    return max(2, min(period, n // 2))


def fold(x, period):
    """Fold 1D array into 2D by period, truncating to full cycles."""
    num_cycles = len(x) // period
    return x[:num_cycles * period].reshape(num_cycles, period)


def plot_comparison(ticker, prices, rsi_period, span_name):
    period = detect_period(prices)
    num_cycles = len(prices) // period

    # Method 1: fold(RSI(price)) — true RSI folded
    true_rsi = relative_strength(prices, rsi_period)
    true_rsi_folded = fold(true_rsi, period)

    # Method 2: RSI(fold(price)) — RSI computed per cycle independently
    price_folded = fold(prices, period)
    approx_rsi_folded = np.zeros_like(price_folded)
    for i in range(num_cycles):
        row = price_folded[i]
        if len(row) > rsi_period:
            approx_rsi_folded[i] = relative_strength(row, rsi_period)
        else:
            approx_rsi_folded[i] = np.nan

    # Difference
    diff = true_rsi_folded - approx_rsi_folded

    # --- Plot ---
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f'{ticker} — RSI({rsi_period}) Fold Comparison  '
        f'[{span_name}, dominant period={period}d, {num_cycles} cycles]',
        fontsize=13, fontweight='bold', y=0.98)

    # Row 1: full time series comparison
    ax_ts = fig.add_axes([0.08, 0.82, 0.87, 0.12])
    n_plot = num_cycles * period  # only the portion that was folded
    ax_ts.plot(true_rsi[:n_plot], color='darkgoldenrod', linewidth=0.7,
               label='True RSI')
    ax_ts.plot(np.ravel(approx_rsi_folded), color='steelblue', linewidth=0.7,
               alpha=0.7, label='Approx RSI (per-cycle)')
    ax_ts.axhline(70, color='gray', linestyle='--', alpha=0.5)
    ax_ts.axhline(30, color='gray', linestyle='--', alpha=0.5)
    ax_ts.set_ylabel('RSI')
    ax_ts.set_title('Time Series Overlay', fontsize=10)
    ax_ts.legend(loc='upper right', fontsize=8)

    # Row 2: three heatmaps
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.94,
                          bottom=0.45, top=0.76, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(true_rsi_folded, cmap='RdYlGn', aspect='auto',
                     vmin=0, vmax=100, interpolation='nearest')
    ax1.set_title('fold( RSI(price) )\nTrue RSI, folded', fontsize=10)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Cycle')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(approx_rsi_folded, cmap='RdYlGn', aspect='auto',
                     vmin=0, vmax=100, interpolation='nearest')
    ax2.set_title('RSI( fold(price) )\nPer-cycle approx', fontsize=10)
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Cycle')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 2])
    vmax = max(np.nanmax(np.abs(diff)), 1)
    im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax3.set_title('Difference\n(true \u2212 approx)', fontsize=10)
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Cycle')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Row 3: per-cycle statistics
    gs2 = fig.add_gridspec(1, 3, left=0.06, right=0.94,
                           bottom=0.06, top=0.38, wspace=0.25)

    # Correlation per cycle
    ax4 = fig.add_subplot(gs2[0, 0])
    corrs = []
    for i in range(num_cycles):
        t = true_rsi_folded[i]
        a = approx_rsi_folded[i]
        mask = ~np.isnan(a)
        if mask.sum() > 2:
            corrs.append(np.corrcoef(t[mask], a[mask])[0, 1])
        else:
            corrs.append(np.nan)
    colors = ['#2ecc71' if c > 0.5 else '#e74c3c' for c in corrs]
    ax4.bar(range(num_cycles), corrs, color=colors, alpha=0.8)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('Cycle')
    ax4.set_ylabel('Pearson r')
    ax4.set_title('Correlation per cycle', fontsize=10)
    ax4.set_ylim(-1, 1)

    # MAE per cycle
    ax5 = fig.add_subplot(gs2[0, 1])
    mae_per_cycle = np.nanmean(np.abs(diff), axis=1)
    ax5.bar(range(num_cycles), mae_per_cycle, color='steelblue', alpha=0.8)
    ax5.set_xlabel('Cycle')
    ax5.set_ylabel('MAE (RSI points)')
    ax5.set_title('Mean absolute error per cycle', fontsize=10)

    # Scatter: true vs approx (all points)
    ax6 = fig.add_subplot(gs2[0, 2])
    t_flat = true_rsi_folded.ravel()
    a_flat = approx_rsi_folded.ravel()
    mask = ~np.isnan(a_flat)
    ax6.scatter(t_flat[mask], a_flat[mask], s=3, alpha=0.3,
                color='darkgoldenrod')
    ax6.plot([0, 100], [0, 100], 'k--', linewidth=0.8, alpha=0.5)
    overall_corr = np.corrcoef(t_flat[mask], a_flat[mask])[0, 1]
    overall_mae = np.nanmean(np.abs(t_flat[mask] - a_flat[mask]))
    ax6.set_xlabel('True RSI')
    ax6.set_ylabel('Approx RSI')
    ax6.set_title(f'All points  (r={overall_corr:.3f}, MAE={overall_mae:.1f})',
                  fontsize=10)
    ax6.set_xlim(0, 100)
    ax6.set_ylim(0, 100)
    ax6.set_aspect('equal')

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare true RSI vs per-cycle approximation from folded price')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols')
    parser.add_argument('--span', default='daily',
                        choices=['daily', 'weekly', 'monthly'])
    parser.add_argument('--rsi-period', type=int, default=7,
                        help='RSI lookback period (default: 7)')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--save', action='store_true',
                        help='Save to Output/ instead of showing')
    args = parser.parse_args()

    for ticker in args.tickers:
        s = Security.load(ticker, offline=args.offline)
        if s is None:
            print(f'Skipping {ticker} (no cached data)')
            continue

        dataset = getattr(s, args.span, s.daily)
        prices = dataset.adj_close.values

        fig = plot_comparison(ticker, prices, args.rsi_period, args.span)

        if args.save:
            fname = f'Output/{ticker}-rsi-fold-comparison.png'
            fig.savefig(fname, dpi=150)
            print(f'Saved {fname}')
            plt.close(fig)

    if not args.save:
        plt.show()
