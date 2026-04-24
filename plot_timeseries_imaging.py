"""
plot_timeseries_imaging : convert price/RSI time series into 2D images
using four approaches:
  1. Recurrence Plot
  2. Gramian Angular Field (GAF)
  3. Markov Transition Field (MTF)
  4. TimesNet fold-by-period

Usage:
    uv run python plot_timeseries_imaging.py AAPL
    uv run python plot_timeseries_imaging.py --signal rsi NVDA
    uv run python plot_timeseries_imaging.py --offline --span weekly GLD
"""

import argparse
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.security import Security
from util.indicators import relative_strength

warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)


# --- Imaging transforms ---

def recurrence_plot(x, eps=None):
    """Build a recurrence matrix: R[i,j] = 1 if |x_i - x_j| < eps.

    Diagonal lines => periodic/quasi-periodic structure.
    Vertical/horizontal lines => laminar (stuck) states.
    """
    n = len(x)
    D = np.abs(x[:, None] - x[None, :])
    if eps is None:
        eps = 0.1 * np.std(x)
    return (D < eps).astype(float)


def gramian_angular_field(x, method='summation'):
    """Encode temporal correlation via angular representation.

    1. Rescale x to [-1, 1]
    2. Convert to polar: phi_i = arccos(x_i)
    3. GAF[i,j] = cos(phi_i +/- phi_j)

    'summation' (GASF) preserves temporal correlation.
    'difference' (GADF) highlights temporal transitions.
    """
    # rescale to [-1, 1]
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min == 0:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    x_scaled = np.clip(x_scaled, -1, 1)

    phi = np.arccos(x_scaled)

    if method == 'summation':
        return np.cos(phi[:, None] + phi[None, :])
    else:
        return np.sin(phi[:, None] - phi[None, :])


def markov_transition_field(x, n_bins=8):
    """Encode transition probabilities as a 2D image.

    1. Quantize x into n_bins bins
    2. Build Markov transition matrix W (bin_i -> bin_j probability)
    3. MTF[i,j] = W[bin(x_i), bin(x_j)] -- the probability of
       transitioning from x_i's state to x_j's state
    """
    # quantize into bins
    bins = np.linspace(np.min(x), np.max(x) + 1e-9, n_bins + 1)
    binned = np.digitize(x, bins) - 1
    binned = np.clip(binned, 0, n_bins - 1)

    # build transition matrix
    W = np.zeros((n_bins, n_bins))
    for i in range(len(binned) - 1):
        W[binned[i], binned[i + 1]] += 1

    # normalize rows to probabilities
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W /= row_sums

    # construct MTF
    n = len(x)
    mtf = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf[i, j] = W[binned[i], binned[j]]

    return mtf


def fold_by_period(x, top_k=3):
    """TimesNet approach: use FFT to find dominant periods, then fold
    the 1D signal into 2D tensors of shape (num_periods x period_length).

    Returns list of (period, 2D_array) tuples for the top_k periods.
    """
    n = len(x)
    # FFT, skip DC component
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x)))[1:]
    freqs = np.fft.rfftfreq(n)[1:]

    # top-k dominant frequencies
    top_indices = np.argsort(fft_vals)[::-1][:top_k]

    results = []
    for idx in top_indices:
        period = int(np.round(1.0 / freqs[idx]))
        period = max(2, min(period, n // 2))

        # fold: truncate to full periods, reshape
        num_periods = n // period
        trimmed = x[:num_periods * period]
        folded = trimmed.reshape(num_periods, period)
        results.append((period, folded))

    return results


# --- Plotting ---

def plot_all(ticker, signal_data, signal_name, dates):
    """Time series on top spanning full width, 4 imaging plots in 2x2 below."""

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle(f'{ticker} — Time Series Imaging ({signal_name})',
                 fontsize=14, fontweight='bold', y=0.98)

    # Top row: original signal spanning full width
    ax_ts = fig.add_axes([0.08, 0.78, 0.88, 0.16])
    ax_ts.plot(dates, signal_data, color='darkgoldenrod', linewidth=0.8)
    ax_ts.set_title(f'Original Signal ({len(signal_data)} points)', fontsize=11)
    ax_ts.set_ylabel(signal_name)
    for label in ax_ts.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')

    # Bottom 2x2 grid
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.92,
                          bottom=0.05, top=0.72, hspace=0.3, wspace=0.3)

    # Recurrence Plot
    ax = fig.add_subplot(gs[0, 0])
    rp = recurrence_plot(signal_data)
    ax.imshow(rp, cmap='binary', origin='lower', aspect='equal')
    ax.set_title('Recurrence Plot')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')

    # Gramian Angular Summation Field
    ax = fig.add_subplot(gs[0, 1])
    gasf = gramian_angular_field(signal_data, method='summation')
    im = ax.imshow(gasf, cmap='RdBu_r', origin='lower', aspect='equal')
    ax.set_title('Gramian Angular Field (GASF)')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Markov Transition Field
    ax = fig.add_subplot(gs[1, 0])
    mtf = markov_transition_field(signal_data, n_bins=8)
    im = ax.imshow(mtf, cmap='inferno', origin='lower', aspect='equal')
    ax.set_title('Markov Transition Field')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Time index')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # TimesNet fold-by-period
    ax = fig.add_subplot(gs[1, 1])
    folds = fold_by_period(signal_data, top_k=3)
    period, folded = folds[0]
    im = ax.imshow(folded, cmap='viridis', aspect='auto',
                   interpolation='nearest')
    ax.set_title(f'Fold-by-Period (p={period}d)')
    ax.set_xlabel(f'Phase within period (0\u2013{period - 1})')
    ax.set_ylabel('Cycle number')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # annotate other detected periods
    other_periods = [p for p, _ in folds[1:]]
    if other_periods:
        ax.text(0.02, 0.02,
                f'Other periods: {", ".join(str(p) + "d" for p in other_periods)}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig


def extract_signal(security, span, signal_name):
    """Extract the named signal from a Security object."""
    dataset = getattr(security, span, security.daily)

    if signal_name == 'price':
        return dataset.adj_close.values, dataset.index

    if signal_name == 'rsi':
        rsi = relative_strength(dataset.adj_close.values, 7)
        return rsi, dataset.index

    if signal_name == 'returns':
        prices = dataset.adj_close.values
        returns = np.diff(np.log(prices))
        return returns, dataset.index[1:]

    if signal_name == 'volume':
        return dataset.volume.values, dataset.index

    raise ValueError(f'Unknown signal: {signal_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert time series to 2D images using RP, GAF, MTF, and fold-by-period')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols to analyze')
    parser.add_argument('--signal', default='price',
                        choices=['price', 'rsi', 'returns', 'volume'],
                        help='Which signal to image (default: price)')
    parser.add_argument('--span', default='daily',
                        choices=['daily', 'weekly', 'monthly'],
                        help='Time span (default: daily)')
    parser.add_argument('--offline', action='store_true',
                        help='Use cached data only')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to Output/ instead of showing')
    args = parser.parse_args()

    for ticker in args.tickers:
        s = Security.load(ticker, offline=args.offline)
        if s is None:
            print(f'Skipping {ticker} (no cached data)')
            continue

        signal_data, dates = extract_signal(s, args.span, args.signal)
        fig = plot_all(ticker, signal_data, args.signal, dates)

        if args.save:
            fname = f'Output/{ticker}-{args.signal}-imaging.png'
            fig.savefig(fname, dpi=150)
            print(f'Saved {fname}')
            plt.close(fig)

    if not args.save:
        plt.show()
