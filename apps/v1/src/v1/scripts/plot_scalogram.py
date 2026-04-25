"""
plot_scalogram : wavelet scalogram that encodes all indicator behaviors
of a stock in a single image.

Each row = a timescale. Each column = a point in time.
Read horizontally to see how a specific timescale evolves.
Read vertically to see all timescales active at a moment.

Annotated with the project's indicator scales:
  RSI(7), EMA(10), MACD fast(12), SMA(20), BBands(21), MACD slow(26),
  weekly(5), monthly(21), Fibonacci(90).

Usage:
    uv run python plot_scalogram.py --offline AAPL
    uv run python plot_scalogram.py --offline --save AAPL NVDA MSFT
    uv run python plot_scalogram.py --offline --wavelet morlet TSLA
"""

import argparse
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import signal as scipy_signal

from v1.models.security import Security
from v1.util.indicators import relative_strength, moving_average, moving_average_convergence

warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)

# Indicator scales used in this project
INDICATOR_SCALES = {
    'Weekly (5d)': 5,
    'RSI (7d)': 7,
    'RSI EMA (10d)': 10,
    'MACD fast (12d)': 12,
    'SMA (20d)': 20,
    'BBands (21d)': 21,
    'MACD slow (26d)': 26,
    'Fib lookback (90d)': 90,
}


def compute_scalogram(prices, scales, wavelet='morlet'):
    """Compute continuous wavelet transform scalogram.

    Returns the magnitude of wavelet coefficients: shape (len(scales), len(prices)).
    """
    # Normalize prices to zero-mean unit-variance for comparable wavelet power
    x = (prices - np.mean(prices)) / (np.std(prices) + 1e-9)

    if wavelet == 'morlet':
        # scipy doesn't have morlet CWT directly, use complex morlet via convolution
        coeffs = np.zeros((len(scales), len(x)), dtype=complex)
        for i, s in enumerate(scales):
            # Morlet wavelet: exp(i*w0*t) * exp(-t^2/2), w0=5
            w0 = 5.0
            t = np.arange(-4 * s, 4 * s + 1) / s
            wavelet_data = np.exp(1j * w0 * t) * np.exp(-t ** 2 / 2)
            wavelet_data /= np.sqrt(s)
            conv = np.convolve(x, wavelet_data, mode='same')
            coeffs[i] = conv[:len(x)]
        power = np.abs(coeffs) ** 2
    else:
        # Ricker (Mexican hat) — real-valued, good for detecting peaks
        coeffs = np.zeros((len(scales), len(x)))
        for i, s in enumerate(scales):
            points = min(10 * s, len(x))
            t = np.arange(-points // 2, points // 2 + 1) / s
            wavelet_data = (1 - t ** 2) * np.exp(-t ** 2 / 2)
            wavelet_data /= np.sqrt(s)
            conv = np.convolve(x, wavelet_data, mode='same')
            coeffs[i] = conv[:len(x)]
        power = coeffs ** 2

    return power, coeffs


def compute_indicators(prices):
    """Compute all project indicators for overlay comparison."""
    rsi = relative_strength(prices, 7)
    rsi_ma10 = moving_average(rsi, 10, type='exponential')
    slow, fast, macd = moving_average_convergence(prices)
    sma20 = moving_average(prices, 20, type='simple')
    return {
        'rsi': rsi,
        'rsi_ma10': rsi_ma10,
        'macd': macd,
        'sma20': sma20,
    }


def plot_scalogram(ticker, prices, dates, wavelet_name):
    """Single-image scalogram with indicator annotations and comparison strips."""

    # Scales from 2 to 200 trading days, log-spaced for visual clarity
    scales = np.unique(np.logspace(np.log10(2), np.log10(200), 120).astype(int))
    power, coeffs = compute_scalogram(prices, scales, wavelet=wavelet_name)

    indicators = compute_indicators(prices)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'{ticker} — Wavelet Scalogram ({wavelet_name})',
                 fontsize=14, fontweight='bold', y=0.98)

    # --- Row 1: Price ---
    ax_price = fig.add_axes([0.10, 0.87, 0.78, 0.08])
    ax_price.plot(dates, prices, color='black', linewidth=0.7)
    ax_price.set_ylabel('Price', fontsize=9)
    ax_price.set_title(f'Adj Close ({len(prices)} points)', fontsize=10)
    ax_price.tick_params(labelbottom=False)
    ax_price.set_xlim(dates[0], dates[-1])

    # --- Row 2: Main scalogram ---
    ax_sg = fig.add_axes([0.10, 0.42, 0.78, 0.42])

    # Plot power as log-scale heatmap
    log_power = np.log10(power + 1e-12)
    im = ax_sg.pcolormesh(dates, scales, log_power,
                          cmap='inferno', shading='auto')
    ax_sg.set_yscale('log')
    ax_sg.set_ylabel('Scale (trading days)', fontsize=10)
    ax_sg.set_xlim(dates[0], dates[-1])
    ax_sg.invert_yaxis()
    ax_sg.tick_params(labelbottom=False)

    # Annotate indicator scales
    cbar = fig.colorbar(im, ax=ax_sg, fraction=0.02, pad=0.01)
    cbar.set_label('log₁₀ power', fontsize=9)

    for label, scale in INDICATOR_SCALES.items():
        if scales[0] <= scale <= scales[-1]:
            ax_sg.axhline(scale, color='white', linewidth=0.6, alpha=0.7,
                          linestyle='--')
            ax_sg.text(dates[2], scale * 0.88, label,
                       color='white', fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15',
                                 facecolor='black', alpha=0.5))

    # --- Row 3: Indicator strips extracted from scalogram vs actual ---
    gs = fig.add_gridspec(3, 1, left=0.10, right=0.88,
                          bottom=0.04, top=0.38, hspace=0.35)

    # RSI comparison
    ax_rsi = fig.add_subplot(gs[0])
    # Extract scale=7 power as proxy for RSI activity
    idx_7 = np.argmin(np.abs(scales - 7))
    scalogram_rsi_proxy = power[idx_7] / np.max(power[idx_7]) * 100
    ax_rsi.plot(dates, indicators['rsi'], color='darkgoldenrod',
                linewidth=0.8, label='RSI(7)')
    ax_rsi.plot(dates, indicators['rsi_ma10'], color='steelblue',
                linewidth=0.8, alpha=0.7, label='RSI EMA(10)')
    ax_rsi2 = ax_rsi.twinx()
    ax_rsi2.fill_between(dates, scalogram_rsi_proxy, alpha=0.15,
                         color='red', label='Scale=7 power')
    ax_rsi2.set_ylabel('Power', fontsize=8, color='red')
    ax_rsi2.tick_params(labelcolor='red', labelsize=7)
    ax_rsi.axhline(70, color='gray', linestyle=':', alpha=0.4)
    ax_rsi.axhline(30, color='gray', linestyle=':', alpha=0.4)
    ax_rsi.set_ylabel('RSI', fontsize=9)
    ax_rsi.set_title('Scale=7 power vs actual RSI(7)', fontsize=9)
    ax_rsi.legend(loc='upper left', fontsize=7)
    ax_rsi.tick_params(labelbottom=False)
    ax_rsi.set_xlim(dates[0], dates[-1])

    # MACD comparison
    ax_macd = fig.add_subplot(gs[1])
    # MACD ≈ difference between scale=12 and scale=26
    idx_12 = np.argmin(np.abs(scales - 12))
    idx_26 = np.argmin(np.abs(scales - 26))
    if wavelet_name == 'morlet':
        scalogram_macd_proxy = np.real(coeffs[idx_12]) - np.real(coeffs[idx_26])
    else:
        scalogram_macd_proxy = coeffs[idx_12] - coeffs[idx_26]
    # Normalize both to same range for comparison
    macd_norm = indicators['macd']
    macd_norm = macd_norm / (np.max(np.abs(macd_norm)) + 1e-9)
    proxy_norm = scalogram_macd_proxy / (np.max(np.abs(scalogram_macd_proxy)) + 1e-9)
    ax_macd.plot(dates, macd_norm, color='darkgoldenrod',
                 linewidth=0.8, label='MACD(12,26)')
    ax_macd.plot(dates, proxy_norm, color='steelblue',
                 linewidth=0.8, alpha=0.7, label='Scale 12−26 diff')
    ax_macd.axhline(0, color='gray', linewidth=0.5)
    corr = np.corrcoef(macd_norm, proxy_norm)[0, 1]
    ax_macd.set_ylabel('Normalized', fontsize=9)
    ax_macd.set_title(f'Scale(12)−Scale(26) vs actual MACD  (r={corr:.3f})',
                      fontsize=9)
    ax_macd.legend(loc='upper left', fontsize=7)
    ax_macd.tick_params(labelbottom=False)
    ax_macd.set_xlim(dates[0], dates[-1])

    # Volatility / BBands comparison
    ax_vol = fig.add_subplot(gs[2])
    idx_21 = np.argmin(np.abs(scales - 21))
    scalogram_vol = power[idx_21]
    scalogram_vol_norm = scalogram_vol / (np.max(scalogram_vol) + 1e-9)
    # Actual rolling std as BBands proxy
    rolling_std = np.array([np.std(prices[max(0, i - 21):i + 1])
                            for i in range(len(prices))])
    rolling_std_norm = rolling_std / (np.max(rolling_std) + 1e-9)
    corr_vol = np.corrcoef(rolling_std_norm, scalogram_vol_norm)[0, 1]
    ax_vol.plot(dates, rolling_std_norm, color='darkgoldenrod',
                linewidth=0.8, label='Rolling σ(21)')
    ax_vol.plot(dates, scalogram_vol_norm, color='steelblue',
                linewidth=0.8, alpha=0.7, label='Scale=21 power')
    ax_vol.set_ylabel('Normalized', fontsize=9)
    ax_vol.set_title(f'Scale=21 power vs actual BBands width  (r={corr_vol:.3f})',
                     fontsize=9)
    ax_vol.legend(loc='upper left', fontsize=7)
    ax_vol.set_xlim(dates[0], dates[-1])
    for label in ax_vol.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wavelet scalogram encoding all indicator behaviors')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols')
    parser.add_argument('--span', default='daily',
                        choices=['daily', 'weekly', 'monthly'])
    parser.add_argument('--wavelet', default='morlet',
                        choices=['morlet', 'ricker'],
                        help='Wavelet type (default: morlet)')
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
        dates = dataset.index

        fig = plot_scalogram(ticker, prices, dates, args.wavelet)

        if args.save:
            fname = f'Output/{ticker}-scalogram-{args.wavelet}.png'
            fig.savefig(fname, dpi=150)
            print(f'Saved {fname}')
            plt.close(fig)

    if not args.save:
        plt.show()
