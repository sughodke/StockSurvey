"""Wavelet scalogram heatmap with indicator-scale annotations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# Common technical-indicator timescales overlaid as horizontal lines on
# the scalogram for visual reference.
INDICATOR_SCALES: dict[str, int] = {
    'Weekly (5d)': 5,
    'RSI (7d)': 7,
    'RSI EMA (10d)': 10,
    'MACD fast (12d)': 12,
    'SMA (20d)': 20,
    'BBands (21d)': 21,
    'MACD slow (26d)': 26,
    'Fib lookback (90d)': 90,
}


def plot_scalogram_heatmap(
    power: np.ndarray,
    scales: np.ndarray,
    dates: np.ndarray,
    *,
    title: str = 'Wavelet Scalogram',
    annotate_indicators: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Axes:
    """Render `(n_scales, n_dates)` power array as a log-scale heatmap.

    If `ax` is omitted, a new figure is created. Returns the axes so the
    caller can attach more layers (price overlay, custom annotations).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    log_power = np.log10(power + 1e-12)
    im = ax.pcolormesh(dates, scales, log_power, cmap='inferno', shading='auto')
    ax.set_yscale('log')
    ax.set_ylabel('Scale (trading days)')
    ax.set_xlim(dates[0], dates[-1])
    ax.invert_yaxis()
    ax.set_title(title)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('log10 power')

    if annotate_indicators:
        for label, scale in INDICATOR_SCALES.items():
            if scales[0] <= scale <= scales[-1]:
                ax.axhline(scale, color='white', linewidth=0.6,
                           alpha=0.7, linestyle='--')
                ax.text(dates[2], scale * 0.88, label,
                        color='white', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='black', alpha=0.5))

    if save_path:
        ax.figure.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
        plt.close(ax.figure)
    return ax
