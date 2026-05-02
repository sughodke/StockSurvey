"""Three-panel reconstruction figure: price, RSI, MACD-line per ticker."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction(
    title_subject: str,
    dates: np.ndarray,
    gt: dict[str, np.ndarray],
    recon: dict[str, np.ndarray],
    stats: dict[str, dict[str, float]], *,
    rsi_n: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    n_features: int,
    decoder: str,
    window_cols: int,
    include_zscore_stats: bool,
    include_returns: bool = False,
    include_return_sign: bool = False,
    vol_window: int = 20,
    cci_n: int = 20,
) -> plt.Figure:
    panel_specs = {
        'price': ('Close', None),
        'rsi': (f'RSI({rsi_n})', (30, 70)),
        'macd': (f'MACD({macd_fast},{macd_slow},{macd_signal}) line', (0,)),
        'vol': (f'RealizedVol({vol_window})', None),
        'cci': (f'CCI({cci_n})', (-100, 0, 100)),
    }
    panels = [(key, *panel_specs[key]) for key in panel_specs if key in gt]

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(13, 3 * len(panels)),
        sharex=True, squeeze=False)
    axes = axes.flatten()
    extras = ''.join([
        ' +zscore-stats' if include_zscore_stats else '',
        ' +returns' if include_returns else '',
        ' +return-sign' if include_return_sign else '',
    ])
    fig.suptitle(
        f'{title_subject} — CWT-slice reconstruction vs full-series '
        f'ground truth ({decoder}, K={window_cols}{extras}, '
        f'{n_features} features)',
        fontsize=13, fontweight='bold')

    for ax, (key, label, hlines) in zip(axes, panels):
        s = stats[key]
        title = (f'{label}    '
                 f'R²={s["r2"]:.4f}  RMSE={s["rmse"]:.3e}  '
                 f'max|Δ|={s["max_abs"]:.3e}')
        ax.plot(dates, gt[key], color='black', linewidth=0.7, alpha=0.6,
                label=f'{label} ground truth')
        ax.plot(dates, recon[key], color='crimson', linewidth=0.9,
                linestyle='--',
                label=f'{label} reconstructed from CWT slice')
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=9, loc='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(dates[0], dates[-1])
        if hlines is not None:
            for y in hlines:
                ax.axhline(y, color='gray', linestyle=':', alpha=0.4)

    fig.tight_layout()
    return fig
