"""Equity-curve comparison plot for backtest results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_comparison(
    equity_curves: dict[str, pd.Series],
    *,
    title: str = 'Equity Curves',
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> None:
    """Overlay multiple equity curves on a single axis.

    `equity_curves` maps strategy name to a pd.Series of cumulative
    equity (typically starting at 1.0). All series are plotted with
    automatic colors; legend uses the dict keys.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=name, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
        plt.close(fig)
    else:
        plt.show()
