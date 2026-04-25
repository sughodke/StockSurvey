"""Training-time plots and printouts.

Used by the regime trainer (and any future trainer) to render the
Sharpe trajectory and the learned scale-weight distribution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def print_scale_weights(
    scales: list[int],
    softmax_weights: np.ndarray,
    *,
    temperature: float | None = None,
    train_sharpe: float | None = None,
    val_sharpe: float | None = None,
) -> None:
    """ASCII bar chart of learned softmax weights over CWT scales."""
    print('\nLearned scale weights:')
    for s, w in zip(scales, softmax_weights):
        bar = '#' * int(round(float(w) * 60))
        print(f'  scale={s:3d}d  {float(w):.4f}  {bar}')
    if temperature is not None:
        print(f'Temperature : {temperature:.4f}')
    if train_sharpe is not None:
        print(f'Train Sharpe: {train_sharpe:+.4f}')
    if val_sharpe is not None:
        print(f'Val   Sharpe: {val_sharpe:+.4f}')


def plot_training_curves(
    train_history: list[float],
    val_history: list[tuple[int, float]],
    *,
    title_suffix: str = '',
    save_path: str | None = None,
) -> None:
    """Plot train/val Sharpe trajectories.

    `train_history` is one Sharpe per Adam step; `val_history` is a
    list of `(step_index, val_sharpe)` tuples sampled every N steps.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(train_history)), train_history,
            color='tab:blue', linewidth=1.0, label='Train Sharpe')
    if val_history:
        vs_x, vs_y = zip(*val_history)
        ax.plot(vs_x, vs_y, color='tab:orange', linewidth=1.0,
                label='Val Sharpe (out-of-sample)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Adam step')
    ax.set_ylabel('Sharpe (annualized)')
    title = 'JAX differentiable optimizer'
    if title_suffix:
        title = f'{title}  ({title_suffix})'
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
        plt.close(fig)
    else:
        plt.show()
