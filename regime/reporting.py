"""Pretty-printing and matplotlib output for a `TrainResult`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from regime.trainer import TrainResult


def print_results(result: TrainResult) -> None:
    """Print learned scale weights, temperature, and final Sharpes."""
    sw = np.asarray(jax.nn.softmax(result.params['scale_log_weights']))
    temp = float(jnp.exp(result.params['log_temperature']))

    print('\nLearned scale weights:')
    for s, w in zip(result.scales, sw):
        bar = '#' * int(round(w * 60))
        print(f'  scale={s:3d}d  {w:.4f}  {bar}')
    print(f'Temperature : {temp:.4f}')
    print(f'Train Sharpe: {result.train_sharpe:+.4f}')
    print(f'Val   Sharpe: {result.val_sharpe:+.4f}')


def plot_training(result: TrainResult, save_path: str | None = None) -> None:
    """Plot train/val Sharpe trajectories, save to disk or show interactively."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(result.train_history)), result.train_history,
            color='tab:blue', linewidth=1.0, label='Train Sharpe')
    if result.val_history:
        vs_x, vs_y = zip(*result.val_history)
        ax.plot(vs_x, vs_y, color='tab:orange', linewidth=1.0,
                label='Val Sharpe (out-of-sample)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Adam step')
    ax.set_ylabel('Sharpe (annualized)')
    ax.set_title(
        f'JAX differentiable regime optimizer  '
        f'(train: {result.train_sharpe:+.3f}  |  val: {result.val_sharpe:+.3f})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
        plt.close(fig)
    else:
        plt.show()
