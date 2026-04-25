"""Thin wrappers that adapt a `TrainResult` to the generic plotting helpers.

Kept inside the regime app (not in `ss_plotting`) because the wrappers
know about `TrainResult`'s shape — the underlying plot/print functions
in `ss_plotting` stay generic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from regime.research.optimize_adam import TrainResult
from ss_plotting import plot_training_curves, print_scale_weights


def print_results(result: TrainResult) -> None:
    """Print learned scale weights, temperature, and final Sharpes."""
    sw = np.asarray(jax.nn.softmax(result.params['scale_log_weights']))
    temp = float(jnp.exp(result.params['log_temperature']))
    print_scale_weights(
        result.scales, sw,
        temperature=temp,
        train_sharpe=result.train_sharpe,
        val_sharpe=result.val_sharpe,
    )


def plot_training(result: TrainResult, save_path: str | None = None) -> None:
    """Plot train/val Sharpe trajectories, save to disk or show interactively."""
    plot_training_curves(
        result.train_history, result.val_history,
        title_suffix=(
            f'train: {result.train_sharpe:+.3f}  |  '
            f'val: {result.val_sharpe:+.3f}'),
        save_path=save_path,
    )
