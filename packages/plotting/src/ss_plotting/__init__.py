"""ss_plotting: shared matplotlib helpers.

  * `plot_training_curves`  — train/val Sharpe trajectory.
  * `print_scale_weights`   — pretty bar chart of learned softmax weights.
  * `plot_equity_comparison`— overlay multiple equity curves on one axis.
  * `plot_scalogram_heatmap`— log-power CWT heatmap with indicator-scale
                              annotations.

These are deliberately small, single-purpose helpers — callers wrap
them in their own figures when needed. The complex multi-panel
indicator/buy-sell layouts that the legacy v1 workflow uses are
*not* re-exported here; they remain inside `v1.plotter`.
"""

from ss_plotting.equity import plot_equity_comparison
from ss_plotting.scalogram import INDICATOR_SCALES, plot_scalogram_heatmap
from ss_plotting.training import plot_training_curves, print_scale_weights

__all__ = [
    'INDICATOR_SCALES',
    'plot_equity_comparison',
    'plot_scalogram_heatmap',
    'plot_training_curves',
    'print_scale_weights',
]
