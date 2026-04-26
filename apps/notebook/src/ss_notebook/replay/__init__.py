"""CWT-slice reconstruction probe — split into focused submodules.

Public API
----------
- `main()`              : CLI entry point (`uv run ss-replay`).
- `fit_and_evaluate`    : multi-ticker pooled fit + per-ticker eval.
- `reconstruct_indicators` : single-train-ticker convenience wrapper.
- `TickerData`, `load_ticker` : data bundle + loader for one ticker.
- `build_features_and_targets`, `compute_scalogram`,
  `build_lagged_features`, `rolling_zscore_stats` : feature builders.
- `fit_ols`, `fit_mlp`, `fit_cnn` : decoder fits.
- `fit_stats`           : R²/RMSE/max-|Δ| metric.
- `plot_reconstruction` : 3-panel reconstruction figure.
"""
from ss_notebook.replay.cli import main
from ss_notebook.replay.decoders import fit_cnn, fit_mlp, fit_ols
from ss_notebook.replay.features import (
    TARGET_NAMES,
    TickerData,
    build_features_and_targets,
    build_lagged_features,
    compute_scalogram,
    load_ticker,
    rolling_zscore_stats,
)
from ss_notebook.replay.metrics import fit_stats
from ss_notebook.replay.plot import plot_reconstruction
from ss_notebook.replay.reconstruct import (
    fit_and_evaluate,
    reconstruct_indicators,
)

__all__ = [
    'TARGET_NAMES',
    'TickerData',
    'build_features_and_targets',
    'build_lagged_features',
    'compute_scalogram',
    'fit_and_evaluate',
    'fit_cnn',
    'fit_mlp',
    'fit_ols',
    'fit_stats',
    'load_ticker',
    'main',
    'plot_reconstruction',
    'reconstruct_indicators',
    'rolling_zscore_stats',
]
