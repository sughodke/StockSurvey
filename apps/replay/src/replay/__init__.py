"""CWT-slice reconstruction probe — split into focused submodules.

Public API
----------
- `main()`              : CLI entry point (`uv run ss-replay`).
- `fit_and_evaluate`    : multi-ticker pooled fit + per-ticker eval.
- `reconstruct_indicators` : single-train-ticker convenience wrapper.
- `TickerData`, `load_ticker` : data bundle + loader for one ticker.
- `build_features_and_targets`, `compute_scalogram`,
  `build_lagged_features`, `rolling_zscore_stats` : feature builders.
- `fit_ols`, `fit_mlp`, `fit_cnn`, `fit_cnn_multihead` : decoder fits.
  `fit_cnn_multihead` is what `fit_and_evaluate` actually calls when
  `decoder='cnn'` — shared backbone, per-target heads.
- `fit_stats`           : R²/RMSE/max-|Δ| metric.
- `plot_reconstruction` : 3-panel reconstruction figure.
"""
from replay.cli import main
from replay.decoders import (
    fit_cnn, fit_cnn_masked_ae, fit_cnn_multihead, fit_mlp, fit_ols,
)
from replay.features import (
    TARGET_NAMES,
    TickerData,
    build_features_and_targets,
    build_lagged_features,
    compute_scalogram,
    load_ticker,
    rolling_zscore_stats,
)
from replay.metrics import fit_stats
from replay.plot import plot_reconstruction
from replay.reconstruct import (
    fit_and_evaluate,
    fit_and_evaluate_ssl,
    reconstruct_indicators,
)

__all__ = [
    'TARGET_NAMES',
    'TickerData',
    'build_features_and_targets',
    'build_lagged_features',
    'compute_scalogram',
    'fit_and_evaluate',
    'fit_and_evaluate_ssl',
    'fit_cnn',
    'fit_cnn_masked_ae',
    'fit_cnn_multihead',
    'fit_mlp',
    'fit_ols',
    'fit_stats',
    'load_ticker',
    'main',
    'plot_reconstruction',
    'reconstruct_indicators',
    'rolling_zscore_stats',
]
