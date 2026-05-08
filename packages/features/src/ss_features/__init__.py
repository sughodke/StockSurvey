"""Shared feature primitives consumed by apps/notebook + apps/factor + apps/replay.

- `TickerData`, `load_prices` — per-ticker container + adjusted-close
  loader (Stooq archive / Kaggle Nasdaq3347 / yfinance).
- `realized_vol`, `log_returns` — causal rolling-vol primitives.
- `Backbone`, `load_backbone` — npz I/O for the SSL-pretrained CNN
  backbone produced by `ss-replay --decoder cnn`. Numpy data only;
  the runtime forward pass lives in `factor.backbone`.
- `compute_scalogram`, `rolling_zscore_stats`, `log_return_signs`,
  `channels_per_lag`, `build_lagged_features`,
  `build_features_and_targets`, `load_ticker`, `TARGET_NAMES` — CWT
  feature builders. Both apps use these to construct the input
  bundle the SSL-pretrained backbone expects (channels lag-windowed
  over `K = window_cols` bars).
- `fit_stats` — R²/RMSE/max-|Δ| for prediction-vs-truth comparison.
  Lifted from `replay.metrics` so factor + relational don't need to
  depend on replay just to compute eval stats.
"""
from ss_features.backbone_io import Backbone, load_backbone
from ss_features.compression import (
    Compression,
    compress_tiles,
    compress_tiles_2d_dct_zigzag,
    compress_tiles_2d_dwt,
)
from ss_features.cwt_features import (
    TARGET_NAMES, build_features_and_targets, build_lagged_features,
    channels_per_lag, compute_scalogram, load_ticker, log_return_signs,
    rolling_zscore_stats,
)
from ss_features.metrics import fit_stats
from ss_features.ticker import DEFAULT_STOOQ_DIR, TickerData, load_prices
from ss_features.vol import log_returns, realized_vol, realized_vol_matrix
from ss_features.walkforward import (
    CalendarWindow, block_windows, calendar_windows,
)

__all__ = [
    'Backbone',
    'CalendarWindow',
    'Compression',
    'DEFAULT_STOOQ_DIR',
    'TARGET_NAMES',
    'TickerData',
    'block_windows',
    'build_features_and_targets',
    'build_lagged_features',
    'calendar_windows',
    'channels_per_lag',
    'compress_tiles',
    'compress_tiles_2d_dct_zigzag',
    'compress_tiles_2d_dwt',
    'compute_scalogram',
    'fit_stats',
    'load_backbone',
    'load_prices',
    'load_ticker',
    'log_return_signs',
    'log_returns',
    'realized_vol',
    'realized_vol_matrix',
    'rolling_zscore_stats',
]
