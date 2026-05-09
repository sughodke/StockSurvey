"""Back-compat shim. CWT feature builders moved to `ss_features`.

Existing `from replay.features import ...` callers keep
working through these re-exports. New code should import directly from
`ss_features` so the dependency direction is explicit.
"""
from ss_features import (
    CHANNELS_PER_SCALE,
    TARGET_NAMES,
    TickerData,
    build_features_and_targets,
    build_lagged_features,
    channels_per_lag,
    compute_scalogram,
    compute_scalogram_polar,
    load_ticker,
    log_returns,
    realized_vol,
)

__all__ = [
    'CHANNELS_PER_SCALE',
    'TARGET_NAMES',
    'TickerData',
    'build_features_and_targets',
    'build_lagged_features',
    'channels_per_lag',
    'compute_scalogram',
    'compute_scalogram_polar',
    'load_ticker',
    'log_returns',
    'realized_vol',
]
