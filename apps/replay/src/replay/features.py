"""Back-compat shim. CWT feature builders moved to `ss_features`.

Existing `from replay.features import ...` callers keep
working through these re-exports. New code should import directly from
`ss_features` so the dependency direction is explicit.
"""
from ss_features import (
    TARGET_NAMES,
    TickerData,
    build_features_and_targets,
    build_lagged_features,
    channels_per_lag,
    compute_scalogram,
    load_ticker,
    log_return_signs,
    log_returns,
    realized_vol,
    rolling_zscore_stats,
)

__all__ = [
    'TARGET_NAMES',
    'TickerData',
    'build_features_and_targets',
    'build_lagged_features',
    'channels_per_lag',
    'compute_scalogram',
    'load_ticker',
    'log_return_signs',
    'log_returns',
    'realized_vol',
    'rolling_zscore_stats',
]
