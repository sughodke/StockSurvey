"""Shared feature primitives consumed by apps/notebook + apps/factor.

- `TickerData`, `load_prices` — per-ticker container + adjusted-close
  loader (Stooq archive / Kaggle Nasdaq3347 / yfinance).
- `realized_vol`, `log_returns` — causal rolling-vol primitives.
- `Backbone`, `load_backbone` — npz I/O for the SSL-pretrained CNN
  backbone produced by `ss-replay --decoder cnn`. Numpy data only;
  the runtime forward pass lives in `factor.backbone`.
"""
from ss_features.backbone_io import Backbone, load_backbone
from ss_features.ticker import DEFAULT_STOOQ_DIR, TickerData, load_prices
from ss_features.vol import log_returns, realized_vol

__all__ = [
    'Backbone',
    'DEFAULT_STOOQ_DIR',
    'TickerData',
    'load_backbone',
    'load_prices',
    'log_returns',
    'realized_vol',
]
