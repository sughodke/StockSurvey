"""Cross-sectional stock scorer on top of the replay CNN backbone.

Pipeline
--------
1. Load a replay multi-head npz (`apps/notebook` -> `ss-replay --decoder
   cnn`); strip the per-target heads and keep only the shared conv
   backbone via `load_backbone`.
2. Build `TickerData` per universe ticker using `replay.features.
   load_ticker` with the *same* scales / window_cols / include_zscore_
   stats / include_returns / lookback / rsi_n / etc. recorded in the
   backbone's `_meta` blob — otherwise the input shape won't match.
3. Call `train_scorer(tickers, backbone, ...)`. **Stage 1** runs the
   frozen backbone forward once up front and trains only the head. If
   `finetune_steps > 0`, **Stage 2** unfreezes the backbone, minibatches
   over rebalance bars, and updates head + backbone jointly (backbone
   at `learning_rate * finetune_lr_scale`). The fine-tuned backbone
   weights come back on `TrainResult.backbone_params`.
4. Training objective is per-rebalance Pearson IC against forward
   log-returns; Sharpe is tracked on val as an eval-only signal.

No-backbone alternative (`indicator_features` module)
-----------------------------------------------------
Skip steps 1-2 entirely. `load_ticker_indicators(name, cfg=...)` builds
`TickerData` from a wide stack of deterministic technical indicators
(strided RSI/CCI grids, MACD over fast-period grid, realized vol over
window grid). `train_scorer_indicators(tickers, cfg, ...)` synthesizes
an `identity_backbone(K=1, F=cfg.feature_width())` and routes through
the same `train_scorer` machinery. Use as an ablation against the
pretrained backbone, or as a standalone scorer when no SSL pretrain is
available.

Public surface
--------------
- `Backbone`, `load_backbone`, `apply_backbone`, `apply_backbone_pytree`,
  `backbone_to_pytree` — backbone module.
- `AlignedTickers`, `align_tickers`, `forward_log_returns` — data prep.
- `IndicatorGridConfig`, `build_indicator_features`,
  `load_ticker_indicators`, `make_indicator_backbone`,
  `train_scorer_indicators` — deterministic-indicator alternative.
- `init_linear`, `apply_linear`, `init_mlp`, `apply_mlp`, `get_scorer`,
  `SCORERS` — scoring heads.
- `pearson_rank_ic`, `block_sharpe` — objectives.
- `TrainResult`, `train_scorer`, `precompute_inputs`, `predict`
  — training loop + helpers.
"""
from ss_notebook.scoring.backbone import (
    Backbone, apply_backbone, apply_backbone_pytree, backbone_to_pytree,
    compute_input_stats, identity_backbone, load_backbone,
)
from ss_notebook.scoring.data import (
    AlignedTickers, align_tickers, forward_log_returns,
)
from ss_notebook.scoring.indicator_features import (
    IndicatorGridConfig, build_indicator_features, load_ticker_indicators,
    make_indicator_backbone, train_scorer_indicators,
)
from ss_notebook.scoring.objectives import block_sharpe, pearson_rank_ic
from ss_notebook.scoring.scorers import (
    SCORERS, apply_linear, apply_mlp, get_scorer, init_linear, init_mlp,
)
from ss_notebook.scoring.train import (
    TrainResult, precompute_inputs, predict, train_scorer,
)

__all__ = [
    'AlignedTickers',
    'Backbone',
    'IndicatorGridConfig',
    'SCORERS',
    'TrainResult',
    'align_tickers',
    'apply_backbone',
    'apply_backbone_pytree',
    'apply_linear',
    'apply_mlp',
    'backbone_to_pytree',
    'block_sharpe',
    'build_indicator_features',
    'compute_input_stats',
    'forward_log_returns',
    'get_scorer',
    'identity_backbone',
    'init_linear',
    'init_mlp',
    'load_backbone',
    'load_ticker_indicators',
    'make_indicator_backbone',
    'pearson_rank_ic',
    'precompute_inputs',
    'predict',
    'train_scorer',
    'train_scorer_indicators',
]
