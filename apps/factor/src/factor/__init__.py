"""Cross-sectional rank-IC scorer.

Two paths feed the same scoring head + IC objective:

Pretrained CNN backbone (apps/notebook → `ss-replay --decoder cnn`)
------------------------------------------------------------------
1. Load a replay multi-head npz; strip per-target heads, keep the
   shared conv backbone via `load_backbone`.
2. Build `TickerData` per ticker using the SSL pretrain's matching
   scales / window_cols / etc. (recorded in the npz's `_meta`).
3. `train_scorer(tickers, backbone, ...)`. **Stage 1** runs the frozen
   backbone forward once and trains only the head. If
   `finetune_steps > 0`, **Stage 2** unfreezes the backbone and updates
   head + backbone jointly (backbone at
   `learning_rate * finetune_lr_scale`). Fine-tuned weights come back
   on `TrainResult.backbone_params`.
4. Training objective is per-rebalance Pearson IC against forward
   log-returns; Sharpe is tracked on val as an eval-only signal.

Deterministic-indicator alternative (`indicator_features`)
----------------------------------------------------------
Skip steps 1-2 entirely. `load_ticker_indicators(name, cfg=...)` builds
`TickerData` from a wide stack of strided RSI/CCI grids, MACD over a
fast-period grid, and realized vol over a window grid.
`train_scorer_indicators(tickers, cfg, ...)` synthesizes an
`identity_backbone(K=1, F=cfg.feature_width())` and routes through the
same `train_scorer` machinery. Use as an ablation against the
pretrained backbone, or as a standalone scorer when no SSL pretrain is
available.

Public surface
--------------
- `Backbone`, `load_backbone` — re-exports from `ss_features` for
  convenience; that package owns the on-disk npz format.
- `apply_backbone`, `apply_backbone_pytree`, `backbone_to_pytree`,
  `compute_input_stats`, `identity_backbone` — tinygrad runtime.
- `AlignedTickers`, `align_tickers`, `align_tickers_at_rebal`,
  `forward_log_returns` — data prep.
- `IndicatorGridConfig`, `build_indicator_features`,
  `load_ticker_indicators`, `make_indicator_backbone`,
  `train_scorer_indicators` — deterministic-indicator alternative.
- `init_linear`, `apply_linear`, `init_mlp`, `apply_mlp`, `get_scorer`,
  `SCORERS` — scoring heads.
- `pearson_rank_ic`, `block_sharpe`, `block_sharpe_long_short`,
  `long_short_weights` — objectives.
- `TrainResult`, `train_scorer`, `precompute_inputs`, `predict`
  — training loop + helpers.
"""
from factor.backbone import (
    Backbone, apply_backbone, apply_backbone_pytree, backbone_to_pytree,
    compute_input_stats, identity_backbone, load_backbone,
)
from factor.data import (
    AlignedTickers, align_tickers, align_tickers_at_rebal,
    forward_log_returns, forward_log_returns_multi,
    forward_robust_z, forward_sign_demeaned, forward_vol_innovation,
)
from factor.horizon import (
    IrregularRunResult,
    simulate_fixed_horizon_daily_pnl,
    simulate_irregular_daily_pnl,
    simulate_oracle_daily_pnl,
)
from factor.indicator_features import (
    IndicatorGridConfig, build_indicator_features, load_ticker_indicators,
    make_indicator_backbone, train_scorer_indicators,
    train_scorer_indicators_walkforward,
)
from factor.cl_encoders import (
    MiniRocketGridConfig, SpectralGridConfig,
    build_minirocket_features, build_spectral_features,
    load_ticker_minirocket, load_ticker_spectral,
    make_minirocket_backbone, make_spectral_backbone,
    train_scorer_minirocket, train_scorer_minirocket_walkforward,
    train_scorer_spectral, train_scorer_spectral_walkforward,
)
from factor.objectives import (
    block_ir_vs_ew, block_sharpe, block_sharpe_long_short,
    horizon_mixture_loss, long_short_weights, masked_mse,
    pearson_rank_ic, per_bar_pearson_ic,
)
from factor.scorers import (
    SCORERS, apply_linear, apply_mlp, apply_mlp_horizon, apply_mlp_multitask,
    get_scorer, init_linear, init_mlp, init_mlp_horizon, init_mlp_multitask,
)
from factor.train import (
    TrainResult, precompute_inputs, predict, train_scorer,
)
from factor.train_horizon_walkforward import (
    HorizonWalkForwardResult, HorizonWindow,
    train_scorer_horizon_walkforward,
)
from factor.train_walkforward import (
    WalkForwardResult, WalkForwardWindow, train_scorer_walkforward,
)
from factor.cwt_gru_walkforward import (
    CwtGruResult, CwtGruWindow, build_cwt_panel, gru_final_hidden,
    init_cwt_gru, load_ticker_cwt, train_cwt_gru_walkforward,
)

__all__ = [
    'AlignedTickers',
    'Backbone',
    'CwtGruResult',
    'CwtGruWindow',
    'HorizonWalkForwardResult',
    'HorizonWindow',
    'IndicatorGridConfig',
    'IrregularRunResult',
    'MiniRocketGridConfig',
    'SCORERS',
    'SpectralGridConfig',
    'TrainResult',
    'WalkForwardResult',
    'WalkForwardWindow',
    'align_tickers',
    'align_tickers_at_rebal',
    'apply_backbone',
    'apply_backbone_pytree',
    'apply_linear',
    'apply_mlp',
    'apply_mlp_horizon',
    'apply_mlp_multitask',
    'backbone_to_pytree',
    'block_ir_vs_ew',
    'block_sharpe',
    'block_sharpe_long_short',
    'build_cwt_panel',
    'build_indicator_features',
    'build_minirocket_features',
    'build_spectral_features',
    'compute_input_stats',
    'gru_final_hidden',
    'init_cwt_gru',
    'forward_log_returns',
    'forward_log_returns_multi',
    'forward_robust_z',
    'forward_sign_demeaned',
    'forward_vol_innovation',
    'get_scorer',
    'horizon_mixture_loss',
    'identity_backbone',
    'init_linear',
    'init_mlp',
    'init_mlp_horizon',
    'init_mlp_multitask',
    'load_backbone',
    'load_ticker_cwt',
    'load_ticker_indicators',
    'load_ticker_minirocket',
    'load_ticker_spectral',
    'long_short_weights',
    'make_indicator_backbone',
    'make_minirocket_backbone',
    'make_spectral_backbone',
    'masked_mse',
    'pearson_rank_ic',
    'per_bar_pearson_ic',
    'precompute_inputs',
    'predict',
    'simulate_fixed_horizon_daily_pnl',
    'simulate_irregular_daily_pnl',
    'simulate_oracle_daily_pnl',
    'train_cwt_gru_walkforward',
    'train_scorer',
    'train_scorer_horizon_walkforward',
    'train_scorer_indicators',
    'train_scorer_indicators_walkforward',
    'train_scorer_minirocket',
    'train_scorer_minirocket_walkforward',
    'train_scorer_spectral',
    'train_scorer_spectral_walkforward',
    'train_scorer_walkforward',
]
