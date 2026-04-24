"""regime: differentiable CWT-regime portfolio strategy.

A standalone implementation of the symmetric-KL regime-shift signal,
optimized end-to-end via JAX autograd over CWT scale weights and a
softmax temperature. Includes data loading, Corwin-Schultz liquidity
filtering, training, checkpointing, and live trading via Alpaca.

CLI:
    python -m regime train --data-dir ./Nasdaq3347 --save-params model.json
    python -m regime live  --params model.json --dry-run

Note: `regime.broker` and `regime.live` import `alpaca-py` at module
load. They are not re-exported here so the training/research path
stays usable when alpaca-py isn't installed; import them directly
(`from regime.live import run_live`) when needed.
"""

from regime.cwt import ALL_SCALES, causal_cwt, precompute_windows
from regime.data import corwin_schultz_spread, load_price_matrix
from regime.inference import apply_position_cap, target_weights
from regime.persist import Checkpoint, load_checkpoint, save_checkpoint
from regime.reporting import plot_training, print_results
from regime.strategy import (
    TRADING_DAYS,
    block_sharpe_with_costs,
    regime_scores,
)
from regime.trainer import TrainResult, train

__all__ = [
    'ALL_SCALES',
    'Checkpoint',
    'TRADING_DAYS',
    'TrainResult',
    'apply_position_cap',
    'block_sharpe_with_costs',
    'causal_cwt',
    'corwin_schultz_spread',
    'load_checkpoint',
    'load_price_matrix',
    'plot_training',
    'precompute_windows',
    'print_results',
    'regime_scores',
    'save_checkpoint',
    'target_weights',
    'train',
]
