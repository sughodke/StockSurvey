"""regime: differentiable CWT-regime portfolio strategy.

A standalone implementation of the symmetric-KL regime-shift signal,
optimized end-to-end via JAX autograd over CWT scale weights and a
softmax temperature. Includes data loading, Corwin-Schultz liquidity
filtering, training, and reporting.

CLI:
    python -m regime --data-dir ./Nasdaq3347
"""

from regime.cwt import ALL_SCALES, causal_cwt, precompute_windows
from regime.data import corwin_schultz_spread, load_price_matrix
from regime.reporting import plot_training, print_results
from regime.strategy import (
    TRADING_DAYS,
    block_sharpe_with_costs,
    regime_scores,
)
from regime.trainer import TrainResult, train

__all__ = [
    'ALL_SCALES',
    'TRADING_DAYS',
    'TrainResult',
    'block_sharpe_with_costs',
    'causal_cwt',
    'corwin_schultz_spread',
    'load_price_matrix',
    'plot_training',
    'precompute_windows',
    'print_results',
    'regime_scores',
    'train',
]
