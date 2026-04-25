"""regime: differentiable CWT-regime portfolio strategy.

A standalone implementation of the symmetric-KL regime-shift signal,
optimized end-to-end via JAX autograd over CWT scale weights and a
softmax temperature. Includes training, checkpointing, inference, and
live trading via Alpaca.

CLI:
    regime train --data-dir ./Nasdaq3347 --save-params model.json
    regime live  --params model.json --dry-run

Note: `regime.broker` and `regime.live` import `alpaca-py` at module
load. They are not re-exported here so the training/research path
stays usable when alpaca-py isn't installed; import them directly
(`from regime.live import run_live`) when needed.

Data, indicators, wavelets, portfolio metrics, and shared plotting
helpers live in their own workspace packages: `ss_loaders`,
`ss_indicators`, `ss_wavelets`, `ss_portfolio`, `ss_plotting`. Import
from there directly rather than going through `regime`.
"""

from regime.inference import target_weights
from regime.persist import Checkpoint, load_checkpoint, save_checkpoint
from regime.reporting import plot_training, print_results
from regime.research.optimize_adam import TrainResult, train

__all__ = [
    'Checkpoint',
    'TrainResult',
    'load_checkpoint',
    'plot_training',
    'print_results',
    'save_checkpoint',
    'target_weights',
    'train',
]
