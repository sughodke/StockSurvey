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
# TODO(review #9): `optimize_adam` is parked-and-broken since the
# ss_indicators numpy migration (gradient flow severed at the
# get_divergence boundary). Re-exported here as a public API surface,
# which misleads readers — `from regime import train` succeeds at
# import but call-fails at `jax.value_and_grad`. Either delete the
# re-export and the file (preferred — see TODO.md "Port
# block_sharpe_with_costs to tinygrad" path B), or stub `train` to
# raise NotImplementedError with a pointer to the canonical Optuna
# trainer at `regime.trainer.train`.
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
