"""regime: CWT-regime portfolio strategy.

Optuna+vectorbt walk-forward search over discrete strategy hyperparameters
(lookback, n_tail, top_n, divergence, scale subset, optional rsi_n).
Persists a JSON checkpoint and trades live via Alpaca.

CLI:
    regime train --data-dir ./StooqData --save-params model.json
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
from regime.persist import Checkpoint, load_checkpoint

__all__ = [
    'Checkpoint',
    'load_checkpoint',
    'target_weights',
]
