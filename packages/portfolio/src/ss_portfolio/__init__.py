"""ss_portfolio: portfolio metrics + weighting utilities.

  * `block_sharpe_with_costs` — differentiable Sharpe for the regime
    trainer (JAX).
  * `annualized_sharpe`       — plain numpy Sharpe of a daily return series.
  * `cagr`, `max_drawdown`, `sortino`, `calmar` — standard performance
    metrics on a daily return series.
  * `softmax_weights`           — temperature-scaled softmax with mask.
  * `apply_position_cap`        — water-fill per-name weight cap.
"""

from ss_portfolio.metrics import (
    annualized_sharpe,
    cagr,
    calmar,
    max_drawdown,
    sortino,
)
from ss_portfolio.sharpe import TRADING_DAYS, block_sharpe_with_costs
from ss_portfolio.weights import apply_position_cap, softmax_weights

__all__ = [
    'TRADING_DAYS',
    'annualized_sharpe',
    'apply_position_cap',
    'block_sharpe_with_costs',
    'cagr',
    'calmar',
    'max_drawdown',
    'softmax_weights',
    'sortino',
]
