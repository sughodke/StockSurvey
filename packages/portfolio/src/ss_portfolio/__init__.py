"""ss_portfolio: portfolio metrics + weighting utilities + strategies + backtester.

  * `block_sharpe_with_costs`   — block-level Sharpe with transaction
    costs (numpy; forward-only metric used by Optuna trials).
  * `annualized_sharpe`, `cagr`, `max_drawdown`, `sortino`, `calmar`
    — standard performance metrics on a daily return series.
  * `softmax_weights`           — temperature-scaled softmax with mask.
  * `select_top_n_matrix`       — hard top-N equal-weight allocation
    from a `(n_dates, n_tickers)` score matrix.
  * `apply_position_cap`        — water-fill per-name weight cap.
  * `apply_spread_mask`, `apply_nan_mask` — universe screening masks
    for ranking strategies.
  * `weights_regime` (+ `log_returns_matrix`) — canonical CWT-divergence
    top-N weights builder; used by `regime.trainer` and as the baseline
    in `relational.research`.
  * `vbt_backtest`              — vectorbt-backed daily-return backtest;
    used by the production trainer at `regime.trainer`.

The `bt`-library helpers live in the `ss_portfolio.bt_helpers` submodule
and are not re-exported from the top-level package — `bt` is an optional
dep, importing it eagerly would break installs that don't need backtests.
Import as `from ss_portfolio.bt_helpers import build_strategy, ...`.
"""

from ss_portfolio.backtest import vbt_backtest
from ss_portfolio.deflated import (
    MetricBlock,
    expected_max_sharpe,
    probabilistic_sharpe,
    standardize_oos,
)
from ss_portfolio.metrics import (
    annualized_sharpe,
    cagr,
    calmar,
    max_drawdown,
    sortino,
)
from ss_portfolio.screening import apply_nan_mask, apply_spread_mask
from ss_portfolio.sharpe import TRADING_DAYS, block_sharpe_with_costs
from ss_portfolio.sharpe_diff import SharpeDiffCI, sharpe_difference_ci
from ss_portfolio.strategies import log_returns_matrix, weights_regime
from ss_portfolio.weights import (
    apply_position_cap,
    select_top_n_matrix,
    softmax_weights,
)

__all__ = [
    'MetricBlock',
    'TRADING_DAYS',
    'annualized_sharpe',
    'apply_nan_mask',
    'apply_position_cap',
    'apply_spread_mask',
    'block_sharpe_with_costs',
    'cagr',
    'calmar',
    'expected_max_sharpe',
    'log_returns_matrix',
    'max_drawdown',
    'probabilistic_sharpe',
    'select_top_n_matrix',
    'SharpeDiffCI',
    'sharpe_difference_ci',
    'softmax_weights',
    'sortino',
    'standardize_oos',
    'vbt_backtest',
    'weights_regime',
]
