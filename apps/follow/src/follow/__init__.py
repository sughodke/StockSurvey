"""follow — leadership-tier congressional-disclosure follower strategy.

Public surface:
    * ``follow.data``       — congress disclosure → price-panel join, leadership filter.
    * ``follow.backtest``   — vectorized walk-forward roster + return computation.
    * ``follow.cli``        — ``ss-follow`` argparse entrypoint.
"""

from follow.data import (
    build_eligible_disclosures,
    DisclosurePanel,
)
from follow.backtest import (
    build_position_history,
    BacktestResult,
    run_backtest,
)

__all__ = [
    'DisclosurePanel',
    'BacktestResult',
    'build_eligible_disclosures',
    'build_position_history',
    'run_backtest',
]
