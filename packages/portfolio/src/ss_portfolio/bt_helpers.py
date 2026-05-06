"""Shared helpers for `bt`-library backtests.

Every research diagnostic in `apps/relational/` and `apps/regime/` was
re-implementing the same `bt.Strategy` template + flat per-side
commission function. This module collapses that pattern into one
import. Lives in `ss_portfolio` (rather than its own package) since
`bt` is already in the consumer apps' transitive dep graph; the `bt`
import is local to this module so the rest of `ss_portfolio` stays
bt-free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import bt
import pandas as pd

if TYPE_CHECKING:
    from bt.backtest import Backtest


def make_commission_fn(bps: float):
    """Flat per-side commission as a fraction of notional.

    Returns a `bt`-compatible `commission(q, p)` closure that charges
    `|q| * p * (bps / 1e4)` per fill.
    """
    frac = bps / 10_000.0

    def commission(q, p):
        return abs(q) * p * frac

    return commission


def bt_safe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward+back-fill NaN prices for `bt`'s price feed.

    `bt`'s rebalance solver raises if a held position's price becomes
    NaN mid-holding (e.g. a ticker delists between rebal dates).
    Wide-universe diagnostics need this; small fixed-universe ones
    typically don't.
    """
    return prices.ffill().bfill()


def print_rebalance_events(
    weight_df: pd.DataFrame, name: str, rebal_days: int,
) -> None:
    """Per-event log: date, holdings (with weights), adds, removes.

    Used by the regime backtest harness to render a human-readable
    audit trail of the strategy's rebalance decisions.
    """
    rebal_weights = weight_df.iloc[::rebal_days]
    prev_holdings: set[str] = set()
    for date, row in rebal_weights.iterrows():
        held = row[row > 0].sort_values(ascending=False)
        current = set(held.index)
        added = current - prev_holdings
        removed = prev_holdings - current
        tickers_str = ', '.join(f'{t} ({w:.0%})' for t, w in held.items())
        changes: list[str] = []
        if added:
            changes.append(f'+{",".join(sorted(added))}')
        if removed:
            changes.append(f'-{",".join(sorted(removed))}')
        change_str = f'  [{" | ".join(changes)}]' if changes else ''
        print(f'  [{name}] {date.date()}  {tickers_str}{change_str}')
        prev_holdings = current


def build_strategy(
    name: str,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    rebal_days: int = 5,
    commission_bps: float = 10,
    drop_empty: bool = False,
    safe_prices: bool = False,
    verbose: bool = False,
) -> 'Backtest':
    """Wrap a weight DataFrame in a `bt.Backtest`.

    Args:
      name: backtest label.
      prices: `(n_dates, n_tickers)` close-price frame.
      weights: `(n_dates, n_tickers)` target-weight frame; only
        every `rebal_days`-th row is used as a rebalance event.
      rebal_days: stride between rebalances.
      commission_bps: flat per-side commission (basis points of
        notional).
      drop_empty: if `True`, drop rebalance rows whose absolute-weight
        sum is below 0.1 (used by pair-trade-style diagnostics where
        the score isn't fully populated in the first window).
      safe_prices: if `True`, forward+back-fill NaN prices before
        feeding them to bt (needed for wide universes with delistings).
      verbose: if `True`, print the rebalance event log.
    """
    rebal_weights = weights.iloc[::rebal_days]
    if drop_empty:
        nonzero = rebal_weights.abs().sum(axis=1) > 0.1
        if nonzero.any():
            rebal_weights = rebal_weights.loc[nonzero]
    if verbose:
        print_rebalance_events(weights, name, rebal_days)
    strategy = bt.Strategy(name, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])
    if safe_prices:
        prices = bt_safe_prices(prices)
    return bt.Backtest(
        strategy, prices,
        commissions=make_commission_fn(commission_bps),
        integer_positions=False,
    )
