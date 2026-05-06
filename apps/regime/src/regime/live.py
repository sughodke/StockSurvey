"""Live orchestration: checkpoint + broker -> rebalance.

This is the production loop: load a checkpoint, fetch enough recent OHLC
from the broker to score the universe, compute target weights, diff
against current positions, apply risk rails, and submit (or print, in
dry-run mode).

Risk rails — each one aborts the run with a clear reason rather than
silently coercing values, so an operator can decide what to do:

  1. Kill-switch file present  -> abort, no orders submitted.
  2. Latest bar staler than N   -> abort, prevents trading on a frozen feed.
  3. Per-name weight cap        -> clip + renormalize before sizing.
  4. Dry-run mode               -> compute and log everything, submit nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from regime.broker import Account, AlpacaBroker, Trade
from regime.inference import target_weights
from regime.persist import Checkpoint, load_checkpoint
from ss_portfolio import apply_position_cap
from ss_wavelets import KERNEL_HALF_EXTENT


DEFAULT_KILLSWITCH: str = '~/.regime-killswitch'


@dataclass
class LiveRunResult:
    """Result of one rebalance pass."""
    timestamp: str
    checkpoint_path: str
    dry_run: bool
    account: Account
    last_bar_date: str
    n_universe: int
    target_weights: pd.Series
    trades: list[Trade]
    submitted_order_ids: list[str] = field(default_factory=list)
    aborted_reason: str | None = None


def run_live(
    checkpoint_path: str | Path,
    *,
    broker: AlpacaBroker | None = None,
    dry_run: bool = True,
    max_position: float = 0.25,
    max_data_age_days: int = 3,
    killswitch_path: str | Path = DEFAULT_KILLSWITCH,
    bar_buffer_days: int = 60,
) -> LiveRunResult:
    """Run one rebalance pass and return a structured summary.

    Parameters
    ----------
    checkpoint_path :
        Path to a JSON checkpoint produced by `persist.save_checkpoint`.
    broker :
        Pre-configured `AlpacaBroker`. Constructed from environment
        credentials if omitted.
    dry_run :
        If True, compute and log trades but do not submit. Defaults to
        True so a misconfigured cron entry never accidentally trades.
    max_position :
        Per-name weight cap, in [0, 1]. Applied before sizing.
    max_data_age_days :
        Abort if the most recent bar is older than this many calendar days.
    killswitch_path :
        Abort if this file exists (allows an operator to halt trading
        without touching the cron entry).
    bar_buffer_days :
        Extra trading-day safety margin on top of the wavelet support.
        Total trading bars requested = lookback + KERNEL_HALF_EXTENT *
        max(scales) + bar_buffer_days, so the latest bar's CWT has full
        kernel support — not zero-padded as it would be with a tighter
        fetch. Default 60 trading days ≈ 3 calendar months of cushion.
    """
    cp = load_checkpoint(checkpoint_path)
    broker = broker or AlpacaBroker()
    account = broker.get_account()
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')

    ks = Path(killswitch_path).expanduser()
    if ks.exists():
        return LiveRunResult(
            timestamp=timestamp, checkpoint_path=str(checkpoint_path),
            dry_run=dry_run, account=account,
            last_bar_date='', n_universe=0,
            target_weights=pd.Series(dtype=float), trades=[],
            aborted_reason=f'kill-switch present at {ks}')

    # Per-day data dependency for `coeffs[scale, t]` is
    # KERNEL_HALF_EXTENT*scale + lookback bars (see CLAUDE.md
    # "Important implementation notes"). Without this, the latest-bar
    # CWT runs against zero-padded history and silently degrades vs
    # train. RSI checkpoints have empty scales -> max=0 -> no addition.
    max_scale = max(cp.scales) if cp.scales else 0
    n_trading_bars = cp.lookback + KERNEL_HALF_EXTENT * max_scale + bar_buffer_days
    prices, highs, lows = broker.get_recent_bars(
        cp.universe, n_days=n_trading_bars)
    last_bar = prices.index[-1]
    # `pd.Timestamp.utcnow()` is deprecated in pandas 2.x; the canonical
    # replacement is `Timestamp.now('UTC')`. Strip tz before diffing
    # against the tz-naive bar index.
    now_naive = pd.Timestamp.now('UTC').tz_convert(None).normalize()
    age_days = (now_naive - last_bar).days
    if age_days > max_data_age_days:
        return LiveRunResult(
            timestamp=timestamp, checkpoint_path=str(checkpoint_path),
            dry_run=dry_run, account=account,
            last_bar_date=str(last_bar.date()), n_universe=prices.shape[1],
            target_weights=pd.Series(dtype=float), trades=[],
            aborted_reason=(
                f'stale data: last bar {last_bar.date()} is {age_days}d old '
                f'(>{max_data_age_days})'))

    raw_weights = target_weights(prices, highs, lows, cp)
    capped = apply_position_cap(raw_weights, max_position)
    capped = capped[capped > 1e-6]

    current_positions = broker.get_positions()
    last_prices = prices.iloc[-1]
    trades = broker.build_trades(
        target_weights=capped,
        last_prices=last_prices,
        current_positions=current_positions,
        equity=account.equity,
    )

    order_ids: list[str] = []
    if not dry_run and trades:
        order_ids = broker.submit_orders(trades)

    return LiveRunResult(
        timestamp=timestamp,
        checkpoint_path=str(checkpoint_path),
        dry_run=dry_run,
        account=account,
        last_bar_date=str(last_bar.date()),
        n_universe=prices.shape[1],
        target_weights=capped,
        trades=trades,
        submitted_order_ids=order_ids,
    )


def format_run(result: LiveRunResult) -> str:
    """Render a human-readable summary of a `LiveRunResult`."""
    lines = [
        f'regime live @ {result.timestamp}',
        f'  checkpoint : {result.checkpoint_path}',
        f'  mode       : {"DRY-RUN" if result.dry_run else "LIVE"} '
        f'({"paper" if result.account.paper else "REAL MONEY"})',
        f'  account    : equity=${result.account.equity:,.2f}  '
        f'cash=${result.account.cash:,.2f}',
    ]
    if result.aborted_reason:
        lines.append(f'  ABORTED    : {result.aborted_reason}')
        return '\n'.join(lines)

    lines += [
        f'  last bar   : {result.last_bar_date}  ({result.n_universe} symbols)',
        f'  positions  : {len(result.target_weights)} target names, '
        f'{len(result.trades)} trades',
    ]
    if result.target_weights.size:
        top = result.target_weights.sort_values(ascending=False).head(10)
        lines.append('  top weights:')
        for sym, w in top.items():
            lines.append(f'    {sym:<6s} {w:6.2%}')
    if result.trades:
        lines.append('  trades:')
        for t in result.trades[:20]:
            lines.append(
                f'    {t.side.upper():<4s} {t.symbol:<6s} '
                f'{t.qty:>10.4f} sh  ${t.notional:>10,.2f}  '
                f'({t.current_weight:.2%} -> {t.target_weight:.2%})')
        if len(result.trades) > 20:
            lines.append(f'    ... {len(result.trades) - 20} more')
    if result.submitted_order_ids:
        lines.append(f'  submitted  : {len(result.submitted_order_ids)} orders')
    return '\n'.join(lines)
