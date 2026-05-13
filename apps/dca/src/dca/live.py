"""Live orchestration for DCA — checkpoint + broker → rebalance.

Mirrors `regime/live.py` and `relational/live.py` but adds a fifth
risk rail specific to fixed-target DCA: the cadence + drift gate.

Risk rails (each aborts with a clear reason rather than silently
continuing):

  1. Kill-switch file present  -> abort, no orders submitted.
  2. Latest bar staler than N   -> abort, prevents trading on a frozen feed.
  3. Cadence + drift gate       -> abort if not yet due AND drift below threshold.
  4. Per-name weight cap        -> clip + renormalize before sizing.
  5. Dry-run mode               -> compute and log everything, submit nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from dca.persist import DCACheckpoint, load_checkpoint
from dca.state import DCAState, load_state, save_state, DEFAULT_STATE_PATH
from ss_portfolio import apply_position_cap
from ss_portfolio.broker import Account, AlpacaBroker, OrderRejection, Trade


DEFAULT_KILLSWITCH: str = '~/.dca-killswitch'

# How many trading bars of history to fetch from the broker. We only
# need the latest close + a small safety margin for ffill in case a
# symbol missed its most recent print. 10 trading days ≈ 2 weeks is
# plenty for an ETF basket.
DEFAULT_BAR_FETCH_DAYS: int = 10


@dataclass
class LiveRunResult:
    """Result of one rebal pass — same shape as regime/relational."""
    timestamp: str
    checkpoint_path: str
    strategy: str
    dry_run: bool
    account: Account
    last_bar_date: str
    n_universe: int
    target_weights: pd.Series
    current_weights: pd.Series
    max_drift: float
    days_since_rebal: int | None
    rebal_reason: str
    trades: list[Trade]
    submitted_order_ids: list[str] = field(default_factory=list)
    rejected_orders: list[OrderRejection] = field(default_factory=list)
    aborted_reason: str | None = None


def _compute_current_weights(
    positions: dict[str, float],
    last_prices: pd.Series,
    equity: float,
    universe: list[str],
) -> pd.Series:
    """Current weights at end-of-day mark, indexed by the union of held
    names and the target universe so the operator can see drift on
    *every* target name (including ones at zero weight today)."""
    if equity <= 0:
        return pd.Series(0.0, index=sorted(set(universe) | set(positions)),
                          name='current_weight')
    idx = sorted(set(universe) | set(positions))
    out = pd.Series(0.0, index=idx, name='current_weight', dtype=float)
    for sym in idx:
        qty = float(positions.get(sym, 0.0))
        if qty == 0:
            continue
        price = float(last_prices.get(sym, 0.0))
        if price <= 0:
            continue
        out[sym] = (qty * price) / equity
    return out


def _evaluate_cadence_gate(
    state: DCAState,
    today: date,
    target_weights: pd.Series,
    current_weights: pd.Series,
    min_rebal_days: int,
    drift_threshold: float,
) -> tuple[bool, str, float, int | None]:
    """Decide whether to rebalance now.

    Returns `(should_rebal, reason, max_drift, days_since_rebal)`.
    Two ways to fire: cadence floor met OR drift exceeds threshold.
    """
    aligned_target = target_weights.reindex(current_weights.index, fill_value=0.0)
    drift = (current_weights - aligned_target).abs()
    max_drift = float(drift.max()) if len(drift) else 0.0

    if state.last_rebal_date is None:
        return True, 'no prior rebal recorded — first run', max_drift, None

    days_since = (today - state.last_rebal_date).days

    if days_since >= min_rebal_days:
        return (True,
                f'cadence floor met ({days_since}d ≥ {min_rebal_days}d)',
                max_drift, days_since)

    if max_drift >= drift_threshold:
        return (True,
                f'drift {max_drift*100:.2f}% ≥ threshold {drift_threshold*100:.2f}%',
                max_drift, days_since)

    next_due = min_rebal_days - days_since
    return (False,
            f'gate held: {days_since}d since last rebal '
            f'(next due in ~{next_due}d), max drift '
            f'{max_drift*100:.2f}% < threshold {drift_threshold*100:.2f}%',
            max_drift, days_since)


def run_live(
    checkpoint_path: str | Path,
    *,
    broker: AlpacaBroker | None = None,
    dry_run: bool = True,
    max_position: float = 0.15,
    max_data_age_days: int = 3,
    killswitch_path: str | Path = DEFAULT_KILLSWITCH,
    state_path: str | Path = DEFAULT_STATE_PATH,
    bar_fetch_days: int = DEFAULT_BAR_FETCH_DAYS,
    force_rebal: bool = False,
) -> LiveRunResult:
    """Run one DCA rebal pass.

    Parameters
    ----------
    checkpoint_path :
        Path to a JSON checkpoint produced by `persist.save_checkpoint`.
    broker :
        Pre-configured `AlpacaBroker`. Constructed from environment
        credentials if omitted.
    dry_run :
        If True, compute and log trades but do not submit. Default True
        so a misconfigured cron entry never accidentally trades.
    max_position :
        Per-name weight cap, in (0, 1]. Default 0.15 — well above the
        0.077 target on a 13-asset EW so the cap is a *diagnostic* rail
        only (would only fire on a corrupted checkpoint).
    max_data_age_days :
        Abort if the most recent bar is older than this many calendar
        days. Default 3 (matches `regime` / `relational`).
    killswitch_path :
        Abort if this file exists.
    state_path :
        Local JSON file tracking last rebal date. Created on first
        live (non-dry-run) rebal.
    bar_fetch_days :
        Trading bars of history to request from broker. Default 10 —
        enough to ffill any missing recent print on a single ETF.
    force_rebal :
        Bypass the cadence + drift gate. Use sparingly (operator's
        manual rebal command).
    """
    cp: DCACheckpoint = load_checkpoint(checkpoint_path)
    broker = broker or AlpacaBroker()
    account = broker.get_account()
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    today_local = date.today()
    target = pd.Series(cp.target_weights, name='target_weight', dtype=float)

    def _abort(reason: str, last_bar='', n=0,
               current_w=None, max_drift=0.0, days=None) -> LiveRunResult:
        return LiveRunResult(
            timestamp=timestamp, checkpoint_path=str(checkpoint_path),
            strategy=cp.name, dry_run=dry_run, account=account,
            last_bar_date=last_bar, n_universe=n,
            target_weights=target,
            current_weights=current_w if current_w is not None
                            else pd.Series(dtype=float),
            max_drift=max_drift, days_since_rebal=days,
            rebal_reason='', trades=[], aborted_reason=reason,
        )

    # Rail 1: kill-switch
    ks = Path(killswitch_path).expanduser()
    if ks.exists():
        return _abort(f'kill-switch present at {ks}')

    # Fetch bars (lightweight — only need latest close)
    prices, _highs, _lows = broker.get_recent_bars(cp.universe, n_days=bar_fetch_days)
    last_bar = prices.index[-1]
    now_naive = pd.Timestamp.now('UTC').tz_convert(None).normalize()
    age_days = (now_naive - last_bar).days

    # Rail 2: data freshness
    if age_days > max_data_age_days:
        return _abort(
            f'stale data: last bar {last_bar.date()} is {age_days}d old '
            f'(>{max_data_age_days})',
            last_bar=str(last_bar.date()), n=prices.shape[1])

    last_prices = prices.iloc[-1]
    positions = broker.get_positions()
    current_w = _compute_current_weights(
        positions, last_prices, account.equity, cp.universe)

    # Rail 3: cadence + drift gate
    state = load_state(state_path)
    should_rebal, reason, max_drift, days_since = _evaluate_cadence_gate(
        state, today_local, target, current_w,
        cp.min_rebal_days, cp.drift_threshold,
    )
    if force_rebal:
        should_rebal = True
        reason = f'force_rebal=True (overriding gate: {reason})'
    if not should_rebal:
        # Gate held — no orders, but return a populated result so the
        # operator can still see drift + days-until-next-rebal.
        return LiveRunResult(
            timestamp=timestamp, checkpoint_path=str(checkpoint_path),
            strategy=cp.name, dry_run=dry_run, account=account,
            last_bar_date=str(last_bar.date()), n_universe=prices.shape[1],
            target_weights=target, current_weights=current_w,
            max_drift=max_drift, days_since_rebal=days_since,
            rebal_reason=reason, trades=[],
        )

    # Rail 4: per-name cap (water-fill). Should be a no-op on a 13-name
    # 1/13 EW basket as long as max_position > 1/n.
    capped = apply_position_cap(target.copy(), max_position)
    capped = capped[capped > 1e-6]

    # Build trades against current positions
    trades = broker.build_trades(
        target_weights=capped,
        last_prices=last_prices,
        current_positions=positions,
        equity=account.equity,
    )

    # Rail 5: dry-run gate
    order_ids: list[str] = []
    rejections: list[OrderRejection] = []
    if not dry_run and trades:
        order_ids, rejections = broker.submit_orders(trades)
        # Persist last_rebal_date *only* on a real (non-dry-run) submission
        # AND only if at least one order succeeded. Otherwise the cadence
        # state would advance based on a no-op or all-rejected run.
        if order_ids:
            save_state(
                DCAState(last_rebal_date=today_local,
                         last_rebal_checkpoint=str(checkpoint_path)),
                state_path,
            )

    return LiveRunResult(
        timestamp=timestamp,
        checkpoint_path=str(checkpoint_path),
        strategy=cp.name,
        dry_run=dry_run,
        account=account,
        last_bar_date=str(last_bar.date()),
        n_universe=prices.shape[1],
        target_weights=capped,
        current_weights=current_w,
        max_drift=max_drift,
        days_since_rebal=days_since,
        rebal_reason=reason,
        trades=trades,
        submitted_order_ids=order_ids,
        rejected_orders=rejections,
    )


def format_run(result: LiveRunResult) -> str:
    """Render a human-readable summary."""
    lines = [
        f'dca live @ {result.timestamp}',
        f'  checkpoint : {result.checkpoint_path}',
        f'  strategy   : {result.strategy}',
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
        f'  drift      : max {result.max_drift*100:.2f}%',
    ]
    if result.days_since_rebal is not None:
        lines.append(f'  cadence    : {result.days_since_rebal}d since last rebal')
    if result.rebal_reason:
        lines.append(f'  decision   : {result.rebal_reason}')

    # Side-by-side current vs target for full universe
    if result.target_weights.size:
        joined = pd.concat([
            result.current_weights.rename('current'),
            result.target_weights.rename('target'),
        ], axis=1).fillna(0.0)
        joined['drift'] = joined['current'] - joined['target']
        joined = joined.sort_values('target', ascending=False)
        lines.append('  positions  :')
        lines.append(f'    {"sym":<6s} {"current":>8s} {"target":>8s} {"drift":>8s}')
        for sym, row in joined.iterrows():
            lines.append(
                f'    {sym:<6s} {row["current"]*100:>7.2f}% '
                f'{row["target"]*100:>7.2f}% {row["drift"]*100:>+7.2f}%'
            )

    if result.trades:
        lines.append(f'  trades ({len(result.trades)}):')
        for t in result.trades:
            lines.append(
                f'    {t.side.upper():<4s} {t.symbol:<6s} '
                f'{t.qty:>10.4f} sh  ${t.notional:>10,.2f}  '
                f'({t.current_weight*100:5.2f}% -> {t.target_weight*100:5.2f}%)')
    elif not result.aborted_reason and not result.rebal_reason.startswith('gate held'):
        lines.append('  trades     : (none — already at target within $1 min-notional)')

    if result.submitted_order_ids:
        lines.append(f'  submitted  : {len(result.submitted_order_ids)} orders')
    if result.rejected_orders:
        lines.append(f'  REJECTED   : {len(result.rejected_orders)} orders')
        for rej in result.rejected_orders[:5]:
            lines.append(f'    {rej.symbol:<6s} qty={rej.qty:.4f}: {rej.reason}')
        if len(result.rejected_orders) > 5:
            lines.append(f'    ... {len(result.rejected_orders) - 5} more')
    return '\n'.join(lines)


__all__ = [
    'DEFAULT_KILLSWITCH',
    'DEFAULT_BAR_FETCH_DAYS',
    'LiveRunResult',
    'run_live',
    'format_run',
]
