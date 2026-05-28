"""Live orchestration for vol-v3 — checkpoint + Alpaca options paper -> rebalance.

Mirrors `regime/live.py` and `relational/live.py`: same four risk
rails plus two vol-specific ones. Differences are mechanical, not
philosophical:

- Inference is short-vol options, not equity weights. Output is a
  list of `Strangle` legs to submit (or unwind).
- Data dependencies are richer: per-name option chains (Alpaca),
  underlying bars (Alpaca), VIX series (FRED via `ss_macro`).
- Per-name cap → per-name vega cap (configured in
  `StranglesConfig.vega_budget_per_name_usd`); portfolio-level cap
  is total vega budget split across the fired names.

Risk rails (each aborts the run with a clear reason):
  1. Kill-switch file present  -> abort, no orders submitted.
  2. Latest underlying bar staler than N days -> abort.
  3. VIX gate must fire today -> abort (do nothing) if gate is closed.
     (This is the v3 deployment recipe — short-vol only fires when
     VIX > 126d-median.)
  4. Total vega budget cap across fired picks -> clip to budget.
  5. Dry-run by default — no orders submitted unless --live.
  6. Realized-friction monitor — abort if rolling-3-rebal mean
     realized c_options_bps exceeds `max_realized_c_bps` (default 250).
     The vol-v3 sleeve-sizing finding (partial-OOS) explicitly
     depends on c_options_bps <= 200; this rail enforces the
     condition against MEASURED bid-ask spreads, not modeled ones.
     Today's rebal contributes to the rolling history regardless of
     whether it triggers an abort; the abort fires on the *prior*
     rolling-3 mean (so the most-recent observation has time to
     persist).

Two more implicit prerequisites enforced earlier:
  - Checkpoint validation via `persist.validate(cp)` at load time.
  - Per-leg liquidity gates inside `strangle.build_short_strangle`
    (open interest, bid size, spread%); names that fail are dropped
    from the trade list, NOT silently zeroed.

This module is the orchestration layer. Today's MVP is dry-run only:
it logs the strangles it WOULD submit. Wiring up the actual Alpaca
options API calls (multi-leg `OptionLegRequest` construction +
submission) is `submit_strangles` and is gated behind `dry_run=False`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vol.persist import VolCheckpoint, load_checkpoint, validate
from vol.strangle import Strangle


DEFAULT_KILLSWITCH: str = '~/.vol-killswitch'

# Rail 6 — realized-friction monitor. The sleeve-sizing finding
# (`apps/docs/docs/findings/vol-sleeve-sizing.md`) is partial-OOS at
# c_options_bps <= 200 and collapses to CI-includes-0 at c_options_bps
# = 400. 250 bps is the locked abort threshold: half-way through the
# danger zone, with rolling-3-rebal averaging so a single noisy quote
# doesn't trip the kill.
DEFAULT_FRICTION_HISTORY: str = '~/.vol-friction-history.csv'
DEFAULT_MAX_REALIZED_C_BPS: float = 250.0
FRICTION_ROLLING_WINDOW_REBALS: int = 3


def _round_trip_c_bps_from_strangles(strangles: list[Strangle]) -> float:
    """Per-rebal realized round-trip c_options_bps, estimated from the
    bid-ask spreads of the chosen legs at construction time.

    Round-trip cost per leg = full spread (cross half-spread on entry
    + half-spread on exit). A short strangle has two legs, so the
    per-strangle round-trip is `call.spread_pct + put.spread_pct`.
    We report the vega-weighted mean across strangles in basis points.

    Returns NaN if no strangles or spreads are NaN — caller treats
    NaN as "no realized signal this rebal" and does not update the
    rolling mean.
    """
    import math
    total_w = 0.0
    weighted = 0.0
    for s in strangles:
        cs = getattr(s.call, 'spread_pct_at_construction', float('nan'))
        ps = getattr(s.put,  'spread_pct_at_construction', float('nan'))
        if not (math.isfinite(cs) and math.isfinite(ps)):
            continue
        w = abs(s.net_vega)
        if w <= 0:
            continue
        rt_pct = cs + ps                  # full round-trip per strangle
        weighted += rt_pct * w
        total_w += w
    if total_w == 0:
        return float('nan')
    return (weighted / total_w) * 10_000.0  # pct -> bps


def _load_friction_history(path: Path) -> list[float]:
    """Read the rolling c_bps log; empty list if file missing.
    File format: one float per line, oldest first."""
    if not path.exists():
        return []
    try:
        vals = []
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            try:
                vals.append(float(ln.split(',')[-1]))
            except ValueError:
                continue
        return vals
    except OSError:
        return []


def _append_friction_history(path: Path, timestamp: str, c_bps: float,
                             max_keep: int = 100) -> None:
    """Append today's measured c_bps to the rolling history file.
    Keeps the last `max_keep` entries (auto-truncates from the front)
    so the file doesn't grow unbounded."""
    import math
    if not math.isfinite(c_bps):
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text().splitlines() if path.exists() else []
    existing.append(f'{timestamp},{c_bps:.2f}')
    if len(existing) > max_keep:
        existing = existing[-max_keep:]
    path.write_text('\n'.join(existing) + '\n')


# --- Result types ----------------------------------------------------------

@dataclass
class GateState:
    fired: bool
    vix_now: float
    rolling_median: float
    lookback_days: int


@dataclass
class LiveRunResult:
    """Result of one rebalance pass.

    Carries the full audit trail an operator needs to verify what the
    bot did (or would have done) — picks, strangles, rejections,
    portfolio-level vega exposure.
    """
    timestamp: str
    checkpoint_path: str
    dry_run: bool
    paper: bool
    account_equity: float
    account_cash: float
    last_bar_date: str
    n_universe: int
    n_eligible: int                     # names that survived liquidity gates
    gate: GateState
    top_k_picks: pd.Series              # symbol -> predicted iv_rv_gap
    strangles: list[Strangle] = field(default_factory=list)
    submitted_order_ids: list[str] = field(default_factory=list)
    rejected_orders: list[tuple[str, str]] = field(default_factory=list)
    aborted_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    # Rail 6 audit
    realized_c_bps_today: float = float('nan')
    realized_c_bps_rolling: float = float('nan')


# --- Orchestration ---------------------------------------------------------

def run_live(
    checkpoint_path: str | Path,
    *,
    broker=None,
    options_data=None,
    bars_data=None,
    vix_loader=None,
    dry_run: bool = True,
    max_total_vega_usd: float = 5000.0,
    max_data_age_days: int = 3,
    killswitch_path: str | Path = DEFAULT_KILLSWITCH,
    max_realized_c_bps: float = DEFAULT_MAX_REALIZED_C_BPS,
    friction_history_path: str | Path = DEFAULT_FRICTION_HISTORY,
) -> LiveRunResult:
    """Run one vol-v3 rebalance pass and return a structured summary.

    Parameters
    ----------
    checkpoint_path : path to a VolCheckpoint JSON
    broker : alpaca-py TradingClient or compatible (None -> env-vars)
    options_data : alpaca-py OptionHistoricalDataClient or compatible
    bars_data : alpaca-py StockHistoricalDataClient or compatible
    vix_loader : callable -> pd.Series (None -> ss_macro.load_fred_series)
    dry_run : if True, log only; do not submit. Default True.
    max_total_vega_usd : portfolio-level vega-budget cap. Total |net_vega|
        across all submitted strangles is clipped to this.
    max_data_age_days : abort if latest underlying bar is older than this.
    killswitch_path : abort if this file exists.
    max_realized_c_bps : rail 6 — abort if rolling-N-rebal mean realized
        round-trip c_options_bps exceeds this (default 250). Derived from
        constructed strangle leg bid-ask spreads; today's value is
        appended to `friction_history_path` so the next run sees it.
    friction_history_path : CSV log of `timestamp,c_bps` for rail 6.

    Returns
    -------
    LiveRunResult with the full audit trail.

    The broker/options-data/bars-data wiring is parameterized so this
    is testable with mocks; the CLI fills them with real Alpaca
    clients.
    """
    # Lazy imports — only the live CLI path actually needs Alpaca
    # configured. Tests can pass mocks for all four dependencies.
    cp = load_checkpoint(checkpoint_path)
    validate(cp)

    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    notes: list[str] = []

    # Rail 1: kill-switch
    ks = Path(killswitch_path).expanduser()
    if ks.exists():
        return _aborted(checkpoint_path, timestamp, dry_run,
                        f'kill-switch present at {ks}')

    # Lazy-load Alpaca clients + VIX
    if broker is None or options_data is None or bars_data is None:
        broker, options_data, bars_data = _default_alpaca_clients()
    if vix_loader is None:
        from ss_macro import load_fred_series
        vix_loader = lambda: load_fred_series(cp.gate_fred_series)

    # Account snapshot
    acct = broker.get_account()
    paper = _is_paper_account(broker)

    # Rail 3: VIX gate (computed first because if the gate is closed,
    # everything else is wasted compute — vol-v3 short-vol only
    # deploys on fired rebals).
    vix = vix_loader().dropna()
    from vol.inference import gate_fires
    fired, vix_now, vix_med = gate_fires(vix, cp.gate_lookback_trading_days)
    gate = GateState(fired=fired, vix_now=vix_now, rolling_median=vix_med,
                     lookback_days=cp.gate_lookback_trading_days)
    if not fired:
        return LiveRunResult(
            timestamp=timestamp, checkpoint_path=str(checkpoint_path),
            dry_run=dry_run, paper=paper,
            account_equity=float(acct.equity), account_cash=float(acct.cash),
            last_bar_date='', n_universe=len(cp.universe), n_eligible=0,
            gate=gate, top_k_picks=pd.Series(dtype=float), strangles=[],
            aborted_reason=(
                f'VIX gate closed: VIX={vix_now:.2f} <= '
                f'{cp.gate_lookback_trading_days}d-median {vix_med:.2f}'),
            notes=notes)

    # Build the feature row from real-time Alpaca chains. This is the
    # data-heavy step — defer to a helper to keep orchestration clean.
    from vol.iv_compute import build_feature_row, atm_iv_from_chain, realized_vol_from_bars
    features, last_bar_date, n_eligible = _build_feature_panel(
        cp, options_data, bars_data, atm_iv_from_chain, realized_vol_from_bars)

    # Rail 2: data freshness
    if last_bar_date:
        last_dt = pd.Timestamp(last_bar_date)
        age = (pd.Timestamp.utcnow().tz_convert(None).normalize() - last_dt).days
        if age > max_data_age_days:
            return _aborted(
                checkpoint_path, timestamp, dry_run,
                f'stale underlying bar: {last_bar_date} is {age}d old '
                f'(>{max_data_age_days})')

    # Inference: predict, top-K, construct strangles
    from vol.inference import predict_iv_rv_gap, select_top_k
    pred = predict_iv_rv_gap(features, cp)
    picks = select_top_k(pred, cp.top_k)
    notes.append(f'predicted {len(pred)} names, selected top {len(picks)}')

    strangles = _build_strangles_for_picks(
        picks, cp, options_data, bars_data, notes)

    # Rail 4: total vega cap
    total_vega = sum(abs(s.net_vega) for s in strangles)
    if total_vega > max_total_vega_usd:
        scale = max_total_vega_usd / total_vega
        strangles = _scale_strangles(strangles, scale)
        notes.append(
            f'total |vega| ${total_vega:.0f} > cap ${max_total_vega_usd:.0f}; '
            f'scaled by {scale:.3f}')

    # Rail 6: realized-friction monitor (vol-specific, post-build).
    # Measure today's c_options_bps from the constructed strangles'
    # bid-ask spreads, then check the rolling-3-rebal mean against
    # `max_realized_c_bps`. The vol-v3 sleeve-sizing finding's
    # partial-OOS verdict depends on c_options_bps <= 200; this rail
    # enforces against MEASURED spreads. The history is loaded BEFORE
    # appending today's reading, so the abort fires on the prior
    # window's mean (avoids today's noise self-tripping the kill).
    fric_path = Path(friction_history_path).expanduser()
    prior_history = _load_friction_history(fric_path)
    realized_c_bps_today = _round_trip_c_bps_from_strangles(strangles)
    rolling = prior_history[-FRICTION_ROLLING_WINDOW_REBALS:]
    rolling_mean = sum(rolling) / len(rolling) if rolling else float('nan')
    # Persist today's reading regardless of abort decision so the
    # next run can see it.
    import math as _math
    if _math.isfinite(realized_c_bps_today):
        _append_friction_history(fric_path, timestamp, realized_c_bps_today)
        notes.append(
            f'realized c_options_bps today={realized_c_bps_today:.1f} '
            f'(prior rolling-{FRICTION_ROLLING_WINDOW_REBALS} '
            f'mean={rolling_mean:.1f} bps)')
    if (_math.isfinite(rolling_mean)
            and len(rolling) >= FRICTION_ROLLING_WINDOW_REBALS
            and rolling_mean > max_realized_c_bps):
        result = _aborted(
            checkpoint_path, timestamp, dry_run,
            f'rail 6: rolling-{FRICTION_ROLLING_WINDOW_REBALS} mean '
            f'c_options_bps {rolling_mean:.1f} > cap '
            f'{max_realized_c_bps:.0f}; sleeve-sizing finding does '
            f'not support deployment at this friction level')
        result.realized_c_bps_today = realized_c_bps_today
        result.realized_c_bps_rolling = rolling_mean
        result.gate = gate
        result.notes = notes
        return result

    # Always run the submit path — dry_run=True produces synthetic
    # DRY_RUN_* order ids so the operator sees what *would* be submitted.
    # The real-money guard is the `dry_run` flag passed through.
    order_ids: list[str] = []
    rejections: list[tuple[str, str]] = []
    if strangles:
        order_ids, rejections = submit_strangles(broker, strangles, dry_run=dry_run)
        notes.append(
            f'{"would submit" if dry_run else "submitted"} '
            f'{len(order_ids)} multi-leg orders; {len(rejections)} rejected')

    return LiveRunResult(
        timestamp=timestamp,
        checkpoint_path=str(checkpoint_path),
        dry_run=dry_run, paper=paper,
        account_equity=float(acct.equity), account_cash=float(acct.cash),
        last_bar_date=last_bar_date, n_universe=len(cp.universe),
        n_eligible=n_eligible, gate=gate,
        realized_c_bps_today=realized_c_bps_today,
        realized_c_bps_rolling=rolling_mean,
        top_k_picks=picks, strangles=strangles,
        submitted_order_ids=order_ids, rejected_orders=rejections,
        notes=notes,
    )


# --- Helpers (extracted so `run_live` reads top-down) ----------------------

def _aborted(checkpoint_path, timestamp, dry_run, reason) -> LiveRunResult:
    # paper=True is the safe display default — we abort before
    # contacting the broker, so we can't know the account's real
    # mode. Better to under-claim than to print "REAL MONEY" when
    # nothing was submitted.
    return LiveRunResult(
        timestamp=timestamp, checkpoint_path=str(checkpoint_path),
        dry_run=dry_run, paper=True, account_equity=0.0, account_cash=0.0,
        last_bar_date='', n_universe=0, n_eligible=0,
        gate=GateState(fired=False, vix_now=float('nan'),
                       rolling_median=float('nan'), lookback_days=0),
        top_k_picks=pd.Series(dtype=float), strangles=[],
        aborted_reason=reason)


def _default_alpaca_clients():
    """Wire up the three Alpaca clients from environment variables.

    Honors `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and
    `ALPACA_BASE_URL` (paper if substring 'paper'). Same convention
    as `ss_portfolio.broker.AlpacaBroker`.
    """
    import os
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    key = os.environ['ALPACA_API_KEY']
    secret = os.environ['ALPACA_SECRET_KEY']
    base = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    paper = 'paper' in base.lower()
    trading = TradingClient(api_key=key, secret_key=secret, paper=paper, url_override=base)
    options = OptionHistoricalDataClient(api_key=key, secret_key=secret)
    stocks = StockHistoricalDataClient(api_key=key, secret_key=secret)
    return trading, options, stocks


def _is_paper_account(broker) -> bool:
    """Heuristic; the broker object may or may not expose a flag."""
    return bool(getattr(broker, '_paper', getattr(broker, 'paper', True)))


def _build_feature_panel(cp, options_data, bars_data, iv_fn, rv_fn):
    """Pull per-name Alpaca chains + bars, build the 4-feature predictor input.

    Returns ``(features, last_bar_date, n_eligible)`` where:
      features : DataFrame indexed by symbol with the four
                 `LIVE_FEATURE_NAMES` columns.
      last_bar_date : ISO date string of the freshest underlying close
                      seen across the universe.
      n_eligible : how many names actually produced finite features
                   (the rest are dropped — the predictor only scores
                   what it has).
    """
    from vol.alpaca_chain import (
        fetch_option_snapshot_chain, fetch_underlying_bars,
    )
    from vol.iv_compute import build_feature_row
    from vol.iv_history import append_snapshot, load_history

    # Step 1: underlying bars for the whole universe in one batch.
    bars = fetch_underlying_bars(bars_data, cp.universe, days=40)
    if bars.empty:
        raise RuntimeError(
            f'no underlying bars returned for {len(cp.universe)} universe '
            'symbols; check Alpaca credentials and market-data permissions')
    last_bar_date = str(bars.index.max().date())

    # Step 2: per-name chain query + ATM IV synth + realized vol.
    iv_now: dict[str, float] = {}
    hv_now: dict[str, float] = {}
    n_chain_ok = 0
    for sym in cp.universe:
        try:
            chain = fetch_option_snapshot_chain(
                options_data, sym,
                target_tenor_days=cp.strangle.target_tenor_days,
                tenor_tolerance_days=cp.strangle.tenor_tolerance_days,
            )
        except Exception as e:
            # Per-name failures don't kill the run; the name just
            # gets dropped from the predictor input. Logged so an
            # operator can investigate.
            print(f'  [chain] {sym}: skipped ({type(e).__name__}: {e})',
                  flush=True)
            continue
        if not chain:
            continue
        if sym not in bars.columns:
            continue
        und_px_today = float(bars[sym].dropna().iloc[-1])
        atm_iv = iv_fn(chain, und_px_today,
                       target_tenor_days=cp.strangle.target_tenor_days,
                       tenor_tolerance_days=cp.strangle.tenor_tolerance_days)
        rv = rv_fn(bars[sym].dropna(), window=20)
        if not (atm_iv == atm_iv) or atm_iv <= 0:  # NaN-safe check
            continue
        if not (rv == rv) or rv <= 0:
            continue
        iv_now[sym] = atm_iv
        hv_now[sym] = rv
        n_chain_ok += 1

    iv_now_s = pd.Series(iv_now, dtype=float)
    hv_now_s = pd.Series(hv_now, dtype=float)

    # Step 3: persist today's snapshot to the local history cache
    # so the next run sees us. Idempotent — overwrites prior same-day rows.
    append_snapshot(iv_now_s, hv_now_s, as_of=bars.index.max())

    # Step 4: load enough back-history for the 4-week change features.
    iv_hist, hv_hist = load_history(n_weeks=8)
    # If the local cache is too thin (first runs), the build_feature_row
    # helper falls back to whatever it has — the diff features will be
    # NaN on those rows and the predictor drops them. The operator
    # bootstraps once via `bootstrap_from_dolthub` before deployment.

    features = build_feature_row(iv_now_s, hv_now_s, iv_hist, hv_hist)
    n_eligible = int(features.notna().all(axis=1).sum())
    return features, last_bar_date, n_eligible


def _build_strangles_for_picks(picks, cp, options_data, bars_data, notes):
    """For each pick, query the chain and build a short strangle.

    Drops names that fail the strangle's liquidity gates rather than
    silently zeroing them. Returns a list of `Strangle`.
    """
    from vol.alpaca_chain import fetch_option_snapshot_chain
    from vol.strangle import build_short_strangle

    strangles: list[Strangle] = []
    # Pull underlying prices once for all picks (small batch).
    syms = list(picks.index)
    from vol.alpaca_chain import fetch_underlying_bars
    bars = fetch_underlying_bars(bars_data, syms, days=3)
    if bars.empty:
        notes.append(f'strangle build: no bars for picks; dropping all')
        return []

    for sym in syms:
        try:
            chain = fetch_option_snapshot_chain(
                options_data, sym,
                target_tenor_days=cp.strangle.target_tenor_days,
                tenor_tolerance_days=cp.strangle.tenor_tolerance_days,
            )
        except Exception as e:
            notes.append(f'strangle build {sym}: chain fetch failed '
                         f'({type(e).__name__})')
            continue
        if not chain or sym not in bars.columns:
            notes.append(f'strangle build {sym}: empty chain or no bar')
            continue
        und_px = float(bars[sym].dropna().iloc[-1])
        s = build_short_strangle(sym, und_px, chain, cp.strangle)
        if s is None:
            notes.append(f'strangle build {sym}: failed liquidity gates')
            continue
        strangles.append(s)
    return strangles


def _scale_strangles(strangles, scale):
    """Scale all strangle qtys down to fit the total-vega budget.
    Reduces each leg's `qty` proportionally; drops legs that round to 0.
    """
    out = []
    for s in strangles:
        new_qty_call = max(0, int(round(s.call.qty * scale)))
        new_qty_put  = max(0, int(round(s.put.qty  * scale)))
        if new_qty_call == 0 or new_qty_put == 0:
            continue
        # Replace the legs immutably (frozen dataclass)
        from dataclasses import replace
        new_call = replace(s.call, qty=new_qty_call)
        new_put = replace(s.put, qty=new_qty_put)
        new_strangle = replace(
            s, call=new_call, put=new_put,
            net_vega=s.net_vega * scale,
            net_delta=s.net_delta * scale)
        out.append(new_strangle)
    return out


def submit_strangles(broker, strangles: list[Strangle], *,
                     dry_run: bool = False,
                     ) -> tuple[list[str], list[tuple[str, str]]]:
    """Submit each strangle as a multi-leg `LimitOrderRequest`.

    Returns (order_ids, rejections). Per-name failures are captured,
    not raised — so a partial outage doesn't lose the rest of the
    rebalance. Same submit-and-report-rejections shape as
    `ss_portfolio.broker.AlpacaBroker.submit_orders`.

    `dry_run=True` returns synthetic `DRY_RUN_*` order ids for each
    strangle without touching Alpaca; this is what `run_live(dry_run=True)`
    passes through. The real-money guard is upstream in `run_live`.
    """
    from vol.alpaca_chain import submit_short_strangle
    order_ids: list[str] = []
    rejections: list[tuple[str, str]] = []
    for s in strangles:
        sub = submit_short_strangle(broker, s, dry_run=dry_run)
        if sub.order_id is not None:
            order_ids.append(sub.order_id)
        if sub.rejection_reason is not None:
            rejections.append((sub.underlier, sub.rejection_reason))
    return order_ids, rejections


# --- Formatting ------------------------------------------------------------

def format_run(result: LiveRunResult) -> str:
    lines = [
        f'vol live @ {result.timestamp}',
        f'  checkpoint : {result.checkpoint_path}',
        f'  mode       : {"DRY-RUN" if result.dry_run else "LIVE"} '
        f'({"paper" if result.paper else "REAL MONEY"})',
        f'  account    : equity=${result.account_equity:,.2f}  '
        f'cash=${result.account_cash:,.2f}',
        f'  VIX gate   : {result.gate.vix_now:.2f} vs '
        f'{result.gate.lookback_days}d-median {result.gate.rolling_median:.2f}'
        f' -> {"FIRED" if result.gate.fired else "closed"}',
    ]
    import math as _math
    if _math.isfinite(result.realized_c_bps_today) or _math.isfinite(result.realized_c_bps_rolling):
        lines.append(
            f'  friction   : today c_bps={result.realized_c_bps_today:.1f} '
            f'rolling-{FRICTION_ROLLING_WINDOW_REBALS}={result.realized_c_bps_rolling:.1f}')
    if result.aborted_reason:
        lines.append(f'  ABORTED    : {result.aborted_reason}')
        return '\n'.join(lines)

    lines += [
        f'  universe   : {result.n_universe} names; {result.n_eligible} eligible '
        f'(passed liquidity gates)',
        f'  last bar   : {result.last_bar_date}',
        f'  picks      : {len(result.top_k_picks)} (top-K)',
        f'  strangles  : {len(result.strangles)} constructed',
    ]
    if result.strangles:
        total_vega = sum(abs(s.net_vega) for s in result.strangles)
        lines.append(f'  total |vega|: ${total_vega:,.0f}')
        lines.append('  top 10 by predicted iv_rv_gap:')
        top = result.top_k_picks.sort_values(ascending=False).head(10)
        for sym, p in top.items():
            lines.append(f'    {sym:<6s}  predicted_gap={p:+.4f}')
    if result.submitted_order_ids:
        lines.append(f'  submitted  : {len(result.submitted_order_ids)} multi-leg orders')
    if result.rejected_orders:
        lines.append(f'  REJECTED   : {len(result.rejected_orders)} orders')
        for sym, reason in result.rejected_orders[:5]:
            lines.append(f'    {sym}: {reason}')
    if result.notes:
        lines.append('  notes:')
        for n in result.notes:
            lines.append(f'    - {n}')
    return '\n'.join(lines)


__all__ = [
    'DEFAULT_KILLSWITCH', 'DEFAULT_FRICTION_HISTORY',
    'DEFAULT_MAX_REALIZED_C_BPS', 'FRICTION_ROLLING_WINDOW_REBALS',
    'GateState', 'LiveRunResult', 'run_live',
    'submit_strangles', 'format_run',
]
