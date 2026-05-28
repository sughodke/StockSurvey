"""apps/cnc — crypto-and-carry: perp funding-rate cash-and-carry basis trade.

Mechanism: when perp price > spot (positive premium), longs pay shorts a
funding payment every funding interval. A long-spot + short-perp position
of equal notional is delta-neutral to first order and collects the
funding stream. With negative funding, the inverse (long-perp +
short-spot) collects the inverse stream.

This arc tests whether top-K-most-funded perps, rebalanced periodically,
earn a positive Sharpe net of round-trip leg friction. Canonical
references: He-Manela-Ross 2023 (crypto basis), Brunnermeier-Pedersen
2009 (funding liquidity).

Venue: Hyperliquid (the only top-N perp venue whose public funding-rate
history endpoint is reachable from this host without auth; Binance &
Bybit return HTTP 451/403, OKX exposes only ~3 months). Funding cadence
on HL is hourly (24/day). Eval substrate is daily-summed.

Modeling approximation: the basis-trade PnL stream is computed as
`sum(funding_rate_per_day * weight)` minus rebal friction. This is the
academic-clean approximation: with equal-notional spot & perp legs
re-hedged daily, the price-delta cancels and the funding payment IS
the per-day PnL. Real-world deployment carries residual basis tracking
error, hedge-rebalance slippage, and asymmetric borrow on the short
leg — out of scope for the eval-substrate test.

Public API:
- `cnc.data.build_panels(...)` — fetch + align funding/close panels.
- `cnc.backtest.run_carry(...)` — vectorized walk-forward carry backtest.
- `cnc.cli.main()` — `ss-cnc backtest ...` entrypoint.
"""
from cnc.data import CarryPanels, build_panels
from cnc.backtest import CarryResult, run_carry

__all__ = [
    'CarryPanels',
    'CarryResult',
    'build_panels',
    'run_carry',
]
