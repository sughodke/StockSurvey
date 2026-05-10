"""apps/pairs — pair-spread mean reversion.

Different prediction problem from `apps/factor` and `apps/gate`:
per-pair time-series mean reversion of `s_t = log(P_A / P_B) − β·...`,
not cross-sectional return ranking and not regime gating. Trade
signal is the classic z-score crossing of the residual spread:
long-A / short-B (dollar-neutral) when spread is high vs train mean
and predicted to revert; flip on the other side.

Three structural differences from the equity-only apps:
  - Each "trade" pairs one long leg + one short leg — friction
    is 2× equity (each leg pays bid/ask + commission).
  - Long-short by construction → benchmark is zero, not market-EW.
    The CLAUDE.md "alpha vs passive EW" rule doesn't apply here.
  - Cointegration is regime-specific. Per-window screening on
    train only (no peeking) is load-bearing — the same lesson the
    relational analog scorer learned the hard way.

Public API:
- `engle_granger_test(y, x)` — OLS hedge-ratio + ADF on residuals.
- `screen_pairs(prices_panel, train_slice, ...)` — per-window
  candidate generation + cointegration filtering.
- `compute_spread(p_a, p_b, beta)` — log spread + z-score.
- `trade_signals(z_score, ...)` — classical entry/exit threshold
  rule.
- `backtest_pair(p_a, p_b, beta, train_stats, ...)` — per-pair PnL.
- `aggregate_walkforward(...)` — rolling train/val harness.
"""
from pairs.cointegration import (
    EngleGrangerResult, engle_granger_test,
)
from pairs.spread import (
    SpreadStats, compute_spread, spread_stats,
)
from pairs.pair_universe import (
    PairCandidate, screen_pairs,
)
from pairs.predictor import trade_signals
from pairs.backtest import (
    PairBacktestResult, backtest_pair, aggregate_pair_pnl,
)


__all__ = [
    'EngleGrangerResult',
    'PairBacktestResult',
    'PairCandidate',
    'SpreadStats',
    'aggregate_pair_pnl',
    'backtest_pair',
    'compute_spread',
    'engle_granger_test',
    'screen_pairs',
    'spread_stats',
    'trade_signals',
]
