"""apps/gate — aggregate drawdown forecaster as an EW-exposure regime gate.

The hypothesis being tested: cross-sectional return prediction at
+0.005 to +0.012 IC can't beat passive EW
([`passive-ew-benchmark`](../../docs/findings/passive-ew-benchmark.md),
[`factor-rankic-long-only-mismatch`](../../docs/findings/factor-rankic-long-only-mismatch.md),
[`factor-loss-pivot`](../../docs/findings/factor-loss-pivot.md)),
but a separate prediction problem might. Forward-aggregate-drawdown
is a different prediction class:

  - **Time-series, not cross-sectional.** One series per universe per
    date, not one row per name per bar.
  - **Different target.** Forward 20-day max drawdown of an EW
    aggregate, not next-period log return.
  - **Different deployment shape.** A scalar gate `g(t) ∈ [0,1]`
    that scales overall exposure (1.0 = fully invested in EW; 0.0 =
    flat in cash). Consumed by EW (default), or any other strategy
    that wants drawdown-aware sizing.

The strategy: `gated_returns_t = g(t) · ew_return_t`. If the gate has
any skill at predicting drawdowns, sitting out high-DD windows
should lift Sharpe materially even at modest accuracy — drawdowns
are asymmetric, so even a noisy gate that catches a fraction of
drawdown events earns more in avoided losses than it loses in
missed gains.

Public API:
- `build_aggregate(prices, rebal_days)` — pre-compute EW return series
  + per-date features.
- `forward_max_drawdown(daily_log_ret, horizon)` — supervised target.
- `train_predictor(features, target, ...)` — linear / MLP head.
- `apply_gate(predicted_dd, threshold)` — convert prediction to
  `gate_t ∈ [0,1]`.
- `gated_returns(ew_returns, gate)` — apply gate to EW return series.
"""
from gate.aggregate import build_ew_aggregate, build_aggregate_features
from gate.target import forward_max_drawdown
from gate.predictor import (
    PredictorResult, train_predictor, predict, apply_gate,
)
from gate.backtest import (
    GatedBacktestResult, gated_returns, evaluate_gated_arm,
)


__all__ = [
    'GatedBacktestResult',
    'PredictorResult',
    'apply_gate',
    'build_aggregate_features',
    'build_ew_aggregate',
    'evaluate_gated_arm',
    'forward_max_drawdown',
    'gated_returns',
    'predict',
    'train_predictor',
]
