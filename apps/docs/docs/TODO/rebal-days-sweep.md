# Rebal-days sweep (gates the event-driven trade trigger)

Open question: is the analog-kNN signal genuinely monthly (per the
"regime signal works on monthly-to-biannual horizons" finding in
CLAUDE.md), or does the DWT-L1 daily-Sharpe edge mean the compressed
fingerprint can act on a faster cadence?

Sweep `rebal_days ∈ {5, 10, 20, 40}` on the same Phase-2 universe,
both baseline and DWT-L1 arms. Two outcomes:

1. **Shorter rebal wins** (rebal_days=5 Sharpe ≥ 20-day): the signal
   supports faster action. Then run a divergence-trigger variant —
   daily compute weights, only act if `max(|target - current|) > θ`,
   sweep θ. If trigger beats both fixed-cadence arms net of
   commission, deploy event-driven.
2. **20-day wins or ties**: the underlying signal is monthly. Daily
   cron is for monitoring only; trades stay 20-day fixed.

This experiment **gates** the Modal-cron-event-driven design. Without
it, deploying event-driven is shipping a new strategy with new
hyperparameters and zero backtest evidence.

Cost: 4 rebal-days values × 2 arms = 8 backtests, ~20 min total with
the CWT cached. Run after the current 8-arm and walk-forward results
land.
