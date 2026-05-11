# Different prediction problem — pair-spread / drawdown / IV-vs-realized

**Status: resolved 2026-05-10** — all three orthogonal prediction
problems were tested as v0 walk-forwards. Closing arc-level
synthesis:
[`prediction-problem-pivot-arc`](../findings/prediction-problem-pivot-arc.md).
Headline: three independent partial-OOS results at consistent
magnitudes (mean alpha +0.07 to +0.10, 4-5/5-6 positive windows),
all regime-conditional. No standalone shipper but the arc
established that **predictions with non-zero multivariate signal
but regime-conditional deployment performance need a regime
filter, not a richer predictor** (operational rule added to
CLAUDE.md). v0 results:

- [`gate-drawdown-v0`](../findings/gate-drawdown-v0.md) —
  `partial-OOS` (mean alpha +0.067, val r +0.26)
- [`pairs-classical-v0`](../findings/pairs-classical-v0.md) —
  `confirmed-null` per pre-reg (mean alpha +0.099)
- [`vol-surface-v0`](../findings/vol-surface-v0.md) —
  `inconclusive`, 5/5 positive (mean alpha +0.089, val r +0.12)

The original framing is preserved below for audit.

---

The [+0.012 ceiling](../findings/factor-indicator-baseline.md) is for
*cross-sectional return direction* at 297 tickers / 20d. Other targets
may carry more signal:

- **Pair-spread mean reversion** — high-IC, low-cap. Pick correlated pairs, predict spread reversion.
- **Drawdown forecasting** — directly relevant to sizing; positive signal here would ship as a risk overlay even at modest IC.
- **IV-vs-realized** — DoltHub IV data is on hand from the relational arc. Predict whether implied vol over-/underestimates realized.

Different prediction problems have different data ceilings; not all
are bounded by the
[+0.012 cross-sectional return-IC limit](../findings/factor-indicator-baseline.md)
we hit on indicators / CWT / wider universe / longer horizon.

## Why this is now top-priority

Three independent 2026-05-10 tests converge on the conclusion that
**signal magnitude is the binding constraint** for cross-sectional
return prediction at our universe / horizon, not the constructor or
loss:

1. [`passive-ew-benchmark`](../findings/passive-ew-benchmark.md) —
   no model row clears its universe's passive EW Sharpe.
2. [`factor-rankic-long-only-mismatch`](../findings/factor-rankic-long-only-mismatch.md)
   — long-short constructor on the rank-IC head delivers val Sharpe
   −0.067 (alpha −0.345 vs long-only). The "discarded short signal"
   hypothesis is falsified.
3. [`factor-loss-pivot`](../findings/factor-loss-pivot.md) —
   Sharpe-aligned and IR-vs-EW-aligned training losses *destroy*
   val Sharpe by ~0.37 vs the rank-IC baseline. The "wrong loss"
   hypothesis is falsified in the opposite-direction sense
   (rank-IC's spread-thin behavior was inadvertent risk control).

The next test must change the underlying prediction problem — what
the head is being asked to predict, not how the prediction is
monetized. The three suggestions below remain the right shortlist;
priority order should now be informed by which target's data
ceiling we have least evidence for.
