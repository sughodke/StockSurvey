# Different prediction problem — pair-spread / drawdown / IV-vs-realized

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
