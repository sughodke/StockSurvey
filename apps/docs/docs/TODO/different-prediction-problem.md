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
