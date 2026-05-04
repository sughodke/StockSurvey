## Trade types

Statistical Arbitrage (StatArb): The broader category of quantitative strategies to which pair trading belongs
Relative Value Trade: Focuses on the price difference (spread) between two related instruments rather than their absolute direction
Convergence Trading: A strategy betting that the prices of two divergent assets will converge back to their historical relationship
Mean Reversion Trading: Specifically refers to trading on the expectation that the spread between the pairs will return to its historical mean
Spread Trading: Used when trading the difference between two related instruments, such as a long and short position, often in futures
Market-Neutral Strategy: Highlights the goal of eliminating overall market risk

### Related Concepts

Long-Short Equity: A general strategy of buying undervalued stocks and selling overvalued ones
Cointegration Trading: A specialized, technical approach to identifying pairs that maintain a long-term stable relationship
Hedged Strategy: A form of hedging intended to minimize risk, often called an alpha extension


## Uses of CWT bundle
 Relational (cross-sectional)

  ┌─────┬───────────────────────────────────────────────────┬─────────────────────────────────────┬────────────────────────────┐
  │  #  │                       Idea                        │          Why it might work          │ Distinguishes from shipped │
  ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────┤
  │     │ Optimal-transport stress index — Sinkhorn         │ Captures cloud shape shift, not     │ C uses centroid distance,  │
  │ R1  │ distance from today's fingerprint cloud to a      │ just first moment; per-name         │ not external regime anchor │
  │     │ rolling "calm regime" reference cloud             │ contribution = transport cost       │                            │
  ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────┤
  │     │ Spectral graph community scoring — kNN graph +    │ Handles non-convex clusters k-means │ A uses k-means (assumes    │
  │ R2  │ Laplacian eigenmap, excess-divergence vs spectral │  misses                             │ ball clusters)             │
  │     │  community                                        │                                     │                            │
  ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────┤
  │     │ Cluster-membership transition signal — flag       │ Derivative of relational position   │ First time-derivative use  │
  │ R3  │ tickers that just switched cluster IDs            │ is a stronger event than level      │ of relational structure    │
  │     │ (Hungarian-matched across rebals)                 │                                     │                            │
  ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────┤
  │     │ Attention-style peer-weighted forward signal —    │ Densifies what B couldn't get on a  │ B looks back in history;   │
  │ R4  │ softmax over fingerprint similarity × peers'      │ 21-name basket                      │ this looks across current  │
  │     │ recent returns                                    │                                     │ peers                      │
  ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────┤
  │ R5  │ Persistent-homology vol-regime classifier — H₀/H₁ │ Topology change = market            │ None of the shipped        │
  │     │  persistence diagram features on the cloud        │ fragmentation; orthogonal axis      │ scorers use topology       │
  ├─────┼───────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────┤
  │     │ UMAP/Isomap manifold peer score — embed           │ If true peer structure is on a      │ C is Euclidean; this is    │
  │ R6  │ fingerprints to 2D, score = manifold-aware peer   │ low-dim manifold, full 168-d L2     │ geodesic                   │
  │     │ distance                                          │ dilutes signal                      │                            │
  └─────┴───────────────────────────────────────────────────┴─────────────────────────────────────┴────────────────────────────┘

  Non-relational (per-name)

  ┌─────┬─────────────────────────────────────────────┬───────────────────────────────────┬────────────────────────────────────┐
  │  #  │                    Idea                     │         Why it might work         │     Distinguishes from shipped     │
  ├─────┼─────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │     │ Scale-energy ratios — short/long energy     │ Trend exhaustion = short spike    │ Shipped scorers use                │
  │ N1  │ ratio + Shannon entropy across scales as    │ with flat long; entropy collapses │ whole-fingerprint distance; this   │
  │     │ features for forward vol/return             │  before regime breaks             │ extracts physically interpretable  │
  │     │                                             │                                   │ stats                              │
  ├─────┼─────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │     │ Effective rank / participation ratio of     │ Concentrated power = clear        │                                    │
  │ N5  │ scale distribution — (Σλ)² / Σλ² over       │ regime, dispersed = noise;        │ One scalar per (ticker, date)      │
  │     │ scale-energy spectrum                       │ cheapest possible regime gauge    │                                    │
  ├─────┼─────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │     │ Bayesian online changepoint detection over  │ Explicitly targets regime-break   │ Time-causal; shipped ideas are     │
  │ N2  │ per-ticker fingerprint stream               │ events — could be the missing     │ snapshot                           │
  │     │                                             │ forward-vol predictor             │                                    │
  ├─────┼─────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │     │ Per-ticker autoencoder reconstruction error │ Anomalies = high error → tail     │                                    │
  │ N3  │  on fingerprints                            │ moves; learns the per-ticker      │ Reuses apps/replay AE infra        │
  │     │                                             │ "normal manifold"                 │                                    │
  ├─────┼─────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │     │ Matched-filter pattern templates —          │                                   │                                    │
  │ N4  │ cross-correlate fingerprints against        │ Hypothesis-driven,                │ Templates encode prior knowledge;  │
  │     │ template scalograms (head-and-shoulders,    │ scale-invariant by construction   │ data-free                          │
  │     │ blow-off)                                   │                                   │                                    │
  ├─────┼─────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │     │ CWT-power term structure — treat scale axis │ Per-stock analog of VIX-curve     │ Factorizes scalogram into 3        │
  │ N6  │  as vol-surface tenor; PCA loadings of      │ term-structure trades             │ interpretable features             │
  │     │ log(power) vs log(scale)                    │                                   │                                    │
  ├─────┼─────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │     │ Tiny transformer over fingerprint sequence  │ Sequence-aware; captures temporal │ Higher infra cost; reuses          │
  │ N7  │ → forward return distribution (quantile     │  evolution none of the shipped    │ apps/replay                        │
  │     │ head)                                       │ scorers exploit                   │                                    │
  └─────┴─────────────────────────────────────────────┴───────────────────────────────────┴────────────────────────────────────┘

  Best bets (agent's recommendation)

  1. R1 (OT stress index) — direct answer to the diagnostic finding that nothing predicts vol surprise. OT to a calm reference is a
  vol-regime distance. One new dep (pot).
  2. N1 + N5 (energy ratio + effective rank) — essentially free numpy one-liners on the cached coeffs. If either correlates with
  forward realized vol above the trailing anchor, it's a real new edge with zero infra cost.


## Conceptual alternatives to rebalancing
- Target-volatility / risk-parity sizing — adjust position sizes continuously based on a vol estimate rather than rebalancing to fixed weights.
- Threshold/band rebalancing — only trade when actual weights deviate from target by more than X%. Reduces turnover.
- Volatility-triggered rebalancing — rebalance when realized vol crosses a threshold rather than on a calendar.
- Signal-triggered rebalancing — rebalance only when the score itself changes meaningfully, not on a fixed cadence.
