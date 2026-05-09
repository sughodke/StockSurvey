# Ablation — disentangle why long-period RSI underperforms

The CSCO zero-shot RSI(n) sweep on the 30-ticker / `n_grid={5,7,9,13,21}`
/ K=64 run showed a sharp degradation at the long end:

| n  | in-grid | R²    |
|----|---------|-------|
| 9  | yes     | 0.964 |
| 13 | yes     | 0.902 |
| 18 | no      | 0.690 |
| 21 | yes     | 0.520 |

Two factors were proposed (see chat 2026-04-27):

1. **Grid spacing** — n=21 sits at the conditioning maximum with no
   right-neighbor; gap to its left-neighbor n=13 is 8 (vs spacings of
   2 below). The linear conditioning has fewer interpolation pairs
   here.
2. **Effective lookback** — Wilder RSI(n) has effective memory ~3×n
   bars. RSI(21) ≈ 63–84 bars; K=64 is at the edge. The model has
   long-horizon info via the rolling z-score stats and long CWT scales,
   but the *direct per-lag* path is window-bounded.

Three runs to disentangle (each is one CLI flag tweak from the
existing `ss-replay … --rsi-n-grid 5,7,9,13,21 …` cell):

| Run | `--rsi-n-grid`              | `--window-cols` | tests       |
|-----|-----------------------------|-----------------|-------------|
| A   | `5,7,9,13,17,21,25`         | 64              | factor 1    |
| B   | `5,7,9,13,21`               | 96              | factor 2    |
| C   | `5,7,9,13,17,21,25`         | 96              | combined    |

If A recovers RSI(21) R², factor 1 is dominant; spacing matters more
than lookback. If B recovers it, factor 2 is dominant; longer K is the
fix. C is the upper bound.

Beyond fixing one ticker's RSI numbers, this informs grid-design
heuristics for any future parameter-conditioned head — both grid
density and the input-bundle's effective lookback need to be
matched to the longest target parametrization.

**Out of scope** for the same diagnostic:

- Non-linear conditioning (sin/cos of n, or a small MLP on n). If A+B
  both fail, that's the next architectural lever.
- Re-running with the (w, n) 2D conditioning — that's a separate
  capability test, not a disentanglement of the existing failure.
