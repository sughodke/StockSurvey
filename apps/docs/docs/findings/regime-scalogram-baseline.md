# Regime-app scalogram strategy — universe-agnostic walk-forward baseline

**Operational rule.** Retire the `scalogram` strategy from the regime
app's `regime live --strategy scalogram` dispatch on the
universe-agnostic wide universe. `weights_scalogram` produces a
top-N basket that *loses to passive EW by 0.36 ann Sharpe* on
`stooq_us_long`, with every single one of 12 grid cells delivering
negative alpha. The CWT direction × momentum × coherence machinery
actively destroys value vs the simpler RSI ranking on the same
universe (RSI is `confirmed-null` at α≈−0.05; scalogram is
`reversed-OOS` at α=−0.36). This is the second of three regime-app
heads to land its first formal walk-forward eval the same week, and
the strongest negative result of the three so far.

**Verdict:** [`reversed-OOS`](../leaderboard.md#verdict-labels).
**Status:** Step 1 (per-regime universe Optuna) NOT triggered per
pre-registered gate — `reversed-OOS` halts the arc before universe
search.

## Background

The regime app exposes three weight builders through its CLI:
`weights_regime` (CWT-power-distribution divergence — the *original*
arc, has multiple findings), `weights_rsi` (top-N most-oversold by
Wilder RSI — first formally eval'd 2026-05-25 the same day as this
run, see [`regime-rsi-baseline.md`](regime-rsi-baseline.md)), and
`weights_scalogram` (this page). All three are wired into the live
harness with the same four risk rails; all three have shipped in
`cli.py`'s strategy dispatch table for the lifetime of the regime app.

`weights_scalogram` had NO formal walk-forward eval. The strategy
docstring claims a "direction − momentum × coherence" composite —
counter-trend bet, picks lowest scoring names (most-negative direction
on incoherent timescales), framed as a mean-reversion-on-CWT
construction. The universe-agnostic question is the prerequisite to
any per-regime variant: until the wide-universe baseline has a
verdict, asking "does it work better in some regimes" is premature.

## Pre-registered bar

Locked into the driver source before the eval ran (see
`apps/regime/scripts/scalogram_universe_agnostic.py:47`):

```
confirmed-OOS:  mean val alpha ≥ +0.20 Sharpe AND ≥4/6 positive AND DSR-t > +1.5
partial-OOS:    mean val alpha ≥ +0.05 Sharpe AND ≥3/6 positive
confirmed-null: alpha < +0.05 Sharpe AND DSR-t < +1.0
reversed-OOS:   mean val alpha < −0.10 Sharpe
diagnostic:     anything else
```

The Step 1 gate: only `partial-OOS` or better proceeds to per-regime
universe Optuna. Step 0 reversed-OOS halts the arc.

## Eval setup

- **Universe:** stooq_us_long (312 names, 2000-01 → 2025-12, full
  history, the canonical walk-forward wide-equity universe).
- **Windowing:** 6 windows of 1260-train / 780-val / 780-step (daily
  bars) — identical to the RSI sibling row, the gate v0 row, and the
  cfr Phase-1 row. Direct DSR-comparable.
- **Strategy params (baseline):** `lookback=252`, `n_tail=21`,
  `top_n=20`, `scales=[5,21,90]` (the `DEFAULT_SCALES` fallback in
  `trainer.py` when all 3 Optuna boolean flags are False).
- **Robustness grid (12 cells):** 4 scale-subsets × 3 top_n values
    - scale subsets: `short-mid=[5,21]`, `mid-long=[21,90]`,
      `default=[5,21,90]`, `all-spread=[5,21,63,126]`
    - top_n ∈ {10, 20, 50}
- **Frictions:** 20-day rebal cadence, 10 bps round-trip commission
  on L1 turnover.
- **Baseline:** passive EW on the same universe (canonical per
  [`passive-ew-benchmark`](passive-ew-benchmark.md)).
- **Compute:** local, ~80s wall, single-threaded numpy + ss_wavelets
  causal CWT.

## Result

### Baseline arm (scales=[5,21,90], top_n=20, n_tail=21)

| Window | Val start → end | scal Sh | EW Sh | alpha | scal max-DD |
|-------:|:---|---:|---:|---:|---:|
| 0 | 2005-01-06 → 2008-02-12 | −0.078 | +0.708 | **−0.786** | −0.340 |
| 1 | 2008-02-13 → 2011-03-17 | +0.089 | +0.466 | −0.377 | −0.599 |
| 2 | 2011-03-18 → 2014-04-24 | +0.945 | +1.017 | −0.072 | −0.224 |
| 3 | 2014-04-25 → 2017-05-30 | +0.102 | +0.925 | **−0.823** | −0.248 |
| 4 | 2017-05-31 → 2020-07-06 | +0.512 | +0.440 | **+0.073** | −0.421 |
| 5 | 2020-07-07 → 2023-08-10 | +0.886 | +1.045 | −0.159 | −0.217 |
| **mean** | — | **+0.409** | **+0.767** | **−0.357** | — |

1/6 windows positive-alpha. Two windows (w0 dot-com hangover, w3 mid-cycle bull) deliver near-catastrophic underperformance vs EW (−0.79 and −0.82 Sharpe respectively).

DSR-t (rough overlay, n_obs=4680, n_trials=12): **−1.17**.

### Robustness grid (12 cells, sorted by alpha)

| Scales | top_n | mean α | pos windows | scal Sharpe |
|:---|---:|---:|---:|---:|
| mid-long | 10 | −0.424 | 1/6 | +0.343 |
| default | 10 | −0.360 | 1/6 | +0.407 |
| default | 20 | −0.357 | 1/6 | +0.409 |
| mid-long | 20 | −0.268 | 2/6 | +0.499 |
| default | 50 | −0.256 | 1/6 | +0.511 |
| mid-long | 50 | −0.252 | 1/6 | +0.515 |
| short-mid | 10 | −0.282 | 0/6 | +0.484 |
| short-mid | 20 | −0.203 | 1/6 | +0.563 |
| short-mid | 50 | −0.148 | 0/6 | +0.619 |
| all-spread | 10 | −0.297 | 2/6 | +0.470 |
| all-spread | 20 | −0.192 | 2/6 | +0.575 |
| **all-spread** | **50** | **−0.106** | 1/6 | +0.661 |

Best cell (`all-spread top_n=50`) is α=−0.106 — still below the
−0.10 reversed threshold. Median alpha across the grid is −0.262;
spread min/max = −0.424 / −0.106.

**Every single grid cell loses to passive EW by ≥0.10 ann Sharpe.**
This is not a single-cell artifact; it is the strategy on this
universe.

## Mechanism

The strategy picks the **lowest** score names (`ascending=True` in
`select_top_n_matrix`) where score = `direction − momentum × coherence`:

- `direction` = trailing-21 mean of the shortest-scale signed CWT
  coefficient. Negative = recent price weakness.
- `momentum` = trailing-21 mean of |coeffs|² across all scales —
  recent volatility magnitude.
- `coherence` = Pearson correlation between shortest-scale and
  longest-scale power, clipped to [0,1].

Picking lowest = "negative direction, high volatility magnitude, low
coherence between timescales" — a hand-crafted mean-reversion signal.

On the wide stooq_us_long universe, the lowest-score names are
post-crash mid/large-cap survivors whose CWT short-scale signal
correlates with **persistent weakness**, not transient dislocation.
Buying them earns the falling-knife premium (negative). Passive EW
holds every name and benefits from the universe-wide drift; the
top-20 scalogram basket concentrates risk into 20 names selected by
a signal that, on this universe, picks names with negative forward
drift — actively *worse* than a random 20-name basket.

The expensive CWT cube (4 × 312 × 6500 = ~8M coefficients per call)
is doing work, but the work selects against the dominant
cross-sectional risk premium (long-only equity drift) and finds no
compensating short-term reversal. The effort costs nothing in
backtest but in deployment buys negative alpha.

## Why not Step 1?

The pre-registered gate stipulates Step 1 (per-regime universe
Optuna) only runs on `partial-OOS` or better. With `reversed-OOS`,
per-regime pre-selection isn't the question — the strategy is
actively destroying value, and slicing into regimes can at best find
sub-regions where the destruction is smaller, not regions where it
flips sign with enough margin to clear the (+0.30 vs Step 0) Step 1
bar.

The single positive-alpha window (w4 2017-05 → 2020-07, the
late-cycle tech-momentum stretch) is the *least* helpful evidence for
a per-regime rescue: a counter-trend strategy posting +0.07 alpha
during a momentum-friendly market is statistically indistinguishable
from sampling noise. If the strategy had a real regime niche, we
would expect the positive window to be a vol-spike or
mean-reversion-friendly regime — not a momentum bull run.

## Cross-strategy comparison

Same universe (stooq_us_long), same windowing (6w-1260tr-780val-780step),
same frictions (rebal_days=20, commission_bps=10), same passive EW
benchmark:

| Strategy | Mean val Sharpe | Mean alpha | Pos windows | Verdict |
|:---|---:|---:|---:|:---|
| `weights_rsi` (RSI top-N most-oversold) | +0.716 | −0.051 | 4/6 | `confirmed-null` |
| `weights_scalogram` (CWT direction − momentum × coherence) | +0.409 | **−0.357** | **1/6** | **`reversed-OOS`** |

Both are counter-trend top-N constructions in the same family. The
RSI variant is a near-tie with EW (small drift-vs-rebal-drag negative).
The scalogram variant — which uses the same family of price input,
just transformed through the CWT cube + direction/momentum/coherence
composite — performs **dramatically worse**. The added machinery is
*subtracting* from a near-null baseline. This is not what wavelet
multi-resolution proponents would predict; on this universe, the
CWT composite is overfitting to the persistence-of-weakness signal
that ranks more aggressively negative than mean RSI does.

## Three surprises

**Every single one of 12 robustness grid cells is negative.** I
expected the standard story where the canonical config is unlucky
and at least one (scales, top_n) corner rescues the verdict. Instead,
the worst cell (mid-long, top_n=10) is α=−0.424 and the *best* cell
is α=−0.106 — still below the reversed-OOS floor. There is no
hyperparameter combination on the searched grid that produces a
basket better than EW − 0.10 Sharpe. The strategy is not failing
because we picked the wrong knob; it is failing structurally on this
universe.

**The only positive-alpha window is the worst-fit regime for a
counter-trend bet.** Window 4 (2017-05 → 2020-07) covers the
late-cycle tech-momentum bull then snaps through the COVID crash
recovery. A mean-reversion strategy posting +0.073 alpha during the
single largest momentum-extension stretch in the sample is, on its
face, the wrong sign of evidence for a real mean-reversion edge.
If anything, this suggests the +0.073 is more likely an artifact of
which names happened to be momentary laggards entering 2020-Q2 — i.e.
sampling noise — than a genuine niche where the signal works.

**The CWT cube actively makes the picks worse than the 1D RSI.** I
expected scalogram and RSI to land near each other (both counter-trend
top-N, broadly similar bet on mean reversion). Instead scalogram is
worse by 0.31 mean alpha. The interpretation: the CWT
direction × momentum × coherence composite is a more discriminating
ranker than mean RSI, and on this universe what it discriminates
*toward* (negative direction + incoherent timescales) correlates more
tightly with persistent forward weakness than mean RSI does. The
CWT machinery is delivering on its claim of "richer information" —
but the information is, on this prediction problem, anti-signal.

## Master walk-forward log pointer

Row in [`leaderboard.md`](../leaderboard.md) dated 2026-05-25,
verdict [`reversed-OOS`](../leaderboard.md#verdict-labels). DSR-ladder
entry in `apps/docs/scripts/compute_dsr.py` under key
`regime-scalogram-universe-agnostic`. Companion finding for the same
day's RSI eval: [`regime-rsi-baseline`](regime-rsi-baseline.md).
Pre-registered bar embedded in driver source
`apps/regime/scripts/scalogram_universe_agnostic.py`.
