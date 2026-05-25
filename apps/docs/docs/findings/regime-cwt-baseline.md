---
tags:
  - regime-2000-25
  - stooq-us-long
  - reversed-OOS
  - hypothesis-system
---

# Regime CWT-divergence strategy — universe-agnostic walk-forward baseline

**Operational rule.** The regime app's namesake CWT-divergence head
(`ss_portfolio.strategies.weights_regime`) does NOT clear passive
equal-weight on the wide stooq_us_long universe at its canonical
`findings/regime-baselines.md` defaults. Universe-agnostic mean val
alpha is **−0.195 Sharpe (1/6 positive-alpha windows, DSR-t −0.78,
ladder deflated-t −3.48)** — `reversed-OOS` against the locked
pre-registration bar. Retire `weights_regime` from the
`regime live` dispatch on the wide universe; the strategy remains
defensible on the mega-cap phase-2 universe per
`findings/regime-baselines.md` and the universe-shift mechanism
in `findings/relational-universe-shift.md`, but should not surface
through the live harness on stooq_us_long without per-regime
conditioning.

The pre-registration bar was locked in driver source before the eval
fired:

```
confirmed-OOS: mean val alpha ≥ +0.20 AND ≥4/6 pos AND DSR-t > +1.5
partial-OOS:   mean val alpha ≥ +0.05 AND ≥3/6 pos
confirmed-null: alpha < +0.05 AND DSR-t < +1.0
reversed-OOS:  alpha < −0.10
diagnostic:    else
```

Baseline cell cleared the `reversed-OOS` threshold cleanly
(α = −0.195, DSR-t = −0.78, ladder deflated-t = −3.48).

## Eval setup

- **Strategy:** `ss_portfolio.strategies.weights_regime` — top-N basket
  by recent-vs-historical CWT-power divergence (KL/JS/cosine/L2),
  raw-close input (log-returns is `reversed-OOS` per
  `findings/log-returns-vs-raw-close.md` — flag preserved but not
  flipped).
- **Baseline config (locked):** scales = `LONG_SCALES =
  [42, 50, 63, 90, 126]`, `lookback=120`, `n_tail=20`, `top_n=20`,
  `divergence='kl'`. Mirrors the `findings/regime-baselines.md`
  "default params" config (lookback=120, n_tail=20, top_n=20, KL).
- **Universe:** stooq_us_long manifest (312 names; 2000-01-03 →
  2025-12-11, full 6,527-bar history).
- **Windowing:** 6 walk-forward windows train=1260 / val=780 /
  step=780 daily bars. Identical to the RSI and scalogram sibling
  baselines.
- **Costs:** `rebal_days=20`, `commission_bps=10`, one-sided turnover
  on rebal dates.
- **Benchmark:** passive EW on the same universe via
  `passive_ew_daily_returns` (matches the canonical
  `findings/passive-ew-benchmark.md` construction).
- **Robustness grid:** 24 cells = 4 divergences × 3 top_n × 2
  (lookback, n_tail) configurations.

The arm builder constructs the global `weights_regime` panel **once**
per `(scales, lookback, n_tail, top_n, divergence)` configuration and
slices into per-window val streams — safe because `weights_regime`
is fully causal (causal CWT + cumsum z-norm). 24 cells × 312
tickers × 6,527 bars completes in ~25 seconds locally.

## Per-window numbers (baseline cell)

| Window | Val range            | Regime Sh | EW Sh   | Alpha   | Regime DD |
|-------:|:---------------------|----------:|--------:|--------:|----------:|
| 0      | 2005-01-06→2008-02-12 | +0.211    | +0.711  | −0.499  | −0.214    |
| 1      | 2008-02-13→2011-03-17 | +0.420    | +0.493  | −0.073  | −0.614    |
| 2      | 2011-03-18→2014-04-24 | +0.648    | +1.019  | −0.371  | −0.215    |
| 3      | 2014-04-25→2017-05-30 | +1.330    | +0.924  | **+0.405** | −0.113 |
| 4      | 2017-05-31→2020-07-06 | +0.374    | +0.440  | −0.066  | −0.335    |
| 5      | 2020-07-07→2023-08-10 | +0.478    | +1.046  | −0.568  | −0.245    |
| **Mean** |                    | **+0.577**| **+0.772** | **−0.195** | (1/6 pos) |

Only w3 (2014–2017) has positive alpha. The negative tail
(w0 −0.499, w5 −0.568, w2 −0.371) swamps it.

## Robustness grid (24 cells)

| divergence | top_n | (lb, nt) | mean α | pos / 6 | regime Sh |
|:-----------|------:|:---------|-------:|--------:|----------:|
| kl         | 10    | (120, 20) | −0.150 | 2       | +0.622    |
| kl         | 10    | (60, 10)  | +0.071 | 5       | +0.843    |
| kl         | 20    | (120, 20) | **−0.195** | 1   | +0.577 *(baseline)* |
| kl         | 20    | (60, 10)  | +0.011 | 4       | +0.783    |
| kl         | 50    | (120, 20) | −0.147 | 1       | +0.625    |
| kl         | 50    | (60, 10)  | +0.025 | 5       | +0.797    |
| js         | 10    | (120, 20) | −0.068 | 2       | +0.704    |
| js         | 10    | (60, 10)  | +0.095 | 4       | +0.867    |
| js         | 20    | (120, 20) | −0.151 | 3       | +0.621    |
| js         | 20    | (60, 10)  | +0.018 | 4       | +0.790    |
| js         | 50    | (120, 20) | −0.149 | 2       | +0.623    |
| js         | 50    | (60, 10)  | +0.037 | 5       | +0.809    |
| cosine     | 10    | (120, 20) | −0.130 | 2       | +0.642    |
| cosine     | 10    | (60, 10)  | −0.086 | 2       | +0.686    |
| cosine     | 20    | (120, 20) | −0.122 | 2       | +0.650    |
| cosine     | 20    | (60, 10)  | +0.095 | 5       | +0.867    |
| cosine     | 50    | (120, 20) | −0.070 | 3       | +0.702    |
| cosine     | 50    | (60, 10)  | +0.007 | 3       | +0.779    |
| l2         | 10    | (120, 20) | −0.111 | 3       | +0.661    |
| l2         | 10    | (60, 10)  | **+0.136** | 5   | +0.908 *(grid-best)* |
| l2         | 20    | (120, 20) | −0.158 | 3       | +0.614    |
| l2         | 20    | (60, 10)  | +0.020 | 4       | +0.792    |
| l2         | 50    | (120, 20) | −0.077 | 2       | +0.695    |
| l2         | 50    | (60, 10)  | +0.018 | 4       | +0.790    |

Two observations dominate:

1. **The (lookback=60, n_tail=10) sub-band is uniformly less-bad
   than (120, 20)** across all four divergences. Best grid cell is
   `l2 / top_n=10 / lb=60 / nt=10` at α = +0.136 (5/6 pos) — but it
   still does not clear the partial-OOS bar (+0.05 AND ≥3/6 pos
   would require both; +0.05 alone is not enough at n_trials=24).
2. **Divergence choice is roughly invariant.** Within each
   (lookback, n_tail) sub-cell, all four divergences agree on sign
   and rough magnitude to within ±0.05 alpha. The result is robust
   to the divergence axis, *fragile* to the window-length axis.

## Mechanism

The strategy buys top-N by **highest** CWT-power divergence (biggest
regime shift). On stooq_us_long's mid/large-cap survivor universe,
this systematically picks names mid-vol-spike — which on a survivor
panel means buying into recently-collapsed names without a
confirming directional horizon. Long lookback × long n_tail (the
locked baseline) sets a long historical baseline against a long
recent window; the recent window inherits enough of the historical
shape that only the largest directional moves register as
divergence, biasing picks toward the falling-knife tail.

Short configurations (lb=60, nt=10) register higher-frequency
shifts and pick somewhat different names, which posts less-negative
alpha — but no cell crosses the partial bar at any axis.

## Three-way cross-comparison: regime / RSI / scalogram

All three regime-app strategy heads were evaluated on the same
universe/windowing/benchmark the same day (2026-05-25). All three
are `confirmed-null` or worse:

| Head      | Mean α | Pos / 6 | DSR-t | Verdict          |
|-----------|-------:|--------:|------:|:-----------------|
| RSI       | −0.051 | 4 / 6   | −0.81 | `confirmed-null` |
| **regime-CWT** | **−0.195** | **1 / 6** | **−0.78** | **`reversed-OOS`** |
| scalogram | −0.357 | 1 / 6   | −1.17 | `reversed-OOS`   |

regime-CWT sits *between* the other two: worse than RSI (the
CWT-divergence machinery makes more aggressive picks → bigger
negative tails) but better than scalogram (random vs counter-trend
wins on this universe; scalogram's `direction − momentum × coherence`
ordering picks systematically worse names). **The CWT machinery is
not the load-bearing differentiator** — the 312-name survivor
universe selectively penalizes top-N counter-trend baskets vs
passive EW regardless of which counter-trend ranker (RSI / CWT /
scalogram) is on the front.

## Surprises

1. **The canonical `findings/regime-baselines.md` defaults — the same
   `(lookback=120, n_tail=20, top_n=20, kl)` config that posted
   Sharpe +0.63 on Kaggle Nasdaq 2013–2025 — lose by 0.20 Sharpe on
   stooq_us_long 2000–2025.** Different universe, different
   windowing (the GFC is now in-sample), different benchmark
   (passive EW vs the prior finding's bare CAGR). The gap is
   universe-shift consistent with `findings/relational-universe-shift.md`:
   strategies that look strong on phase-2 mega-caps degrade on the
   wide universe.
2. **(lookback=60, n_tail=10) uniformly less-bad than (120, 20)
   across all 4 divergences.** Short windows pick up genuine regime
   change before the historical baseline absorbs it; long windows
   require a larger move to register, biasing toward falling-knife
   picks. If you wanted to keep this strategy alive, the next
   experiment is *shorter* (lookback, n_tail) — but at α ≈ +0.04
   median for that sub-band, the head wouldn't clear the partial
   bar even after a wider search of that axis.
3. **Divergence choice barely matters.** All four divergences
   (KL/JS/cosine/L2) agree on sign and magnitude per (lookback,
   n_tail) cell. The `findings/regime-baselines.md`-era assumption
   that divergence kind was a load-bearing hyperparameter is
   contradicted — the lever is window length, not divergence
   function.

## Next-experiment per CLAUDE.md's verdict→next-experiment table

The `reversed-OOS` row asks: *what killed val?* Three failure modes
to distinguish:

- **Overfit (DOF too high).** Not the case here — `weights_regime`
  is parameter-free per window; the only fitted axis is the
  hyperparameter grid which the locked baseline didn't search over.
- **Regime-specific.** The "regime" axis to split on is *universe*,
  not market-state: this head posts +0.63 Sharpe on phase-2
  mega-caps (Kaggle 2013–2025) and −0.20 alpha on stooq_us_long
  (full 2000–2025). The split is `findings/relational-universe-shift.md`-style.
  This is the most plausible failure mode.
- **Pipeline bug.** Eliminated: the global-CWT-once arm builder is
  verified against the smoke test (30 tickers × 2 windows produced
  α = +0.088); the basket/EW arithmetic matches the RSI sibling row
  bit-for-bit.

**Recommended next experiment (if any).** Restrict the universe to
the phase-2 21-name mega-cap subset and re-run the same 6-window
walk-forward. Hypothesis: regime-CWT replicates the
`findings/regime-baselines.md` +0.63 Sharpe finding when restricted
to mega-caps, falsifying the wide-universe result as universe-shift
rather than strategy failure. Falsification bar: mean val Sharpe
≥ +0.60 AND mean val alpha vs phase-2 passive EW ≥ +0.05 →
`partial-OOS` rescue; otherwise the wide-universe result generalizes
and the head is retired regardless of universe.

The brief's Step 1 (per-regime universe Optuna with K=2 turbulence
regime axis) is **NOT triggered** per the locked pre-reg gate:
universe-agnostic baseline must be `partial-OOS` or better to
proceed, and `reversed-OOS` shortcircuits to next-experiment.

## Operational implication

Retire `weights_regime` from `regime live --strategy regime` dispatch
on stooq_us_long. The strategy remains defensible on phase-2
mega-caps per `findings/regime-baselines.md` (where it was
originally evaluated) but should not surface through the live
harness on the wide universe.

**No change to deployment recommendation.** DCA + sized vol_v3
sleeve remains canonical per `findings/vol-sleeve-sizing.md` and
`findings/meta-allocator-no-vol-v3.md`. The meta-allocator 5-arc
panel does not gain a regime-CWT arm.

## Master walk-forward log

Master walk-forward log: [Leaderboard](../leaderboard.md) (the
2026-05-25 regime CWT-divergence universe-agnostic row —
[`reversed-OOS`](../leaderboard.md#verdict-labels)).

Driver: `apps/regime/scripts/regime_cwt_universe_agnostic.py`
(no Modal; ~25s local wall).
Artifacts: `Output/regime-cwt-universe-agnostic-walkforward.{npz,json}`.
DSR ladder spec: `apps/docs/scripts/compute_dsr.py` key
`regime-cwt-universe-agnostic`.
Companions: [`findings/regime-rsi-baseline`](regime-rsi-baseline.md),
[`findings/regime-scalogram-baseline`](regime-scalogram-baseline.md).
Parent (mega-cap universe baseline this row contradicts on
stooq_us_long): [`findings/regime-baselines`](regime-baselines.md).
Mechanism reference: [`findings/relational-universe-shift`](relational-universe-shift.md).
