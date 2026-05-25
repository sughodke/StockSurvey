# Relational regime-velocity scorer — universe-agnostic walk-forward baseline

**Operational rule.** The relational app's velocity head
(`weights_velocity_magnitude` / `weights_axis_alignment`) is
**`confirmed-null`** as a standalone universe-agnostic strategy on
`stooq_us_long`. Across 6 walk-forward windows (1260-train /
780-val / 780-step, rebal_days=20, 10bps), the canonical Phase-11
config (magnitude variant, top_n=20, w_delta=20, fp_window=21,
lookback=120, scales=[5,7,10,12,21,26,50,90]) yields **mean val
Sharpe +0.774** vs **passive EW +0.772** — **mean alpha +0.002**, 3/6
positive-alpha windows, DSR-t (overlay, n_trials=30) −0.15 / ladder
deflated-t −2.15. A 30-cell robustness grid produces 29 cells with
α ≤ 0; the single positive-alpha cell (magnitude, top_n=10,
w_delta=20, full scales) lifts to α=+0.045, still short of the
+0.05 partial-OOS bar. Per the pre-registered bar this is
`confirmed-null`.

Do not pursue the "per-regime universe pre-selection lifts velocity
to `confirmed-OOS`" question. The baseline produces a basket
statistically indistinguishable from passive EW — there is no signal
that universe filtering can amplify. The canonical
`relational-velocity.json` checkpoint should not be deployed as a
standalone live strategy on this universe.

## Why this finding existed (and why it didn't already)

The relational app exposes six scoreboard scorers — empirical, gmm,
analog, farthest, diversified, and **velocity**. The first five live
on the Phase-2 21-name mega-cap universe and were stamped via the
phase-2 walk-forward arc (`findings/relational-arc-synthesis.md`,
`findings/relational-universe-shift.md`). The velocity scorer was
designed specifically for the wider universe and pinned to
`stooq_us_long` in `build_canonical_checkpoints.py`, with
`train_sharpe=0.60 / val_sharpe=0.60` claimed on the Phase-11
`2010-2025` split.

The trigger was the apples-to-apples reality-check series running
the same scaffold on the three regime-app heads (rsi / scalogram /
regime-CWT) the same day. Velocity is the natural fourth — same
universe, same windowing, same benchmark, same pre-reg bar — so the
four rows line up directly. The Phase-11 verdict ("Sharpe 0.60") was
on a single phase split with no walk-forward; this finding fills the
walk-forward receipt at the apples-to-apples scaffold.

## Eval setup

- **Strategies under test.** Two variants exposed by
  `relational.regime_velocity`:
  - `weights_velocity_magnitude(prices, lookback, top_n, scales,
    fp_window, w_delta)` — top-N by `||fp[t, i] − fp[t-W, i]||` where
    `fp` is the z-normed flattened CWT-scalogram window.
  - `weights_axis_alignment(prices, ..., n_axes, train_window_days)`
    — top-N by `max_k |fp_velocity · SVD_axis_k|`; SVD axes are fit
    once on the first `train_window_days=252` post-lookback velocity
    rows then frozen (look-ahead-free per the module's
    construction).
- **Universe.** `stooq_us_long` (312 names, 2000-01-03 →
  2025-12-11) — the canonical wide-universe panel that
  `build_canonical_checkpoints.py::build_velocity_checkpoint` pins
  for the Phase-11 scorer.
- **Walk-forward.** 6 windows, 1260-bar train / 780-bar val /
  780-bar step. The strategy is parameter-free per-window in the
  walk-forward sense (single global SVD fit at the start of the
  panel); the train slices exist purely to align the windowing
  convention with the parallel regime-head agents
  (RSI / scalogram / regime-CWT).
- **Trading.** rebal_days=20, commission_bps=10. One-sided turnover
  cost on each rebal (full L1 entry on the first rebal, 0.5×L1Δ
  thereafter). Returns are lagged by 1 bar from weight construction
  (no same-bar peek).
- **Benchmark.** Passive EW rebalanced on the same cadence over the
  *active* (non-NaN) names. Canonical per
  [`passive-ew-benchmark`](passive-ew-benchmark.md).
- **Hyperparameter grid (robustness, 30 cells).**
  variant ∈ {magnitude, axis_alignment(n_axes=5)} × top_n ∈ {10, 20,
  50} × w_delta ∈ {10, 20} × scales ∈ {full, short=[5,10,21,50],
  long=[10,21,50,90]}. The magnitude variant skips the 'long'
  scale-set to keep the grid manageable; effective cells = 12
  magnitude + 18 axis_alignment = 30.

### Pre-registered verdict bar (LOCKED before running)

| label | criteria |
|---|---|
| `confirmed-OOS` | mean val alpha vs EW ≥ +0.20 Sharpe **AND** ≥4/6 positive alpha windows **AND** DSR-t > +1.5 |
| `partial-OOS`   | mean val alpha vs EW ≥ +0.05 Sharpe **AND** ≥3/6 positive alpha windows |
| `confirmed-null` | alpha < +0.05 Sharpe **AND** DSR-t < +1.0 |
| `reversed-OOS`  | mean val alpha < −0.10 Sharpe |
| `diagnostic`    | anything else |

The pre-registered bar is persisted as the
`pre_registered_bar` string field in
`Output/regime-velocity-universe-agnostic-walkforward.npz`.

## Per-window results — canonical baseline (magnitude, top_n=20, w_delta=20, scales=full)

| win | val_start | val_end | vel Sharpe | EW Sharpe | alpha | vel maxDD |
|---:|---|---|---:|---:|---:|---:|
| 0 | 2005-01-06 | 2008-02-12 | +0.603 | +0.711 | **−0.107** | −0.200 |
| 1 | 2008-02-13 | 2011-03-17 | +0.383 | +0.493 | **−0.110** | −0.572 |
| 2 | 2011-03-18 | 2014-04-24 | +1.293 | +1.019 | **+0.274** | −0.174 |
| 3 | 2014-04-25 | 2017-05-30 | +1.093 | +0.924 | **+0.168** | −0.119 |
| 4 | 2017-05-31 | 2020-07-06 | +0.456 | +0.440 | +0.016 | −0.432 |
| 5 | 2020-07-07 | 2023-08-10 | +0.818 | +1.046 | **−0.228** | −0.165 |
| **mean** | | | **+0.774** | **+0.772** | **+0.002** | |

3/6 windows are positive-alpha. The two largest positive windows
(w2 +0.274, w3 +0.168) cluster in the calmer 2011-2017 stretch; the
three negative windows include the GFC tail (w1 −0.110) and the
2020-2023 vol-spike-then-bull (w5 −0.228). The aggregate effect is
zero alpha — velocity is not a counter-trend bet (those bleed
negatively on this universe; see siblings) but also not a
positive-expectancy momentum bet.

## Robustness grid — 30 cells

Best/median/worst:

| metric | variant | top_n | w_delta | scales | value |
|---|---|---:|---:|---|---:|
| best alpha | magnitude | 10 | 20 | full  | **+0.045** |
| median Sharpe | magnitude | 50 | 10 | short | +0.649 (α −0.123) |
| worst alpha | axis_alignment | 10 | 10 | short | **−0.332** |

- **Grid alpha spread:** min −0.332, max +0.045, median −0.131.
- **29/30 cells** are at or below zero alpha; the single positive
  cell (+0.045) misses the `partial-OOS` bar (+0.05).
- **axis_alignment uniformly worse than magnitude.** Across all 6
  matched (top_n × w_delta × scales) cells in the magnitude ∩
  axis_alignment overlap, axis_alignment alpha is lower by an
  average of 0.08 Sharpe. The literal "word2vec-style stable
  behavioral axis" projection underperforms the simpler magnitude
  norm.
- **w_delta=20 > w_delta=10** uniformly. Longer fingerprint-motion
  windows (20d) carry less measurement noise than 10d.
- **scales=full > scales=short ≈ scales=long.** Concentrating into
  just short or just long scales drops alpha by ~0.10–0.15; the
  full 8-scale stack is what the canonical checkpoint locks.

## DSR-t and ladder placement

Cross-arc DSR-t via `compute_dsr.py` (`n_trials=30` deflation,
`sharpe_std_ann=0.072` workspace default, overlay mode with passive
EW as benchmark):

```
arc                              mode       trials  annSh   DSR-t
regime-velocity-universe-agnostic overlay        30  +0.005  −2.150
```

The "annSh +0.005" is the annualized Sharpe of the (velocity − EW)
edge stream; the negative DSR-t reflects that the edge stream's
mean is essentially zero and is then deflated for the 30-cell
grid + cross-trial dispersion. Pre-reg's DSR-t < +1.0 criterion is
satisfied with margin.

## Mechanism — why the strategy averages to zero

Unlike the three counter-trend regime-app heads
(RSI / scalogram / regime-CWT), velocity's `magnitude` variant
picks names with the *largest* fingerprint motion — names whose
CWT-shape changed most over the last 20 days. On a 312-name
survivor universe, this selects a mix of three populations:

1. **Genuine regime-shifters** whose price action just transitioned
   (positive expectancy if the new regime is up-trending).
2. **Idiosyncratic-news names** with one-bar shocks that revert
   within a rebal cycle (negative expectancy after costs).
3. **Post-crash decay continuations** — names that just took a
   structural hit and continue to decline (negative expectancy).

The net averages to ~zero, not the counter-trend negative drift
seen in RSI / scalogram / regime-CWT (all three of which
preferentially pick population (3) and ignore population (1)).

The `axis_alignment` variant fits SVD axes on the first 252
post-lookback velocity rows then freezes them — a deliberate
look-ahead-free design that turns into an overfit problem at long
horizons. The "stable behavioral axes" turn out to encode 2003-era
fingerprint motion shapes that don't generalize across the 2008
GFC, the 2017-19 low-vol stretch, the 2020 COVID shock, and the
2022 rate cycle. By 2020 the SVD basis is "stable" only in name.

## Cross-strategy comparison — the four-way regime/relational head row

Same universe, same windowing, same benchmark, run the same day:

| head | construction | mean alpha | pos α windows | verdict |
|---|---|---:|---:|---|
| velocity | top-N highest fingerprint motion | **+0.002** | 3/6 | confirmed-null |
| RSI | top-N most-oversold | −0.051 | 4/6 | confirmed-null |
| regime-CWT | top-N highest CWT-power divergence | −0.195 | 1/6 | reversed-OOS |
| scalogram | top-N counter-trend dir/mom/coherence | −0.357 | 1/6 | reversed-OOS |

Velocity is the **least-bad** of the four. The reason is structural:
velocity is the one head that does not bias toward post-decline
names by construction. The other three (RSI oversold, scalogram
counter-trend, regime-CWT high-divergence) all preferentially pick
names that just moved against the wider market; velocity picks
names that moved a lot in *either* direction. On a 312-name
survivor universe with positive long-run drift, "moved a lot" is
closer to neutral than "moved down a lot."

But "least bad" still means zero alpha after costs. The wide
universe + 20-day rebal + 10bps friction is a hostile environment
for top-N basket strategies in this feature class.

## Mega-cap dependence check

Unlike `relational-analog` and `relational-empirical` (which posted
Sharpe ~1.07 on Phase-2 mega-caps but collapsed to ~0.40 on
`stooq_us_long` per `findings/relational-universe-shift.md`),
velocity was *designed* for the wide universe — the canonical
checkpoint metadata in `build_canonical_checkpoints.py` already
pins `train_sharpe=0.60 / val_sharpe=0.60` on `stooq_us_long` at
the Phase-11 (2010-2025) split. This walk-forward baseline
broadens the eval to 2005-2023 with 6 non-overlapping val windows
and finds mean Sharpe +0.774 ≈ passive EW +0.772 — consistent with
the Phase-11 0.60 number once GFC-inclusive windows are folded in.

The lack of universe-shift hurt is not a win: the wide-universe
baseline is already at zero alpha, so there is no Phase-2-era
benchmark to fall from. Velocity does not have a hidden
mega-cap-specific edge waiting to be discovered.

## What this means for the per-regime question

The user's original Step 1 question ("would per-regime universe
pre-selection lift velocity to `confirmed-OOS`?") is
**null-conditioned by this Step 0 finding.** Per-regime
pre-selection routes the strategy to its strongest sub-regime —
but velocity's strongest sub-regime (2011-2017 cluster) does not
share an obvious causal regime feature with the next-best (w4
2017-2020 +0.016). The two negative-alpha tails (w0 GFC tail, w5
post-COVID vol) are macro-distinct in ways a regime detector
could in principle separate, but the magnitudes (−0.11, −0.23) are
small enough that even perfect regime routing into the positive
windows would lift mean alpha to ~+0.13 — barely above the
`partial-OOS` bar and below the `confirmed-OOS` bar by a wide
margin.

Pre-selection can amplify a real positive signal; it cannot
manufacture one. Per CLAUDE.md's verdict-table for
`confirmed-null`:

> Stop testing variations of the same lever — find an orthogonal one.

Step 1 (per-regime universe Optuna) is **not triggered**. The next
experiment is not a refinement of the velocity head.

## Proposed next-experiment per CLAUDE.md verdict-table

For `confirmed-null`, the canonical next-step is an orthogonal
lever. Velocity is the fourth and final regime/relational top-N
basket head to be falsified on `stooq_us_long`; the family is
exhausted. The orthogonal levers remaining are already in motion
at the workspace level:

1. **Retire the velocity arm from the relational app's
   six-scoreboard surface for universe-agnostic deployment.**
   The arm remains in `build_canonical_checkpoints.py` for
   reproducibility but should not be wired through the
   `ss-relational live` dispatch on `stooq_us_long` without a
   conditioning layer.
2. **Workspace pivot already underway.** The arc-level
   pivot from cross-sectional return forecasting to
   prediction-problem variants (`apps/gate` drawdown forecast,
   `apps/pairs` mean-reversion, `apps/vol` IV-vs-RV, `apps/dca`
   passive basket) is the orthogonal-lever response; this finding
   does not introduce a new lever, it confirms the existing pivot.
3. **No change to deployment recommendation.** DCA + vol_v3 sleeve
   remains canonical; the meta-allocator panel does not gain a new
   arm. The four-way regime/relational-head reality check
   collectively closes out the wide-universe top-N basket family.

## Reproduce

```bash
uv run python apps/relational/scripts/velocity_universe_agnostic.py
```

Wall time: ~3 min local (Intel Mac, no Modal) — CWT cache is
reused across the 30-cell grid; first-call cache fill is ~12s for
312 tickers × 8 scales. Caffeinate optional.

## Artifacts

- Driver: `apps/relational/scripts/velocity_universe_agnostic.py`.
- NPZ: `Output/regime-velocity-universe-agnostic-walkforward.npz`
  (carries `oos_block_returns`, `oos_ew_returns`,
  `pre_registered_bar`, `periods_per_year=252`,
  `verdict='confirmed-null'`, plus grid metadata).
- JSON summary:
  `Output/regime-velocity-universe-agnostic-walkforward.json`
  (per-window + full 30-cell grid).
- DSR ladder entry: `compute_dsr.py` ArcSpec
  `key='regime-velocity-universe-agnostic'`, n_trials=30,
  mode='overlay', benchmark_key='oos_ew_returns'.

## Master walk-forward log pointer

[`confirmed-null`](../leaderboard.md#verdict-labels) — see the
2026-05-25 leaderboard row for the relational regime-velocity
universe-agnostic walk-forward.
