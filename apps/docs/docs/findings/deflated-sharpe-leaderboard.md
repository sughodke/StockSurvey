# Cross-arc comparability: the Deflated Sharpe Ratio

**Operational rule.** A raw annualized Sharpe is *not* a fair cross-arc
ranking key, and neither is the leaderboard's "mean alpha" /
"mean val Sharpe" — those are **means of per-window Sharpe ratios**,
which can carry the opposite sign from the Sharpe of the actual
deployable return stream. Rank arcs on the **deflated-Sharpe t-stat**
(`ss_portfolio.standardize_oos`), computed on each arc's OOS *net
return stream*, with the multiple-testing deflation set by the number of
configurations the arc tried. Only rows that **form a return stream**
can carry this number; meta-evaluations that compose scalar Sharpes and
non-portfolio diagnostics are `DSR N/A` by construction.

## Why the existing Sharpe column isn't apples-to-apples

The leaderboard's Sharpe column mixes conventions across arcs — daily vs
block Sharpe, gross vs net, absolute vs alpha, long-only vs long-short,
21 mega-caps vs 2073 equities vs a 13-ETF basket vs an options panel.
All are annualized (shared `sqrt(252)` / `sqrt(252/rebal_days)`
convention), but annualization is the *only* thing they share. Worse,
many rows report a **mean of per-window Sharpe ratios**, which is a
statistic of statistics — not the Sharpe of the strategy you would
actually deploy.

The Deflated Sharpe Ratio (Bailey & López de Prado 2014) normalizes all
of this to one unit-free number by operating on the return stream and
correcting for three things a naive Sharpe ignores:

- **higher moments** — fat tails / skew inflate a naive Sharpe (PSR
  term);
- **sample length** — a 5-window arc is noisier than a 6-window one
  (the `sqrt(N-1)` term);
- **selection bias** — the more configurations an arc tried, the higher
  the Sharpe it should be *expected* to produce by chance (the
  expected-maximum-Sharpe deflation, `n_trials`).

The reported `deflated_tstat` is the z behind `DSR = Phi(z)`; it is the
cross-arc ranking key.

## Harness

- `ss_portfolio.standardize_oos(returns, *, periods_per_year, n_trials,
  trial_sharpes=None, sharpe_std=None, benchmark=None) -> MetricBlock` —
  the single source of truth. Self-contained PSR/DSR math (normal
  CDF/PPF implemented in-module; `ss_portfolio` stays numpy-only). 13
  unit tests in `packages/portfolio/tests/test_deflated.py`.
- `apps/docs/scripts/compute_dsr.py` — reads each arc's
  `Output/<arc>-returns.npz`, runs the harness, writes the ranked
  `Output/dsr-leaderboard.json`.
- Each deployable arc's eval driver gained a `--dump-returns` flag that
  concatenates the per-window OOS **net** return stream and writes the
  npz.

**Standalone vs overlay framing.** For rows claiming an absolute Sharpe
(factor, relational, vol) DSR is computed on the strategy's own stream.
For *overlay* rows claiming alpha over a benchmark (gate, any
timing/exposure overlay) DSR is computed on the **excess** stream — the
claimed edge — so "is the claimed edge real" stays comparable across
both kinds.

## Results so far

`compute_dsr.py` ranks every arc that has been re-run with a return
dump. Completed (local) arcs:

| arc | mode | n_trials | stream ann. Sharpe | skew | kurt | E[max SR] | DSR | deflated t |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pairs-v0 | standalone | 4 | +0.203 | +0.61 | 20.81 | 0.015 | 0.429 | −0.180 |
| gate-v0 (overlay) | overlay | 6 | −0.100 | +0.40 | 51.24 | 0.019 | 0.042 | −1.727 |

Both confirm-and-sharpen the existing verdicts, and both expose the
mean-of-Sharpes artifact:

- **gate v0** — the row's headline "alpha +0.067 Sharpe" is a mean of
  per-window Sharpe *differences*. The actual excess return stream
  (gated − unconditional EW, 4680 OOS days) has annualized Sharpe
  **−0.10**, deflated t **−1.73**, DSR **0.04**. The overlay's claimed
  edge is not skill — and the excess-kurtosis-51 fat tail (the gate
  concentrating into crises) makes the naive Sharpe especially
  misleading. Sharpens the prior `partial-OOS` reading toward "no edge".
- **pairs v0** — row reports mean agg val Sharpe +0.099; the
  concatenated stream Sharpe is +0.203, DSR 0.43, deflated t −0.18 — not
  skill, consistent with the existing `confirmed-null`.

## Scope: which rows can carry a DSR

Re-running surfaced that the 96 leaderboard rows fall into three classes
by **whether the eval ever forms a return stream**:

1. **Stream-bearing strategy arcs** (gate ✅, pairs ✅, factor,
   relational, vol, DCA / cfr-phase4d) — get a true returns-based DSR.
2. **Meta-evaluations** that compose *scalar* per-window Sharpes (cfr
   macro-gate, the oracle arms, sizing/overlay diagnostics, regime
   Optuna best-params) — `DSR N/A`; even re-running can't produce a
   stream because the eval is arithmetic on Sharpes.
3. **Non-portfolio diagnostics** (replay R², macro-regime Pearson,
   compression error) — `DSR N/A` (Sharpe undefined; already tagged).

A deflated Sharpe is *defined* only on a return stream, so the rankable
ladder is the stream-bearing strategy arcs; the rest stay as
falsification history.

## Remaining work (staged)

- **factor** — numpy stream mirrors `block_port_returns_np` /
  `block_port_returns_long_short_np` added to `factor.objectives` and
  validated against the tinygrad `block_sharpe` scalars (match to
  float32 precision). Next: capture the per-val-window streams in
  `train_walkforward`, ship them back through the Modal entrypoint, run.
- **vol** — short-vol book stream dump + Modal run.
- **relational** — phase-2 8-arm winner (analog cross_ticker) equity
  curve → val return stream; Modal.
- **DCA / cfr-phase4d** — local backtests → streams.
- **Trial-count reconstruction** — ✅ pinned per arc in
  `compute_dsr.py` SPECS (conservative; round up when ambiguous, since
  under-counting trials weakens the deflation):

  | arc | rows | `n_trials` | basis |
  |---|---:|---:|---|
  | gate | 4 | 6 | v0 threshold sweep {q=.85,.90,.95} × {binary, sigmoid} |
  | pairs | 4 | 4 | pre-reg config + screening-param variants |
  | factor | 38 | 50 | horizons × representations × losses × heads × universes |
  | relational | 14 | 16 | 8-arm scorer × ±DWT × {cross_ticker, per_ticker} |
  | vol | 10 | 12 | v0→v3.1 × sizing × OI filters × regime gates |
  | cfr / dca | 12 | 12 / 4 | CFR phase ladder; DCA = Phase 4a–d basket |
- **Leaderboard backfill** — once all stream arcs are computed, add the
  `deflated t-stat` in one append-only, provenance-tagged pass (mirrors
  the 2026-05-18 Sharpe backfill contract: additive, no
  verdicts/numbers altered), make it the primary sort key, and tag the
  meta/diagnostic rows `DSR N/A — <reason>`.

## Master walk-forward log

The DSR augments — does not replace — existing rows; see the
[leaderboard](../leaderboard.md) and its
[verdict labels](../leaderboard.md#verdict-labels).
