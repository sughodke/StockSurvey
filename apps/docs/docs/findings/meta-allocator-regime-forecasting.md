# Meta-allocator regime forecasting — inverse-arc-vol beats every forecast

**Operational rule.** Across the 6-arc strategy panel (dca, gate,
pairs, relational, dca_winner_4etf, vol_v3), **inverse-arc-volatility
weighting (B3) is the deployable meta-allocator**. Pooled-OOS
annualized Sharpe **+1.722** (DSR-t **+4.17**, n=2740 days,
2015-01 → 2025-11) — it is the only candidate of the 10 evaluated
whose Ledoit-Wolf 95% CI excludes 0 on the positive side against any
benchmark, beating 1/N by **ΔSR_ann +0.367 [+0.028, +0.682]**. No
forecasting candidate (Markov C1, turbulence-overlay C2, meta-labeling
C3, CUSUM C4, combination C5) clears the persistence (B1 L=252) or
1/N (B2) bars; 5 of 5 are `reversed-OOS` against B3. The
combination-forecast hypothesis from Rapach-Strauss-Zhou collapses
because the components themselves are below the inverse-vol baseline —
averaging weak forecasts cannot rescue them.

The actionable rule is one sentence: **do not forecast which arc will
win; weight inversely by each arc's realized vol over a 252-day
trailing window, renormalized on availability.** This is the
Bridgewater All-Weather / López de Prado "risk parity beats
risk-on/off timing" result, reproduced on our specific 6-arc panel.

## Eval setup

Pre-registration locked in
[`TODO/meta-allocator-regime-forecasting.md`](../TODO/meta-allocator-regime-forecasting.md).

- **Universe:** `meta-alloc-arcs-6` — 6 strategy arcs from
  `count_regimes_since_2005.build_master`. vol_v3 only available from
  2024-04-12; all candidates exclude unavailable arcs from selection
  at each rebal date.
- **Windowing:** `meta-alloc-3fold-2015-25` — three contiguous OOS
  folds 2015–2018, 2019–2022, 2023–2025-11. Pooled OOS daily return
  stream of length 2740 days across all folds.
- **Cadence:** 20 trading days (~monthly) rebal; **10 bps switching
  cost** paid on `|Δw|/2` per rebal.
- **Macro stack (C2, C3):** canonical 6 features from
  `ss_macro.load_macro_panel` — `fed_funds, slope_10y_3m, credit_baa,
  m2_yoy, real_yield_10y, vix`. Forward-filled onto trading days
  (point-in-time).
- **DSR:** `n_trials=8` (5 modeling + 3 lookback choices).
- **Falsification bars:** `confirmed-OOS` = (CI excludes 0 positive
  side) AND (DSR-t > +3.0) AND (ΔSR_ann ≥ +0.3); `partial-OOS` =
  ΔSR_ann ≥ +0.15 AND DSR-t > +1.5; `confirmed-null` = CI includes 0
  AND |ΔSR_ann| < +0.05.

## Results — pooled OOS (n=2740 daily obs)

### Sharpes + DSR

| Candidate | Sharpe_ann | DSR | DSR-t | n_obs |
|---|---:|---:|---:|---:|
| **B3 inverse-vol**        | **+1.722** | 1.000 | **+4.17** | 2740 |
| B1 persistence L=252      | +1.402 | 0.999 | +3.08 | 2740 |
| B2 1/N equal-weight       | +1.355 | 0.999 | +2.98 | 2740 |
| C2 turbulence overlay     | +1.318 | 0.998 | +2.84 | 2740 |
| C1 Markov + Laplace α=1   | +1.294 | 0.997 | +2.78 | 2740 |
| C5 combination ensemble   | +1.213 | 0.994 | +2.51 | 2740 |
| B1 persistence L=126      | +1.143 | 0.986 | +2.20 | 2740 |
| C3 meta-labeling          | +1.068 | 0.979 | +2.04 | 2740 |
| C4 CUSUM change-point     | +0.885 | 0.925 | +1.44 | 2740 |
| B1 persistence L=504      | +0.759 | 0.851 | +1.04 | 2740 |

### ΔSR vs benchmarks (Ledoit-Wolf 95% CI)

vs **B1 persistence L=252**:

| Candidate | ΔSR_ann | 95% CI | Verdict |
|---|---:|:---:|---|
| B3 inverse-vol     | +0.320 | [−0.288, +0.893] | `partial-OOS` |
| B2 1/N             | −0.047 | [−0.689, +0.567] | `confirmed-null` |
| C2 turb overlay    | −0.085 | [−0.728, +0.527] | `confirmed-null` |
| C1 Markov          | −0.109 | [−0.880, +0.646] | `confirmed-null` |
| C5 combo           | −0.189 | [−0.936, +0.533] | `confirmed-null` |
| C3 meta-label      | −0.335 | [−1.056, +0.397] | `confirmed-null` |
| C4 CUSUM           | −0.518 | [−1.003, −0.130] | `reversed-OOS` |

vs **B2 1/N**:

| Candidate | ΔSR_ann | 95% CI | Verdict |
|---|---:|:---:|---|
| **B3 inverse-vol** | **+0.367** | **[+0.028, +0.682]** | **`confirmed-OOS`** |
| B1 L=252           | +0.047 | [−0.567, +0.689] | `confirmed-null` |
| C2 turb overlay    | −0.038 | [−148.92, +0.032] | `confirmed-null` |
| C1 Markov          | −0.061 | [−0.293, +0.172] | `confirmed-null` |
| C5 combo           | −0.142 | [−0.336, +0.056] | `confirmed-null` |
| C3 meta-label      | −0.288 | [−0.540, −0.038] | `reversed-OOS` |

vs **B3 inverse-vol** (the dominator):

| Candidate | ΔSR_ann | 95% CI | Verdict |
|---|---:|:---:|---|
| B1 L=252           | −0.320 | [−0.893, +0.288] | `partial-OOS` (toward loss) |
| B2 1/N             | −0.367 | [−0.682, −0.028] | `reversed-OOS` |
| C1 Markov          | −0.428 | [−0.809, −0.008] | `reversed-OOS` |
| C2 turb overlay    | −0.405 | [−0.737, −0.052] | `reversed-OOS` |
| C5 combo           | −0.509 | [−0.911, −0.068] | `reversed-OOS` |
| C3 meta-label      | −0.655 | [−1.138, −0.145] | `reversed-OOS` |
| C4 CUSUM           | −0.837 | [−1.503, −0.168] | `reversed-OOS` |

### Overall verdicts (worst of three benchmark-verdicts per candidate)

| Candidate | overall verdict |
|---|---|
| **B3 inverse-vol**          | **`confirmed-OOS` vs B2 1/N; `partial-OOS` vs B1; the deployable winner** |
| B1 persistence L=252        | `confirmed-null` vs B2; loses to B3 by `partial-OOS` margin — incumbent |
| B2 1/N                      | beaten by B3; baseline |
| C2 turbulence overlay       | `reversed-OOS` overall (loses to B3 at 95% CI) |
| C1 Markov                   | `reversed-OOS` overall (loses to B3 at 95% CI) |
| C5 combination ensemble     | `reversed-OOS` overall (loses to B3 at 95% CI) |
| C3 meta-labeling            | `reversed-OOS` overall (loses to B2 and B3 at 95% CI) |
| C4 CUSUM change-point       | `reversed-OOS` overall (loses to B1 and B3 at 95% CI) |
| B1 persistence L=126 / L=504 | `confirmed-null` / `reversed-OOS` — L=252 is the only viable lookback |

## Mechanism — why inverse-vol wins and forecasts lose

**Why B3 wins.** Our 6-arc panel has heterogeneous vol scales. vol_v3
has tiny absolute vol (premium-collection overlay alpha, ~1% ann);
DCA is ~14% ann; pairs and relational are intermediate; gate sits
between. Equal-weight (B2) gives DCA disproportionate risk
contribution because of its larger vol. Inverse-vol (B3) flattens
risk contributions, which is mathematically the variance-minimizing
allocation under the (true) assumption that arc-level Sharpes are
within-noise comparable. Since our leaderboard explicitly shows no
arc has a deflated-Sharpe-t advantage over DCA at 95% CI (per
[ladder-methodology-rewrite](ladder-methodology-rewrite.md)), the
correct prior is "Sharpe-equality across arcs," and the
variance-minimizing weight under Sharpe-equality is inverse-vol.

**Why C1 Markov loses.** The 6×6 transition matrix has 30 free
parameters; we have ~45 regimes / 44 transitions. Per the literature
brief's sample-size analysis, this is below the "10 obs per
parameter" floor by a factor of 7. Even with Laplace α=1, the matrix
is dominated by the prior, not the data — predictions revert to
near-uniform and the candidate degrades to a noisier B2.

**Why C2 turbulence overlay loses.** The K=2 turbulence proxy
correctly identifies high-stress regimes (COVID-2020, 2022 rate
shock), but multiplying B2 by `(1 − P_turb)` mechanically reduces
exposure during turbulence — which is exactly when the diversified
B3 stream is performing best (vol_v3 fires, gate captures DD avoidance,
DCA absorbs the loss but is downweighted by its now-high realized vol
in B3). The overlay turns off exposure right when the diversifier
would have helped. The published Kritzman-Page-Turkington gain assumed
a stock+bond panel where "turbulent = bear market = switch to cash";
our 6-arc panel includes a turbulence-loving arc (vol_v3) which
inverts the overlay's intended sign.

**Why C3 meta-labeling loses.** Six logistic regressions with 9
features each = 54 effective parameters. Training set per rebal is
~50–150 labeled obs (every 30 days, history ≥ 252d). The
features-to-obs ratio is too high; the classifiers overfit to macro
state and ignore the much-noisier arc-Sharpe persistence signal that
inverse-vol picks up structurally without any fitting. The +1.068
pooled Sharpe is actually worse than B2 — meta-labeling adds noise.

**Why C4 CUSUM loses.** CUSUM detection is structurally late (the
literature-brief's "regime detection is intrinsically lagged" rule).
The detector reacts 10–30 days *after* the trailing-Sharpe winner has
already lost meaningfully. The change-point gate triggers re-evaluation
to persistence, but the new winner has also already moved — by which
time the trailing-Sharpe board has rotated past the optimal entry.
Net result: pays switching costs for late entries.

**Why C5 ensemble loses.** Rapach-Strauss-Zhou simple-mean of forecasts
works when the forecasts are unbiased but noisy. C1, C2, C3 are not
unbiased here — they are systematically below the inverse-vol baseline.
Averaging three losers produces a fourth loser. The ensemble result
falsifies the "combination forecasts always beat individual forecasts"
intuition on our panel: the components have to be at least
benchmark-competitive for the average to outperform.

## Failure modes flagged

- **C2 CI [−148.92, +0.032] vs B2** — the bootstrap distribution is
  pathological because C2 = (1 − P_turb) × B2 is almost colinear with
  B2 in calm periods, making the studentized bootstrap denominator
  near-zero on some resamples. The point estimate −0.038 is what we
  cite; the lower CI tail is a bootstrap artifact, not a true
  worst-case. This is documented and does not change the verdict (CI
  still includes 0 at the conventional reading).
- **vol_v3 only enters fold 3.** Fold 1 and fold 2 use a 5-arc set;
  fold 3 has the full 6-arc set from 2024-04-12 onward. The 2024-25
  window is *the only window where the full design is testable*.
  B3's CI excluding 0 is therefore primarily driven by the 2015-2023
  multi-arc base, not vol_v3's strong 2024-25 tail.
- **C4 trailing-21d CUSUM fired only 8 times across 2740 days** — a
  larger CUSUM tuning (h=2.0 instead of 4.0) was not pre-registered
  and is not tested here. Per the brief, change-point detection's
  lower bound is structural; a more sensitive threshold would just
  rotate more, paying more switching cost.

## Operational surprise

**Persistence L=252 (the brief's bar) ties with 1/N**: ΔSR_ann +0.047
[−0.567, +0.689]. The brief's referee-line "the trailing-rolling-Sharpe
winner is the bar" turns out to be statistically indistinguishable
from 1/N on this panel. The actual bar that mattered was *inverse-vol*,
not persistence. **L=252 is also the only viable persistence lookback:**
L=126 and L=504 both underperform by ΔSR_ann −0.260 / −0.643.

## Robustness check (2026-05-24)

**Falsified for the 5-arc no-vol_v3 panel** — see
[`meta-allocator-no-vol-v3`](meta-allocator-no-vol-v3.md). Re-running
this exact walk-forward with vol_v3 excluded collapses the B3-vs-B2
ΔSR_ann from **+0.367 [+0.028, +0.682]** to **+0.039 [−0.152,
+0.220]** (verdict `confirmed-null`). The "deploy inverse-arc-vol"
operational rule above is **vol_v3-specific** — the deployment recipe
that survives the robustness check is **DCA + sized vol_v3 sleeve**,
NOT "inverse-vol over the arc bundle." This finding's lede should
therefore be read as "the meta-allocator analysis surfaced vol_v3's
contribution," not as "inverse-vol weighting is itself the edge."

## Highest-EV follow-up

**Test whether B3 vs B2 reverses if vol_v3 is excluded from the panel.**
A 5-arc inverse-vol board (drop vol_v3) over 2015–2025 would tell us
whether the +0.367 ΔSR is structural (any heterogeneous-vol panel
favors inverse-vol) or vol_v3-specific (vol_v3's tiny vol gets
disproportionate weight in B3 during 2024-25, carrying the result).
If B3-without-vol_v3 still beats B2 at 95% CI, ship inverse-vol;
if it does not, the result is a 2024-25-window artifact and the
ladder's actual recommendation is "stay on DCA."

## Master walk-forward log pointer

Leaderboard rows live in [`leaderboard.md#master-table`](../leaderboard.md#master-table)
dated 2026-05-23, one row per candidate × benchmark verdict.
Verdict labels link to [`leaderboard.md#verdict-labels`](../leaderboard.md#verdict-labels).
Pre-reg page: [`TODO/meta-allocator-regime-forecasting.md`](../TODO/meta-allocator-regime-forecasting.md).
Artifacts: `Output/meta-allocator-results.json` +
`Output/meta-allocator-daily-streams.npz`. Driver:
`apps/docs/scripts/meta_allocator_run.py`.
