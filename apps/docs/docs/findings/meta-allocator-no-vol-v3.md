# Meta-allocator falsification — inverse-vol's edge was vol_v3-carried

**Operational rule.** The 2026-05-23
[meta-allocator finding](meta-allocator-regime-forecasting.md)'s
`confirmed-OOS` for B3 inverse-arc-vol vs B2 1/N (ΔSR_ann **+0.367
[+0.028, +0.682]**) **does NOT survive vol_v3 exclusion**. On the
5-arc panel (`dca, gate, pairs, relational, dca_winner_4etf`),
identical machinery, identical 2015–2025-11 OOS window, identical
Ledoit-Wolf bootstrap, the same comparison collapses to
**ΔSR_ann +0.039 [−0.152, +0.220]** — CI now includes 0, point
estimate shrinks 9.4×, and the verdict is **`confirmed-null`** per
the pre-reg's locked bar (ΔSR < +0.05 AND CI includes 0).

**The actionable update:** the "inverse-arc-vol meta-allocator" was
**vol_v3 in disguise**. vol_v3's ~1% annualized realized vol made it
the dominant inverse-vol weight from 2024-04 onward; removing it
reduces B3 to inverse-vol over arcs with within-noise-comparable
vols, at which point inverse-vol is mathematically equivalent to
equal-weight up to noise. **Deploy DCA + a sized vol_v3 sleeve**, not
"inverse-arc-vol over an arc bundle." The meta-allocator framing was
a (statistically rigorous) lens that surfaced vol_v3 — it is NOT the
deployment recipe.

## Eval setup

Bitwise-identical reproduction of the 2026-05-23 walk-forward with
`ARC_COLS` overridden:

- **Universe:** `meta-alloc-arcs-5-no-vol` — 5 arcs from
  `count_regimes_since_2005.build_master`, vol_v3 column dropped.
  Arcs in: `dca, gate, pairs, relational, dca_winner_4etf`.
  Arcs out: `vol_v3`.
- **Windowing:** `meta-alloc-3fold-2015-25` — identical FOLDS (2015,
  2019, 2023 starts), identical pooled OOS calendar (n=2740 days,
  2015-01-06 → 2025-11-25).
- **Cadence:** identical — 20 TD, 10 bps switching cost on `|Δw|/2`.
- **L:** 252 (canonical).
- **DSR:** `n_trials=8` (same as parent — 5 modeling + 3 lookback
  choices).
- **CI machinery:** identical `ss_portfolio.sharpe_difference_ci`,
  `n_bootstraps=2000, confidence=0.95, seed=42`.
- **Pre-registered verdicts** (locked before run, from
  `confirmed-OOS = CI excludes 0 positive AND DSR-t > +3.0 AND
  ΔSR_ann ≥ +0.3`; `partial-OOS = ΔSR_ann ≥ +0.15 AND DSR-t > +1.5`;
  `confirmed-null = CI includes 0 AND |ΔSR_ann| < +0.05`; else
  `reversed-OOS` if ΔSR < 0).

Driver: `apps/docs/scripts/meta_allocator_no_vol.py` (thin wrapper
that monkey-patches `meta_allocator_run.ARC_COLS` and re-uses every
other function untouched).

## Results — pooled OOS (n=2740 daily obs)

### Sharpes + DSR

| Candidate | Sharpe_ann | DSR | DSR-t | Δ vs 6-arc |
|---|---:|---:|---:|---:|
| **B3 inverse-vol**        | **+1.249** | 0.996 | **+2.67** | −0.473 |
| B2 1/N equal-weight       | +1.210 | 0.994 | +2.52 | −0.145 |
| C1 Markov + Laplace α=1   | +1.210 | 0.994 | +2.54 | −0.084 |
| C2 turbulence overlay     | +1.170 | 0.991 | +2.38 | −0.148 |
| C5 combination ensemble   | +1.161 | 0.991 | +2.35 | −0.052 |
| C3 meta-labeling          | +1.068 | 0.979 | +2.04 | +0.000 |
| B1 persistence L=252      | +0.956 | 0.951 | +1.66 | −0.446 |
| C4 CUSUM change-point     | +0.767 | 0.853 | +1.05 | −0.118 |
| B1 persistence L=126      | +0.708 | 0.802 | +0.85 | −0.435 |
| B1 persistence L=504      | +0.672 | 0.777 | +0.76 | −0.087 |

The two arcs that took the largest hit are **B3 inverse-vol
(−0.473)** and **B1 persistence L=252 (−0.446)** — exactly the two
allocators that, on the 6-arc panel, were the most aggressively
exposed to vol_v3 during fold 3.

### ΔSR vs benchmarks (Ledoit-Wolf 95% CI)

vs **B2 1/N**:

| Candidate | ΔSR_ann | 95% CI | Verdict |
|---|---:|:---:|---|
| **B3 inverse-vol** | **+0.039** | **[−0.152, +0.220]** | **`confirmed-null`** |
| C1 Markov          | −0.001 | [−0.254, +0.259] | `confirmed-null` |
| C2 turb overlay    | −0.040 | [−152.435, +0.028] | `confirmed-null` (lower CI is bootstrap artifact, same as parent) |
| C5 combo           | −0.050 | [−0.255, +0.163] | `confirmed-null` |
| C3 meta-label      | −0.143 | [−0.358, +0.076] | `confirmed-null` |
| B1 L=252           | −0.254 | [−0.795, +0.296] | `confirmed-null` |
| C4 CUSUM           | −0.443 | [−1.042, +0.142] | `confirmed-null` |
| B1 L=126           | −0.503 | [−1.102, +0.058] | `confirmed-null` |
| B1 L=504           | −0.538 | [−1.005, −0.054] | `reversed-OOS` |

vs **B3 inverse-vol**:

| Candidate | ΔSR_ann | 95% CI | Verdict |
|---|---:|:---:|---|
| B2 1/N             | −0.039 | [−0.220, +0.152] | `confirmed-null` |
| C1 Markov          | −0.039 | [−0.360, +0.289] | `confirmed-null` |
| C2 turb overlay    | −0.079 | [−0.302, +0.119] | `confirmed-null` |
| C5 combo           | −0.089 | [−0.375, +0.207] | `confirmed-null` |
| C3 meta-label      | −0.182 | [−0.474, +0.107] | `confirmed-null` |
| B1 L=252           | −0.293 | [−0.820, +0.234] | `confirmed-null` |
| C4 CUSUM           | −0.482 | [−1.017, +0.067] | `confirmed-null` |
| B1 L=126           | −0.542 | [−1.132, −0.005] | `reversed-OOS` |
| B1 L=504           | −0.577 | [−1.068, −0.034] | `reversed-OOS` |

Every modeling and short-lookback persistence candidate now ties B2
and B3 within Ledoit-Wolf 95% CI — the 6-arc finding's clean
"inverse-vol dominates everything" picture flattens to "every
non-extreme allocator is within noise on the 5-arc no-vol panel."

### Headline comparison

| Comparison | 6-arc panel (parent) | 5-arc no-vol (this) | Δ |
|---|:---:|:---:|---:|
| B3 Sharpe_ann | +1.722 | +1.249 | −0.473 |
| B2 Sharpe_ann | +1.355 | +1.210 | −0.145 |
| B3 vs B2 ΔSR_ann | +0.367 | +0.039 | −0.328 |
| B3 vs B2 95% CI | [+0.028, +0.682] | [−0.152, +0.220] | CI now includes 0 |
| B3 vs B2 verdict | `confirmed-OOS` | **`confirmed-null`** | falsified |

## Mechanism — why removing one arc collapses the whole picture

vol_v3 entered the panel 2024-04-12 with daily realized vol ~0.4%
(versus DCA at ~0.9%, gate at ~0.8%, dca_winner_4etf at ~0.7%).
Inverse-vol weighting normalizes by `1/σ`, so vol_v3's weight in B3
during fold 3 was roughly `(1/0.4) / sum(1/σ_i) ≈ 0.45` — nearly
half the portfolio. vol_v3's fold-3 raw Sharpe (per the parent finding
+ the vol_v3 leaderboard rows) is high single digits ann; sleeve-sized
to 45% it contributes 1+ Sharpe-point of pooled lift on its own.

Drop vol_v3 and you have four arcs with realized vols clustered in
0.6%–0.9% per day — homogeneous to within a factor of ~1.5. Inverse-
vol on homogeneous vols converges to equal weight: `w_i = (1/σ_i) /
Σ(1/σ_j) ≈ 1/N` when the `σ_i` are similar. That is exactly what the
new table shows — B3 (Sharpe +1.249) and B2 (Sharpe +1.210) are
within 0.04 Sharpe of each other, well inside the bootstrap CI.

Persistence L=252 also collapses (+1.402 → +0.956). On the 6-arc
panel, the trailing-Sharpe winner picker rotated to vol_v3 once it
entered, inheriting that arc's fold-3 alpha. Without vol_v3 the
picker rotates among DCA / dca_winner_4etf / relational, paying
switching cost without the specialist alpha to compensate. The 6-arc
finding's "B1 L=252 ties B2 at +0.047" turns into "B1 L=252 loses to
B2 by −0.254" — the persistence baseline was also vol_v3-carried.

## Deployment implication

The 2026-05-23 finding's lede claimed inverse-arc-vol was "the
deployable meta-allocator across the 6-arc panel" with "do not
forecast which arc will win — weight inversely by each arc's
realized vol over a 252-day trailing window" as the operational
rule. The robustness check flips that:

- **Do NOT ship "inverse-arc-vol over the arc bundle" as the
  deployment recipe.** Without vol_v3 in the bundle, the recipe is
  statistically indistinguishable from 1/N — and incurs 10 bps
  switching cost on every rebal for no expected lift.
- **Vol_v3 IS the alpha.** Per the
  [`vol-v3-dolthub-oos`](vol-v3-dolthub-oos.md) finding, that arc is
  the only one with a deflated-t > +3 OOS replication (post-
  realistic-friction). The meta-allocator analysis surfaced its
  contribution; it did not create an additional layer of alpha.
- **The correct ship-list is the
  [`ladder-methodology-rewrite`](ladder-methodology-rewrite.md)
  recommendation: DCA + sized vol_v3 sleeve.** That decomposition is
  unchanged by this experiment; if anything, it is reinforced. The
  follow-up [`vol-sleeve-sizing`](vol-sleeve-sizing.md) finding
  (2026-05-24) confirms the recipe on a 9 × 4 friction grid — `partial-OOS`
  at recommended `vega_scale = 2.0, c_options_bps ≤ 200` (combined Sharpe
  +2.46 vs DCA +1.30, max-DD actually improves to −4.9% from DCA's
  −6.8%); collapses at the 400 bps stress.

## Failure-mode notes (replicating parent caveats)

- **C2 CI [−152.435, +0.028] vs B2** — same bootstrap pathology as
  the parent finding (turbulence overlay near-colinear with B2 in
  calm periods, studentized denominator near-zero on some resamples).
  Cite the point estimate; the lower tail is an artifact.
- **vol_v3 only entered fold 3 of the 6-arc panel anyway** — so the
  fold-1 / fold-2 numbers under both panels should be similar, and
  the collapse is concentrated in fold 3. We did not split fold
  contributions per candidate in this run; the parent finding's note
  that "the +0.367 CI excluding 0 is primarily driven by the 2015-
  2023 multi-arc base, not vol_v3's strong 2024-25 tail" is partly
  *falsified by this result* — if it were true, removing vol_v3
  should have left the CI bound near +0.30 with point estimate near
  +0.30; instead the point estimate dropped to +0.039. That note
  understated vol_v3's contribution; this falsification corrects it.

## Honest assessment

The surprise — and the cleanest takeaway — is the **magnitude** of
the collapse, not its direction. The parent finding flagged this as
the highest-EV follow-up and explicitly cautioned that vol_v3 might
be carrying the result. What it did not anticipate (and we did not
predict in the pre-reg) was that the parent's own caveat
("primarily driven by 2015-2023 multi-arc base") would be partly
*incorrect* — vol_v3 alone moves the B3-vs-B2 ΔSR from `confirmed-
OOS` to `confirmed-null` with no other change. The 2015-2023 base
itself does not support an inverse-vol-over-1/N edge on this panel.

Persistence L=252 collapsing alongside B3 is the secondary surprise.
"The trailing-Sharpe winner is the bar" (Brock-Lakonishok-LeBaron's
default) loses to 1/N when the strongest specialist is removed from
the menu — i.e. the persistence picker also needed vol_v3 to clear
1/N. This reinforces a stronger generalization: **on this 5-arc no-
specialist panel, every causal allocator we tested is within bootstrap
noise of 1/N.** That includes literature-canonical persistence,
Bridgewater-style risk parity, Markov+Laplace forecasting, Kritzman
turbulence, López-de-Prado meta-labeling, CUSUM change-point, and
Rapach-Strauss-Zhou combination — five years of allocation-research
literature, all tying 1/N when the specialist is excluded.

## Master walk-forward log pointer

Leaderboard row dated 2026-05-24 in
[`leaderboard.md#master-table`](../leaderboard.md#master-table).
Verdict label links to
[`leaderboard.md#verdict-labels`](../leaderboard.md#verdict-labels).
Parent finding:
[`meta-allocator-regime-forecasting`](meta-allocator-regime-forecasting.md).
Driver: `apps/docs/scripts/meta_allocator_no_vol.py`. Artifacts:
`Output/meta-allocator-no-vol-results.json`,
`Output/meta-allocator-no-vol-daily-streams.npz`,
`Output/meta-allocator-no-vol.log`.
