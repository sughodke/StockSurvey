---
tags:
  - relational
  - arc-synthesis
---

# Relational arc synthesis — what the 12-phase CWT-dislocation research closed out

Compresses the durable findings from a 12-phase research arc through
`apps/relational/` (originally tracked in
`apps/relational/NO_OPTIONS.md`, lifted here and deleted from the
app dir 2026-05-10). The arc opened with the question *"do the
CWT-bundle scorers — already proven on long-only equity — extract
forecastable vol surprise vs IV?"* It closed with three durable
results: one shippable strategy, two falsified strategy classes,
and one operational rule about how to use the fingerprint
embedding.

## What's shippable

| Strategy | Universe | Sharpe | Notes |
|---|---|---:|---|
| **Long-only Phase-2** equal-weight top-10 by `empirical` / `farthest` / `analog` | 21 mega-caps | **1.07–1.13** | Phase-2-specific. Same scorers degrade to Sharpe ~0.4 on stooq_us_long. |
| **Transition-triggered rebal of `analog`** | stooq_us_long (312) | **0.63** | 25 rebals over 26 years (vs 325 for scheduled-20d which scored 0.42). Fingerprint cluster transitions are the rebal trigger. |
| **Velocity-magnitude scorer** | stooq_us_long (312) | **0.60** | Continuous version of the transition-triggered idea: rank by `‖fp[t,i] − fp[t-W,i]‖`. Beats `farthest` (snapshot) by +0.24. |

[Passive EW benchmark](passive-ew-benchmark.md) regrades all of
these against the 0.85 stooq_us_long EW baseline — the wider-
universe winners (Sharpe 0.6) are alpha-negative vs passive on
that universe, and the Phase-2 winners deliver only +0.07 alpha
within noise. **Operationally that means: the relational arc's
"shippable" set is shippable in the sense of "real positive
Sharpe", not in the sense of "clears the EW gate."** Decide
deployment vs the gate, not vs zero.

## What was falsified (with mechanism)

### Vol-surprise / short-vol prediction (Phases 1-5)

CWT scorers do not predict the forward IV/RV gap better than the
unconditional universe-wide short-vol baseline. **Universe-wide
short-vol baseline: Sharpe 0.51, MaxDD −83%, 69% win rate.**
9 scorers (5 dislocation + 4 brainstorm) settled in t-stats
`[-1.08, +1.49]` on the DoltHub IV anchor. The `n1_entropy` arm's
Phase-2 result (t = +1.49) was the best individual scorer.

The 47-rebal gauss314 result that initially looked promising
(idea C at t = −2.21) **collapsed to t = −0.63 when 31 more
rebals from DoltHub were added** — a window artifact. Lesson
preserved separately as a cross-validation discipline rule:
*single-window IV results are noise unless reproduced across
two independent IV sources*.

`apps/vol`'s
[surface-features v0](vol-surface-v0.md) partially refutes the
"IV market efficiently incorporates the dislocation" framing: it
incorporates dislocation at the **ATM-IV level** the prior arc
tested, but leaves residuals at the **full-surface level**
(skew + smile + multi-horizon IV/HV + OI imbalance + VIX-spread)
the prior arc didn't compute.

### Sizing overlays (Phase 6)

12 variants (3 scorers × {equal, RP, VT, RP+VT}). **Equal-weight
wins on Sharpe.**

| rank | strategy | Sharpe | CAGR | max DD |
|---|---|---:|---:|---:|
| 1 | empirical \| equal | 1.13 | 22.4% | −38.0% |
| 1 | farthest \| equal | 1.13 | 21.0% | −32.2% |
| 3 | baseline \| equal | 1.07 | 20.7% | −38.8% |
| 4 | farthest \| vt | 1.06 | 31.7% | −58.0% |
| 12 | baseline \| rp+vt | 0.85 | 25.3% | −68.2% |

Risk-parity dilutes the alpha-rich high-vol picks; vol-targeting
amplifies noise via the leverage knob and adds turnover; combined
RP+VT compounds both. **Operational rule: the dislocation alpha
lives in the score, not in sizing. Equal-weight is near-Pareto
on this universe.**

### Pair trades / market-neutral / cluster-pair / rank-spread (Phases 7-8)

Phase-2:

| strategy | Sharpe | CAGR | max DD |
|---|---:|---:|---:|
| empirical \| long-only | 1.13 | 22.4% | −38.0% |
| **empirical \| mkt-neutral** | **0.16** | **0.8%** | −18.9% |
| empirical \| rank-spread | 0.07 | 0.2% | −37.3% |
| empirical \| cluster-pair | −0.07 | n/a | −22% |

stooq_us_long (worse):

| strategy | Sharpe |
|---|---:|
| baseline \| mkt-neutral | −0.29 |
| empirical \| mkt-neutral | −0.58 |
| empirical \| cluster-pair | −0.62 |

**Drawdown reduction works** (Phase-2 mkt-neutral max DD −38% →
−19%) **but kills CAGR** (22.4% → 0.8%). The dislocation alpha
is mostly long-side directional — top-N picks are high-beta
names having a good period, not idiosyncratic divergers.
Hedging out market beta removes the largest alpha source.
[`apps/pairs`](pairs-classical-v0.md) confirmed independently:
classical Engle-Granger pair trading on factor-narrow is
`confirmed-null` per pre-reg.

### NN-pair / word2vec hedge (Phase 12)

Per-pick hedge: at each rebal, for each top-N long, find its
closest behavioral peer NOT in top-N and short that specific
name.

| strategy | Sharpe | max DD |
|---|---:|---:|
| **empirical \| nn-pair** | **−1.12** | **−99%** |
| farthest \| nn-pair | −0.20 | −75% |

Sanity checks confirmed the construction was correct (0
distinctness violations, healthy distance distributions). The
**premise** failed.

**Mechanism:** empirical's score is excess-divergence vs cluster
aggregate — top-N picks are stocks with the most idiosyncratic
move *relative to their cluster peers*. The "nearest behavioral
peer" is by construction another stock from the same cluster — a
name that is *correlated*, not anti-correlated, with the long.
Shorting it doesn't hedge; it doubles the bet (and pays
commissions both ways). −99% drawdown follows.

**Operational rule:** *the word2vec analogy works for similarity
selection (find behaviorally similar names → expect similar
behavior) but not for anti-hedging (the nearest behavioral peer
is the worst possible short for a name that's outperforming
because of behavioral cohort effects).*

## What partially worked but didn't transform anything

### GMM cluster softening (Phase 10)

Replaced k-means with `sklearn.mixture.GaussianMixture(diag)` to
fix boundary jitter. Lifts long-only Sharpe by **+0.03** (small
but real) and recovers cluster-pair from −0.62 to −0.40
(+0.22, but cluster-pair stays negative regardless).
Hard-vs-soft cluster-aggregate correlation mean = 0.58 — meaningfully
different but not transformative. Conclusion: jitter was a real
second-order issue, but cluster-pair structure fundamentally
isn't where the alpha lives; softening doesn't rescue it.

## Combined synthesis — three word2vec-analog tests

| construction | physical meaning | Sharpe lift vs baseline | verdict |
|---|---|---:|---|
| Phase 11 — velocity magnitude | continuous motion | **+0.06** | works |
| Phase 9 — cluster transitions | discrete motion (rebal trigger) | **+0.21** | works clearly |
| Phase 12 — NN-pair | nearest-peer hedge | **−1.5 to −1.6** | fails badly |

**Pattern:** the fingerprint embedding has real predictive
content for **positional dynamics** (where a stock is moving,
when it crosses regime boundaries) but **negative content for
hedge selection** (the nearest peer is the most dangerous short,
not the safest).

**Operational rule preserved in CLAUDE.md:** *use the
fingerprint embedding for selection and timing; do not use it
for hedging.*

## Reproducing

Drivers still live in `apps/relational/src/relational/research/`:

```bash
# Forward vs IV anchor — Phase 3 (kills the Phase 2 signal)
uv run python -m relational.research.diagnostic_dislocation_vs_vol \
    --data-dir ./StooqData --scorer all \
    --iv-anchor --iv-source dolthub \
    --start 2018-06-01 --end 2026-04-30

# Brainstorm scorers on IV diagnostic — Phase 4
uv run python -m relational.research.diagnostic_dislocation_vs_vol \
    --data-dir ./StooqData --scorer brainstorm \
    --iv-anchor --iv-source dolthub

# Short-vol P&L leaderboard — Phase 5
uv run python -m relational.research.diagnostic_short_vol_pnl \
    --data-dir ./StooqData --iv-source dolthub

# Wide-universe pair-trade — Phase 8
uv run python -m relational.research.diagnostic_pair_trades_wide

# Transition-triggered rebal — Phase 9 (the unlock)
uv run python -m relational.research.diagnostic_transition_triggered

# GMM vs k-means — Phase 10
uv run python -m relational.research.diagnostic_gmm_vs_kmeans

# Regime velocity — Phase 11
uv run python -m relational.research.diagnostic_velocity

# NN-pair — Phase 12 (the falsified hedge)
uv run python -m relational.research.diagnostic_nn_pairs

# Sizing overlays — Phase 6
uv run python -m relational.research.diagnostic_sizing_overlays \
    --data-dir ./StooqData

# Pair-trade overlays — Phase 7
uv run python -m relational.research.diagnostic_pair_trades \
    --data-dir ./StooqData
```

Outputs land in `Output/relational-{vol-expansion-diagnostic,
short-vol-pnl,sizing-overlays,pair-trades,...}-*.{txt,png}`.

The IV loaders + short-vol PnL primitives live in `ss_iv`
([`packages/iv/`](https://github.com/sughodke/StockSurvey/tree/master/packages/iv))
since 2026-05-10 — promoted from `apps/relational` when
[`apps/vol`](../apps/vol.md) became the second consumer.
