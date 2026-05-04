# NO_OPTIONS — why CWT-dislocation scorers don't (yet) ship as options strategies

Record of a research arc through `apps/relational/` that asked whether
the CWT-bundle scorers (baseline regime, A empirical, B analog, C
farthest, #1 GICS) — already proven on long-only equity — could be
rotated into options or market-neutral structures. They can't, on this
universe, with this data. The long-only scoreboard winners remain the
shippable artifact.

This file exists so the next person (or next-you) doesn't redo the work
to re-derive the negative result. Diagnostics, primitives, and
data-loaders that produced it all ship in `relational/`; rerun with the
same Phase-2 tickers, same Stooq + DoltHub-IV inputs, to reproduce.

## Hypothesis (entering)

The CWT-dislocation scorers had already shown long-only Sharpe gains in
the relational scoreboard. The session-opening question: do they
extract forecastable **vol surprise vs IV** — i.e., does dislocation
predict an options edge?

## Phase 1 — diagnostic with trailing-realized anchor

Built a per-(rebalance, ticker) test: do top-N picks show forward
realized vol systematically above their trailing realized vol?
**Result: no.** All five scorers' t-stats within `|t| < 1.2`.
Trailing-realized was a poor proxy because it ignored the vol risk
premium baked into IV.

## Phase 2 — real IV anchor (gauss314, 47 rebals, 2019-2023)

Swapped to actual ATM IV from the HF gauss314 dataset
(`load_atm_iv` in `iv_data.py`). Idea C surfaced **t = -2.21** (top-N
has forward < IV more than rest → short-vol edge). Empirical (A) and
GICS (#1) leaned positive at t = +1.5. Looked promising.

## Phase 3 — validation on DoltHub IV (78 rebals, 2019-2026)

Cloned the DoltHub `post-no-preference/options` repo (~7.5 GB), exported
`volatility_history` to a 22 MB parquet (`load_dolthub_iv_parquet`), and
re-ran. **Idea C collapsed to t = -0.63.** Adding 31 rebalances of
out-of-sample data killed the apparent signal. The 47-rebal result was
a window artifact, not a robust edge.

## Phase 4 — brainstorm scorers (N1, N2, R1)

Built three new scorers from the brainstorm shortlist explicitly
targeting vol-regime change:
- N1 — `scale_energy.py`: short/long energy ratio + Shannon entropy
- N2 — `bocpd.py`: Adams-MacKay online Bayesian changepoint
- R1 — `ot_stress.py`: Sinkhorn distance to a calm-reference cloud

Best result: **n1_entropy at t = +1.49** — interpretable
(top-N = high-entropy / noisy regime → bigger vrp capture) but not
significant. All 9 scorers settled in [-1.08, +1.49] on DoltHub IV.

## Phase 5 — short-vol P&L (reframed)

Stopped looking at t-stats and computed actual vol-points P&L per
cycle (`short_vol.py` + `diagnostic_short_vol_pnl.py`). Headline:

| strategy | Sharpe | mean P&L / cycle | win rate | cum P&L | max DD |
|---|---|---|---|---|---|
| n1_entropy bot-asc (best combo) | 0.73 | +1.72% | 74% | +1.34 | -70% |
| **UNIVERSE (sell vol on everything)** | **0.51** | +1.36% | 69% | +1.06 | -83% |
| n1_entropy top-desc (worst) | 0.28 | +0.88% | 71% | +0.69 | -95% |

**Universe-wide baseline alone reaches Sharpe 0.51 / win-rate 69%** with
no scorer required. Most scorer-direction combos cluster within ±0.1
Sharpe of universe; the scorers add noise more than alpha. Max
drawdowns of -70 to -95% mean any short-vol implementation needs a real
risk overlay (e.g., suspend during VIX > 25) before deployment.

## Phase 6 — sizing overlays on long equity

Pivoted from options to vol-targeting / risk-parity overlays atop the
existing scoreboard winners (`sizing.py` +
`diagnostic_sizing_overlays.py`). Tested 12 variants
(3 scorers × {equal, RP, VT, RP+VT}). **Equal-weight wins on Sharpe.**

| rank | strategy | Sharpe | CAGR | max DD | Calmar |
|---|---|---|---|---|---|
| 1 | empirical \| equal | 1.13 | 22.4% | -38.0% | 0.59 |
| 2 | farthest \| equal | 1.13 | 21.0% | -32.2% | 0.65 |
| 3 | baseline \| equal | 1.07 | 20.7% | -38.8% | 0.53 |
| 4 | farthest \| vt | 1.06 | 31.7% | -58.0% | 0.55 |
| 12 | baseline \| rp+vt | 0.85 | 25.3% | -68.2% | 0.37 |

Risk-parity dilutes the alpha-rich high-vol picks; vol-targeting
amplifies noise via the leverage knob and adds turnover; combined
RP+VT compounds both. The dislocation alpha lives in the score, not
in sizing. Equal-weight is already near-Pareto on this universe.

## Phase 7 — pair trades / market-neutral

Tested universe-wide hedge and rank-spread on 3 scorers
(`pairs.py` + `diagnostic_pair_trades.py`). Results on Phase-2:

| strategy | Sharpe | CAGR | max DD |
|---|---|---|---|
| empirical \| long-only | 1.13 | 22.4% | -38.0% |
| **empirical \| mkt-neutral** | **0.16** | **0.8%** | **-18.9%** |
| empirical \| rank-spread | 0.07 | 0.2% | -37.3% |

**Drawdown reduction works** (max DD -38% → -19% on
`empirical|mkt-neutral`) **but kills CAGR** (22.4% → 0.8%). The
dislocation alpha is mostly long-side directional — top-N picks are
high-beta names having a good period, not idiosyncratic divergers.
Hedging out market beta removes the largest alpha source.
Rank-spread is even worse: bot-N has no negative alpha (it's just
other up-trending names), so the short bleeds money in bull markets.

V2 of this diagnostic added idea-B long-short and an idea-A natural
cluster-aware pair (`cluster_pair_weights` in `pairs.py`), specifically
to test two follow-up hypotheses:

1. *A forecast-driven scorer (idea B) should preserve more alpha in
   long-short* — directionally correct but tiny: `analog|rank-spread`
   Sharpe **+0.03** vs `farthest|rank-spread` **-0.04**. Both still
   bleed slightly negative.
2. *Intra-cluster hedging (idea A) should preserve more alpha than
   universe hedging* — **falsified**: `empirical|cluster-pair` Sharpe
   **-0.07** vs `empirical|mkt-neutral` **+0.16**. With k=11 clusters
   on 21 names, average cluster size ≈ 2 reduces "long winner / short
   the rest" to noisy 1-vs-1 spreads, comparable DD (-22% vs -19%) but
   worse Sharpe.

Bonus finding: `analog|long-only` is real long-only alpha at Sharpe
**1.07**, comparable to baseline. Worth keeping in the shippable set
alongside empirical and farthest.

## Phase 8 — wider universe (`stooq_us_long`, 312 tickers, 2000-2026)

Re-ran the pair-trade matrix on the curated 312-ticker subset to test
the structural prediction that diversified factor exposure should let
intra-cluster (cluster-pair) and rank-spread variants extract real
cross-sectional alpha. **Falsified.**

| strategy | Sharpe | CAGR | max DD |
|---|---|---|---|
| baseline \| long-only | 0.54 | 9.6% | -61% |
| empirical \| long-only | 0.42 | 6.9% | -59% |
| farthest \| long-only | 0.36 | 5.7% | -73% |
| baseline \| mkt-neutral | -0.29 | -2.8% | -66% |
| empirical \| mkt-neutral | -0.58 | -5.2% | -81% |
| empirical \| cluster-pair | -0.62 | -7.0% | -87% |

Two new findings:
1. **Phase-2's Sharpe-1.13 long-only result was mega-cap-specific
   alpha.** Same scorers drop to 0.4-0.5 on the diversified 312-ticker
   universe. The "shippable" claim earlier in this doc is universe-
   conditional and should be qualified accordingly.
2. **Pair-trade variants are even worse on the wider universe** than
   they were on Phase-2. Diversification didn't rescue the
   cross-sectional construction; it amplified the negative result.

## Phase 9 — transition-triggered rebal (the one that worked)

Replaced fixed 20-day rebal cadence with **rebal triggers on cluster-
membership transitions**. A stock moving from one Hungarian-stabilized
empirical cluster to another (with 5-day persistence to filter
boundary jitter) fires a rebal. Idea-A scorer, 312-ticker universe.

| variant | Sharpe | CAGR | max DD | n_rebals |
|---|---|---|---|---|
| **transition-only** | **0.63** | 11.8% | -54% | **25** |
| scheduled-20d | 0.42 | 6.9% | -59% | 325 |
| transition-or-20d | 0.41 | 6.6% | -58% | 345 |

**Triggering on the 25 actual cluster-transition events over 26 years
beats the 325-trade scheduled cadence by +0.21 Sharpe.** Catching
transitions early is meaningful; the scheduled-20d cadence dilutes the
alpha by trading on dates that aren't synchronized with the
fingerprint-space regime changes the scorer is actually detecting.

This is the first scoreboard-class result of this arc that came from
a *rebal-timing* change rather than a scorer or sizing change.
Worth elevating in the shippable set.

## Phase 10 — GMM soft-cluster replacement

Replaced k-means with `sklearn.mixture.GaussianMixture(diag)` to test
whether boundary-jitter was hurting cluster-pair (the v2 cluster-pair
result was -0.07 Sharpe on Phase-2). 312-ticker universe.

| strategy | Sharpe | CAGR | max DD |
|---|---|---|---|
| empirical \| gmm | 0.45 | 7.5% | -62% |
| empirical \| kmeans | 0.42 | 6.9% | -59% |
| empirical \| gmm \| cluster-pair | -0.40 | -4.6% | -77% |
| empirical \| kmeans \| cluster-pair | -0.62 | -7.0% | -87% |

GMM lifts long-only Sharpe by +0.03 (real but small) and recovers
cluster-pair from -0.62 to -0.40 (+0.22) — but cluster-pair stays
negative regardless. Hard-vs-soft cluster-aggregate correlation
mean=0.58: meaningfully different but not transformative. Conclusion:
jitter was a real second-order issue, but cluster-pair structure
fundamentally isn't where the alpha lives; softening doesn't rescue it.

## Phase 11 — regime velocity (motion in fingerprint space)

The cluster-transition result (Phase 9) suggested fingerprint-space
*motion* matters more than snapshot position. Tested two continuous
versions of that idea (`regime_velocity.py`):
- **velocity-magnitude**: `||fp[t, i] - fp[t-W, i]||` — undirected
  motion ("how aggressively is this stock moving through fingerprint
  space?")
- **axis-alignment**: `max_k |v · axis_k|` over top-K SVD axes from a
  training window — directed motion along stable behavioral axes

312-ticker universe, same defaults, W=20:

| strategy | Sharpe | CAGR | max DD | Calmar |
|---|---|---|---|---|
| **velocity-magnitude** | **0.60** | 10.9% | -60% | 0.18 |
| baseline | 0.54 | 9.6% | -61% | 0.16 |
| axis-alignment | 0.52 | 9.0% | -64% | 0.14 |
| farthest (snapshot) | 0.36 | 5.7% | -73% | 0.08 |

**Velocity-magnitude beats baseline by +0.06 and farthest by +0.24.**
Two structural findings: (1) motion > position — the trajectory
through fingerprint space carries more signal than where the stock
currently sits; (2) magnitude > direction — undirected ||v|| beat
the SVD-axis-projection variant. The signal is in *how much* the
fingerprint is moving, not which interpretable axis it's moving along.

This is the second independent confirmation (after Phase 9) that
fingerprint-space motion is a real signal beyond snapshot dislocation.
Velocity is the continuous + scheduled version; transition-triggered
is the discrete + signal-triggered version. Both work.

## Phase 12 — nearest-neighbor pair (the word2vec hedge falsified)

Per-pick hedge construction: at each rebalance, for each top-N long,
find its closest behavioral peer NOT in the top-N and short that
specific name. The cleanest version of "trade the behavioral spread"
from the word2vec-analog framing.

312-ticker universe:

| strategy | Sharpe | CAGR | max DD |
|---|---|---|---|
| empirical \| long-only | 0.42 | 6.9% | -59% |
| farthest \| long-only | 0.36 | 5.7% | -73% |
| farthest \| nn-pair | -0.20 | -4.3% | -75% |
| farthest \| mkt-neutral | -0.41 | -6.5% | -83% |
| empirical \| mkt-neutral | -0.57 | -5.0% | -80% |
| **empirical \| nn-pair** | **-1.12** | **-14.5%** | **-99%** |

Sanity checks confirmed the construction was correct (0 distinctness
violations, 0 collapses, healthy distance distributions). The
*premise* failed.

**Why empirical|nn-pair was catastrophic.** Empirical's score is
already excess-divergence vs cluster aggregate — its top-N picks are
stocks with the most idiosyncratic move *relative to their cluster
peers*. The "nearest behavioral peer" is by construction another
stock from the same cluster — a name that is *correlated*, not
anti-correlated, with the long. Shorting it doesn't hedge; it doubles
the bet (and pays commissions both ways). -99% drawdown follows.

**Why farthest|nn-pair was less bad.** Farthest picks names far from
the universe centroid; the "nearest non-top peer" is the next-most-
distant ticker, whose idiosyncratic moves are less correlated. Some
hedging happens — just not enough to clear costs.

**The word2vec analogy works for *similarity selection* (find behaviorally
similar names → expect similar behavior) but not for *anti-hedging*
(the nearest behavioral peer is the worst possible short for a name
that's outperforming because of behavioral cohort effects).** Document
as a falsified hypothesis with a clear mechanism, not a noisy null.

## Combined synthesis — three word2vec-analog tests

| construction | physical meaning | Sharpe lift vs baseline | verdict |
|---|---|---|---|
| Phase 11 — velocity magnitude | continuous motion | **+0.06** | works |
| Phase 9 — cluster transitions | discrete motion (rebal trigger) | **+0.21** | works clearly |
| Phase 12 — NN-pair | nearest-peer hedge | **-1.5 to -1.6** | fails badly |

Pattern: the fingerprint embedding has real predictive content for
**positional dynamics** (where a stock is moving and when it crosses
regime boundaries), but **negative content for hedge selection** (the
nearest peer is the most dangerous short, not the safest). Use the
embedding for selection and timing; do not use it for hedging.

## Verdict on the original hypothesis

**Refuted on this universe** for cross-sectional / pair-trade / options
constructions; **confirmed and elevated** for signal-triggered timing
of the existing long-only scorer. The CWT scorers produce real long-
only equity alpha and the fingerprint space carries genuinely useful
motion / regime-change information. What it doesn't have is enough
cross-sectional alpha to make pair trades, market-neutral hedges, or
options strategies clear transaction costs. The IV market and the
cross-sectional equity market both efficiently incorporate the
dislocation information from the same CWT-style features.

## What's shippable

- **Long-only equity on Phase-2 only**, equal-weight, top-10 by
  empirical (A), farthest (C), or analog (B): Sharpe 1.07-1.13, CAGR
  21-22%, max DD 32-38%, Calmar 0.56-0.65 over 13 years (2013-2025,
  10 bps, 20-day rebal). Phase-9 showed this is mega-cap-specific —
  same scorers degrade to Sharpe ~0.4 on the 312-ticker universe.
- **Transition-triggered rebal of idea A on the wider 312-ticker
  universe**: Sharpe 0.63, CAGR 11.8%, max DD -54%, only **25 rebals
  over 26 years**. The right strategy on the wider universe isn't a
  pair trade — it's signal-triggered timing of the existing scorer.
- **Velocity-magnitude scorer on the wider 312-ticker universe**:
  Sharpe 0.60, CAGR 10.9%, max DD -60%, scheduled-20d. The continuous
  analog of the transition-triggered finding — same physics
  (fingerprint-space motion), different mechanic (magnitude as
  ranking score rather than transition as rebal trigger). Independent
  confirmation that motion is the signal.
- **Universe-wide short-vol overlay** if vol options are in scope:
  Sharpe ~0.5, BUT max DD -83% cumulative — needs a vol-spike
  suspension / drawdown stop overlay before deployment.

## What won't work — confirmed by Phase 8

- **Pair-trade / market-neutral / rank-spread / cluster-pair on either
  Phase-2 or stooq_us_long.** Phase-9's confirmation: the wider
  universe didn't rescue any cross-sectional construction; some got
  worse. The dislocation alpha is fundamentally long-side directional.
- **Cluster-pair specifically.** Phase-10 showed that GMM softening
  (which fixes the boundary-jitter root cause) recovers Sharpe from
  -0.62 to -0.40 — meaningful improvement on the construction's
  weakest link, but the construction is still negative.
- **Options strategies beyond "sell vol on the universe with risk
  overlay"** — the IV market efficiently incorporates whatever
  dislocation information is in our CWT bundle.

## All diagnostics complete

(Earlier draft of this doc had `regime_velocity` and `nn_pairs` as
"code-only / unrun" — Phases 11 and 12 above are their results.)

## Reproducing

```bash
# Forward vs trailing-realized — null result (Phase 1)
uv run python -m relational.research.diagnostic_dislocation_vs_vol \
    --data-dir ./StooqData --scorer all

# Forward vs IV anchor (gauss314 daily, 2019-2023) — Phase 2
uv run python -m relational.research.diagnostic_dislocation_vs_vol \
    --data-dir ./StooqData --scorer all \
    --iv-anchor --iv-source gauss314 \
    --start 2018-01-02 --end 2023-07-28

# Forward vs IV anchor (DoltHub weekly, 2019-2026) — Phase 3 (kills the signal)
# Requires: nix develop && (cd .iv-cache && dolt clone post-no-preference/options
#                           && cd options && dolt sql -r parquet \
#                              -q 'SELECT date, act_symbol, iv_current, hv_current
#                                  FROM volatility_history' \
#                              > ../volatility_history.parquet)
uv run python -m relational.research.diagnostic_dislocation_vs_vol \
    --data-dir ./StooqData --scorer all \
    --iv-anchor --iv-source dolthub \
    --start 2018-06-01 --end 2026-04-30

# Brainstorm scorers on IV diagnostic — Phase 4
uv run python -m relational.research.diagnostic_dislocation_vs_vol \
    --data-dir ./StooqData --scorer brainstorm \
    --iv-anchor --iv-source dolthub \
    --start 2018-06-01 --end 2026-04-30

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

# NN-pair — Phase 12
uv run python -m relational.research.diagnostic_nn_pairs

# Sizing overlays on long equity — Phase 6
uv run python -m relational.research.diagnostic_sizing_overlays \
    --data-dir ./StooqData

# Pair-trade overlays — Phase 7
uv run python -m relational.research.diagnostic_pair_trades \
    --data-dir ./StooqData
```

Outputs land in `Output/relational-{vol-expansion-diagnostic,short-vol-pnl,sizing-overlays,pair-trades}-*.{txt,png}`.
