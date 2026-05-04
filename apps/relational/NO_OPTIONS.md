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

## Verdict on the original hypothesis

**Refuted on this universe.** The CWT scorers produce real long-only
equity alpha (empirical / farthest at Sharpe 1.13) but not enough
*cross-sectional* alpha to support market-neutral, pair-trade, or
vol-arbitrage strategies that survive transaction costs. The IV market
appears to efficiently incorporate dislocation-style information from
the same CWT bundle we have access to.

## What's shippable

- **Long-only equity, equal-weight, top-10 by empirical (idea A),
  farthest (idea C), or analog (idea B)** on Phase-2: Sharpe
  1.07-1.13, CAGR 21-22%, max DD ~32-38%, Calmar 0.56-0.65 over 13
  years (2013-2025, 10 bps commission, 20-day rebal). Baseline is
  also competitive (Sharpe 1.07).
- **Universe-wide short-vol overlay** as a separate book if vol options
  are in scope: Sharpe ~0.5, BUT max DD -83% cumulative — needs a real
  risk overlay (vol-spike suspension, drawdown stop) before any
  capital is allocated.

## What won't work without more data

- Anything market-neutral, pair-trade, or stat-arb on Phase-2's 21
  mega-caps. Universe is too small and too correlated. Wider universe
  (DoltHub's 2,276 tickers) is the unlock that wasn't tested in this
  arc.
- Any options strategy beyond "sell vol on the universe with risk
  overlay" — needs an informational edge against the IV market that
  the CWT bundle doesn't seem to provide on these names.

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

# Sizing overlays on long equity — Phase 6
uv run python -m relational.research.diagnostic_sizing_overlays \
    --data-dir ./StooqData

# Pair-trade overlays — Phase 7
uv run python -m relational.research.diagnostic_pair_trades \
    --data-dir ./StooqData
```

Outputs land in `Output/relational-{vol-expansion-diagnostic,short-vol-pnl,sizing-overlays,pair-trades}-*.{txt,png}`.
