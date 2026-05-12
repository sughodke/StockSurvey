---
tags:
  - cfr
  - meta-allocator
---

# `apps/cfr` — Deep CFR meta-allocator across the existing strategy menu

Status: **Phase 1 partial-OOS (2026-05-12)** — tabular CFR cleared
the pre-registered Phase 1 PASS cut against trailing-best-greedy
(+0.609 Sharpe in 6/6 windows) but ties naive uniform mix (+0.002)
and undershoots passive EW (alpha −0.093). See
[`cfr-phase1`](../findings/cfr-phase1.md) for full analysis. Phase
2 (13F imitation pretrain + deep CFR) is the architectural
correction.

The architectural premise — *predictions with regime-conditional
deployment performance need a regime filter, not a richer
predictor* — comes from the
[prediction-problem-pivot arc](../findings/prediction-problem-pivot-arc.md)
and is refined by the
[macro-regime diagnostic](../findings/macro-regime-diagnostic.md):
*macro is a real but graduated regime signal; use as a continuous
deployment scaler at the meta-level, not as direct predictor
inputs*. `apps/cfr` is the architectural answer — a learned
meta-allocator that decides which existing scorer to listen to (and
how loud) in which regime.

See the [`apps/cfr` TODO](../TODO/apps-cfr.md) for the full phased
falsifiable-experiment ladder, pre-registered cuts, and design
rationale.

## Why CFR

Three rare attributes trading shares with multiplayer no-limit
poker:

1. **Imperfect information.** We see history; we don't see counter-
   parties' inventories, hidden orders, or future fundamentals.
2. **Multi-agent.** Prices are the interaction of many strategic
   players. There's no single "environment" to plan against.
3. **Post-hoc counterfactual observability.** *This is the unusual
   one.* In poker, computing counterfactual regret requires either
   game-tree traversal (CFR variants) or sampling (MCCFR). In
   trading at price-taker scale, **counterfactual regret is
   directly computable from price history** — our trade doesn't
   move the market, so we can replay every action's realized return
   over any historical block without re-simulating the future.

That third property is what makes CFR exceptionally tractable for
this problem. The regret signal that's the central bottleneck for
poker CFR is free for us.

| Algorithm family | Match for trading |
|---|---|
| AlphaZero / MuZero (MCTS + NN) | Wrong — markets are imperfect-info; planning forks future states by action choice, but at price-taker scale our actions don't move markets, so MCTS branching collapses. |
| Standard supervised return-prediction | What we already have; ceilings at +0.005 to +0.012 IC at our universe / horizon. |
| Standard RL (PPO, DQN) | Reward is dominated by exogenous noise; credit assignment over long holding periods is hard. |
| **Deep CFR** | Native fit. Closed-form regret. Mixed strategies emerge naturally in low-edge regimes (correct behavior). |
| ReBeL / Player of Games | AlphaZero-shape architecture but with CFR-style updates. Where the field is going; the right reference if compute allows. |

## Phase 0 architecture (shipped)

The Phase 1 algorithm is **tabular CFR** over discrete `(infoset, action)`
buckets. Both axes are small enough to enumerate; Deep CFR's neural
function approximation kicks in at Phase 2+ once the menu grows
beyond ~20 actions or the state grows beyond ~10 buckets.

### Modules

| Module | Purpose |
|---|---|
| `cfr.menu` | `ActionMenu` enumerator + universe-agnostic modes (EW, top-K momentum / reversal / low-vol / high-vol, cash). `action_to_weights(a, t) → ℝ^N` is precomputed once over the panel for speed. |
| `cfr.state` | `InfosetBuilder` — buckets trailing market vol × cross-sectional dispersion into discrete regime labels. 3×3 = 9 cells by default. **Bucket cutoffs frozen on train**; val uses the same cutoffs. |
| `cfr.regret` | `compute_block_regrets(block_logret, action_weights, played)` — closed-form per-action realized log return over a forward block, minus the played action's. `regret_matching(R) → π` turns positive cumulative regret into a probability distribution. |
| `cfr.tabular` | `TabularCFR` — cumulative regret + cumulative strategy table per infoset. Exposes `current_policy(i)` (regret matching) and `average_policy(i)` (time-averaged, the no-regret limit). |
| `cfr.baselines` | `PassiveEW`, `TrailingBestGreedy`, `NaiveUniform` — match the [passive-EW benchmark](../findings/passive-ew-benchmark.md) convention. |
| `cfr.walkforward` | `CFRWalkForward` orchestrates train-CFR / eval-CFR / baselines across rolling windows. Reports per-window alpha vs passive + CFR-vs-trailing-best lift. |
| `cfr.persist` | JSON checkpoint I/O. Same shape as the other apps' checkpoints. |
| `cfr.cli` | `ss-cfr smoke` and `ss-cfr walkforward` subcommands. |

### Action menu (default)

`default_phase1_menu(top_k=20)` produces 16 actions:

```
cash
ew @ {0.5, 1.0, 2.0}
mom @ {0.5, 1.0, 2.0}    21d momentum, top-K equal-weight
rev @ {0.5, 1.0, 2.0}    5d reversal,  top-K equal-weight
lowv @ {0.5, 1.0, 2.0}   21d low-vol,  top-K equal-weight
highv @ {0.5, 1.0, 2.0}  21d high-vol, top-K equal-weight
```

Gross=0 across all modes dedups to a single canonical `cash`
action. The menu is universe-agnostic — no saved checkpoints
required — so Phase 1 validates the *algorithm* independently of
any one scorer's quality. Phase 2 adds 13F-imitation-pretrained
modes that wrap existing relational / factor checkpoints.

### Infoset

3 × 3 = 9 regime cells + 1 warmup cell:

- **vol axis** — trailing 21d stdev of EW universe log return,
  bucketed into low/mid/high
- **dispersion axis** — trailing 21d cross-sectional stdev of
  per-ticker log returns, bucketed into low/mid/high

Both cutoffs are train-period quantiles, frozen for val. This is
the macro-regime-diagnostic's *VIX × dispersion* axis encoded
universe-internally — sidesteps the train/val distribution-shift
problem that killed the
[v1a macro-direct-features arm](../findings/macro-regime-diagnostic.md).

### Training loop

```
for window in walkforward.windows:
    train_prices, val_prices = window
    builder.fit(train_prices)
    table = TabularCFR(n_infosets, n_actions)
    for t in train_rebal_indices:
        infoset = builder.transform(train_prices[:t+1])
        pi = table.current_policy(infoset)
        played = sample(pi)
        block_logret = log(prices[t+rebal]/prices[t])      # (N,)
        regrets = compute_block_regrets(
            block_logret, action_weights[t], played)         # closed-form
        table.update(infoset, regrets, pi)
    # Eval: at each val rebal, use table.average_policy()
    # for the mixed target portfolio.
```

The price-taker assumption means we update regret for *every*
action at every visit, not just the sampled one — so the regret
estimator has zero sampling variance from the played-action axis.

## Phase 1 result (2026-05-12, Modal CPU 8c, 23s wall)

6 walk-forward windows on the canonical `stooq_us_long` (312
tickers, 2000-2025, 5y train / 3y val / 3y step, 1 training pass):

| win | val_dates | CFR Sh | Passive EW | Trailing-best | Naive uniform | α vs EW | CFR vs trailing |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2005-01 → 2008-02 | +0.279 | +0.529 | −0.727 | +0.321 | −0.251 | **+1.006** |
| 1 | 2008-02 → 2011-03 | **+0.596** | +0.331 | −0.312 | +0.416 | **+0.265** | **+0.908** |
| 2 | 2011-03 → 2014-04 | +0.817 | +0.928 | +0.174 | +0.711 | −0.111 | +0.643 |
| 3 | 2014-04 → 2017-05 | +0.727 | +0.916 | +0.256 | +0.637 | −0.189 | +0.471 |
| 4 | 2017-05 → 2020-07 | +0.440 | +0.440 | +0.060 | +0.288 | +0.000 | +0.380 |
| 5 | 2020-07 → 2023-08 | +0.697 | +0.968 | +0.452 | +1.172 | −0.271 | +0.245 |
| **mean** | | **+0.593** | **+0.685** | **−0.016** | **+0.591** | **−0.093** | **+0.609** |

**Headline reads** (full mechanism in
[cfr-phase1](../findings/cfr-phase1.md)):

1. **PASS vs trailing-best-greedy** by +0.609 Sharpe (threshold
   was +0.10), in 6/6 windows. The algorithm reliably refuses to
   follow the regime-mismatched "switch into whatever just won"
   heuristic.
2. **Tied with naive uniform mix** within noise (+0.593 vs
   +0.591). CFR doesn't add information over 1/16 uniform mixing
   on this menu × universe.
3. **−0.093 alpha vs passive EW**, within ±0.10 noise band, 1/6
   positive windows (window 1 = GFC, the same outlier window
   carrying alpha across the pivot arc).

**Architectural read:** the algorithm works correctly — regret
matching converges to the Cover universal-portfolio uniform-mix
limit when no infoset has clearly positive cumulative regret,
which is the right behavior. The binding constraint is the
**action menu**: universe-agnostic top-K factor exposures
(momentum / reversal / vol-rank) on a 312-name universe are too
close to alpha-zero to reward concentration. Phase 2's natural
move is to add alpha-positive modes (13F-imitation-pretrained
scorers, sector-restricted variants) so regret matching has
something to discover beyond uniform.

## Running

```bash
# Local sanity smoke (~10s + load)
uv run python apps/cfr/scripts/smoke.py --data-dir apps/notebook/data/stooq_us_long

# Phase 1 canonical 6-window on full StooqData/ (local, slow load)
uv run python apps/cfr/scripts/run_walkforward.py \
    --data-dir ./StooqData \
    --output Output/cfr-phase1.json

# Phase 1 on Modal CPU (recommended; 23s total wall after image cache)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py
uvx modal run apps/cfr/scripts/modal/run_phase1.py

# Tests (23 unit tests)
uv run pytest apps/cfr/tests/
```

## Next — Phase 2 (13F imitation pretrain + deep CFR)

The Phase 1 result reframes the Phase 2 priorities. The TODO had
the order as Phase 0 scaffold → Phase 1 baseline → Phase 2 = add
imitation. The "tied with naive uniform" caveat from Phase 1 says
**add alpha-positive modes first**, *then* worry about deep CFR.
The pre-registered Phase 2 cut becomes: **CFR with 13F-imitation
modes > Phase 1 CFR by ≥ +0.10 mean Sharpe**.

Concrete next steps:

1. **Build the 13F loader** (`packages/edgar` or
   `apps/cfr/data/`). SEC EDGAR XML, per-quarter / per-fund /
   per-ticker aggregation. ~1 week build estimate.
2. **Add `mode_long_13f_consensus` (top-N fund consensus by
   trailing Sharpe) to the action menu.** Wraps the 13F
   aggregation as an action that gives target weights at any
   bar. Should be alpha-positive in some regimes by construction
   (it's an imitation of presumed-skilled traders).
3. **Re-run Phase 1 eval with the expanded menu.** If CFR's
   regret matching now concentrates on `mode_long_13f_consensus`
   in some infosets (and we see CFR > naive uniform by
   ≥ +0.10 Sharpe in ≥ 4/6 windows), the algorithm has been
   validated against a more meaningful baseline.
4. **Then** deep CFR + multi-modal encoder per the [original
   Phase 3 design](../TODO/apps-cfr.md).

The current Phase 1 numbers are a real baseline against which
every Phase 2+ variant will be measured.
