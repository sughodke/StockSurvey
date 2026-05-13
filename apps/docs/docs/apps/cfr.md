---
tags:
  - cfr
  - meta-allocator
---

# `apps/cfr` — Deep CFR meta-allocator across the existing strategy menu

Status: **Phase 2 confirmed-null (2026-05-12)** — three Modal walk-
forwards landed same day. Phase 1 (16-action universe-agnostic menu)
ties naive uniform; Phase 2a (28 actions, +4 documented-alpha modes)
and Phase 2b (31 actions, +real SEC 13F-HR consensus mode from new
`packages/edgar`) both failed the menu-enrichment cut. The Cover
universal-portfolio diagnosis from Phase 1 is now confirmed at the
menu axis: **the binding constraint is tabular regret-table sample
density, not menu content**. Full per-window analysis of all three
phases in [`cfr-phase2`](../findings/cfr-phase2.md). Notable nuance:
Phase 2b's late window (2020-2023, full 13F coverage) posts CFR
alpha +0.277 vs Phase 1 — the 13F signal IS real where data exists,
but tabular-CFR can't extract it cleanly. Phase 3 (deep CFR with
`regret_net(state, action_emb) → R` MLP + learned multi-modal
encoder) is the architectural correction.

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

## Phase 1 / 2a / 2b result (2026-05-12, Modal CPU 8c)

Three walk-forwards on the canonical `stooq_us_long` (312
tickers, 2000-2025, 5y train / 3y val / 3y step). Same algorithm,
infoset, friction, windowing — only the action menu changes.

| Metric | Phase 1 (16 act) | Phase 2a (28 act) | Phase 2b (31 act) |
|---|---:|---:|---:|
| mean CFR Sharpe | **+0.593** | +0.573 | +0.583 |
| mean passive EW | +0.685 | +0.685 | +0.685 |
| mean trailing-best | −0.016 | +0.044 | +0.064 |
| mean naive uniform | +0.591 | +0.632 | **+0.652** |
| **CFR vs trailing-best** | **+0.609** | +0.529 | +0.520 |
| **CFR vs naive uniform** | **+0.002** | −0.059 | **−0.069** |
| **CFR alpha vs EW** | −0.093 | −0.112 | −0.103 |
| Verdict | partial-OOS | confirmed-null | confirmed-null |

**Read across the row** (full mechanism in
[cfr-phase2](../findings/cfr-phase2.md)):

1. **CFR Sharpe is essentially flat** across phases — adding
   documented-alpha modes (2a) or real 13F data (2b) doesn't lift
   CFR's mean Sharpe.
2. **Naive uniform mix Sharpe rises monotonically** with each
   enrichment (+0.591 → +0.632 → +0.652) — the 1/N benefits from
   each new diversifying action regardless of alpha content.
3. **CFR's lift over naive uniform turns negative** (+0.002 → −0.069):
   the richer menus help the baseline more than the algorithm.
4. **Per-window subtlety in Phase 2b:** window 5 (val 2020-2023,
   the only window with full 13F coverage in train + val) posts
   CFR alpha **+0.006 vs Phase 1's −0.271 — a +0.277 lift from
   the 13F mode** in the regime where it has data. The 13F signal
   IS real where coverage exists; the binding constraint that
   washes it out at the mean is tabular-CFR's sample density, not
   menu content.

**Architectural read across all three phases:** the algorithm
works correctly — regret matching converges to the Cover universal-
portfolio uniform-mix limit when no infoset has clearly positive
cumulative regret, which is the right behavior. Adding more actions
to a tabular table where most cells already lack signal **makes
naive uniform stronger** (more diversification) **and CFR weaker**
(more noise dimensions in the regret table). The Cesa-Bianchi &
Lugosi O(√(log n)/√T) bound predicts this. **Phase 3 must move from
tabular CFR to deep CFR** (`regret_net(state_vec, action_emb) → R`
MLP that shares statistical strength across actions) over a
learned multi-modal encoder — not a richer table.

The Phase 1 per-window detail and the deeper Phase 1 analysis
([cfr-phase1](../findings/cfr-phase1.md)) explain how the algorithm
behaves window-by-window. Phase 2's per-window deltas are in
[cfr-phase2](../findings/cfr-phase2.md).

## Running

```bash
# Local sanity smoke (~10s + load)
uv run python apps/cfr/scripts/smoke.py --data-dir apps/notebook/data/stooq_us_long

# Phase 1 canonical 6-window — local
uv run python apps/cfr/scripts/run_walkforward.py \
    --menu phase1 --data-dir ./StooqData --output Output/cfr-phase1.json

# Phase 1 on Modal CPU (recommended; 23s total wall after image cache)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py
uvx modal run apps/cfr/scripts/modal/run_phase1.py

# Phase 2a — Phase 1 + 4 documented-alpha modes (no extra data)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py     # if not already
uvx modal run apps/cfr/scripts/modal/run_phase2a.py

# Phase 2b — Phase 2a + real SEC 13F-HR consensus mode
uv run python apps/cfr/scripts/modal/prep_phase1_data.py     # if not already
uv run python apps/cfr/scripts/modal/prep_phase2b_data.py    # ~4 min cold cache
uvx modal run apps/cfr/scripts/modal/run_phase2b.py

# Tests (33 unit tests, cfr + ss-edgar)
uv run pytest apps/cfr/tests/ packages/edgar/tests/
```

## Next — Phase 3 (deep CFR with learned encoder)

Phase 2 demonstrated that tabular menu enrichment is the wrong
lever — Phase 3's pre-registered cut is tightened accordingly:

> **PASS** — deep CFR mean Sharpe ≥ Phase 1 CFR + 0.15 (i.e.,
> ≥ +0.74 absolute) AND CFR > naive uniform mix on Phase 2b menu
> by ≥ +0.10. The +0.15 floor (vs +0.10 in Phase 2) reflects that
> Phase 3 is a more ambitious architectural change.

Concrete deltas vs Phase 2:

1. **Replace `(infoset, action) → cumulative regret` table with a
   `regret_net(state_vec, action_emb) → R` MLP.** Tinygrad,
   ~50K-200K params. Shares statistical strength across (state,
   action) pairs that have similar structure — solving the
   sample-density problem Phase 2 hit.
2. **Replace 9-cell discrete infoset with a learned encoder** over
   the multi-modal state vector: per-ticker price CWT + macro
   panel (FRED) + cross-sectional dispersion + 13F-overlap
   indicator + portfolio-state. The discrete cuts in Phase 1-2
   throw away continuous regime information.
3. **Keep the closed-form counterfactual regret signal** (it's
   the load-bearing reason CFR is tractable here) but train via
   SGD over the deep regret-net rather than accumulating into a
   sparse table.
4. **Fix the Phase 2b cash-equivalent-no-dedup issue** — era-
   dependent menu (drop `top13f` pre-2013) or explicit menu-time
   mask in `ActionMenu` for actions with no data.

Estimated: 2-4 weeks of focused work for the encoder + regret-net
+ training-loop wiring, then one Modal walk-forward for the
verdict. The Phase 1/2 results are the baseline against which
Phase 3 will be measured.
