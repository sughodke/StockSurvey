---
tags:
  - cfr
  - meta-allocator
---

# `apps/cfr` — Deep CFR meta-allocator across the existing strategy menu

Status: **Phase 0 shipped (2026-05-12)** — tabular CFR scaffold,
universe-agnostic action menu, walk-forward driver, baselines,
tests, smoke. Phase 1 walk-forward against the canonical 6-window
spec is the next concrete eval.

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

## Phase 0 smoke result (2026-05-12)

3-window walk-forward on the curated `stooq_us_long` subset (312
tickers, 2010-2025, 5y train / 3y val / 3y step):

| Window | val_dates | CFR Sh | Passive EW | Trailing-best | Naive uniform | α vs EW | CFR vs trailing |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2015-01 → 2018-02 | +0.508 | +0.904 | +0.204 | +0.562 | −0.396 | **+0.304** |
| 1 | 2018-02 → 2021-03 | +0.982 | +0.738 | +0.227 | +0.819 | +0.244 | **+0.755** |
| 2 | 2021-03 → 2024-04 | −0.042 | +0.435 | −0.244 | +0.250 | −0.476 | **+0.203** |
| **mean** | | **+0.483** | **+0.692** | **+0.062** | **+0.544** | **−0.209** | **+0.420** |

**Reads:**

1. **CFR beats trailing-best-greedy by +0.42 Sharpe across 3 / 3
   windows.** The algorithm earns its keep against the naive
   "pick the recent winner" ensemble.
2. **CFR Sharpe ≈ naive uniform mix.** At 3 windows the CFR
   policy isn't materially better than mixing across all actions
   uniformly. Two interpretations: (a) the menu's modes are
   close enough to alpha-zero that mixing is near-Pareto;
   (b) the algorithm needs more training data than 3 windows ×
   1260 train bars to converge to a useful policy. Phase 1's
   canonical 6-window 25y span (full StooqData/) will resolve
   this.
3. **No baseline clears passive EW.** Consistent with the
   [passive-EW benchmark](../findings/passive-ew-benchmark.md):
   passive EW on this universe has a +0.69 mean Sharpe that no
   action-menu mix has surpassed. Beating passive EW
   structurally requires either (a) modes that are themselves
   alpha-positive (not the case for momentum / reversal /
   vol-ranked here), or (b) regime-conditional deployment that
   sits in cash during EW's worst stretches. Phase 1 will tell us
   whether the regime-conditional argument has legs.

Smoke is a smoke — no leaderboard row, no verdict. Validates
end-to-end mechanics and that the algorithm beats its weakest
baseline.

## Running

```bash
# Phase 0 sanity (fast, ~10s + load)
uv run python apps/cfr/scripts/smoke.py --data-dir apps/notebook/data/stooq_us_long

# Phase 0 multi-window on curated subset (~10s + load)
uv run python apps/cfr/scripts/run_walkforward.py \
    --data-dir apps/notebook/data/stooq_us_long \
    --start 2010-01-01 --end 2025-12-11 \
    --output Output/cfr-smoke-multiwin.json

# Phase 1 canonical 6-window on full StooqData/ (slow load, fast compute)
uv run python apps/cfr/scripts/run_walkforward.py \
    --data-dir ./StooqData \
    --output Output/cfr-phase1.json

# Tests
uv run pytest apps/cfr/tests/
```

## Next

Per the [`apps/cfr` TODO](../TODO/apps-cfr.md) Phase 0 → Phase 1
chain: run the canonical 6-window walk-forward on the full
`stooq_us_long` universe (25 years of data, 6 windows at 5y/3y/3y).
The pre-registered Phase 1 cuts:

- **PASS** — CFR ≥ trailing-best-greedy + 0.10 mean **AND**
  CFR > trailing in ≥ 4/6 windows → proceed to Phase 2 (13F
  imitation pretrain).
- **MARGINAL** — within ±0.10 → diagnose where the policies
  diverge per window.
- **FAIL** — CFR < trailing-best-greedy − 0.10 **OR** CFR >
  trailing in < 3/6 windows → confirmed-null, park the arc.

The smoke run already clears the PASS lift (+0.42 vs needed +0.10)
on a 3-window subset, which is encouraging but not conclusive —
the canonical 6-window eval is the load-bearing test.
