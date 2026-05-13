---
tags:
  - cfr
  - meta-allocator
---

# `apps/cfr` — Deep CFR meta-allocator across the existing strategy menu

Status: **Phase 3 MARGINAL (2026-05-12)** — five phase variants
shipped same day. Phase 1 (tabular, 16 actions): partial-OOS, ties
naive uniform mix. Phase 2a (28 actions, +4 documented-alpha modes)
and Phase 2b (31 actions, +real SEC 13F-HR consensus from new
`packages/edgar`): both confirmed-null on tabular menu enrichment.
Phase 2b-fixed (action-availability mask bugfix): +0.017 lift,
still confirmed-null. **Phase 3 (Deep CFR with tinygrad regret_net
over a 10-feature continuous state vector incl. 4 macro features
from FRED): MARGINAL** — mean CFR Sharpe **+0.614** (best of all
phases), CFR alpha vs EW **−0.071** (32% improvement over Phase 1's
−0.093), and **window 2 flips −0.111 → +0.127** (the cleanest deep-
architecture win). Cumulative Phase 1 → Phase 3 lift is **+0.021**
— far short of the +0.15 PASS floor. **The binding constraint is no
longer architecture or menu**; it's signal-availability in the
menu × universe × horizon combination. Phase 4 should change the
prediction problem (different universe / horizon), not the meta-
allocator's representation. Full per-phase analysis in
[`cfr-phase3`](../findings/cfr-phase3.md).

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

## Phase 1 / 2a / 2b / 2b-fixed / 3 result (2026-05-12, Modal CPU 8c)

Five walk-forwards on the canonical `stooq_us_long` (312
tickers, 2000-2025, 5y train / 3y val / 3y step). Same algorithm
shape, friction, windowing — only the action menu and
representation change.

| Metric | Phase 1 (tabular, 16 act) | Phase 2a (28 act) | Phase 2b (31 act) | 2b-fixed (avail mask) | **Phase 3 (deep)** |
|---|---:|---:|---:|---:|---:|
| mean CFR Sharpe | +0.593 | +0.573 | +0.583 | +0.600 | **+0.614** |
| mean passive EW | +0.685 | +0.685 | +0.685 | +0.685 | +0.685 |
| mean trailing-best | −0.016 | +0.044 | +0.064 | +0.064 | +0.064 |
| mean naive uniform | +0.591 | +0.632 | +0.652 | +0.652 | +0.652 |
| **CFR vs trailing-best** | **+0.609** | +0.529 | +0.520 | +0.536 | +0.550 |
| **CFR vs naive uniform** | **+0.002** | −0.059 | −0.069 | −0.052 | **−0.038** |
| **CFR alpha vs EW** | −0.093 | −0.112 | −0.103 | −0.085 | **−0.071** |
| Pos α windows | 1/6 | 1/6 | 2/6 | 2/6 | **2/6** |
| Verdict | partial-OOS | confirmed-null | confirmed-null | (cleanup) | **MARGINAL** |

**Read across the row** (full mechanisms in
[cfr-phase1](../findings/cfr-phase1.md) /
[cfr-phase2](../findings/cfr-phase2.md) /
[cfr-phase3](../findings/cfr-phase3.md)):

1. **Tabular menu enrichment hurts** (Phase 1 → 2a → 2b): mean
   CFR drifts down +0.593 → +0.573 → +0.583 while naive uniform
   rises +0.591 → +0.632 → +0.652. Cesa-Bianchi-Lugosi O(√(log
   n)/√T) regret bound predicts this — more actions, same T,
   sparser regret-table estimator.
2. **Phase 2b-fixed availability mask** removes phantom-cash
   contamination from pre-2013 bars (where `Top13FConsensusMode`
   returned all-zero weights without dedup), giving +0.017 lift.
3. **Phase 3 deep CFR** lifts the algorithm by another **+0.014**
   via parameter sharing across the 31-action × 9-state-region
   space. Window 2 (2011-2014, post-GFC recovery) flips
   −0.111 → +0.127 alpha — direct evidence that the continuous
   regime encoder finds something the discrete tabular grid missed.
4. **Cumulative Phase 1 → 3 lift is +0.021** mean Sharpe, after
   testing 4 distinct architectural levers. Diminishing returns
   are clean: every architectural improvement wins a small chunk
   of variance reduction; none manufactures alpha.

**Architectural read across all five phases:** the algorithm
works correctly. Across menu sizes 16 → 28 → 31 and tabular
→ deep, the regret matching policy converges to the Cover
universal-portfolio uniform-mix limit when no action has clearly
positive cumulative regret. Phase 3 narrowed the gap to passive
EW from −0.093 → −0.071 (32% improvement) but still doesn't
clear the operational floor. **The binding constraint is no
longer architecture or menu** — it's signal availability in the
menu × universe × horizon combination. Phase 4 should change
the prediction problem (different universe, horizon, or composite-
regime action class), not the meta-allocator's representation.

## Running

```bash
# Local sanity smoke (~10s + load)
uv run python apps/cfr/scripts/smoke.py --data-dir apps/notebook/data/stooq_us_long

# Phase 1 canonical 6-window — local
uv run python apps/cfr/scripts/run_walkforward.py \
    --menu phase1 --data-dir ./StooqData --output Output/cfr-phase1.json

# Phase 1 on Modal CPU (~23s after image cache)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py
uvx modal run apps/cfr/scripts/modal/run_phase1.py

# Phase 2a — Phase 1 + 4 documented-alpha modes
uvx modal run apps/cfr/scripts/modal/run_phase2a.py

# Phase 2b — Phase 2a + real SEC 13F-HR consensus mode
uv run python apps/cfr/scripts/modal/prep_phase2b_data.py    # ~4 min cold cache
uvx modal run apps/cfr/scripts/modal/run_phase2b.py

# Phase 3 — Deep CFR with tinygrad regret_net + macro state
uv run python apps/cfr/scripts/modal/prep_phase3_data.py     # ~30s (FRED cache)
uvx modal run apps/cfr/scripts/modal/run_phase3.py           # ~80s end-to-end

# Tests (53 unit tests across cfr + ss-edgar)
uv run pytest apps/cfr/tests/ packages/edgar/tests/
```

## Next — Phase 4 candidates (all change the prediction problem, not the meta-allocator)

The architectural progression Phase 1 → 3 hit a +0.02 ceiling:
every variant moves CFR Sharpe by ~±0.02 with no transformational
change. The Cesa-Bianchi-Lugosi bound at our T=6,000 / n=31
predicts this. **Don't iterate on the meta-allocator
representation.** Four pre-registered Phase 4 candidates:

1. **Hybrid Phase 3 + macro v1b VIX gate.** Keep deep CFR but
   only deploy when VIX > 1y rolling median (per the
   [macro-regime diagnostic](../findings/macro-regime-diagnostic.md)
   v1b: pooled per-app-z-scored lift +0.215 from this gate alone
   on the pivot-arc apps). 1-line wiring change. Pre-registered
   cut: combined Sharpe ≥ Phase 3 + 0.15 mean.
2. **Different universe.** Sector ETFs (XLK, XLF, etc.) instead
   of 312 mega-caps. Sector rotation alpha is documented; no need
   to discover it from per-name picking. Pre-reg cut: ≥ +0.20
   alpha vs EW-of-sectors.
3. **Different horizon.** Daily rebal instead of 20-day. Cuts
   transaction-cost erosion that probably eats the marginal alpha
   the +0.02 architecture progression keeps finding.
4. **Composite-regime menu.** Replace deterministic factor modes
   with sector × style × cap-tilt combinations that have known
   regime-conditional alpha (e.g., "tech overweight in rising-rate
   periods", "small-cap underweight near recessions"). Even if
   each composite is alpha-positive only ~30% of the time,
   regret matching can concentrate on the active set.

None of these is a "scale up the model" answer. The architectural
ceiling at this universe + horizon is ~+0.6 mean Sharpe, ~−0.07
alpha vs passive EW. To break that ceiling, the prediction
problem itself has to change — same conclusion the prediction-
problem-pivot arc reached for the per-app predictors.
