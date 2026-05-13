---
tags:
  - cfr
  - meta-allocator
---

# `apps/cfr` — Deep CFR meta-allocator across the existing strategy menu

Status: **Arc fully closed (2026-05-13)** — confirmed-null on
realistic-alpha basis AND on the window-level macro-gated
composition. Phase 4d's raw +0.056 alpha vs EW collapsed to
**+0.015 net** once realistic deployment friction was applied
([`cfr-vs-dca-realistic`](../findings/cfr-vs-dca-realistic.md)).
The final-swing experiment composing Phase 4d with a
window-level VIX-above-1y-rolling-median gate
([`cfr-macro-gate-final`](../findings/cfr-macro-gate-final.md))
recovered only +0.053 net alpha and was positive in only **1/5
windows** — failed both pre-registered cuts. **Three independent
arcs converge:** raw friction collapse, Phase 4a bar-level gate
failure, and the window-level gate's memory-heavy mis-classification
of w0 (post-GFC recovery, CFR's best window, threw away +0.422
because the 1y median was inflated by GFC). **Canonical live
strategy is [`apps/dca`](dca.md)**. CFR research preserved as
documented body of work for future regime-shift re-deployment;
no further pivots planned.

Prior status (2026-05-12) — **Phase 4d PASS** on raw-Sharpe basis,
nine phase variants shipped same day. Phase 1-3 hit a +0.02
architectural ceiling on US equities (full Phase 1-3 analysis in
[`cfr-phase3`](../findings/cfr-phase3.md)). Phase 4 sweep tested
4 orthogonal axes: 4a (VIX bar-gate, FAIL), 4b (sector ETF
universe, partial-OOS, first positive alpha vs EW at +0.015), 4c
(5-day rebal, partial-OOS), and **4d (13-asset multi-asset
universe = 9 sector ETFs + TLT/IEF + GLD/DBC)** posted mean CFR
Sharpe +0.861, CFR vs naive uniform +0.101, mean alpha vs EW
+0.056, 3/5 positive alpha windows. The PASS verdict was correct
on raw-Sharpe basis but did not survive realistic friction. Full
Phase 4 analysis in [`cfr-phase4`](../findings/cfr-phase4.md).

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

## All 9 phases — full result table (2026-05-12, Modal CPU 8c)

Same algorithm + Phase 3 architecture across Phases 4a-d; only
the universe / horizon / gate changes. 5-6 windows depending on
universe history availability.

| Metric | P1 tab | P2a 28act | P2b 31act | P2b-fix | P3 deep | **P4a** VIX | **P4b** sectors | **P4c** 5d | **P4d** multi-asset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean CFR Sharpe | +0.593 | +0.573 | +0.583 | +0.600 | +0.614 | **+0.383** | +0.780 | +0.574 | **+0.861** |
| mean passive EW | +0.685 | +0.685 | +0.685 | +0.685 | +0.685 | +0.685 | +0.765 | +0.688 | +0.805 |
| mean naive uniform | +0.591 | +0.632 | +0.652 | +0.652 | +0.652 | +0.652 | +0.747 | +0.560 | +0.760 |
| **CFR vs naive** | **+0.002** | −0.059 | −0.069 | −0.052 | −0.038 | **−0.269** | +0.034 | +0.013 | **+0.101** |
| **CFR alpha vs EW** | −0.093 | −0.112 | −0.103 | −0.085 | −0.071 | **−0.302** | +0.015 | −0.114 | **+0.056** |
| Pos α windows | 1/6 | 1/6 | 2/6 | 2/6 | 2/6 | 1/6 | 3/5 | 3/6 | **3/5** |
| Verdict | partial-OOS | conf-null | conf-null | (fix) | MARGINAL | **FAIL** | partial-OOS | partial-OOS | **PASS** |

**Read across the table** (full mechanisms in
[cfr-phase1](../findings/cfr-phase1.md) /
[cfr-phase2](../findings/cfr-phase2.md) /
[cfr-phase3](../findings/cfr-phase3.md) /
[cfr-phase4](../findings/cfr-phase4.md)):

1. **Architectural variance is bounded at ±0.02 mean Sharpe.**
   Phases 1 → 3 (tabular → enriched menu → deep CFR) move CFR
   only from +0.59 to +0.61. Cesa-Bianchi-Lugosi O(√(log n)/√T)
   regret bound predicts this; the algorithm works correctly,
   the menu/universe doesn't have enough edge for it to find.
2. **Universe shifts move the needle by 5-10×.** Phase 4b sector
   ETFs gives mean CFR +0.78 (+0.17 over Phase 3) and the first
   positive alpha vs EW (+0.015). Phase 4d multi-asset gives mean
   CFR +0.86 (+0.25 over Phase 3) and clears the pre-reg cut
   with mean alpha vs EW **+0.056** and CFR vs naive **+0.101**.
3. **Bar-level VIX gating destroys Phase 3.** Phase 4a's per-bar
   VIX-above-median mask suspends 57% of bars and CFR loses
   compounding more than it saves. Window-level gating remains
   viable; bar-level isn't.
4. **5-day rebal eats alpha to friction.** Phase 4c trades 4×
   more SGD samples (better training stability — first phase
   with finite loss across all windows) for 5%/yr friction tax
   that the equity universe doesn't earn back.
5. **Phase 4d PASSES.** Multi-asset (9 sector ETFs + TLT/IEF +
   GLD/DBC) is the architectural correction the meta-allocator
   was missing — uncorrelated asset classes give large per-
   action variance, and cross-asset has documented regime-
   switching alpha that intra-equity doesn't.

**The arc-level lesson:** the meta-allocator is **prediction-
problem bound, not representation bound**. Iterating on the
algorithm hit a +0.02 ceiling. Iterating on the universe broke
through it. The deep CFR architecture earned its keep on the
multi-asset universe where the menu actually has cross-action
alpha for it to find.

## Running

```bash
# Local sanity smoke (~10s + load)
uv run python apps/cfr/scripts/smoke.py --data-dir apps/notebook/data/stooq_us_long

# Phase 1 canonical 6-window — local
uv run python apps/cfr/scripts/run_walkforward.py \
    --menu phase1 --data-dir ./StooqData --output Output/cfr-phase1.json

# All 9 phase variants on Modal CPU (each ~30-180s end-to-end)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py    # close panel
uv run python apps/cfr/scripts/modal/prep_phase2b_data.py   # 13F (~4min cold)
uv run python apps/cfr/scripts/modal/prep_phase3_data.py    # macro from FRED
uv run python apps/cfr/scripts/modal/prep_phase4b_data.py   # sector ETFs
uv run python apps/cfr/scripts/modal/prep_phase4d_data.py   # multi-asset

uvx modal run apps/cfr/scripts/modal/run_phase1.py    # tabular (Phase 1)
uvx modal run apps/cfr/scripts/modal/run_phase2a.py   # +documented-alpha modes
uvx modal run apps/cfr/scripts/modal/run_phase2b.py   # +real SEC 13F
uvx modal run apps/cfr/scripts/modal/run_phase3.py    # deep CFR
uvx modal run apps/cfr/scripts/modal/run_phase4a.py   # +VIX bar-gate
uvx modal run apps/cfr/scripts/modal/run_phase4b.py   # sector ETF universe
uvx modal run apps/cfr/scripts/modal/run_phase4c.py   # 5-day rebal
uvx modal run apps/cfr/scripts/modal/run_phase4d.py   # multi-asset (PASS!)

# Tests (53 unit tests across cfr + ss-edgar)
uv run pytest apps/cfr/tests/ packages/edgar/tests/
```

## Next — Phase 5: build `ss-cfr live` and paper trade Phase 4d

Phase 4d is the first deployable result, but needs the live
integration scaffolding. Phase 5 plan:

1. **`ss-cfr live` subcommand** with the four risk rails per the
   live-trading conventions: kill-switch file
   (`~/.cfr-killswitch`), data freshness (max bar age days),
   per-name position cap via `ss_portfolio.apply_position_cap`,
   `--live` opt-in (dry-run by default).
2. **Save Phase 4d checkpoint** — currently the regret_net is
   per-window and not persisted across the walkforward. Need a
   `cfr.deep_persist` module that serializes (state-vec stats,
   regret_net params, action menu definition) to JSON.
3. **Paper trade for 1-2 quarters.** Daily fetch the 13-asset
   panel via the broker, build state vector, run regret_net
   forward, mix actions, submit through the broker rails. No
   real capital.
4. **Evaluate vs Phase 4d backtest.** If paper-trade aligns with
   walkforward expectations (~+0.06 alpha, mid-single-digit
   variance), promote to small live capital.

Estimated 1-2 weeks for the live wrapper, then 3-6 months of
paper trading to accumulate confidence. Do NOT run real capital
on Phase 4d's +0.056 alpha vs EW — that's positive but too
small to survive one bad regime (w2 2016-2019 was −0.508 in the
backtest). v2 should add a window-level macro-regime gate to
suspend deployment in "everything works passive" eras.

Other Phase 5 candidates worth considering after live integration:

- **Phase 4d extension to longer history.** Use synthetic /
  index-equivalent data for sector/bond/commodity exposures
  pre-2005. More walkforward windows would tighten the +0.056
  alpha estimate and verify the w2 outlier risk.
- **Universe scale-up.** Add international ETFs (EFA, EEM), TIP,
  more commodity slices (USO, SLV). 25-30 cross-asset names
  with the same 28-action menu.
