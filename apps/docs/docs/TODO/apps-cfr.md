# `apps/cfr` — Deep CFR meta-allocator across existing scorers

**Status (2026-05-12):** Phase 0 scaffold + **Phase 1 result shipped
same day.** Tabular CFR cleared the pre-registered PASS cut against
trailing-best-greedy on the canonical 6-window stooq_us_long walk-
forward (mean +0.609 Sharpe lift, 6/6 windows positive vs +0.10
threshold) but **ties naive uniform mix within noise** (Δ +0.002)
and undershoots passive EW (alpha −0.093, only 1/6 positive).
Verdict: [`partial-OOS`](../leaderboard.md#verdict-labels) — PASS
on the pre-registered cut but the practically-useful comparisons
(vs naive uniform, vs passive) don't separate. Architectural read:
the algorithm works (Cover universal-portfolio uniform-mix limit
when no infoset has positive cumulative regret) but the action
menu's universe-agnostic top-K factor exposures are too close to
alpha-zero to reward concentration. **Phase 2 priority shift:
build alpha-positive modes (13F imitation) FIRST, then deep CFR
on top.** See [`cfr-phase1`](../findings/cfr-phase1.md) for the
load-bearing analysis.

**Verdict → next-experiment chain.** The
[prediction-problem-pivot arc](../findings/prediction-problem-pivot-arc.md)
landed three independent partial-OOS results (`gate`, `pairs`, `vol`)
at consistent magnitude (mean alpha +0.07 to +0.10, regime-conditional)
with the operational rule *"predictions with non-zero multivariate
signal but regime-conditional deployment performance need a regime
filter, not a richer predictor."* The
[macro-regime diagnostic](../findings/macro-regime-diagnostic.md) v1
arc then refined this: *"macro is a real but graduated regime signal —
use as a continuous deployment scaler at the meta-level, NOT as direct
predictor inputs."* Both rules point in the same direction:
**the next architectural move is a meta-allocator that decides which
existing scorer to deploy when**, not a richer single predictor.

`apps/cfr` is that meta-allocator, framed as Deep Counterfactual
Regret Minimization over a discrete action menu of existing strategy
modes. CFR is the right algorithmic family because trading shares
three rare attributes with multiplayer no-limit poker — imperfect
information, multi-agent, and **post-hoc counterfactual
observability** (price-takers can compute exactly what every action
would have returned at every historical bar). The first two attributes
make MCTS-style perfect-information planning the wrong family;
the third is what makes CFR *unusually* tractable here — the
counterfactual regret signal that's the central bottleneck for poker
CFR is directly computable from price-history for trading.

This is not an immediate v0. The data dependencies (13F loader, 13D
loader, action-menu plumbing) are non-trivial, the algorithm is novel
to this stack, and the failure modes are different from any prior
experiment — so the work is staged into four phases with pre-registered
cuts at each. Each phase is independently falsifiable; if any phase
fails, the next phase is skipped.

## Why CFR over the alternatives we've considered

| Algorithm family | Game class | Match for trading |
|---|---|---|
| AlphaZero / MuZero (MCTS + NN) | Perfect information | **Wrong** — markets are imperfect-info; planning forks future states by action choice, but at price-taker scale our actions don't move markets, so MCTS branching collapses |
| Standard supervised return-prediction | Regression | What we already have; ceilings at +0.005 to +0.012 IC at our universe / horizon |
| Standard RL (PPO, DQN) | Sequential decision | Reward is dominated by exogenous noise; credit assignment over a recurrent core through 60-day holding periods is a hard RL problem |
| **Deep CFR** | Imperfect-information | Native fit. Counterfactual regret signal is exactly computable from history. Mixed strategies emerge naturally in low-edge regimes (correct behavior). 30-year track record in adjacent finance literature (Cover's universal portfolios as the formal ancestor). |
| ReBeL / Player of Games | Imperfect-info + neural search | Where the field is going; AlphaZero-shape architecture but with CFR-style updates. The "right" reference if compute allows. |

## Runtime architecture

Inputs (assembled per trading day at pre-market, ~7-8 AM ET):

```
markets:           overnight bars + intraday VWAP estimates
macro:             FRED daily series (VIX, yields, gold-VIX, real-yield) +
                   weekly / monthly releases as they hit
expert_positions:  13F filings landed in last 24h (lagged 45d) +
                   13D filings landed in last 24h (lagged 10d, activist)
news_features:     placeholder for future news pipeline (not v1)
calendar:          day-of-week, FOMC week, OpEx, earnings calendar
portfolio_state:   current weights, cash, gross, net, drawdown,
                   days-since-last-rebal
```

Multi-modal encoder (shape borrowed from the AlphaStar / MuZero
architecture family — per-modality encoders fused into a recurrent
core, then auto-regressive heads):

```
per_ticker_cwt    [N, T_lookback, scales, channels] → CNN encoder      → [N, d_emb]
per_ticker_inds   [N, T_lookback, ind_features]     → CNN encoder      → [N, d_emb]
cross_sectional   stack                              → Transformer      → [N, d_ctx]
macro             [T_lookback, 9 macro features]    → MLP encoder      → [d_macro]
market_agg        [T_lookback, ew/breadth/vol/dd]   → MLP encoder      → [d_mkt]
portfolio_state   current weights / gross / net /dd → MLP encoder      → [d_port]
expert_positions  per-ticker 13F-consensus weight   → MLP encoder      → [N, d_exp]
calendar          day/month/event embeddings        → embedding        → [d_cal]
fuse              concat → linear projection                            → state vector s_t
```

Two heads (the CFR-specific machinery):

```
regret_net    R_θ(s_t) → vector [num_actions]   cumulative regret per discrete action
policy_net    π_φ(s_t) → vector [num_actions]   time-averaged strategy (no-regret limit)
```

Action selection at runtime uses **regret matching** on the regret
net's output:

```python
positive_regret = np.maximum(R_θ(s_t), 0)
if positive_regret.sum() > 0:
    π = positive_regret / positive_regret.sum()
else:
    π = np.ones(num_actions) / num_actions  # uniform when no action stands out

# Either sample or take expectation:
target_weights = sum(π[a] * action_to_weights[a] for a in actions)
```

Submit through existing broker rails (`ss_portfolio.broker`) — kill
switch, freshness check, position cap, dry-run-by-default. CFR slots
in *before* the rails; weight construction is unchanged downstream.

## Action menu design (the single most load-bearing piece)

CFR is intractable on raw continuous portfolio weights and impractical
on a fully-flat per-ticker discretization (10⁴⁰ possible portfolios).
The right framing is **action = "which existing scorer + at what gross
level"**, taking advantage of the fact that we already have 6+ trained
scorers. Sketch (numbers indicative, final menu requires sweeping):

```
strategy mode:
  mode_long_factor_indicator     factor.train_scorer_indicators top-K
  mode_long_relational_empirical relational empirical scorer top-K
  mode_long_relational_velocity  regime-velocity scorer top-K
  mode_long_short_relational     empirical scorer LS construction
  mode_pairs_active              pairs-classical pairs (when EG-passing-rate high)
  mode_vol_short                 vol-surface short-vol picks
  mode_gate_flat                 gate-suspended (cash)
  mode_ew_passive                equal-weight benchmark
  mode_cash                      all cash
  ... ~20-50 modes total

gross level:
  0.0, 0.5, 1.0, 1.5, 2.0          5 buckets

total action space: ~100-250 discrete (mode, gross) tuples
```

The continuous portfolio-construction work is delegated to the
existing scorers. CFR is the meta-layer that decides which scorer to
listen to and how loud — exactly the
[regime-filter rule](../findings/macro-regime-diagnostic.md) made
algorithmic.

**Key design questions for the action menu** (need to be locked
before phase 1):

1. How many modes to include — too few = brittle, too many = slow
   convergence and overfitting to specific historical realizations.
2. Whether to include short-only modes (currently underrepresented;
   `mode_long_short_*` covers it but a pure short bucket might help
   the regret net learn when to lean short).
3. Whether to include sector-restricted variants of each scorer
   (e.g. `mode_long_factor_tech_only`) or keep the menu universe-wide.
4. Whether `mode_cash` and `mode_ew_passive` are redundant with
   `gross=0` — they likely are for cash but EW-at-gross-1.0 is a
   distinct action (it's the passive baseline that nothing has cleared
   per the [passive-EW benchmark](../findings/passive-ew-benchmark.md)).

## End-of-day regret signal — why this works for trading

In poker, computing counterfactual regret requires either explicit
search (CFR variants traverse the game tree) or sampling
(Monte-Carlo CFR). In trading, **counterfactual regret is directly
computable from price history**:

```python
realized_return_today[a] = portfolio(a) · returns_today  # for ALL a in menu
played_action_return     = realized_return_today[σ_played]
instantaneous_regret[a]  = realized_return_today[a] - played_action_return
```

This is the bootstrapping property that makes CFR exceptionally
tractable here — the price-taker assumption (our trade doesn't move
the market) means future state is action-independent, so we can
replay every counterfactual without simulation. Append
`(s_t, a, instantaneous_regret[a])` tuples to a replay buffer for
every action — buffer grows by `num_actions` rows per day.

## Training curriculum — phased

Mirror of the AlphaStar / Cicero curriculum (supervised → imitation →
RL fine-tune), adapted for CFR:

**Phase 1 — supervised auxiliary heads.** Pretrain encoders on
auxiliary targets that aren't the regret signal:
forward 20d realized vol per ticker, forward 20d return rank, forward
20d max DD, regime classification on NBER + custom market-state
labels. Same shape as the existing replay backbone's multi-task
training. Goal: get encoders to a useful starting point before
exposing them to the noisy regret signal.

**Phase 2 — imitation pretrain.** Initialize the policy net with
behavioral cloning on 13F-consensus targets — at each quarter-end,
the consensus long portfolio of the top-N funds by trailing Sharpe
becomes the imitation target for the policy head. KL regularizer
during phase 3 keeps RL fine-tune from drifting away from this
prior. 13D event predictions as an auxiliary "did experts just buy
high-conviction?" head.

**Phase 3 — CFR fine-tune.** Walk the historical bars in
chronological order; at each bar compute regret updates for all
actions using the closed-form counterfactual; update the regret net
via MSE against cumulative regrets in the replay buffer; update the
average-policy net to track the regret-matching action distribution.

**Phase 4 (optional) — population-based search.** Train N=8-32
policies with diverse reward weightings (Sharpe vs Calmar vs hit-rate
vs return). Periodically replace worst with mutations of best.
Mitigates single-loss overfit.

## Phased falsifiable experiments

Each phase has pre-registered cuts. If a phase fails (FAIL), the
arc is closed as `confirmed-null` and `apps/cfr` is parked. If it
passes (PASS), we proceed to the next phase. If marginal, additional
diagnostics decide.

### Phase 0 — scaffold infrastructure (no eval, no leaderboard row)

Just the plumbing:

- Action menu enumerator + `action_to_weights(a, prices, scorers)` adapter
- Replay buffer (in-memory + disk-flush, like SSL pretrain)
- Regret net + policy net (tinygrad, small — ~1M params; encoder reuses replay backbone)
- `compute_counterfactual_regret(state, prices_today)` end-of-day routine
- Walk-forward driver (`apps/cfr/scripts/run_walkforward.py`)
- Smoke test: 30-ticker universe, 50 training bars, verify regret signal is non-zero
  and the policy distribution evolves

This is engineering, not experiment. No leaderboard row.

### Phase 1 — naive CFR (no pretraining, no expert imitation)

**Hypothesis.** Even without any pretraining, CFR over the existing
scorer menu should beat trailing-best-greedy ensemble (pick whichever
scorer had the highest Sharpe in the last `T_trail` days), because
CFR's regret signal naturally diffuses across multiple modes when
edge is unclear, while trailing-best concentrates on a single mode
that may be regime-mismatched.

**Test design.**

- **Universe**: `stooq_us_long` (existing canonical for action-menu
  scorers, see [operating conditions](../leaderboard.md#operating-conditions))
- **Windowing**: 6 walk-forward windows matching the
  [pairs](../findings/pairs-classical-v0.md) /
  [vol](../findings/vol-surface-v0.md) protocol — 5y train, 3y val, 3y step
- **Action menu**: minimal — top-3 long-only modes
  (factor-indicator, relational-empirical, EW passive) × 3 gross
  buckets (0, 1.0, 2.0) = 9 actions
- **Pretraining**: none. Random init.
- **Reward**: per-period log return on portfolio (not Sharpe — Sharpe
  is downstream evaluation metric)

**Baselines.**

1. Passive EW (the operational floor)
2. Best single scorer's val Sharpe per window (oracle baseline)
3. Trailing-best-greedy (pick scorer with highest Sharpe over last
   `T_trail=63` days, gross=1.0)
4. Naive uniform-mix (1/N across all modes, gross=1.0)

**Pre-registered cuts.**

- **PASS** — mean val Sharpe ≥ trailing-best-greedy + 0.10 AND
  positive-window count ≥ 4/6 → CFR has signal beyond naive ensemble;
  proceed to Phase 2.
- **MARGINAL** — mean val Sharpe within ±0.10 of trailing-best-greedy
  → diagnose where the policies diverge; decide based on
  per-window stratification whether the signal is regime-conditional
  (and worth Phase 2 with proper pretrain) or genuinely null.
- **FAIL** — mean val Sharpe < trailing-best-greedy − 0.10 OR
  positive-window count < 3/6 → confirmed-null. Either the action menu
  is wrong (doesn't span the regime variation) or CFR's no-pretrain
  bootstrap is too noisy at the 6,250-bar dataset size. Park the arc.

### Phase 2 — add 13F-consensus imitation pretrain

**Hypothesis.** Bootstrapping the policy net with 13F-consensus
imitation should lift mean val Sharpe materially over Phase 1 by
giving CFR a non-random prior to fine-tune from. Same lesson as the
AlphaStar imitation phase.

**Test design.** Same universe / windowing / action menu / baselines
as Phase 1. Add: 13F loader (data dependency below), behavioral
cloning loss on policy net using top-N-by-trailing-Sharpe consensus,
KL regularizer in CFR fine-tune.

**Pre-registered cuts.**

- **PASS** — mean val Sharpe ≥ Phase 1 mean + 0.10 (additional lift
  from imitation pretrain) AND ≥ trailing-best-greedy + 0.20 → expert
  prior helps; proceed to Phase 3.
- **MARGINAL** — Phase 2 mean within ±0.10 of Phase 1 → 13F pretrain
  doesn't help at this resolution / lag; diagnose if the imitation
  data has structure but the algorithm isn't using it.
- **FAIL** — Phase 2 mean < Phase 1 mean → 13F prior actively misleads
  (e.g. 45-day-lagged expert positions are worse than uniform). Park
  the imitation channel; decide on Phase 3 separately.

### Phase 3 — add soft-oracle aux heads + 13D events

**Hypothesis.** Adding the
[soft oracle](../findings/relational-arc-synthesis.md) on aggregate
properties (optimal sector tilt, optimal gross, optimal long-short
balance) as auxiliary supervised heads, plus 13D activist filings
as an event-prediction head, should sharpen the regret net's
state-encoding without inducing the noise-fitting failure mode of
naive hindsight-oracle on per-name picks.

**Test design.** Same universe / windowing / action menu. Add aux
heads to the regret net, weighted at 0.1 of the regret loss
(following the [aux-weight sweep](../findings/factor-multitask-aux-weight-sweep.md)
default).

**Pre-registered cuts.**

- **PASS** — mean val Sharpe ≥ Phase 2 mean + 0.05 AND no degradation
  on any per-window → soft oracle is a useful regularizer; this is
  the v1 shipper.
- **MARGINAL** — within ±0.05 → aux heads neither help nor hurt; ship
  Phase 2 as v1; consider Phase 4.
- **FAIL** — mean val Sharpe < Phase 2 mean − 0.05 → aux signals are
  actively noisy at the meta-allocator level (despite working at the
  per-app level); strip them. Ship Phase 2 as v1.

### Phase 4 (optional) — population-based search + MCTS over portfolio trajectories

**Hypothesis.** Population-based search over reward weightings
(Sharpe / Calmar / hit-rate / return) hedges against single-loss
overfit; MCTS over portfolio-state trajectories (using a learned
forward-DD predictor as the dynamics model) adds risk-aware planning
that scalar regret can't capture.

**Test design.** Independent against the Phase 2 or Phase 3 v1
shipper. Same universe / windowing.

**Pre-registered cuts.**

- **PASS** — mean val Calmar ≥ Phase 3 + 0.20 (using risk-adjusted
  metric, not Sharpe, since the population objective is risk-shape
  diversity) → population helps. Promote.
- **MARGINAL** — within ±0.20 Calmar → not worth the complexity;
  ship Phase 3.
- **FAIL** → confirmed-null on the population-search angle; Phase 3
  is the terminal architecture.

## Data dependencies (longest lead time, build first)

### 13F loader (`packages/edgar` or `apps/cfr/data/`)

EDGAR exposes 13F filings as quarterly XML (Form 13F-HR). Free, no
API key, but rate-limited (10 req/s SEC etiquette).

- **Coverage**: ~5K+ filers, US-listed equity longs ≥$100M AUM
- **Lag**: 45 days from quarter-end (filings dribble in for 45 days)
- **Aggregation**: per-quarter, per-fund, per-ticker `(shares, market_value)`
- **Derived**: top-N-by-trailing-Sharpe consensus weights (the
  imitation target)
- **Cache**: on-disk `.edgar-cache/` mirroring the `.macro-cache/` and
  `.iv-cache/` patterns
- **Tests**: golden-file test on a known fund-quarter (e.g. Berkshire
  Q3 2023, 13F-HR filed 2023-11-14) to catch upstream schema drift
- **Estimated build**: 1 week

### 13D loader (same package)

Schedule 13D filings for activist 5%+ stakes. Filed within 10 days,
much fresher signal but narrower coverage.

- **Coverage**: per-event, not periodic
- **Lag**: 10 days from event
- **Use**: per-name "expert just made a high-conviction bet" feature
  fed into the per-ticker encoder
- **Estimated build**: 3 days (after 13F infra exists)

### Action-menu adapter (`apps/cfr/menu.py`)

Translates `(mode, gross_bucket)` action tuples into target portfolio
weights at a given bar by calling the right scorer's `target_weights`
function:

```python
def action_to_weights(action: tuple[str, float],
                      prices, highs, lows,
                      checkpoints: dict[str, Checkpoint]) -> np.ndarray:
    mode, gross = action
    if mode == 'mode_long_relational_empirical':
        w = relational.inference.target_weights(
            prices, highs, lows, checkpoints['empirical'])
    elif mode == ...
    return w * gross
```

Need: canonical checkpoints for every mode in the menu. Most exist
(`Output/relational-*.json`); a few may need building
(e.g. `mode_long_factor_indicator` needs a saved walk-forward
artifact).

## Open design questions

- **State representation drift / non-stationarity.** Macro
  distributions are non-stationary across train/val
  ([macro v1a finding](../findings/macro-regime-diagnostic.md)); the
  regret net trained on pre-2008 states may misfire on post-2020
  states. Mitigations: (a) include macro features in state encoder so
  net learns regime-conditional regret; (b) z-score state features
  against trailing 5-year window rather than full-sample stats;
  (c) periodically re-evaluate and re-train.
- **Convergence rate.** CFR's O(1/√T) bound at T ≈ 6,250 bars (25
  years × 250) is ~1.3% — meaningful but not negligible. Pre-training
  shifts the starting point; the bound applies only to the regret
  net's improvement from there. The honest ceiling is whatever the
  pretrain achieves; CFR fine-tune adds incremental improvement on
  top.
- **The "no Nash" reality.** Markets aren't strategic opponents —
  there's no Nash equilibrium to converge to. The guarantee we get
  is "no-regret vs the action menu in hindsight" (the Cover
  universal-portfolio analog), which is weaker than poker CFR's
  Nash convergence. Need to be honest about this when claiming
  theoretical grounding.
- **Action menu validation.** No way to know upfront if the menu spans
  the strategy variation that markets reward. If Phase 1 fails, the
  diagnostic question is: was it the algorithm, the menu, or the
  pretraining? Need per-mode regret diagnostics in the eval (which
  modes accumulated positive vs negative regret per window) to
  distinguish.
- **Compute budget.** Each phase's walk-forward at 6 windows × 25y
  data × ~250 actions × ~250 bars/year = ~9M regret tuples per phase.
  Tinygrad on Modal T4 should handle this, but inference per bar
  has to stay <1s for a daily live deployment. Budget the encoder
  size accordingly.
- **Imitation target construction.** Which "top-N by trailing Sharpe"
  funds? Universe selection introduces survivorship bias (today's
  top funds aren't 1995's top funds). One option: re-select top-N
  per-window from data available at val_start. Mitigates bias at the
  cost of more bookkeeping.
- **Continuous-action extension.** If the discrete-menu version works,
  natural follow-up is a hybrid where the discrete head picks the
  mode and a continuous head fine-tunes weights within the mode's
  basket. Punted to Phase 5+ if Phase 3 ships.

## Reference architectures + reading list

The three core papers if anyone picks this up cold:

- **Pluribus** (Brown & Sandholm, 2019, *Science*) — multiplayer NLHE
  AI using CFR + abstractions. Core algorithm we're adapting.
- **Deep CFR** (Brown et al., 2019, ICML) — replaces tabular regret
  tables with neural networks; the version of CFR that scales to
  large state spaces.
- **ReBeL** (Brown, Bakhtin, Lerer, Gong, 2020, NeurIPS) — combines
  self-play RL with depth-limited search and CFR; the field's
  attempt to unify AlphaZero and CFR. Most directly applicable to
  trading.
- **Cicero** (Meta, 2022, *Science*) — non-zero-sum multi-agent
  Diplomacy; relevant for the persistent-state + multi-agent
  inference pieces.
- **Cover's Universal Portfolio** (Cover, 1991, *Mathematical
  Finance*) — the formal trading-as-no-regret-learning ancestor.
  Establishes the no-regret-vs-best-fixed-mix-in-hindsight
  guarantee that's the real theoretical claim we'd be making.
- **Player of Games** (DeepMind, 2021) — unified algorithm handling
  perfect + imperfect-info games using growing-tree CFR + neural
  function approximation. The cleanest "AlphaZero of imperfect-info"
  reference.

## What kicks this off

If/when this becomes top-priority:

1. ~~Build the **13F loader** first (longest lead time, useful even
   standalone for diagnostic work like "does following Berkshire
   beat EW?").~~ — *Deferred to Phase 2 prep; Phase 0 used
   universe-agnostic deterministic modes that don't require an
   expert prior.*
2. ~~Build **Phase 0 scaffold** (action menu, replay buffer, encoder
   + heads, walk-forward driver) in parallel.~~ — **Shipped
   2026-05-12.** `apps/cfr/src/cfr/{menu,state,regret,tabular,
   buffer,baselines,walkforward,persist,cli}.py` +
   `apps/cfr/tests/` (23 unit tests green) +
   `apps/cfr/scripts/{smoke,run_walkforward}.py`. Tabular instead
   of deep at Phase 1 — 9 infosets × 16 actions = 144 table
   entries, dense enough to validate on 25y of data.
3. **Next: Run Phase 1** — canonical 6-window walk-forward on full
   StooqData/ with `stooq_us_long` manifest. Command:
   `uv run python apps/cfr/scripts/run_walkforward.py
   --output Output/cfr-phase1.json`. Pre-registered cuts above
   gate the Phase 2 decision.
4. Decide on Phase 2 (13F imitation pretrain + deep CFR) based on
   Phase 1 result.

Estimated remaining to a Phase 3 verdict: 6-10 weeks of focused
work, mostly in 13F loader + encoder pretraining + walk-forward
debugging. The algorithm itself is now in place; the failure modes
are in data quality and action-menu design.

## Phase 0 scaffold — design decisions

A few choices worth noting for future contributors:

- **Tabular at Phase 1, deep at Phase 2+.** Tabular CFR is well-
  understood, easier to validate (regret matching is one np.maximum
  call), and the 144-entry table converges within the available
  ~6,000 train rebals. Deep CFR replaces the table with a regret_net
  + policy_net at Phase 2+ once 13F imitation pretraining can
  initialize the policy net non-randomly.
- **Universe-agnostic modes, no checkpoints.** Phase 0 ships with
  EW / top-K momentum / reversal / low-vol / high-vol modes that
  are pure-numpy heuristics over the price panel. Sidesteps the
  dependency on saved factor/relational checkpoints (which are
  Phase-2-mega-cap-specific per
  [relational-universe-shift](../findings/relational-universe-shift.md)).
  Phase 2 adds modes wrapping existing scorers as part of the
  imitation prior.
- **Infoset = (vol bucket × dispersion bucket), train-only fit.**
  3×3 = 9 cells encodes the macro-regime-diagnostic's two strongest
  feature axes (VIX-proxy + dispersion) universe-internally. Bucket
  cutoffs frozen on train sidesteps the distribution-shift problem
  that killed
  [macro v1a's direct-feature arm](../findings/macro-regime-diagnostic.md).
- **Closed-form counterfactual regret.** `compute_block_regrets`
  uses `log(weights @ exp(per_ticker_logret))` which is exact for
  arbitrary block sizes. Approximate first-order form (`weights @
  logret`) is also correct within ~1bp at 20-day blocks but
  diverges at longer horizons; keeping the exact form means
  rebal_days is a free hyperparameter without correctness risk.
- **Action mixing at eval, not sampling.** `CFRWalkForward._eval_cfr`
  uses the expectation of action weights under the average policy
  rather than sampling. Removes Monte Carlo variance from val
  Sharpe measurement (the policy is what we want to evaluate, not
  one realization of it).
