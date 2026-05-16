# Map

A visual index of every research arc: what was tried, what it
returned, and how each result gated the next experiment. Nodes are
experiments; edges are the **verdict → next-experiment** decisions
the [leaderboard](leaderboard.md) protocol mandates. Edge labels are
the *reason* the next experiment followed — read them and you have
the research narrative without reading a single findings page.

This page is the cross-cutting view; per-experiment numbers live in
the [Leaderboard](leaderboard.md), prose in
[Findings](findings/index.md). When you add a leaderboard row, add
its node here.

## How to read it

Verdict labels (the
[leaderboard vocabulary](leaderboard.md#verdict-labels)) are encoded
by node colour, so an arc's *shape* is legible at a glance — a long
grey chain is a dead lever; a green node is a validated/shipped
result; an amber node is a live frontier; a blue node is a
diagnostic that re-framed the question.

```mermaid
flowchart LR
    cOOS["confirmed-OOS · validated / shipped"]:::cOOS
    pOOS["partial-OOS · regime-conditional / MARGINAL"]:::pOOS
    rOOS["reversed-OOS · val killed the train signal"]:::rOOS
    cnull["confirmed-null · lever tried, does not move"]:::cnull
    diag["diagnostic · re-frames the question"]:::diag
    pend["pending · run not yet landed"]:::pend
    base["baseline / infra · the bar, not a bet"]:::base
    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef pend fill:#eceff1,stroke:#90a4ae,color:#37474f,stroke-dasharray:4 3
    classDef base fill:#fff,stroke:#90a4ae,color:#37474f
```

## Arc map — how the arcs hook up

The top-level structure. Solid edges = an arc spawned or pivoted
into another; dashed edges = a cross-app diagnostic feeding an
insight forward.

```mermaid
flowchart TD
    rel["Relational arc<br/>12-phase, mostly null<br/>rule: fingerprint for selection, not hedging"]:::cnull
    ppp["Prediction-problem-pivot arc<br/>rule: regime filter &gt; richer predictor"]:::diag
    rel -->|"no relational ckpt clears its passive EW baseline →<br/>pivot the prediction problem itself"| ppp

    ppp -->|"orthogonal v0 #1: drawdown"| gate["Gate arc"]:::pOOS
    ppp -->|"orthogonal v0 #2: pair-spread"| pairs["Pairs arc"]:::cnull
    ppp -->|"orthogonal v0 #3: IV-vs-realized"| vol["Vol arc"]:::cOOS

    factor["Factor arc<br/>representation + objective search"]:::pOOS
    replay["Replay arc<br/>SSL backbone"]:::cOOS
    replay -->|"backbone npz feeds the factor SSL input path"| factor

    orc{{"Oracle survey<br/>cross-app diagnostic:<br/>headroom is real, predictors miss it"}}:::diag
    factor -.->|"+0.11 headroom"| orc
    vol -.->|"+2.86 headroom"| orc
    gate -.->|"+0.32 headroom"| orc
    pairs -.->|"+1.79 headroom"| orc

    orc -->|"can a value function learn it?"| critic["Critic Φ arc<br/>v0 / v0.1 / v0.2 all null"]:::cnull
    critic -.->|"predictor-quality is the binding constraint,<br/>not architecture"| frei["Factor endogenous-horizon<br/>target-side REINFORCE ★"]:::pOOS
    factor --> frei

    cfr["CFR meta-allocator arc<br/>phase 1→4"]:::cOOS
    cfr -->|"raw alpha collapses under realistic friction"| dca["DCA<br/>canonical live strategy"]:::cOOS

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef pend fill:#eceff1,stroke:#90a4ae,color:#37474f,stroke-dasharray:4 3
    classDef base fill:#fff,stroke:#90a4ae,color:#37474f
```

## Factor — endogenous-horizon arc

The deepest single arc: seven experiments chasing a +0.112 oracle
ceiling on state-conditional rebal cadence. Four rescues failed
identically (all damaged the w3 canary); the fifth — target-side
REINFORCE on the Sharpe-residual — is the first positive lever and
the live frontier.

```mermaid
flowchart TD
    v0["v0 mixture<br/>partial-OOS · Δ-fix +0.048"]:::pOOS
    ent["entropy-weight sweep<br/>confirmed-null on rescue"]:::cnull
    rg["regime-gated horizon (VIX)<br/>confirmed-null + score-head-spec reframe"]:::cnull
    orc["hindsight oracle<br/>diagnostic · +0.112 ceiling"]:::diag
    bil["bilevel objective<br/>confirmed-null"]:::cnull
    hal["horizon-aligned grid<br/>confirmed-null"]:::cnull
    rei["target-side REINFORCE β=8<br/>partial-OOS · Δ-fix +0.095 ★"]:::pOOS
    nxt["higher-β β∈{16,32}<br/>pending"]:::pend
    wog["w0 regime gate<br/>pending"]:::pend
    osr["output-side per-horizon heads<br/>pending"]:::pend

    v0 -->|"is π-collapse the cause?"| ent
    v0 -->|"hand-engineered VIX state?"| rg
    v0 -->|"what's the ceiling?"| orc
    orc -->|"+0.112 real → richer π signal"| bil
    bil -->|"per-day return too noisy"| hal
    hal -->|"feature space isn't it →<br/>rank-IC vs Sharpe mismatch"| rei
    rei -->|"first +lift, MARGINAL ·<br/>does the curve clear PASS?"| nxt
    rei -->|"w0 is the only neg window,<br/>fails under every config"| wog
    rei -->|"compound with score-head spec"| osr

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef pend fill:#eceff1,stroke:#90a4ae,color:#37474f,stroke-dasharray:4 3
```

## Factor — representation + objective search (earlier)

The pre-endogenous-horizon work: can a learned backbone or an
alternative objective beat the deterministic-indicator rank-IC bar?
Consistently no.

```mermaid
flowchart TD
    base["deterministic-indicator val-IC<br/>baseline · the bar"]:::base
    ssl["supervised-cnn / SSL walkforward<br/>confirmed-null (doesn't clear bar)"]:::cnull
    f32["f32 forward-return precision<br/>diagnostic · 6× IC regression bug"]:::diag
    aux["multi-task aux head<br/>confirmed-null (+0.012, doesn't clear bar)"]:::cnull
    auxw["aux-weight sweep<br/>reversed-OOS (train fits, val anti-corr)"]:::rOOS
    lp["loss-pivot (Sharpe/IR vs rank-IC)<br/>confirmed-null · rank-IC best"]:::cnull
    s0["sizing-input v0 (MSE-on-alpha)<br/>confirmed-null · calibrates, no info"]:::cnull
    s1["sizing-input v1 (signal-quality → VIX gate)<br/>confirmed-null · too lagged"]:::cnull
    rio["rank-IC long-only mismatch<br/>diagnostic · signed signal, half-executed"]:::diag

    base --> ssl
    base --> aux
    f32 -.->|"fixed silent IC regression"| base
    aux --> auxw
    base -->|"is rank-IC even the right loss?"| lp
    lp -->|"rank-IC's spread-thin = risk control"| s0
    s0 --> s1
    lp -.->|"signed-signal insight"| rio

    classDef base fill:#fff,stroke:#90a4ae,color:#37474f
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
```

## Vol surface arc

The codebase's one big OOS-validated signal — and the one that also
shows how a real edge dies on deployability.

```mermaid
flowchart TD
    v0["v0 multivariate prediction<br/>inconclusive · 5/5 pos, α just below"]:::pOOS
    v1["v1 per-rebal aggregator + costs<br/>confirmed-OOS · α Sh +5.86, 25× shuffle"]:::cOOS
    v2a["v2#1 dollar-PnL (3 sizings)<br/>confirmed-OOS · $-vega +4.60"]:::cOOS
    v2b["v2#2 OI restriction<br/>reversed-OOS · α −0.48 at top-200 OI"]:::rOOS
    v2c["v2#3 DoltHub OOS to 2026-04<br/>confirmed-OOS · 11/11 pos quarters"]:::cOOS
    v3["v3 regime-gated liquid (126d VIX)<br/>partial-OOS · fired-α +2.01, MARGINAL"]:::pOOS
    syn["arc synthesis<br/>partial-OOS · v3 is deployment recipe"]:::pOOS
    bvrp["small-cap illiquid-VRP re-frame<br/>reversed-OOS · 92.5% un-quotable on free data"]:::rOOS

    v0 -->|"is the signal real net of costs?"| v1
    v1 -->|"convert to deployable $-PnL"| v2a
    v1 -->|"restrict to liquid (OI) names"| v2b
    v1 -->|"does it hold on fresh OOS data?"| v2c
    v2b -->|"liquidity kills it → gate the regime"| v3
    v3 --> syn
    syn -.->|"recover the discarded raw edge<br/>at small capacity"| bvrp

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
```

## Gate + Pairs arcs (prediction-problem-pivot spawn)

Two of the three orthogonal v0 tests the prediction-problem-pivot
arc launched. Both have real-but-predictor-bound signal — the oracle
diagnostics revealed large unrealized headroom.

```mermaid
flowchart TD
    gv0["gate drawdown v0<br/>partial-OOS · r +0.26, α +0.07"]:::pOOS
    gorc["gate hindsight oracle<br/>diagnostic · +0.32 headroom, predictor-bound"]:::diag
    gv0 -->|"how much is on the table?"| gorc

    pv0["pairs classical v0<br/>confirmed-null per pre-reg"]:::cnull
    peg["pairs EG-passing-rate gate<br/>confirmed-null · falsified at 3 thresholds"]:::cnull
    porc["pairs hindsight oracle<br/>diagnostic · +1.79 per-pair headroom"]:::diag
    pv1["pairs v1 7-feature LR predictor<br/>confirmed-null · 5.4% capture"]:::cnull
    pv0 -->|"audit: is it a regime gate?"| peg
    peg -->|"w0 in working band yet worst Sharpe →<br/>what's the ceiling?"| porc
    porc -->|"per-pair selection has +1.79 →<br/>can a predictor capture it?"| pv1

    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## Critic Φ arc

The cross-app oracle survey's follow-up: can a learned value
function Φ(state, action) → Sharpe capture the headroom the oracles
revealed? Three arms, all confirmed-null — the cheap version of the
idea is comprehensively falsified, but the diagnosis (predictor-
quality is binding) fed forward into the factor REINFORCE frontier.

```mermaid
flowchart TD
    c0["v0 cross-app window-level Φ<br/>confirmed-null · oracle-detection illusion"]:::cnull
    c01["v0.1 pair-level Φ (rich features)<br/>confirmed-null · ties LR baseline"]:::cnull
    c02["v0.2 policy training vs −Φ<br/>confirmed-null · collapses to rank-by-Φ"]:::cnull
    ins{{"diagnosis: predictor-quality binds at<br/>all stages; not architecture"}}:::diag

    c0 -->|"window-level too coarse →<br/>finer granularity?"| c01
    c01 -->|"Φ ties LR → does a policy help?"| c02
    c02 --> ins

    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## CFR meta-allocator arc → DCA

A four-phase escalation that reached a deployable PASS, then died on
realistic-friction re-eval — and in dying, established DCA as the
canonical live strategy.

```mermaid
flowchart TD
    p1["Phase 1 tabular CFR<br/>partial · clears trailing-best, ties uniform"]:::pOOS
    p2["Phase 2 menu enrichment (13F)<br/>confirmed-null · sample-density bound"]:::cnull
    p3["Phase 3 Deep CFR + continuous state<br/>partial-OOS · +0.02 MARGINAL"]:::pOOS
    p4["Phase 4 13-asset multi-asset<br/>confirmed-OOS · α vs EW +0.056"]:::cOOS
    vd["CFR vs DCA realistic friction<br/>reversed-OOS · net α +0.015, DCA wins"]:::rOOS
    mg["+ window-level VIX gate close-out<br/>confirmed-null · bot fully dead"]:::cnull
    sf["Phase 4d sensitivity follow-up<br/>diagnostic · α never clears +0.10"]:::diag
    dca["DCA — canonical live strategy<br/>confirmed-OOS · Sh +0.673, 20y"]:::cOOS

    p1 -->|"menu the binding constraint?"| p2
    p2 -->|"architecture the binding constraint?"| p3
    p3 -->|"universe the binding constraint?"| p4
    p4 -->|"survives realistic friction?"| vd
    vd -->|"can a macro gate rescue it?"| mg
    vd --> sf
    vd -->|"DCA wins worst-window + ops simplicity"| dca

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## Relational arc + infra (parked / supporting)

The relational 12-phase arc (mostly null, parked behind the
prediction-problem pivot), the replay backbone arc (shipped
compression results that feed factor), and the regime/cross-app
diagnostics that gate the others.

```mermaid
flowchart TD
    rus["relational universe-shift<br/>diagnostic · mega-cap-specific edge"]:::diag
    rdwt["relational DWT-L1 compression<br/>confirmed-null OOS (4 scorers)"]:::cnull
    rmor["relational polar-Morlet bundle<br/>confirmed-null OOS gate"]:::cnull
    rsyn["relational arc synthesis<br/>rule: embedding for selection, not hedging"]:::diag
    rus --> rdwt
    rus --> rmor
    rdwt --> rsyn
    rmor --> rsyn

    rdec["replay decoder options<br/>diagnostic"]:::diag
    rdc["replay 2D-DWT keep-LL<br/>confirmed-OOS · 4× input reduction"]:::cOOS
    rlk["replay length-axis (K) sufficiency<br/>confirmed-OOS · K=96 over-provisioned"]:::cOOS
    rdec --> rdc
    rdec --> rlk

    rb["regime baselines<br/>diagnostic · Optuna instability"]:::diag
    rlr["log-returns CWT input<br/>confirmed-null · degrades Sharpe"]:::cnull
    rb --> rlr

    mac["macro regime diagnostic<br/>diagnostic · 5/6 features predict pivots"]:::diag
    pew["passive EW benchmark<br/>reversed-OOS · relational rows α≤0"]:::rOOS
    dew["delisting-aware EW falsified<br/>confirmed-null · ffill is a no-op"]:::cnull

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## Maintenance

When a leaderboard row lands, add its node to the matching arc
diagram with the right verdict class and an edge from the
experiment that gated it (label = the reason it followed). A node
with no outgoing edge is a live frontier or a terminal close — the
[TODO](TODO/index.md) for that arc says which. If an arc grows past
~10 nodes, split it into a sub-section rather than letting one
flowchart sprawl.
