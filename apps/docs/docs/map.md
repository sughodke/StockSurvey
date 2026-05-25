# Map

A visual index of every research arc: what was tried, what it
returned, and how each result gated the next experiment. Nodes are
experiments (or, where the detail is one click away, *arcs*);
edges are the **verdict → next-experiment** decisions the
[leaderboard](leaderboard.md) protocol mandates. Edge labels are
the *reason* the next experiment followed — read them and you have
the research narrative without reading a single findings page.

This page is the cross-cutting view; per-experiment numbers live in
the [Leaderboard](leaderboard.md), prose in
[Findings](findings/index.md). When you add a leaderboard row, add
its node here — but think first about whether it deserves a node or
collapses into an existing arc.

## How to read it

Verdict labels (the
[leaderboard vocabulary](leaderboard.md#verdict-labels)) are encoded
by node colour, so an arc's *shape* is legible at a glance — a long
grey chain is a dead lever; a green node is a validated/shipped
result; an amber node is a live frontier or marginal; a red node is
a real signal killed in OOS or by deployability; a blue node is a
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

## The frontier — what survives after deflation

The single most important picture in the repo: the
[cross-arc Deflated-Sharpe ladder](leaderboard.md#cross-arc-deflated-sharpe-ranking).
Every deployable arc was re-run to dump its OOS net return stream and
scored on the **deflated-Sharpe t-stat** (Sharpe corrected for fat
tails, sample length, and number of configs tried). The meta-finding:
**raw mean-of-window Sharpe is not apples-to-apples** — once selection
bias is priced in, only one arc clears t > 2.

```mermaid
flowchart TD
    dca["DCA — multi-asset EW basket<br/>deflated t +2.07 · DSR 0.98 · n_trials=4<br/>★ only arc clearing significance"]:::cOOS
    rel["relational analog cross_ticker<br/>t +0.74 · n_trials=16 · mega-cap-specific"]:::pOOS
    vol["vol v3 regime-gated<br/>t +0.13 · headline Sharpe +1.15 collapsed<br/>under 12-trial deflation"]:::pOOS
    fac["factor indicator-baseline 20d<br/>t −1.37 · confirmed-OOS on IC,<br/>negative once ~50-config search priced in"]:::rOOS
    cs["cross-sectional factor zoo<br/>12-1 momentum, low-vol BAB,<br/>shape-kNN reversal, 5d skip-1<br/>all DSR-sub-significant"]:::cnull
    gate["gate v0 drawdown overlay<br/>t −1.73 · excess-stream Sharpe negative<br/>despite +0.07 mean-of-window α"]:::rOOS
    pair["pairs v0<br/>t −0.18 · confirmed-null per pre-reg"]:::cnull

    dsr{{"Cross-arc DSR ladder<br/>= the apples-to-apples meta-finding"}}:::diag
    dsr --> dca
    dsr --> rel
    dsr --> vol
    dsr --> fac
    dsr --> cs
    dsr --> gate
    dsr --> pair

    rule["Operational rule:<br/>headline Sharpe is a hypothesis;<br/>deflated-t is evidence"]:::diag
    dsr -.-> rule

    click dsr "../findings/deflated-sharpe-leaderboard.md"
    click dca "../findings/cfr-vs-dca-realistic.md"

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## Ensemble discovery — can any combination beat DCA?

The natural follow-up to the DSR ladder: if no *single* arc clears
deflation by a wide margin, can a **combination** of the positive-t
arcs honestly produce a higher deflated-t than DCA-alone? Searching
all 91 candidate ensembles (C(7,2) + C(7,3) + C(7,4)) across the
positive-DSR arc set under a single weight rule (inverse-variance,
equal, tangency — all tested) and apples-to-apples
DCA-on-same-overlap reference: **no**. The headline "+5.11 lift over
DCA" for `relational + vol-v3-dolthub` collapses under two
falsifications: (1) vol-v3-dolthub-alone scores **+5.55** on its own
33-obs window — every ensemble containing it strictly *dilutes* the
deflated-t; (2) DCA scores only +1.53 on the 30-obs overlap because
that window (2023-08→2026-03) contains zero crises, vs DCA's
full-sample +1.93 across 20.7 years and 4 regimes. The "lift" is
sample-window selection.

```mermaid
flowchart TD
    dsr["DSR ladder<br/>only DCA clears t > 2"]:::diag
    ask{{"Can a combination beat DCA?<br/>91 candidate ensembles searched"}}:::diag
    dsr --> ask

    topp["Top pair: relational + vol-v3-dolthub<br/>n=30 · ens t +5.11 · DCA-on-overlap +1.53"]:::pOOS
    topt["Top triple: DCA + relational + vol-v3-dolthub<br/>n=30 · ens t +4.95"]:::pOOS
    nullp["Every DCA-free pair (no vol-v3-dolthub):<br/>≤ DCA-on-overlap"]:::cnull
    drel["DCA + relational (ρ +0.79)<br/>ens t +1.42 vs DCA-alone +1.41 ≈ zero lift"]:::cnull

    ask --> topp
    ask --> topt
    ask --> nullp
    ask --> drel

    f1["Falsification 1:<br/>vol-v3-dolthub-alone t +5.55 on same window<br/>→ adding cash books DILUTES the t-stat"]:::diag
    f2["Falsification 2:<br/>overlap window is calm-bull short-vol regime<br/>→ DCA-on-overlap +1.53 vs full-sample +1.93<br/>= window selection, not ensemble lift"]:::diag
    f3["Falsification 3:<br/>short-vol live infra unbuilt<br/>→ ensemble is unreachable even if signal real"]:::diag

    topp -.-> f1
    topp -.-> f2
    topt -.-> f1
    topt -.-> f2

    verdict["Verdict: deploy DCA-alone<br/>0 of 91 ensembles produce a defensible<br/>improvement once selection bias priced in"]:::rOOS
    f1 --> verdict
    f2 --> verdict
    f3 --> verdict

    nextexp["Next-experiment hypothesis:<br/>extend vol-v3-dolthub through next vol crisis<br/>as a standalone — NOT more ensemble search"]:::pend
    verdict --> nextexp

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef pend fill:#eceff1,stroke:#90a4ae,color:#37474f,stroke-dasharray:4 3
```

The mechanism behind the negative result is geometric: combining a
high-t arc (vol-v3-dolthub) with a lower-t arc (anything else)
weights the variance contribution of the lower-t arc into the
ensemble denominator without proportionally lifting the mean. For
the ensemble t to exceed the best component's t, the components need
either (a) negative correlation strong enough to compress combined
variance below either component's own, or (b) the lower-t arc must
itself clear the high bar. Neither holds here — pairwise ρ on the
overlap windows ranges −0.04 to +0.31, and no second arc has the
sample-window-conditional t to compete with vol-v3-dolthub.

The operational rule extracted: **ensembling is not a free-lunch
escape from the deflation wall**. A k-arc ensemble carries
Σ component trials + (k−1) trials of its own, and even before that
penalty the best single component's t is the ceiling unless
correlations are strongly negative. The repo's positive-t arcs are
all weakly-positively-correlated risk-on books, so the ceiling is
binding.

## Arc map — how the arcs hook up

The top-level structure: how each arc spawned, what it pivoted into,
and which cross-app diagnostic re-shaped the question. Solid edges =
an arc spawned or pivoted into another; dashed edges = a cross-app
diagnostic feeding insight forward.

```mermaid
flowchart TD
    rel["Relational arc<br/>12-phase, mostly null<br/>rule: fingerprint for selection, not hedging"]:::cnull
    ppp["Prediction-problem pivot<br/>rule: regime filter &gt; richer predictor"]:::diag
    rel -->|"no relational ckpt clears passive EW →<br/>pivot the prediction problem itself"| ppp

    ppp -->|"v0 #1: drawdown"| gate["Gate arc"]:::pOOS
    ppp -->|"v0 #2: pair-spread"| pairs["Pairs arc"]:::cnull
    ppp -->|"v0 #3: IV-vs-realized"| vol["Vol arc"]:::cOOS

    factor["Factor arc<br/>representation, objective, horizon, REINFORCE"]:::pOOS
    replay["Replay arc<br/>SSL backbone (shipped)"]:::cOOS
    replay -->|"backbone npz → SSL input path"| factor

    orc{{"Oracle survey (cross-app)<br/>headroom is real, predictors miss it"}}:::diag
    factor -.->|"+0.11 headroom"| orc
    vol -.->|"+2.86 headroom"| orc
    gate -.->|"+0.32 headroom"| orc
    pairs -.->|"+1.79 headroom"| orc

    orc -->|"can a value function learn it?"| critic["Critic Φ arc<br/>v0 / v0.1 / v0.2 — all null"]:::cnull
    critic -.->|"predictor-quality, not architecture,<br/>is the binding constraint"| facfront["Factor live frontier<br/>target-side REINFORCE β=8 ★"]:::pOOS
    factor --> facfront

    cfr["CFR meta-allocator arc<br/>phases 1→4"]:::cOOS
    cfr -->|"raw alpha collapses under realistic friction"| dca["DCA — canonical live<br/>only DSR-significant arc"]:::cOOS

    cwt["CWT recursive-compression arc<br/>Kalman vs GRU vs return-coupled GRU"]:::cnull
    cwt -.->|"reconstructible dim ≫ predictable,<br/>recurrence linearity not binding"| factor

    macro["Macro regime diagnostic<br/>5/6 features predict pivot outcomes"]:::diag
    macro -.->|"VIX-pct + slope hypothesis"| gateA2["Gate A1/A2 follow-ups<br/>both confirmed-null"]:::cnull
    gate --> gateA2

    dsr[/"Cross-arc Deflated-Sharpe ladder<br/>= the meta-finding the arcs feed"/]:::diag
    dca --> dsr
    vol --> dsr
    factor --> dsr
    gate --> dsr
    pairs --> dsr
    rel --> dsr

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef pend fill:#eceff1,stroke:#90a4ae,color:#37474f,stroke-dasharray:4 3
    classDef base fill:#fff,stroke:#90a4ae,color:#37474f
```

## Vol surface arc — the deployable-edge case study

The codebase's one big in-sample-validated signal, three OOS
extensions, and two re-frames that demonstrate **deployability is a
separate gate from cost**: an edge can be real and still die because
you cannot quote the instrument.

```mermaid
flowchart TD
    v0["v0 multivariate prediction<br/>inconclusive · 5/5 pos, α just below"]:::pOOS
    v1["v1 per-rebal aggregator + costs<br/>confirmed-OOS · α Sh +5.86, 25× shuffle"]:::cOOS
    v2a["v2#1 dollar-PnL (3 sizings)<br/>confirmed-OOS · $-vega +4.60"]:::cOOS
    v2b["v2#2 OI restriction<br/>reversed-OOS · α −0.48 at top-200 OI"]:::rOOS
    v2c["v2#3 DoltHub OOS to 2026-04<br/>confirmed-OOS · 11/11 pos quarters"]:::cOOS
    v3["v3 regime-gated liquid (126d VIX)<br/>partial-OOS · fired-α +2.01, MARGINAL"]:::pOOS
    syn["arc synthesis<br/>partial-OOS · v3 is deployment recipe"]:::pOOS
    bvrp["Small-cap illiquid-VRP re-frame<br/>reversed-OOS · 92.5% un-quotable on free data"]:::rOOS
    b1a["B1 Phase A — borrow-stress conditioning<br/>in-sample hi-tercile Sh +3.80"]:::pOOS
    b1b["B1 Phase B — OOS confirmation<br/>reversed-OOS · 50× data killed the lift"]:::rOOS
    dsrv["DSR re-rank: vol v3 → t +0.13<br/>(12 trials priced in)"]:::diag

    v0 -->|"net of costs?"| v1
    v1 -->|"$ deployable?"| v2a
    v1 -->|"liquidity restrict?"| v2b
    v1 -->|"fresh OOS data?"| v2c
    v2b -->|"gate the regime"| v3
    v3 --> syn
    syn -.->|"recover discarded raw edge<br/>at small capacity"| bvrp
    bvrp -.->|"deployability gate:<br/>quote-availability ≠ cost"| rule1["Rule: quote-availability is<br/>upstream of transaction cost"]:::diag
    syn -.->|"borrow as second-order conditioner"| b1a
    b1a -->|"pre-reg OOS test"| b1b
    b1b -.->|"in-sample mirage<br/>caught by pre-reg"| rule2["Rule: in-sample number<br/>is a hypothesis, not evidence"]:::diag
    syn --> dsrv

    click bvrp "../findings/vol-borrow-illiquid-vrp-falsified.md"
    click b1b "../findings/vol-borrow-liquid-universe.md"
    click dsrv "../findings/deflated-sharpe-leaderboard.md"

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## Factor arc — three sub-arcs collapsed

Three years of factor work compress into three intertwined questions:
*what features?* (representation), *what loss?* (objective), *what
horizon?* (the late-arc finding that horizon was the binding lever
all along). The endogenous-horizon REINFORCE thread is the live
frontier; every other lever is grey.

```mermaid
flowchart TD
    base["deterministic-indicator val-IC<br/>baseline · the bar"]:::base

    %% Representation + objective (collapsed)
    repr["Representation search (collapsed)<br/>SSL backbone · aux-head · loss-pivot ·<br/>sizing-input v0/v1 · (C,L) spectral/MiniRocket"]:::cnull

    %% Horizon
    horiz["Short-horizon × (C,L) sweep<br/>partial-OOS @ 5d (+0.0212 IC, 6/6)<br/>representation null, horizon was the lever"]:::pOOS
    skip1["5d skip-1 microstructure control<br/>partial-OOS · ≈46% was bid-ask bounce"]:::pOOS
    skip1d["5d skip-1 long-only book<br/>DSR ≈ 0 · fat-tail mirage (1 block +93%)"]:::cnull

    %% Endogenous-horizon REINFORCE chain
    eh["Endogenous-horizon mixture v0<br/>partial-OOS · Δ-fix +0.048"]:::pOOS
    rescues["4 rescues all null<br/>(entropy / VIX gate / bilevel / aligned grid)"]:::cnull
    orc["Hindsight oracle<br/>diagnostic · +0.112 ceiling"]:::diag
    rei["Target-side REINFORCE β=8<br/>partial-OOS · Δ-fix +0.095 ★"]:::pOOS
    reib["β ∈ {16, 32} resolved<br/>partial-OOS · benchmark-artifact-bound"]:::pOOS

    %% CWT recursive-compression (separate arc but feeds factor)
    cwtarc["CWT recursive-compression arc<br/>Kalman ≈ GRU on recon;<br/>return-coupled GRU rank-IC: confirmed-null"]:::cnull

    base --> repr
    base --> horiz
    horiz -->|"5d is the horizon"| skip1
    skip1 -->|"deploy as a book?"| skip1d
    base --> eh
    eh -->|"rescue π collapse"| rescues
    rescues --> orc
    orc -->|"first +lift"| rei
    rei --> reib
    repr -.->|"feature space isn't the lever"| cwtarc

    click horiz "../findings/factor-shorthorizon-representation.md"
    click cwtarc "../findings/cwt-recursive-compression.md"
    click orc "../findings/factor-endogenous-horizon-mixture.md"

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef base fill:#fff,stroke:#90a4ae,color:#37474f
```

## Cross-sectional factor zoo

The pre-registered single-hypothesis arms that test "does the
canonical literature anomaly survive in this repo's OOS protocol?"
Headline: **the deflated-Sharpe ladder collapses them all**, even
the best (low-vol BAB) is sub-significant.

```mermaid
flowchart LR
    mom["12-1 cross-sectional momentum<br/>Jegadeesh-Titman · monthly L/S<br/>confirmed-null · DSR t +0.40"]:::cnull
    bab["Low-volatility / BAB<br/>−trailing-252d vol · long low / short high<br/>confirmed-null · DSR t +0.64 (best cs-arm)"]:::cnull
    shape["Shape-kNN 1mo reversal (lie v3/v4)<br/>per-ticker shape features · long/short<br/>confirmed-null portfolio · wide-universe FAIL"]:::cnull
    skip["5d skip-1 indicator long/short<br/>confirmed-null · DSR t −3.07"]:::cnull
    note["All four are pre-registered single-hypothesis arms;<br/>none clear deflated-t = 2"]:::diag

    mom --> note
    bab --> note
    shape --> note
    skip --> note

    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## Gate + Pairs arcs (prediction-problem-pivot spawn)

Two of the three orthogonal v0 tests from the prediction-problem
pivot. Both have real-but-predictor-bound signal — the oracle
diagnostics revealed large unrealized headroom, and the v1 / A1 / A2
follow-ups falsified that simple lifts close that gap.

```mermaid
flowchart TD
    gv0["gate drawdown v0<br/>partial-OOS · r +0.26, α +0.07<br/>(DSR t −1.73 — overlay is excess-negative)"]:::pOOS
    gorc["gate hindsight oracle<br/>diagnostic · +0.32 headroom"]:::diag
    gv0 -->|"how much on the table?"| gorc
    ga1["A1 — nonlinear predictor + dispersion<br/>confirmed-null"]:::cnull
    ga2["A2 — continuous VIX-percentile meta-gate<br/>confirmed-null"]:::cnull
    gorc -->|"swap OLS for MLP / GBM?"| ga1
    gorc -->|"continuous regime conditioner?"| ga2
    cad["Cadence oracle Stage-0 kill<br/>confirmed-null · +0.004 over daily-EW"]:::cnull
    gv0 -.->|"5th reframing of state-conditional selection"| cad

    pv0["pairs classical v0<br/>confirmed-null per pre-reg"]:::cnull
    peg["pairs EG-passing-rate gate<br/>confirmed-null at 3 thresholds"]:::cnull
    porc["pairs hindsight oracle<br/>diagnostic · +1.79 per-pair headroom"]:::diag
    pv1["pairs v1 — 7-feature LR predictor<br/>confirmed-null · 5.4% capture"]:::cnull
    pv0 -->|"audit: regime gate?"| peg
    peg -->|"ceiling?"| porc
    porc -->|"simple predictor close it?"| pv1

    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## Critic Φ arc (collapsed)

Three arms, all confirmed-null. The diagnosis — *predictor-quality
is the binding constraint, not architecture* — fed forward into the
factor REINFORCE frontier; the arm-level detail is one click away.

```mermaid
flowchart LR
    crit["Critic Φ(state, action) arc<br/>v0 (window-level) · v0.1 (pair-level rich)<br/>· v0.2 (policy vs −Φ)<br/>all confirmed-null"]:::cnull
    ins{{"Diagnosis: predictor-quality binds at all stages;<br/>not architecture"}}:::diag
    crit --> ins
    click crit "../findings/critic-phi-quality-v0.md"
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
```

## CFR meta-allocator arc → DCA → live frontier

A four-phase escalation that reached a deployable PASS, then died on
realistic-friction re-eval — and in dying, established DCA as the
canonical live strategy and (via the DSR ladder) the only arc in the
repo clearing significance.

```mermaid
flowchart TD
    cfr["CFR meta-allocator arc<br/>Phase 1 (tabular) → 2 (menu) → 3 (Deep CFR)<br/>→ 4 (multi-asset, +0.056 raw alpha)"]:::pOOS
    vd["CFR vs DCA realistic friction<br/>reversed-OOS · net α +0.015, DCA wins"]:::rOOS
    extras["macro-gate + Phase 4d sensitivity<br/>confirmed-null + diagnostic close-out"]:::cnull
    dca["DCA — canonical live strategy<br/>confirmed-OOS · Sh +0.673, 20y"]:::cOOS
    dcareg["Regime-scaled DCA overlays<br/>3 arms (passive / vol-target / DD-gate)<br/>confirmed-null · regime signal not monetizable"]:::cnull
    dsr["DSR rank #1<br/>t +2.07 · only significant arc"]:::cOOS

    cfr -->|"realistic friction"| vd
    vd --> extras
    vd -->|"DCA wins worst-window + ops simplicity"| dca
    dca -->|"can a gate scale exposure?"| dcareg
    dca --> dsr

    click cfr "../findings/cfr-phase4.md"
    click vd "../findings/cfr-vs-dca-realistic.md"

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
```

## Relational + replay + infra (parked / supporting)

The relational 12-phase arc (mostly null, parked behind the
prediction-problem pivot), the replay backbone arc (shipped
compression results that feed factor), and the diagnostics + shared
packages that gate the rest of the workspace.

```mermaid
flowchart TD
    rus["relational universe-shift<br/>diagnostic · mega-cap-specific edge"]:::diag
    rarc["relational 12-phase arc (collapsed)<br/>DWT-L1 · polar-Morlet · synthesis<br/>rule: embedding for selection, not hedging"]:::cnull
    rus --> rarc
    relan["relational analog cross_ticker<br/>DSR rank #2 · t +0.74 (mega-cap-specific)"]:::pOOS
    rarc --> relan

    rdec["replay decoder options"]:::diag
    rdwt["replay 2D-DWT keep-LL<br/>confirmed-OOS · 4× input reduction"]:::cOOS
    rk["replay length-axis K sufficiency<br/>confirmed-OOS · K=96 over-provisioned"]:::cOOS
    rdec --> rdwt
    rdec --> rk

    rb["regime baselines<br/>diagnostic · Optuna instability"]:::diag
    rlr["log-returns CWT input<br/>confirmed-null · degrades Sharpe"]:::cnull
    rb --> rlr

    mac["Macro regime diagnostic<br/>5/6 features predict pivots<br/>→ ss_macro package"]:::diag
    pew["passive EW benchmark<br/>reversed-OOS · relational rows α≤0"]:::rOOS
    dew["delisting-aware EW falsified<br/>confirmed-null · ffill is a no-op"]:::cnull

    pkgs[/"Package extractions:<br/>ss_iv (vol+relational shared),<br/>ss_macro (FRED+regime stack)"/]:::base
    mac --> pkgs

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef rOOS fill:#c62828,stroke:#b71c1c,color:#fff
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef base fill:#fff,stroke:#90a4ae,color:#37474f
```

## Meta-narrative — what the map says taken together

```mermaid
flowchart LR
    a["Cross-sectional return-forecasting<br/>(relational + factor representation)<br/>→ all null"]:::cnull
    b["Prediction-problem pivot<br/>drawdown / pair-spread / IV<br/>→ partial-OOS with regime conditioning"]:::pOOS
    c["Oracle survey<br/>headroom is real,<br/>predictors don't capture it"]:::diag
    d["Critic + REINFORCE<br/>architecture isn't the lever,<br/>predictor-quality is"]:::cnull
    e["DSR ladder<br/>even survivors are sub-significant<br/>under selection-bias deflation"]:::diag
    g["Ensemble discovery<br/>0 of 91 combos beat DCA-on-overlap<br/>after honest n_trials + window control"]:::cnull
    f["Next lever:<br/>paid data / capacity-constrained edges<br/>(see research-strategy memory)"]:::pend

    a --> b --> c --> d --> e --> g --> f

    classDef cOOS fill:#2e7d32,stroke:#1b5e20,color:#fff
    classDef pOOS fill:#f9a825,stroke:#f57f17,color:#000
    classDef cnull fill:#9e9e9e,stroke:#616161,color:#fff
    classDef diag fill:#1565c0,stroke:#0d47a1,color:#fff
    classDef pend fill:#eceff1,stroke:#90a4ae,color:#37474f,stroke-dasharray:4 3
```

The narrative the arcs jointly tell: **free-data, cross-sectional,
single-asset-class anomaly hunting hits a deflation wall**. Of the
six stream-bearing strategy arcs on the DSR ladder, DCA is the only
one above t = 2 — and DCA is not a *prediction* edge, it is a
risk-premium-capture buy-and-hold. The published-anomaly arms
(12-1 momentum, low-vol BAB) plus the repo's own predictors all
fail to clear significance once trial counts are priced in. The
forward bet, made explicit in the [research-strategy
memory](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md):
**pursue structurally-persistent edges — capacity-constrained, novel
data, microstructure** — rather than novelty-to-Claude on the same
free OHLC.

## Maintenance

When a leaderboard row lands, decide *before* adding a node:

1. Is this a new arc, or a new arm of an existing arc? Most rows are
   the latter — add them inside the relevant sub-diagram, don't
   create a top-level node.
2. If the arc has grown past ~10 nodes, **collapse** the older arms
   into a single descriptive node (the way "Critic Φ arc",
   "Representation search", and "CFR phases 1→4" are collapsed here)
   and link the finding page from the collapsed node. The map is a
   curated cartography, not a changelog — don't let it grow
   monotonically.
3. The DSR ladder section at the top is the meta-finding view. When
   a new stream-bearing arc lands, add it there *and* in its arc
   diagram. Meta-evaluations and non-portfolio diagnostics do not
   appear on the ladder by construction.
4. Edge labels are the *reason* the next experiment followed —
   re-read them after editing to make sure the narrative still
   reads top-to-bottom.

A node with no outgoing edge is either a live frontier or a terminal
close — the [TODO](TODO/index.md) for that arc says which.
