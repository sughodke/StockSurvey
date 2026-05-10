---
tags:
  - phase-2
  - stooq_us_long
  - factor-narrow
  - factor-wide
  - diagnostic
  - hypothesis-user
---

# Passive equal-weight benchmark — every "shippable" relational row was alpha-zero or alpha-negative

**Operational rule (added 2026-05-10 to
[`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md#operational-rules-extracted-from-findings)):**
any portfolio-level "win" must clear the passive equal-weight
Sharpe on the same universe + window before being treated as
shippable. Use `alpha = model_val_sharpe − passive_val_sharpe` as
the load-bearing column in any new leaderboard row. Raw val
Sharpe is largely market beta of the chosen universe, not skill.

The 2026-05-10 EW benchmark reclassified the three previously
"shippable" relational rows. None of them clear their passive
baseline. The leaderboard's `confirmed-OOS` verdicts before
this date measured *relative to other model variants*, not
relative to passive — and that distinction turns out to be
load-bearing for the live-go-no-go decision.

## Setup

Same canonical Phase-2 train/val split used for every prior
relational row (train 2013-01-29 → 2020-12-31, val 2021-01-01 →
2025-12-11). Two passive arms per universe:

- **`buy_and_hold`** — 1/N at t=0, weights drift with prices, no
  rebalancing, no commission. The truly passive baseline.
- **`ew_rebal20_10bps`** — reset to 1/N every 20 trading bars,
  10 bps commission on L1 turnover at each rebal. Matched to the
  canonical relational checkpoint convention
  (`rebal_days=20, commission_bps=10`) so the comparison vs model
  val Sharpe is apples-to-apples on friction.

Three universes:

- **Phase-2** — 21 mega-cap fixed list (`PHASE2_TICKERS`).
- **stooq_us_long** — 312 long-history names from
  `apps/notebook/data/stooq_us_long/manifest.json`.
- **ex-Phase-2** — 296 stooq_us_long names minus the 21 Phase-2
  mega-caps.

Implementation: `apps/relational/scripts/equal_weight_benchmark.py`
— pure numpy + `ss_loaders` + `ss_portfolio.metrics`. ~5 min wall
on the local Intel Mac, no Modal, no GPU. Prices loaded once per
universe; ffill + drop-leading-NaN per window so val isn't
penalized by ticker IPOs that pre-date val but post-date train.

## Result — val Sharpe (2021-01-01 → 2025-12-11)

| Universe | passive BH | passive EW-rebal20-10bps | model val | alpha (model − BH) |
|---|---:|---:|---:|---:|
| Phase-2 (21 mega-caps) | **+1.079** | +1.066 | +1.146 (analog cross_ticker, Ricker) | **+0.067** |
| stooq_us_long (312) | +0.850 | +0.851 | +0.717 (analog Morlet) | **−0.133** |
| ex-Phase-2 (296) | +0.818 | +0.832 | +0.484 (analog cross_ticker, Ricker) | **−0.334** |
| factor-wide-ish (2162) | **+0.681** | +0.674 | (no model run yet) | n/a |

Train-side comparison (2013-01-29 → 2020-12-31) for context:

| Universe | passive BH train | model train | model alpha (in-sample) |
|---|---:|---:|---:|
| Phase-2 | +1.431 | 1.032 | **−0.399** |
| stooq_us_long | +0.878 | (not directly comparable; per-arm) | — |
| ex-Phase-2 | +0.833 | 0.615 | **−0.218** |

The Phase-2 train-side is the most damning single number on the
page: passive **+1.431** vs model **1.032** in-sample. The
analog cross_ticker model **underperformed passive by ~0.4 of
Sharpe in-sample** on the universe where it was developed. The
"+0.114 train→val Δ" celebrated as `confirmed-OOS` was movement
up the underperformance curve, not toward an alpha-positive
operating point.

## Per-universe reading

### Phase-2 — alpha within noise, but in-sample underperforms passive

Model val 1.146 vs passive BH 1.079 = **+0.067 Sharpe alpha**.
That's about 6 bps of Sharpe — well within single-split eval
noise, and would be wiped out by any cost slippage beyond the
10 bps assumed (bid/ask spreads, market impact, partial fills).
The leaderboard's own pre-test threshold ("if Phase-2 EW ≥ 1.0,
model adds ~0.15 alpha") came in at 1.07, and the model only
delivered **+0.067**, less than half the threshold's expected
lift.

The in-sample story makes the verdict stronger, not weaker.
Passive BH train Sharpe was **+1.431** while the model train
Sharpe was **1.032**. The model could not even fit the train
window above passive — meaning what the optimizer learned was a
worse strategy than holding the universe and doing nothing. The
val-side "win" (+0.067) is what's left after the universe's
mega-cap bull tailwind 2021-2025 carries the passive baseline
down from 1.43 to 1.08; the model rides the same tailwind to
1.146.

The canonical `Output/relational-analog.json` is, operationally,
a high-fee mega-cap index fund with a noisy 6-bps overlay.

### stooq_us_long Morlet — moral victory only

The [polar Morlet wide-universe finding](relational-morlet-failure.md)
celebrated a +0.17 val Sharpe lift for Morlet over Ricker on
this universe (0.547 → 0.717). That comparison was movement
*within an alpha-negative regime*:

- Passive BH val: **+0.850**.
- Morlet val: **+0.717** → **alpha −0.133**.
- Ricker val: **+0.547** → **alpha −0.304**.

Both arms underperform passive. The bundle migration moved a
losing strategy slightly less far in the wrong direction. The
"+0.17 lift" was real but the absolute number was always below
passive — a fact none of the prior runs surfaced because they
compared model arms to other model arms.

This result also weakens the case for the polar Morlet bundle
itself on the relational side: its only previous "clear win"
([relational-morlet-failure.md](relational-morlet-failure.md))
was a +0.17 movement in alpha-negative territory. The bundle
remains canonical for `apps/replay --decoder cnn` reconstruction
R² (where the metric is target-recovery, not portfolio Sharpe),
but the relational case for it just got materially weaker.

### factor-wide extension

The 2026-05-10 follow-up (added after the initial 3-universe
run) loads the `factor-wide` panel (`min_history=3500`,
`first_valid_index ≤ 2010-01-01`, ~2162 names from the full
Stooq archive) via the cached pickle from
`apps/factor/scripts/modal/prep_universe_pivot_data.py`. No
relational model arm has been trained against this universe,
but it completes the passive-side picture and establishes a
monotonic structural pattern:

| Universe | n names | passive BH val Sharpe |
|---|---:|---:|
| Phase-2 | 21 | +1.079 |
| stooq_us_long | 312 | +0.850 |
| ex-Phase-2 | 296 | +0.818 |
| factor-wide-ish | 2162 | **+0.681** |

**Passive val Sharpe drops monotonically as the universe
broadens.** The 2021-2025 passive Sharpe was carried by mega-cap
concentration; widening to mid-caps and beyond dilutes it. This
reframes the prior 3-universe table:

- The Phase-2 model row's "+1.146 val Sharpe" was 94% market beta
  *of a slice with very high market beta*. On the same window
  the broader universe paid 0.68. The model didn't choose a
  high-Sharpe operating point; the universe did.
- The natural follow-on hypothesis (more applicable to small/
  mid-cap names → more alpha?) has a partial answer here:
  broader universes have **lower passive bars** (0.68 vs 1.08),
  so a strategy with even moderate per-name picking skill could
  in principle clear them in absolute Sharpe terms while
  showing a higher *relative* alpha. But:
    1. We have no relational model run on factor-wide. The
       factor-side `universe-pivot` (2026-05-06 leaderboard
       row) tested factor heads on a similar wide universe —
       val IC tied with factor-narrow, `confirmed-null`. That's
       a different model class but the only same-direction data
       point we have, and it argues against the "wider = more
       alpha" intuition for the model classes in this repo.
    2. The friction stack assumed (10 bps round-trip) is roughly
       right for the names that actually clear `min_history=3500`
       — these are still established mid-to-large caps, not
       micro-caps. True penny-stock universes would need 100-500
       bps round-trip friction modeled, which would crater the
       passive Sharpe further (passive ≠ frictionless when
       you're rebalancing micro-caps either) but crater the
       model's even more (the model rebalances every 20 days vs
       passive's never).
    3. The trend is consistent with "model class is wrong",
       not "universe slice is wrong". 3-of-3 universes show the
       model losing to passive; nothing in the trend predicts
       the loss flips to a win at micro-caps. The strategy-class
       falsification test (next-experiment #1 below) is more
       informative than another universe-shift run.

**Caveat on the factor-wide passive number.** The benchmark uses
`pd.DataFrame.ffill().dropna(axis=1)` per window — which holds
delisted names flat at their last quoted price for the rest of
the window. On factor-wide (where 11+ years between
`first_valid_index` and val end means non-trivial delisting)
this is on the optimistic side for passive Sharpe by ~0.05-0.15
depending on the actual delisting rate. The bias affects all
four universes monotonically (Phase-2 has zero delistings;
factor-wide has the most), so the *absolute* Phase-2 → factor-
wide gap may be ~0.05-0.15 narrower than the table shows. The
*relative* alpha measurement vs a model run on the same
universe (with the same ffill convention) is unaffected.

### ex-Phase-2 — catastrophic, exactly as predicted

The leaderboard's pre-test threshold was "if ex-Phase-2 EW ≥
0.6, the model is **negative alpha** outside mega-caps". Passive
came in at **+0.818** — well above 0.6 — and the model came in
at **0.484**, **alpha −0.334**.

This is a /third/ of a Sharpe unit of value destruction vs just
holding the universe. Compounded by 10 bps of commission per
rebal cycle, the model is actively losing money relative to a
do-nothing alternative. The
[universe-shift finding](relational-universe-shift.md) called
the ex-Phase-2 collapse "model has ~0.5 of mega-cap-specific
alpha"; the EW comparison reveals it has *no* alpha and ~0.33
of negative-alpha cost.

The 2026-05-08 8-arm reversed-OOS rows (farthest, diversified,
per_ticker — all with severe train→val Δ) compound this read:
the entire kNN-on-CWT-fingerprint strategy class destroys value
off mega-caps. The "wins" on Phase-2 were narrowness artifacts.

## Implications

1. **No current relational checkpoint should be taken live.**
   All three measured candidates are alpha-negative or
   alpha-noise after passive comparison. The live infrastructure
   (`ss-relational live` + Alpaca broker rails) is wired and
   tested, but there's no shippable strategy to plug into it
   right now.
2. **Operational rule change.** Future leaderboard rows
   claiming live-tradeability must include the passive-EW val
   Sharpe on the same universe + window. `alpha = model − passive`
   becomes the headline column for shippable claims; raw Sharpe
   stays for relative-arm comparisons but is decorative for the
   live-go decision. See the new bullet in
   [`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md#operational-rules-extracted-from-findings).
3. **Compression follow-ups stay research-only.** The
   [DWT compression follow-ups](../TODO/dwt-compression-followups.md)
   (K-only DWT, lossless polar-CWT compression) are informative
   for replay-CNN reconstruction R² and storage budgets, but
   layering compression on a strategy that loses to passive is
   rearranging deck chairs. The compression work is interesting
   on its own merits; it doesn't gate any live decision until
   the underlying strategy clears EW.
4. **Re-examine the factor app's apparent baseline.** The
   2026-04-30 deterministic-indicator linear head shows val IC
   +0.012 / val Sharpe +0.44 on `factor-narrow`, marked
   `confirmed-OOS`. Different metric (IC vs Sharpe) so the
   ceiling claim isn't directly affected, but the matched
   portfolio-side Sharpe of +0.44 needs the same passive-EW
   comparison on `factor-narrow` before being treated as
   shippable. Likely a similar regrade — the factor universe is
   297 stooq_us_long names, where passive EW is ~+0.83 val.
5. **The kNN-on-CWT-fingerprint strategy class may be
   structurally wrong for this objective.** Three universes,
   six prior `confirmed-OOS` / `partial-OOS` arms, and not one
   beats passive. The pattern is consistent: when the candidate
   pool is narrow enough to support overfitting (Phase-2), the
   model approximates passive with extra friction; when the
   pool is wide enough to be a real test (stooq_us_long,
   ex-Phase-2), the model destroys value. The next-experiment
   question this raises is structural — see "Next experiments"
   below.

## What this result is — and isn't

- **Is**: a comparison of model val Sharpe to passive
  equal-weight on the same universe, same window, matched
  friction. Settles whether the model's Sharpe is alpha or
  market beta of the chosen universe.
- **Is not**: a comparison to a market index (SPY, IWM, equal-
  weighted S&P 500). Those would be slightly different
  benchmarks — equal-weighting our specific 312-name pool is
  closer to the model's structural opportunity set than holding
  SPY would be. If anything, comparing to a real broad-market
  index would be even less favorable to the model (SPY 2021-2025
  Sharpe is also in the 0.7–1.0 range with much lower
  friction).
- **Is not**: a single-window walk-forward. Same caveat as the
  [leaderboard reading note #3](../leaderboard.md#reading-the-table)
  applies — single-split eval can flatter or denigrate by
  accident of regime alignment. A rolling-window EW comparison
  would be stronger evidence; a single-split EW comparison is
  still strictly better evidence than the no-EW-comparison the
  prior rows had.

## Next experiments

The verdict label is `diagnostic` (this is a passive baseline,
not a model run). Per the
[`diagnostic` next-experiment rule](../leaderboard.md#verdict-labels):
turn it into falsifiable hypotheses. Three open lines:

1. **Strategy-class falsification.** Hypothesis: no kNN-on-CWT-
   fingerprint variant in the current relational toolbox
   (empirical, gmm, analog, farthest, diversified, velocity)
   beats passive EW on a wide universe. Test: run all six
   scorers against passive EW on stooq_us_long with the same
   train/val split. Expected null. If any clears EW, it's the
   first one we'd actually want live; if none does, the entire
   strategy class is the wrong tool and we stop iterating
   within it.
2. **Friction sensitivity.** Hypothesis: the −0.334 ex-Phase-2
   alpha is dominated by the 20-bar rebal × 10 bps friction
   stack. Test: rerun the model side at rebal_days ∈ {60, 252}
   (quarterly / annual) and recompute alpha vs the same
   passive baseline. If quarterly rebal narrows the gap to <
   0.1, the strategy is still wrong but the operational form
   matters; if it doesn't, the negative alpha is intrinsic.
3. **Benchmark robustness.** Hypothesis: the passive numbers
   above generalize to other long-only baselines (cap-weighted,
   minimum-variance, simple risk-parity). Test: add 2-3 more
   passive arms to the benchmark script and compare. If the
   strategy doesn't beat the *easiest* passive (EW), it's
   unlikely to beat the harder ones either, but worth measuring
   to anchor the future EW-vs-model comparison conventions.

## Notes

- Implementation: `apps/relational/scripts/equal_weight_benchmark.py`,
  pure numpy + `ss_loaders.load_stooq_matrix` +
  `ss_portfolio.metrics.{annualized_sharpe, sortino, cagr,
  max_drawdown}`. ~5 min wall locally.
- Reproducibility:
  `uv run python apps/relational/scripts/equal_weight_benchmark.py`.
- Artifacts: `Output/equal-weight-benchmark.json` (per-universe,
  per-arm, per-window stats).
- Master walk-forward log: three
  [2026-05-10 leaderboard rows](../leaderboard.md), each
  carrying the `diagnostic` verdict and reclassifying a prior
  `confirmed-OOS` / `reversed-OOS` model row.
