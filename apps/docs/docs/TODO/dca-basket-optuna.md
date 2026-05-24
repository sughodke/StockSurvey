# DCA basket Optuna search — pre-registered arc

**Status: `pending` — pre-registration locked 2026-05-23. Eval has
NOT yet run. This page must be committed *before* any Optuna trial
fires; the falsification bar and search space are not editable
post-hoc.**

## Motivation

The canonical DCA basket is the **13-ETF Phase 4d universe** (9 SPDR
sectors + TLT/IEF + GLD/DBC). It was inherited from CFR Phase 4d, not
chosen by a systematic search. The current leaderboard claim is
"DCA stays #2 on the DSR ladder at deflated-t +1.93" but we have no
direct comparison against alternative baskets. **Goal**: test
whether a different basket composition has materially better
deflated-t on held-out OOS data, after honest accounting for the
selection penalty of running Optuna.

## Search space (locked — bucketed, not arbitrary tickers)

Within each bucket, **equal weight** to chosen ETFs. The only lever
is which buckets are present.

| bucket | choices | n |
|---|---|---:|
| `equity_core` | `{9-spdr-sectors-EW (canonical), SPY-only, VTI-only, top-3-sectors-trailing-12m-sharpe}` | 4 |
| `international` | `{none, EFA+EEM, VEU}` | 3 |
| `bonds` | `{none, TLT-only, TLT+IEF (canonical), AGG, TLT+IEF+TIP}` | 5 |
| `commodities` | `{none, GLD-only, GLD+DBC (canonical), GLD+DBC+USO}` | 4 |
| `reits` | `{none, VNQ}` | 2 |
| `rebal_trading_days` | `{21, 63, 80 (canonical), 126, 252}` | 5 |
| `drift_threshold` | `{0.03, 0.05 (canonical), 0.10}` | 3 |

Grid: 4 × 3 × 5 × 4 × 2 × 5 × 3 = **3,600 combinations**.
Optuna's TPE sampler explores this subset; we do not exhaustively
evaluate.

**Excluded** (data-quality / launch-date constraints, locked):
- `XLRE` (launched 2015) — train period needs 2005-on history
- `XLC` (launched 2018) — same reason
- Pre-2005 data on EFA, EEM, VNQ verified before the run (drop if
  missing)
- Single-commodity ETFs other than GLD/SLV/DBC/USO (avoid an
  open-ended ticker universe)

## Trial budget (locked)

**N_TRIALS = 200**. This is the deflation count regardless of how
many additional trials we *would have run* given more compute.
Pre-registration commits us to this budget BEFORE the search begins.

Under the workspace's `sharpe_std_ann=0.25` calibration:
`E[max Sharpe under 200-trial null] ≈ 0.25 × Z(1 − 1/200) = 0.65`
annualized. This is the bar any winner has to clear *just to be
distinguishable from coin flips*. The canonical 13-ETF posts ann
Sharpe +0.69 — so the search needs to find something materially
better than the canonical or honestly conclude null.

## Walk-forward split (locked)

- **Train** (Optuna objective):  **2005-02-25 → 2018-12-31** (~14 yr,
  includes 2008 GFC)
- **Val** (OOS report only):    **2019-01-01 → 2025-12-31** (~7 yr,
  includes 2020 COVID + 2022 Fed-pivot)

Optuna sees ONLY the train slice. The val slice is touched exactly
once at the end, when reporting the winner's OOS deflated-t. No
re-tuning, no peeking.

## Objective (locked)

For each candidate basket:
- Compute daily net returns via `cfr.baselines.PassiveEW(rebal_days=
  basket['rebal_trading_days'], commission_bps=10.0)`.
- Compute deflated-t via `ss_portfolio.standardize_oos(returns,
  periods_per_year=252, n_trials=200, sharpe_std=0.25/sqrt(252))`.
- **Optuna maximizes train-period deflated-t.**

We are NOT maximizing val-period anything — that's the held-out
verdict. Optuna's search target is train-period DSR.

## Falsification bar (locked — before eval runs)

Let `t_canonical_val` = val-period deflated-t of the canonical
13-ETF basket (re-computed under identical methodology). Let
`t_winner_val` = val-period deflated-t of Optuna's train-winner
applied OOS.

| outcome | verdict | action |
|---|---|---|
| `t_winner_val > t_canonical_val + 1.0` | **`confirmed-OOS`** — adopt the new basket | Update `apps/dca/scripts/build_checkpoint.py` defaults, append leaderboard row, rebuild `Output/dca-multiasset.json` |
| `t_winner_val ∈ [t_canonical_val + 0.0, t_canonical_val + 1.0]` | **`partial-OOS`** — interesting but not enough to switch live | Document the candidate; keep canonical as live deployment |
| `t_winner_val ≤ t_canonical_val` | **`confirmed-null`** — canonical basket is defensible | Document the search; this IS the right defense of the current recipe |

**Critically**: the eval also re-computes `t_canonical_val` under
identical method (same walk-forward dates, same DSR calibration, same
commission). The reported "current 13-ETF deflated-t" on the
leaderboard (+1.93) was computed differently (full sample, 5232
daily bars, n_trials=4); we cannot directly compare against it. The
fair reference is `t_canonical_val` from this same eval.

## Datasets / dependencies (locked)

- Daily closes for every ETF in the search space, sourced from
  **Stooq** (`./StooqData`) for consistency with the existing DCA
  arc. Tickers verified to have continuous coverage 2005-02-25 →
  2025-12-31. If any ticker is missing it gets dropped from its
  bucket *before the search begins*, with the drop recorded in this
  page.
- `cfr.baselines.PassiveEW` for the EW rebal engine (same as the
  canonical DCA today).
- `ss_portfolio.standardize_oos` for the deflated-t.

## What does NOT count as a result

- A train-period winner — Optuna will produce one by construction.
  This pre-reg only counts the **val-period deflated-t against
  canonical-on-val**.
- A winner with worse val-period max-drawdown than canonical, EVEN
  IF deflated-t is higher. Implementing such a basket on a $100k
  paper account is a real-money worse path. The bar is *deflated-t*
  AND *val max-DD ≤ canonical-val-max-DD + 5 pp*.
- Any basket that contains a ticker for which Stooq history doesn't
  reach 2005-02-25. We don't backfill; we drop.

## Reproduction

When the eval runs, it lands at:

```bash
uv run python apps/dca/scripts/optuna_basket_search.py \
    --n-trials 200 \
    --train-start 2005-02-25 --train-end 2018-12-31 \
    --val-start 2019-01-01 --val-end 2025-12-31 \
    --out Output/dca-basket-optuna.json
```

Artifacts: `Output/dca-basket-optuna.json` (full Optuna study
serialization), `Output/dca-basket-optuna-summary.txt` (human-readable
top-10 + canonical comparison + verdict), and a finding page
`apps/docs/docs/findings/dca-basket-optuna.md` (the writeup).

## Expected outcome (recorded for honesty)

My honest prior: `confirmed-null` or weak `partial-OOS`. The
canonical 13-ETF spans 4 asset classes and was already implicitly
selected through CFR Phase 4d's findings; the search will likely
not find a +1.0-deflated-t improvement that isn't a snooping
artifact.

The most plausible *real* improvement, if any: **adding international
equity (EFA+EEM) or VEU** lifts val-Sharpe by ~0.05-0.15 in
historical backtests; lower-correlation diversification could
plausibly push the deflated-t over the bar.

Most plausible nulls: dropping bonds (DBC), tightening rebal
cadence to monthly, changing equity_core from sectors to SPY-only.

## Why this matters

This is the cleanest test of the load-bearing assumption "the 13-ETF
basket is good enough to deploy live as the DCA leg of the
DCA + vol-v3 ensemble." If the search confirms-null, we can deploy
the canonical basket with full confidence — the alternative was
considered and rejected. If the search confirms-OOS, we deploy the
improved basket. Either way, the live decision becomes more
defensible than the prose-only argument that motivated this work.

## Cross-links

- Methodology: [`deflated-sharpe-leaderboard`](../findings/deflated-sharpe-leaderboard.md)
- DCA arc baseline: [`cfr-phase4`](../findings/cfr-phase4.md),
  [`cfr-vs-dca-realistic`](../findings/cfr-vs-dca-realistic.md)
- Verdict label vocabulary: [leaderboard](../leaderboard.md#verdict-labels)
