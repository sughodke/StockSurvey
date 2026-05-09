# Leaderboard

Master log of every walk-forward / out-of-sample eval run in this repo,
in chronological order. Append-only — when a finding is upgraded or
downgraded, add a new row referencing the prior; never rewrite history.

The point of this file is to make the *OOS verdict* of every claim
visible at a glance, so an in-sample finding can't sit unchallenged in
[CLAUDE.md](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md)
(or [Findings](findings/index.md)) for weeks before someone notices the
OOS check disagrees. A row here is the source of truth on whether
something is shippable; the prose pages around it are narrative.

## Operating conditions

Recurring universes, windowings, and friction params — referenced by
short tag in the table below so each row stays one line.

### Universes

| tag | size | source | span | filter |
|---|---:|---|---|---|
| **phase-2**       |   21 | Stooq archive `./StooqData/` | 2013-01-29 → 2025-12-11 | mega-cap fixed list (PHASE2_TICKERS) |
| **stooq_us_long** |  312 | `apps/notebook/data/stooq_us_long/manifest.json` | 2000-01-03 → 2026-04-24 | Stooq archive subset, baked-in |
| **factor-narrow** |  297 | stooq_us_long → `min_history_bars=6500` | ~2000 → 2026 | canonical factor walk-forward universe |
| **factor-wide**   | 2073 | full Stooq archive → `min_history=3500`, `start_grace_days=3650` | 2010-01-01 → 2026 | accepts late-listing tickers, NaN early bars |
| **regime-2010-24** | n/a | Stooq archive | 2010-01-01 → 2024-12-31 | regime trainer training span |

PHASE2_TICKERS = `AAPL, AMZN, BA, BAC, CRM, CSCO, DIS, GE, GOOGL, JNJ,
JPM, KO, META, MSFT, NFLX, NVDA, T, TSLA, UNH, WMT, XOM`. 5 of these
(CRM / GOOGL / META / NFLX / TSLA) post-date stooq_us_long's 2000
cutoff and are missing from that subset — Phase-2 only loads cleanly
from the full archive.

### Windowings

| tag | shape | notes |
|---|---|---|
| **6-window factor** | rolling, train=63 blocks (~3y) / val=39 (~2y) / step=39 (~2y) | rebal_days=20 unless noted; AdamW lr=1e-2 wd=1e-3 n_steps=200 default |
| **6-window factor (q)** | rolling, train=20 / val=12 / step=12 | scaled by 20/63 to keep train/val years comparable at rebal_days=63 |
| **regime-3w-optuna** | rolling, 3 windows, 20 Optuna trials/window | regime trainer multi-window search |
| **phase-2 split** | single split, train 2013-01-29 → 2020-12-31, val 2021-01-01 → 2025-12-11 | 8y / 5y; mirrors `build_canonical_checkpoints.py` |
| **70/30 single** | single split, first 70% train / last 30% val | one-shot train/val, not rolling |
| **single train+val ticker** | one trained set + one held-out ticker | not technically walk-forward; SSL reconstruction context |

### Friction / portfolio defaults

`rebal_days=20`, `commission_bps=10.0`, `top_n=10` for portfolio metrics
unless explicitly overridden in a row's notes column.

### Verdict labels

- **confirmed-OOS** — val performance matches or extends the train
  finding. Safe to cite as evidence; safe to ship if other gates pass.
- **reversed-OOS** — train win does not survive in val (or worse, val
  underperforms baseline). Do NOT ship; treat the train finding as
  in-sample artifact.
- **partial-OOS** — some windows / metrics survive, others don't.
  Caveat-laden; not shippable without further qualification.
- **confirmed-null** — both train and val show no edge. Useful as a
  falsification (saves later researchers from re-running it) but does
  not advance the strategy.
- **diagnostic** — stability / structure observation rather than a
  shippable claim. E.g. "Optuna best params bounce window-to-window"
  — informative but no train/val comparison to verdict on.
- **pending** — experiment in flight, no final number yet.

## Master table

Date is the row's recording date (when the result was finalised), not
the experiment's first kickoff. App is the directory under `apps/`.
Metric is what train / val numbers refer to.

| date | app | experiment | universe | windowing | metric | train | val | delta (val − train) | verdict | artifact |
|---|---|---|---|---|---|---:|---:|---:|---|---|
| pre-2026-04 | regime    | Optuna walk-forward over (lookback, n_tail, top_n, divergence) | regime-2010-24 | regime-3w-optuna | val Sharpe (best params/window) | not reported | later windows ~+1.36..+1.63 | n/a | diagnostic | best-params drift wide; signal unstable across windows |
| pre-2026-04 | regime    | JAX-Adam differentiable trainer, single 70/30 split (impl since deleted) | regime-2010-24 | 70/30 single | Sharpe | +1.22 | +0.80 | −0.42 | confirmed-OOS | scale weights collapsed to long horizons (126d 48%); impl removed with JAX dep drop |
| 2026-04-30 | regime    | Log-returns CWT input vs raw close (controlled, kernel_half_extent=3 both arms) | regime-2010-24 | regime-3w-optuna | val Sharpe (median / mean / worst) | n/a | raw close: +0.15 / +0.07 / −0.41 ; log-returns: +0.03 / −0.29 / −1.06 | raw close beats in 3/3 windows | confirmed-OOS | `Output/regime-eval-{rawclose-kernel3,logreturns}.{log,json}` |
| 2026-04-30 | factor    | Deterministic-indicator baseline (linear head) | factor-narrow | 6-window factor | mean val IC ; mean val Sharpe | n/a | +0.0120 ; +0.440 | 5/6 windows positive | confirmed-OOS | `Output/walkforward-{linear,mlp}-s200-wd0.001-windows.npz` |
| 2026-04-30 | factor    | Deterministic-indicator baseline (MLP head) | factor-narrow | 6-window factor | mean val IC | (3× linear train IC) | +0.0081 | 4/6 windows positive | partial-OOS | mlp triples train IC over linear but lower val IC + 2 negative windows — overfitting |
| 2026-05-04 | factor    | sign_demeaned target reduction probe | factor-narrow | 6-window factor | mean val IC | n/a | +0.0088 vs +0.0120 control (Δ = −0.0032) | 5/6 windows degrade | reversed-OOS | falsified; `Output/forecast-probe-{control,probe}-windows.npz` |
| 2026-05-04 | factor    | vol_innovation forecast target | factor-narrow | 6-window factor | mean val IC ; mean val Sharpe | range 0.435..0.536 | +0.4743 ; +0.515 | 6/6 windows positive | confirmed-OOS (for vol IC) | real signal but only +0.075 Sharpe lift; `Output/forecast-probe-probe-vol-windows.npz` |
| 2026-05-04 | factor    | Vol-target overlay (fcst-VT vs trail-VT vs EW)  | factor-narrow | 6-window factor | mean val Sharpe | n/a | EW +0.215, trail-VT +0.231, fcst-VT +0.232 | fcst − trail = +0.001 | confirmed-null | forecast adds nothing over trailing vol for sizing; `Output/vol-overlay-{summary,windows}.{json,npz}` |
| 2026-05-04 | factor    | Pure CWT bundle vs IndicatorGridConfig at matched setup | factor-narrow | 6-window factor (K=1) | mean val IC (return / vol) | n/a | cwt-return +0.0091, indicator-return +0.0120 ; cwt-vol +0.2165, indicator-vol +0.4743 | tied at noise on returns; indicators 2× CWT on vol | confirmed-null (returns) / partial-OOS (vol) | `Output/cwt-bundle-{summary.json,*-windows.npz}` |
| 2026-05-04 | factor    | Feature augmentation: vol forecast as 75th feature for return head | factor-narrow | 6-window factor | mean val IC ; mean val Sharpe | n/a | aug +0.0117 vs base +0.0120 (Δ −0.0004); +0.442 vs +0.446 | head L1-share on forecast feature = 0.014 (uniform) | confirmed-null | `Output/feature-aug-summary.json` |
| 2026-05-04 | factor    | Horizon pivot rebal=63d (quarterly) | factor-narrow | 6-window factor (q) | mean val IC (return / vol) | n/a | q-return −0.0019 vs 20d-return +0.0120 ; q-vol +0.3439 vs 20d-vol +0.4743 | quarterly worse on both | reversed-OOS | quarterly Sharpe rises mechanically from less commission, not skill; `Output/horizon-pivot-*.{json,npz}` |
| 2026-05-06 | factor    | Universe pivot (factor-narrow → factor-wide) | factor-wide | 6-window factor | mean val IC (return / vol) | n/a | wide-return +0.0106 ; wide-vol +0.4091 | tied / slightly worse vs narrow | confirmed-null | 7× universe doesn't lift the +0.012 return ceiling; `Output/universe-pivot-{summary.json, *-windows.npz, close.pkl}` |
| 2026-05-06 | factor    | Regime gate via aggregate forecast vol (gate-70/80/90) | factor-narrow | 6-window factor | mean val Sharpe | n/a | always-on +0.215, gate-70 +0.015, gate-80 +0.195, gate-90 +0.240 | gate-90 +0.025 over always-on (within ±0.17 noise) | confirmed-null | vol forecast doesn't gate return signal; `Output/regime-gate-{summary.json,windows.npz}` |
| 2026-05-07 | replay    | 2D Haar DWT-L1 keep-LL CWT-tile compression vs uncompressed (cwt-only bundle) | stooq_us_long (295-ticker pool, AAPL primary, NVDA val) | single train+val ticker | NVDA val R² (rsi / cci / vol / macd) | not reported | rsi 0.576 vs 0.582 ; cci 0.610 vs 0.603 ; vol −0.30 vs −0.38 ; macd both broken | rsi −0.006 ; cci +0.007 ; vol +0.084 ; macd N/A | partial-OOS | SSL reconstruction R², not a portfolio metric; CSCO zero-shot peaks (RSI 0.92 vs 0.90 ; CCI 0.89 vs 0.85) marginally favor compressed; MACD head broken in both arms (orthogonal bug); `Output/cwtonly-{,dwtL1-}*` |
| 2026-05-07 | relational | Analog-kNN DWT-L1 vs uncompressed fingerprint | phase-2 | phase-2 split | Daily Sharpe | dwt-L1 1.116 vs baseline 1.032 (Δ +0.084) | dwt-L1 1.099 vs baseline 1.146 (Δ −0.046) | val edge reverses sign | **reversed-OOS** | full-period in-sample +0.04 Sharpe was driven entirely by 2013-2020 sub-period; do NOT pin compress_levels=1 on canonical analog checkpoint; `Output/relational-idea-b-analog-knn-dwt-walkforward-{equity.png,stats.txt}` |
| 2026-05-08 | relational | 8-arm distance-scorer × ±DWT-L1 — **analog cross_ticker baseline** | phase-2 | phase-2 split | Daily Sharpe | 1.032 | 1.146 | **+0.114** | **confirmed-OOS** | only arm whose val *exceeds* train; this is the canonical `Output/relational-analog.json`; `Output/relational-dwt-phase2-walkforward.{csv,txt}` |
| 2026-05-08 | relational | 8-arm — analog cross_ticker DWT-L1 (supersedes 2026-05-07 single-arm row above) | phase-2 | phase-2 split | Daily Sharpe | 1.116 | 1.099 | −0.016 | reversed-OOS | confirms prior single-arm walkforward; do NOT pin compress_levels=1 on canonical analog |
| 2026-05-08 | relational | 8-arm — analog **per_ticker** baseline | phase-2 | phase-2 split | Daily Sharpe | 1.305 | 0.824 | **−0.482** | reversed-OOS, severe | per_ticker pool gives best in-sample Sharpe (1.305) of any arm but loses 0.48 OOS; not shippable |
| 2026-05-08 | relational | 8-arm — analog per_ticker DWT-L1 | phase-2 | phase-2 split | Daily Sharpe | 1.320 | 0.876 | **−0.444** | reversed-OOS, severe | DWT-L1 marginally rescues per_ticker val (+0.052 over uncompressed pt), but absolute level is still 0.27 below uncompressed cross_ticker val |
| 2026-05-08 | relational | 8-arm — farthest baseline | phase-2 | phase-2 split | Daily Sharpe | 1.321 | 0.828 | **−0.493** | reversed-OOS, catastrophic | most extreme train→val gap of any arm; train-only artifact |
| 2026-05-08 | relational | 8-arm — farthest DWT-L1 | phase-2 | phase-2 split | Daily Sharpe | 1.102 | 0.833 | −0.269 | reversed-OOS but smaller | DWT compresses train Sharpe back toward val level; val Sharpe nearly identical to uncompressed (0.833 vs 0.828) — DWT doesn't add OOS skill, just reduces in-sample inflation |
| 2026-05-08 | relational | 8-arm — diversified baseline | phase-2 | phase-2 split | Daily Sharpe | 1.222 | 1.002 | −0.220 | partial-OOS | only non-cross_ticker-analog arm with val Sharpe > 1.0; better than DWT version |
| 2026-05-08 | relational | 8-arm — diversified DWT-L1 | phase-2 | phase-2 split | Daily Sharpe | 1.245 | 0.832 | **−0.413** | reversed-OOS | strictly worse than uncompressed diversified on val (0.832 vs 1.002); clearest "compression hurts OOS" signal in the table |
| 2026-05-09 | relational | analog cross_ticker — universe-shift validation (ex-Phase-2; same algo, wider universe) | factor-narrow-ish (296 stooq_us_long names ex-PHASE2_TICKERS) | phase-2 split | Daily Sharpe | 0.615 | 0.484 | **−0.131** | reversed-OOS — **also collapses vs Phase-2 baseline** | val Sharpe 0.484 vs Phase-2 cross_ticker val 1.146 (Δ −0.66) directly confirms the macro-tailwind concern: ~0.5-0.7 of Phase-2's val edge was mega-cap-specific behavior, not generalizable cross-sectional skill. MaxDD also worse (-39% vs -31%). Algorithm: `analog_knn_scores_fast(n_workers=24)` mp.Pool over t-axis, OPENBLAS_NUM_THREADS=1 per worker (12 min wall on Modal cpu=8 vs 2-4h serial estimate). Universe is mid/large-cap-survivor names, not true small caps. Artifacts: `Output/relational-exmegacap-{equity.png,stats.txt,walkforward.csv}`. Repro: `uvx modal run apps/relational/scripts/modal/relational_exmegacap_modal.py` (after `prep_exmegacap_prices.py`). |

## Pending walk-forward checks called out elsewhere

These have hypotheses to test but no results yet; they're tracked in
the [TODO](TODO/index.md) backlog and should land here as new rows when
run:

- **DWT wider-universe validation** (factor-wide or stooq_us_long) —
  superseded by the 2026-05-09 ex-Phase-2 *uncompressed* run: the
  cross_ticker baseline already collapses to val Sharpe 0.48 on the
  wider universe, so layering DWT on top of an already-broken arm
  isn't informative. Skip unless we have a separate reason to test
  compression specifically.
- **Equal-weight benchmark on Phase-2 + ex-Phase-2** (~5 min local).
  Outstanding: we don't yet know how the analog cross_ticker val
  Sharpe (1.146 Phase-2, 0.48 ex-Phase-2) compares to passive
  equal-weight on the same universes. If passive Phase-2 hits 1.0+
  Sharpe, the model adds ~0.15 of marginal alpha. If passive
  ex-Phase-2 hits 0.6+, the model is **negative alpha** outside
  mega-caps. Cheap and answers the most important remaining
  question.
- **Rebal-days sweep** (factor-narrow or phase-2) — sweep
  `rebal_days ∈ {5, 10, 20, 40}` to determine whether DWT-L1 supports
  a faster rebalance cadence than 20d. Gates the daily-cron-with-trigger
  decision in [TODO](TODO/index.md).
- **Non-Haar wavelet sweep** (phase-2) — db2/sym4/coif1 vs Haar at
  L=1. Cheap (~7 min) but only worth it after wider-universe and
  rebal-days resolve.
- **DCT zigzag-keep-top-k** (phase-2 head-to-head) — the deferred
  follow-up to the DWT/DCT compression breakthrough; needs a flat-input
  decoder branch in replay before the replay-side comparison is
  possible. Relational side can run as-is.
- **Modal-cron live deployment dry-run** (Alpaca paper) — first
  ~2-week paper-trade window comparing live decisions vs backtest
  expected positions. Counts as a "verdict-by-paper-trade" row, not
  a backtest row, but lands in the same table for visibility.

## Reading the table

A few interpretation notes:

1. **`confirmed-null` is information.** A row marked `confirmed-null`
   has a *false* hypothesis tested against and confirmed false; the
   data is useful — it tells future-you the lever has been tried and
   doesn't move. Don't re-run.
2. **`reversed-OOS` is the load-bearing column.** When it appears,
   the matching prose entry in [Findings](findings/index.md) should
   already be qualified — if it isn't, that's a doc bug.
3. **Single-split eval (e.g., `phase-2 split`) is weaker than rolling
   walk-forward.** A single train/val window can flatter or
   denigrate a finding by accident of regime alignment. Rolling
   windows are stronger evidence. The current relational result is
   single-split — a rolling-window protocol on Phase-2 would be a
   stronger follow-up, not a separate experiment.
4. **The `partial-OOS` label hides a lot.** When you see it, click
   through to the artifact path and read the per-window stats — the
   "partial" usually means "1 outlier window inflates the mean";
   medians are typically more honest than means in those cases.
5. **Sharpe ≠ skill at low IC.** Several factor rows show val Sharpe
   well above val IC's "alpha-positive" threshold (e.g., quarterly
   horizon at val IC ≈ 0 still posts +0.65 Sharpe). That's commission
   geometry, not signal — a near-zero-IC strategy with low turnover
   beats a higher-IC strategy with high turnover on Sharpe. Always
   read the IC column alongside the Sharpe column.
