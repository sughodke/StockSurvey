---
tags:
  - stooq_us_long
  - partial-OOS
---

# Replay CNN — length-axis (K) sufficiency for indicator reconstruction

**Operational rule:** for the
[`--decoder cnn`](replay-decoders.md) indicator-reconstruction
objective, the CWT bundle's **time-window length `K`** can be halved
from 96 → 48 with no measurable loss on RSI / CCI / vol heads. The
high-frequency content *along the K axis* (bar-to-bar oscillation
within the rolling window) is not load-bearing for these targets.
The next replay backbone can default to a shorter K without quality
loss on this objective; live-bar inference cost along the K axis
drops linearly.

This page is an **extraction** of the K-axis reading from the
[parent 2D DWT compression finding](replay-dwt-compression.md).
The underlying experiment compressed K and the scale axis S
simultaneously, so the K-axis claim here is suggestive — strong
enough to act on for an architectural default, not isolated enough
to call confirmed-OOS for K alone. The
[K-only DWT follow-up](#follow-up-1d-dwt-on-k-only) proposes the
clean isolation.

## Why call it out separately

The 2D DWT keep-LL result (`(K, S) = (96, 15) → (48, 8)` per tile)
was framed as a 4× input shrink with reconstruction quality
preserved. But for downstream design decisions, *which axis was
sufficient to compress* matters:

- **Architectural defaults.** If K-axis compression is what's
  preserving signal, the next backbone iteration should rethink K
  before rethinking S (which is closer to the actual frequency
  content of the wavelet bundle). The two axes have different
  semantic content.
- **Live-trading latency.** Per-bar inference cost in the conv
  stack scales roughly linearly with K. Halving K halves live
  latency on the encoder forward; halving S halves the per-tile
  feature count but the conv kernels span K, not S.
- **Causality.** The K axis is the time-window axis — the
  causality-relevant axis. Compressing K is the axis where
  "doesn't smear future into past" matters most. Haar at L=1 along
  K is a 2-tap low-pass + downsample by 2, fully causal within the
  per-bar tile.

## What the K-axis reduction looks like

Haar L=1 along K maps each output cell to the mean of two
consecutive K cells. So `K'[i] = (K[2i] + K[2i+1]) / 2` plus a
discarded high-pass detail coefficient. The targets'
recoverability under this transform implies the *delta between
adjacent K cells* (i.e., the bar-to-bar change in the CWT power /
phase at a given scale) is approximately reconstructible from the
local mean — and that the heads don't need that delta to
reconstruct RSI / CCI / vol.

This is consistent with what the targets are: trailing rolling
statistics on close prices that themselves change ~daily-bar by
daily-bar with most of the energy in the slowly-varying envelope.
The K-axis high-frequency content of the CWT bundle was over-
provisioned for these reconstruction objectives.

## What the data shows (extracted from the 2D run)

NVDA val R² with `K = 48` (DWT-L1 compressed) vs `K = 96` baseline
(both arms also compressed S, see caveat):

| Target | K' = 48 (compressed) | K = 96 (baseline) | Δ      |
|--------|----------------------|-------------------|--------|
| rsi    | 0.576                | 0.582             | −0.006 |
| cci    | 0.610                | 0.603             | +0.007 |
| vol    | −0.30                | −0.38             | +0.084 |
| macd   | both arms broken     | both arms broken  | N/A    |

CSCO zero-shot peak R² is the same story: compressed equals or
slightly exceeds uncompressed at every non-MACD target.
Side-by-side `(n, w)` sweeps and the FiLM attention shift are in
the [parent 2D finding](replay-dwt-compression.md).

## Caveats and scope

- **Joint K + S compression.** The 2D DWT keep-LL transform halved
  *both* axes simultaneously. The data is consistent with "K-axis
  compression is sufficient" but does not falsify "S-axis
  compression is what's actually doing the work" or "both axes
  contribute equally". A K-only follow-up isolates this — see
  below.
- **Indicator-reconstruction objective only.** This finding is
  about the
  [`--decoder cnn`](replay-decoders.md) indicator-reconstruction
  objective. The
  [factor walk-forward](factor-ssl-walkforward.md) shows the
  encoder doesn't beat the +0.012 deterministic-indicator baseline
  on cross-sectional return rank-IC; whether K-axis sufficiency
  also holds for a return-predictive head is independently
  unanswered.
- **Live-trading transfer pending.** No live-trading run has used a
  K=48 backbone end-to-end. The R² preservation is a strong
  encoder-side signal but doesn't replace the eventual
  paper-trading walk-forward.
- **MACD pathology.** MACD heads break in both arms in this and
  every other replay run since 2026-05-07. Pre-existing bug,
  tracked under
  [DWT compression follow-ups](../TODO/dwt-compression-followups.md#macd-head-pathology-replay).
  This finding is silent on MACD.

## Follow-up: 1D DWT on K only

Proposed clean isolation of the K-axis claim. Tracked separately in
[`TODO/dwt-compression-followups.md`](../TODO/dwt-compression-followups.md#k-only-dwt-isolation).

**Hypothesis.** A 1D Haar-L1 DWT applied **only** to the K axis
(leaving S=15 intact) preserves indicator-reconstruction R² within
±0.02 of the K=96 baseline on RSI / CCI / vol heads. If yes,
K-axis sufficiency is confirmed isolated. If R² drops on the
K-only arm (but the existing K+S 2D arm preserved R²), then S-axis
compression was doing more work than the K axis was, and the
"K-axis is sufficient" reading is wrong.

**Test design.**

- Same harness as
  [`apps/replay/scripts/modal/train_cnn_multihead.py`](https://github.com/sughodke/StockSurvey/blob/master/apps/replay/scripts/modal/train_cnn_multihead.py),
  same 295-ticker stooq_us_long pool, same K=96 / S=15 / 500
  steps / batch 8192.
- Three arms instead of two:
    1. `K=96, S=15` — uncompressed baseline (same as existing
       row 89).
    2. `K=48, S=15` — **K-only 1D DWT-L1**. New transform path:
       1D Haar L=1 along K only, S left untouched. Per-tile shape
       `(48, 15)` instead of `(48, 8)`.
    3. `K=96, S=8` — S-only 1D DWT-L1. The mirror arm. Per-tile
       shape `(96, 8)`.
- Eval: NVDA val R² (RSI / CCI / vol). MACD excluded (broken in
  every arm — separate bug).

**Decision rule.**

- Both 1D arms preserve R² → K-axis and S-axis compression are
  *each* individually sufficient; the 2D win was not specific to
  one. Architectural defaults can shorten K without losing
  anything. The 2D arm's confound becomes a non-issue.
- K-only preserves but S-only degrades → K-axis sufficiency
  confirmed; S carries signal that compression discards.
- K-only degrades but S-only preserves → flip the operational
  rule: shorten S, keep K=96.
- Both degrade → 2D compression's win was in the
  *joint* downsampling pattern (the corner of the
  `(K, S)` tile that survives keep-LL), and neither axis alone is
  sufficient. Most surprising outcome.

**Cost.** ~10 min Modal × 3 arms = ~30 min. Cheap; the CWT cache
warms once. Implementation lift: a `Compression.kind='dwt-1d-K'`
and `'dwt-1d-S'` mode with `axes=(-2,)` / `axes=(-1,)` instead of
`axes=(-2, -1)` in
[`packages/features/src/ss_features/compression.py`](https://github.com/sughodke/StockSurvey/blob/master/packages/features/src/ss_features/compression.py)
— ~10 LoC.

## Master walk-forward log

This finding does not introduce a new leaderboard row — it
extracts the K-axis reading from the existing
[2026-05-07 row](../leaderboard.md) ("2D Haar DWT-L1 keep-LL
CWT-tile compression vs uncompressed", verdict
[`partial-OOS`](../leaderboard.md#verdict-labels)).
The follow-up experiment will land its own row(s) on completion.
