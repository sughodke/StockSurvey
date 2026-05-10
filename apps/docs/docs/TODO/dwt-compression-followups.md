# DWT-compression follow-ups (post 2026-05-07 breakthrough)

The 2D DWT keep-LL compression of CWT tiles + relational fingerprints
landed for replay (`ss_features.Compression` + `ss-replay --compress
dwt`) and relational (`extract_fingerprints(..., compression=...)` +
`compress_levels` in `strategy_kwargs`). Phase-2 analog-kNN val
Sharpe moved from 1.07 → 1.11 with a 168→44 fp_dim shrink — but the
[full eight-arm walk-forward later
reversed that verdict](../findings/relational-dwt-failure.md), and the
canonical `Output/relational-analog.json` was rolled back to
full-resolution fingerprints. The replay-side
[DWT compression result is independent](../findings/replay-dwt-compression.md)
(SSL R², not a portfolio metric). Open threads:

## DCT zigzag-keep-top-k variant

Originally specified by the user but deferred — zigzag top-k yields a
flat coefficient vector (loses the 2D `(K, C)` tile shape the CNN
reshape relies on), so a flat-input decoder branch is needed. Stub
already in `Compression.kind='dct'` raises NotImplementedError. Plan:

- Add a `dct_zigzag_keep_top_k` helper to
  `ss_features.compression`. Use `scipy.fft.dctn` over the per-bar
  `(K, S)` tile, traverse coefficients in standard JPEG zigzag order,
  retain the first `keep_top_k`. Output is a flat
  `(n_dates, keep_top_k)` array per CWT-derived stack.
- Replay decoder: add a `--decoder-flat-input` mode (or new decoder
  type `cnn-flat`) that takes the concatenated flat coefficients and
  runs an MLP rather than reshape-then-conv. Disable the K/C reshape
  validation when this mode is active.
- Relational consumer: a flat-vector fingerprint plays nicely with
  the kNN code as-is — `extract_fingerprints` already returns `(n_dates,
  n_tickers, fp_dim)` regardless of how `fp_dim` was assembled. So
  the DCT path is *cheaper to wire into relational than into replay*.
- Test: the same Phase-2 head-to-head harness (`idea_b_analog_knn_dwt`)
  with arms `analog`, `analog-dwt-L1`, `analog-dct-k20`, `analog-dct-k40`.
  If DCT-top-k beats DWT-L1 on Sharpe, the win was about
  energy-concentration (DCT is closer to optimal for piecewise-smooth
  signals) rather than the multiresolution structure of the LL band.

## MACD head pathology (replay)

The 2026-05-07 Modal A/B (cwt-only bundle, baseline + dwt-L1) showed
MACD reconstruction R² ≈ −400 to −1200 across the (n,w) grid in
**both** arms. RSI / CCI / vol heads behaved sensibly (R² in the
0.45–0.92 range zero-shot on CSCO), but MACD alone exploded. Same
training config produced a usable MACD head in earlier replay runs.

Hypotheses to check (no need for Modal — should reproduce locally on
one ticker):

1. Scaling. MACD is unscaled price-difference, can range ±5..±50.
   The other heads are bounded ([0,100] for RSI/CCI, ~[0,0.05] for
   vol). When the multi-head loss sums un-normalised per-target
   MSEs, MACD's term dominates and the optimiser can blow up by
   over-correcting on a few outlier bars.
2. The `macd_fast_grid={8,12,16,24}` introduced in the multi-head
   training expects `slow=2*fast` and `signal=int(fast*3/4)`, but
   the un-conditioned MACD head reads the canonical `(fast=12,
   slow=26, signal=9)` triple. Possible mismatch between target
   computation and head conditioning.
3. Anchor-target wiring: the head supervised on the canonical target
   may be reading the FiLM-conditioned MACD line instead, which uses
   a different slow/signal ratio. The plot title says "MACD" but the
   target may be off-anchor.

Test plan: run `ss-replay AAPL --decoder cnn --targets macd
--cnn-steps 200` (single target, no FiLM grid) → confirm sensible R².
Then add `--macd-fast-grid 8,12,16,24` → reproduce the explosion. If
yes, the bug is in the grid conditioning path.

This is **blocking** on declaring the apps/replay → apps/factor SSL
backbone pipeline trustworthy for live use. Relational doesn't depend
on the backbone, so live trading on the relational checkpoints is
unaffected — but anyone consuming the replay-trained backbone via
`ss_features.load_backbone` for downstream factor scoring should be
aware that the MACD prefix of the `_meta` is currently unreliable.

## Wider-universe DWT validation

[Phase-2 wins for ideas A/B/C/D drop from Sharpe ~1.1 to ~0.4 when the
same code runs on the wider universe](../findings/relational-universe-shift.md).
The DWT-L1 finding (Sharpe 1.07 → 1.11) was measured *on top of* the
Phase-2-specific strategy. We don't know whether DWT helps, hurts, or
is neutral on the wider universe.

Plan: run the same 8-arm Modal entrypoint
(`relational_dwt_phase2.py`) but loaded against
`stooq_us_long` (or a min_history-filtered subset). The kNN inner
loop scales as O(n_dates × n_tickers × cand_pool); on 312 tickers
that's ~15× the Phase-2 work — each arm becomes ~30-45 min. Bump the
function timeout to 4h or split into separate function calls per
arm. Phase-2 entrypoint is the template; only the prep step
(`prep_phase2_prices.py`) needs to be replaced with a wide-universe
loader.

If DWT-L1 *also* wins on the wide universe, this becomes the first
finding in this codebase that beats the Phase-8 universe-degradation
result — would justify a wide-universe canonical checkpoint, not just
Phase-2. If it ties or loses, the result is mega-cap-specific and the
canonical checkpoints stay Phase-2-only.

## K-only DWT isolation

The
[length-axis sufficiency finding](../findings/replay-length-axis-compression.md)
extracts a K-axis-specific reading from the existing 2D DWT
result, but the underlying experiment compressed both K and S
simultaneously. The K-axis claim ("K=96 was over-provisioned for
indicator reconstruction") is suggestive, not isolated.

**Falsifiable hypothesis.** A 1D Haar-L1 DWT applied **only** to
the K axis (`S=15` left intact) preserves NVDA val R² within ±0.02
of the K=96 baseline on RSI / CCI / vol heads.

**Test design.** Three Modal-T4 arms on the same 295-ticker
stooq_us_long pool, K=96 / S=15 / 500 steps / batch 8192:

1. `(K=96, S=15)` — uncompressed baseline.
2. `(K=48, S=15)` — **K-only 1D DWT-L1**.
3. `(K=96, S=8)` — S-only 1D DWT-L1 (mirror arm).

Eval: NVDA val R² and CSCO zero-shot peak R² on RSI / CCI / vol.
MACD excluded (broken in every arm; tracked separately above).

**Decision rule.**

| Outcome                                    | Reading                                                                    |
|--------------------------------------------|----------------------------------------------------------------------------|
| Both K-only and S-only preserve R²         | each axis individually sufficient; 2D win not specific to either           |
| K-only preserves, S-only degrades          | K-axis sufficiency confirmed; default to shorter K                          |
| K-only degrades, S-only preserves          | flip the operational rule — shorten S, keep K=96                           |
| Both degrade                               | 2D win was in the joint downsampling pattern; neither axis alone enough   |

**Implementation.** Add `Compression.kind='dwt-1d-K'` and
`'dwt-1d-S'` modes to
[`packages/features/src/ss_features/compression.py`](https://github.com/sughodke/StockSurvey/blob/master/packages/features/src/ss_features/compression.py)
that pass `axes=(-2,)` or `axes=(-1,)` instead of the current
`axes=(-2, -1)`. ~10 LoC. Modal cost ~30 min total (~10 min/arm,
CWT cache shared across arms).

**Priority.** Cheaper than the wider-universe DWT validation
below, lands a clean architectural answer regardless of the
outcome (every cell in the decision rule has an actionable
reading). Run before defaulting any new backbone npz to a shorter
K.

## Non-Haar wavelet sweep

Haar is the shortest filter (length 2) and produces the blockiest LL
band. Smoother wavelets (db2, sym4, coif1) have longer filters that
blur the LL across more neighbouring time/scale cells before
downsampling — could either preserve more signal (if the
discontinuities the Haar LL captures are noise artifacts) or destroy
the signal Haar was usefully picking up.

One-arm follow-up on Phase-2: sweep `wavelet ∈ {haar, db2, sym4,
coif1}` at L=1 with the same harness. Cheap (~7 min for 4 arms with
the CWT cached). Only worth doing once the wide-universe result and
the rebal-days sweep have run, since those are higher-information
experiments.
