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

**Polar Morlet caveat (added 2026-05-10).** The original plan
below pre-dates the
[polar Morlet rewrite](polar-morlet-migration.md) and treats the
input as if it were a single real-valued scalogram. The current
bundle is 7 channels per scale: `|c|`, `|c|²`, `cos(arg)`,
`sin(arg)`, `g`, `g²`, `log-L2-amp`. Five are amplitudes (smooth,
bounded, well-behaved under linear smoothing); two are **phase**.
Haar LL is just a 2-tap mean — `mean(cos(θ_t), cos(θ_{t+1})) ≠
cos((θ_t + θ_{t+1})/2)`. At low scales, where phase rotates fast
within a 2-bar K window, the LL of the phase channels is a
mixture that represents neither sample. Compressing all 7
channels uniformly via Haar LL asks the phase channels to do
something they can't do; the 2D DWT result on the books got
bailed out by the channel-mix ratio (5 amp vs 2 phase) without
ever isolating per-channel reconstruction quality.

Three handlings, cheapest to most invasive:

1. **Amp-only compression**, leave `cos`/`sin` at full K.
   Concedes K-shrink is partial; concat compressed-amp +
   uncompressed-phase along the channel axis is awkward for the
   CNN reshape but doable.
2. **Complex-domain compression.** Re-derive `c = |c|·(cos +
   i·sin)` per scale, run `pywt.wavedec` (handles complex), keep
   LL, re-derive `|c|`, `|c|²`, `cos(arg)`, `sin(arg)` from the
   compressed complex coeff. Complex Haar LL = complex mean,
   which is the *right* averaging operation for both magnitude
   and phase jointly. Mathematically clean answer.
3. **Compress all 7 uniformly anyway** (the original plan), but
   stratify the diagnostic R² **per-channel** so the phase
   damage is at least visible. Cheap, but if "R² preserved" is
   just amp channels carrying phase channels we draw the wrong
   conclusion again.

Option 2 is the only one that actually answers "is the K-axis of
the *polar Morlet CWT* compressible".

**Pre-experiment diagnostic (added 2026-05-10) — run before any
Modal training.** A pure-numpy round-trip CWT-reconstruction
test isolates whether the input is even compressible along K
without burning Modal time on a CNN training loop:

1. Pick one ticker (NVDA), compute the polar bundle via
   `ss_features.compute_scalogram_polar`.
2. Compress complex coeffs along K with Haar L=1 (option 2
   above).
3. `pywt.waverec` back to original K with detail bands
   zero-filled.
4. Report per-scale R² and RMSE of reconstructed `|c|`, `arg`
   vs original. Fail-fast threshold: R² < 0.9 at the smallest
   scale → option 2 doesn't preserve enough; option 1
   (amp-only) becomes the plan.

Also emit a 4-panel side-by-side figure (original `|c|` /
reconstructed `|c|` / original `arg` / reconstructed `arg`) at
one mid-scale and one low-scale. The
[parent finding page](../findings/replay-length-axis-compression.md)
embeds 8 downstream-target images but **zero** input-CWT
visualizations, so any future K-axis claim has no visual
grounding for the input transformation; this plot fills the gap.
Lands in `Output/` and folds into the finding page.

**Falsifiable hypothesis.** A 1D Haar-L1 DWT applied **only** to
the K axis (`S=15` left intact, complex-domain per option 2)
preserves NVDA val R² within ±0.02 of the K=96 baseline on
RSI / CCI / vol heads, *and* per-channel reconstruction R² on
`cos(arg)` / `sin(arg)` stays above 0.9 at every scale.

**Test design.** Three Modal-T4 arms on the same 295-ticker
stooq_us_long pool, K=96 / S=15 / 500 steps / batch 8192:

1. `(K=96, S=15)` — uncompressed baseline.
2. `(K=48, S=15)` — **K-only 1D DWT-L1, complex-domain**
   (option 2). Per-tile shape `(48, 15)` after re-deriving the
   7 polar channels from the LL complex coeff.
3. `(K=96, S=8)` — S-only 1D DWT-L1 (mirror arm). S-axis is
   the scale axis — no phase-vs-amp confound applies.

Eval: NVDA val R² and CSCO zero-shot peak R² on RSI / CCI / vol,
plus per-channel CWT-reconstruction R² from the diagnostic above
for arm 2. MACD excluded (broken in every arm; tracked
separately above).

**Decision rule.**

| Outcome                                    | Reading                                                                    |
|--------------------------------------------|----------------------------------------------------------------------------|
| Both K-only and S-only preserve R²         | each axis individually sufficient; 2D win not specific to either           |
| K-only preserves, S-only degrades          | K-axis sufficiency confirmed; default to shorter K                          |
| K-only degrades, S-only preserves          | flip the operational rule — shorten S, keep K=96                           |
| Both degrade                               | 2D win was in the joint downsampling pattern; neither axis alone enough   |

**Implementation.** Add `Compression.kind='dwt-1d-K'` and
`'dwt-1d-S'` modes to
[`packages/features/src/ss_features/compression.py`](https://github.com/sughodke/StockSurvey/blob/master/packages/features/src/ss_features/compression.py).
For `dwt-1d-S` (no phase confound): pass `axis=-1` to
`pywt.wavedec`. For `dwt-1d-K` per option 2: build the complex
coeff stack inside `cwt_features.build_features_and_targets`
*before* the polar derivation, run `pywt.wavedec(complex,
axis=-2)`, keep LL, then re-derive the 7 polar channels from
the compressed complex coeff. The polar derivation move means
~30-40 LoC, not the original ~10 estimate. Modal cost ~30 min
total (~10 min/arm, CWT cache shared across arms).

**Priority.** Run the pre-experiment diagnostic *first* — it's
~30 LoC of pure numpy, no GPU, ~5 min. If reconstruction R² is
clean at every scale, the Modal training run is justified. If
the smallest scales fail, fall back to option 1 (amp-only) or
shrink the scale grid before the training arm. Cheaper than the
wider-universe DWT validation below.

**K' length determinant.** `K' = ceil(K / 2^L)` under
`mode='periodization'` (current default). `K=96, L=1 → K'=48`;
`L=2 → 24`; `L=3 → 12`. Wavelet family is irrelevant to output
length under periodization. Lower bound: K' must be ≥ the
encoder's effective K-receptive field; verify before pushing
past `L=1`.

## Lossless polar-CWT compression

The K-only DWT plan above is **lossy by construction** — it
keeps only the LL band of the DWT and discards detail
coefficients. The
[parent finding](../findings/replay-length-axis-compression.md)
and the existing
[2D DWT result](../findings/replay-dwt-compression.md) are
silent on this; both justify the compression entirely on
downstream R² preservation, never on the round-trip property.
Worth scoping the lossless ceiling separately so the lossy
choice is at least informed.

**Channel-axis lossless (free, ~2.3× immediate win).** The
7-channel polar bundle has 4 redundant channels:

- `|c|²` derives from `|c|`.
- `cos(arg)² + sin(arg)² = 1` — phase is one DOF, not two.
- `g²` derives from `g`.
- `log-L2-amp` derives from `|c|` over the K-window.

Independent DOF count is **3**, not 7. Equivalently, store
`(c = Re + i·Im, g)` — 2 real + 1 real = 3 reals per scale, all
7 channels exactly recoverable. The CNN could be re-trained to
consume the 3-channel form directly with no loss of input
information. Implementation: stop materializing the derived
channels in
[`cwt_features.build_features_and_targets`](https://github.com/sughodke/StockSurvey/blob/master/packages/features/src/ss_features/cwt_features.py)
and add a `(Re, Im, g)` consumer path in the encoder. Storage
shrinks 7→3; no fitting per ticker; truly lossless.

**K-axis lossless (Shannon-bounded, per-scale).** The CWT at
scale `s` is bandlimited along K to ~`±1/s` cycles per bar
(wavelet uncertainty / Heisenberg-Gabor). By Nyquist, the K
axis at scale `s` can be subsampled by ~`s/2` losslessly under
sinc reconstruction:

| Scale `s` | Max lossless K decimation |
|-----------|---------------------------|
| 2         | 1× (no shrink possible)   |
| 8         | 4×                        |
| 32        | 16×                       |
| 64        | 32×                       |

This is the structural insight behind the dyadic Mallat DWT —
why the *full-coefficient* DWT is invertible to float
precision. For the polar Morlet's non-dyadic scale grid,
per-scale polyphase decimation at `~s/2` is the principled
lossless K-shrink. Total storage drops from `K · n_scales` to
`K · Σ(2/s)`. On the current geometric scale grid this is
~3-4× lossless K-shrink; on a linear grid the win shrinks
because small scales (which dominate K storage) can't be
compressed.

**Combined ceiling.** Channel collapse (~2.3×, free) ×
per-scale Nyquist along K (~3-4× on the current grid) →
**~7-9× lossless** total for the polar Morlet bundle, before
any generic byte-level compression (blosc/zlib on float32 adds
another ~2× for storage only).

**Vs. the lossy K-only plan.** The Haar-LL keep-only along K
is ~2× per axis but lossy. Lossless along K via per-scale
Nyquist gives variable shrink (1× at the small end, 32× at the
large end) and requires polyphase resampling instead of a
single Haar pass.

**Falsifiable hypotheses (two, separable).**

1. *Channel collapse.* Re-training the encoder on `(Re, Im, g)`
   instead of the 7-channel polar bundle preserves NVDA val R²
   within ±0.01 on RSI / CCI / vol heads (since the input
   information content is identical and the encoder is
   capacity-unconstrained at current sizes).
2. *Per-scale Nyquist K decimation.* Replacing the uniform
   K=96 input with per-scale-decimated K (variable per scale)
   and reconstructing via sinc interpolation before the
   encoder preserves NVDA val R² within ±0.02 on the same
   heads.

**Test design.** Diagnostic-first, training-second:

1. Pure-numpy: per-scale Nyquist decimation + sinc
   reconstruction on NVDA's polar coeffs; report per-scale R²
   of reconstructed `c` vs original. If R² > 0.99 at every
   scale, the bandlimit assumption holds and the training arm
   is justified.
2. Modal arms (only if diagnostic passes):
    - `7-channel, K=96` — current baseline.
    - `(Re, Im, g)` 3-channel, K=96 — channel-collapse only.
    - `(Re, Im, g)` 3-channel, per-scale K — both losses
      together.

**Priority.** The channel-collapse pre-screen is ~5 min of
work and immediately tells us whether the encoder is using the
redundant channels (it shouldn't be, but worth confirming).
Run after the K-only DWT diagnostic above (overlapping
infrastructure). The per-scale Nyquist arm is a bigger lift
(polyphase resampling + variable-K input plumbing) and only
worth doing if channel-collapse and K-only DWT both come back
clean.

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
