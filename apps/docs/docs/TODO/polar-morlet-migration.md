# Migrate non-research apps to the polar Morlet input bundle

The polar Morlet + Gaussian + log-L2 amplitude bundle is the canonical
SSL/CNN input for **`apps/replay`** (already on it as of 954a88a) and
**`apps/factor`** (consumes the resulting backbone npz via
`ss_features.load_backbone` — no code change needed, the new npz
schema is backward-compatible at the loader layer). The remaining
non-research consumers of `ss_wavelets.causal_cwt` (real Ricker)
should migrate so the workspace converges on a single wavelet family.

Two important caveats up front:

1. **"Bundle" is a CNN-input concept.** It's a `(K, F=7*n_scales)` lag-
   windowed feature stack consumed by a 1-D conv backbone. Apps that
   use CWT as a *scoring primitive* (regime divergences, relational
   kNN distance) don't lag-window — they compute one-shot per-bar
   scalogram outputs and apply scalar reductions. Their migration is
   "swap the wavelet, not the bundle": replace `causal_cwt` with
   `causal_cwt_morlet`, take `np.abs(coeffs)**2` as the power signal,
   optionally add `causal_cwt_gaussian` over cumulative log-returns
   as a trend channel.
2. **Each migration is a strategy change, not a refactor.** The
   prior canonical Sharpe numbers (regime walk-forward 0.6, the six
   [relational scoreboard winners 1.07–1.13 on Phase-2](../findings/relational-universe-shift.md))
   were tuned against Ricker-based scalograms. Switching to Morlet
   `|c|^2` shifts the per-scale frequency response (Morlet at
   `omega0=6` is narrowband at `1/scale`; Ricker is broadband around
   `1/scale`). **Every migrated checkpoint must re-pass walk-forward
   before live deploy** — the
   [Phase-2 analog kNN Morlet attempt already failed this gate](../findings/relational-morlet-failure.md).

Public API the migrations should consume:

```python
from ss_features import (
    TickerData,                       # per-ticker container
    build_features_and_targets,       # CNN-input bundle (replay only)
    compute_scalogram_polar,          # 4-tuple (|c|, cos, sin, g) per ticker
    causal_polar_morlet_matrix,       # matrix-form (C*S, T, N) panel
                                      # for relational / regime use
    load_ticker,                      # one-shot loader
    CHANNELS_PER_SCALE,               # = 7 (CNN bundle)
    RELATIONAL_CHANNELS_PER_SCALE,    # = 4 (kNN-distance bundle)
    channels_per_lag,                 # n_scales -> 7 * n_scales
)
from ss_wavelets import (
    causal_cwt,                       # real Ricker (kept; do not delete)
    causal_cwt_morlet,                # complex Morlet, bandpass + phase
    causal_cwt_gaussian,              # real Gaussian, lowpass / trend
    DEFAULT_MORLET_OMEGA0,            # = 6
    KERNEL_HALF_EXTENT,               # = 3
)
```

Real Ricker `causal_cwt` stays in `ss_wavelets` as the legacy primitive
— don't delete it. Research scripts and the parked v1 app still
reference it.

Note that the matrix-form helper exposes **4 channels per scale**
(`|c|, cos(arg), sin(arg), g`), not 3. The Gaussian companion `g` is
included because for kNN-distance scoring it carries the
trend / level signal the Morlet bandpass kernels structurally cannot.
The redundant `|c|^2` and `g^2` channels (present in the CNN bundle
as monotone convenience nonlinearities) are intentionally dropped —
they add nothing to L2 / cosine distance once `|c|` and `g` are in
the vector. fp_dim therefore goes `S * w → 4 * S * w`, not the 3×
the original plan estimated.

## `apps/regime` (live trading) — STRATEGY CHANGE

Files:

- `apps/regime/src/regime/trainer.py:67` — `from ss_wavelets import
  KERNEL_HALF_EXTENT, causal_cwt, precompute_windows`. Used by
  `weights_regime` and `weights_scalogram` for per-scale CWT power
  (recent-vs-historical window divergence).
- `apps/regime/src/regime/inference.py:28` — same import; used by
  the live scoring path.
- `apps/regime/src/regime/persist.py` — `Checkpoint` dataclass should
  gain a `wavelet: str` field (default `"ricker"` for back-compat;
  set to `"morlet"` after migration). Consume in `inference.py` so
  live can mirror train-time wavelet choice.
- `apps/regime/research/optimize_regime.py`,
  `apps/regime/research/backtest_bt.py`,
  `apps/regime/research/backtest_ranking.py` — same import; wire the
  wavelet flag through.

Migration:

- Replace `causal_cwt(prices, scales, lookback)` with
  `np.abs(causal_cwt_morlet(prices, scales, lookback)) ** 2` to get
  Morlet power. The downstream `precompute_windows` signature is
  unchanged.
- Add `--wavelet {ricker,morlet}` CLI flag on `regime train` and
  thread to the trainer / persist into the checkpoint.

Validation gate:

- Run the existing controlled walk-forward eval (`Output/regime-eval-
  rawclose-kernel3.{log,json}` template) on Stooq 2010-2024, 20
  trials per window, both wavelet arms. The Morlet arm must match or
  beat the Ricker arm's median val Sharpe (currently +0.15, mean
  +0.07). If Morlet regresses, the migration is rejected for live —
  Morlet stays research-only.
- After validation, regenerate `Output/regime-v1.json` with the
  Morlet checkpoint and confirm `regime live --dry-run` produces
  weights consistent with the chosen target portfolio.

Risk: the regime trainer's strongest signal is on long scales —
[126d won 48% of scale weight in the JAX-Adam
run](../findings/regime-baselines.md#jax-differentiable-optimizer-now-removed-finding-preserved).
Morlet narrowband behaviour at long scales may either sharpen or noise
that signal — empirical question.

## `apps/relational` (live trading) — STRATEGY CHANGE, all six checkpoints

### Status — analog migrated, gate FAILED, halt extension — 2026-05-09

- ✅ `ss_features.causal_polar_morlet_matrix` — matrix-form polar
  bundle helper added (4 channels per scale, see API note above).
- ✅ `RelationalCheckpoint.wavelet` field — defaults `'ricker'` for
  back-compat; validates against `SUPPORTED_WAVELETS = ('ricker',
  'morlet')`.
- ✅ `relational.scalogram_cache.load_or_compute_cwt(..., wavelet=)`
  — accepts both kernels; cache key hashes the wavelet name so the
  existing Ricker `.npz` cache is preserved unchanged. Morlet panels
  land at `cwt-morlet-<hash>.npz` next to `cwt-ricker-<hash>.npz`.
- ✅ `analog_knn.{analog_knn_scores, analog_knn_scores_fast,
  weights_regime_analog}` — `wavelet=` arg plumbed through; default
  `'ricker'` keeps the Phase-2 canonical winner unchanged.
- ✅ `relational.inference._build_weights_panel` — forwards
  `cp.wavelet` to the analog dispatch and **raises
  `NotImplementedError` for the other five strategies** if a
  checkpoint pins `wavelet='morlet'`. Paper-trade-safe: a stray
  Morlet checkpoint loaded against an unmigrated strategy fails
  loudly, not silently.
- ✅ Walk-forward Ricker-vs-Morlet A/B harness:
    - Local: `python -m
      relational.research.idea_b_analog_knn_morlet_walkforward
      --data-dir ./StooqData`
    - Modal: `uvx modal run apps/relational/scripts/modal/
      relational_morlet_phase2.py` (after the
      `prep_phase2_prices.py` prep step)
- ❌ **Validation gate FAILED for `analog`** — see
  [findings/relational-morlet-failure](../findings/relational-morlet-failure.md)
  for the full numbers and mechanism hypotheses. Train +0.221 /
  val −0.310 vs Ricker baseline (Ricker: train 1.019 → val 1.146;
  Morlet: train 1.240 → val 0.836). Classic train>val sign-flip
  overfit — the polar bundle's extra channels improve fit on
  2013-2020 but hurt OOS in 2021-2025. Operational verdict:
  `Output/relational-analog.json` stays on Ricker; do not regenerate.
- ⏸️ **Halt extension to other strategies.** The original plan to
  plumb wavelet through `empirical`, `gmm`, `farthest`,
  `diversified`, `velocity` is deferred. Expectation given the analog
  result: most or all will exhibit the same overfit pattern (the
  bundle adds DOF that the small Phase-2 universe doesn't have data
  to constrain). A cheaper next experiment — rerun the analog A/B on
  `stooq_us_long` (N=312, 15× larger candidate pool) — is the
  recommended way to validate the overfit hypothesis before
  committing more plumbing work.
- N/A Canonical checkpoint regen — `Output/relational-{strategy}.json`
  files all stay on Ricker.

Migration scope deviated from the prescribed "regime first" decision
order — the user asked the relational migration first since `analog`
is Phase-2's canonical winner and the SSL bundle is independently
useful there. Regime is still pending; given that the analog gate
failed, the regime migration's expected value drops too — Morlet
provides phase information the SSL CNN can use, but for the regime
trainer's KL/JS divergence math the only thing that matters is the
power spectrum, and Morlet `|c|^2` will drift on the same kind of
narrowband / DOF arguments that broke analog. Regime migration is
not blocked, but should be approached with a similar
"single-strategy A/B before committing canonical" stance.

Files (every CWT-touching scoring module):

- `apps/relational/src/relational/scalogram_cache.py:30,66` — the
  Modal-volume-cached `causal_cwt` wrapper. **Migration entry point**:
  add a `wavelet` parameter (`"ricker"` | `"morlet"`) and key the
  cache by it so the existing Ricker cache doesn't get clobbered.
- `apps/relational/src/relational/regime_velocity.py:46,84` —
  `from ss_wavelets import causal_cwt`; thin wrapper over scalogram_cache.
- `apps/relational/src/relational/scoring.py:22` —
  `from ss_wavelets import causal_cwt, precompute_windows`. Core
  fingerprint + power computation.
- `apps/relational/src/relational/fingerprints.py:43` (docstring
  reference + actual consumer downstream) — fingerprint primitives
  consume scalogram cache output.
- `apps/relational/src/relational/empirical_sectors.py:50`,
  `empirical_sectors_gmm.py:58`, `analog_knn.py`, `farthest.py:?`,
  `diversify.py:34`, `regime_velocity.py` — each `weights_*` builder
  consumes `precompute_windows` output. No direct `causal_cwt`
  imports here, but each transitively depends on the scalogram via
  the cache.

Migration:

- ✅ Add `wavelet: str = "ricker"` to `RelationalCheckpoint` (in
  `apps/relational/src/relational/persist.py`) so live mirrors
  train-time choice. Done in commit `c479ca3`.
- 🟡 Plumb `wavelet` from `RelationalCheckpoint` through
  `scalogram_cache.compute_or_load` → all six `weights_*` builders.
  Cache + `analog` are done; the other five `weights_*` builders are
  still on the legacy `(S, T, N)` Ricker shape and need their own
  fingerprint paths to handle the `(C*S, T, N)` Morlet panel.
- ✅ For Morlet, the fingerprint shape decision landed as
  **`np.concatenate([|c|, cos(arg), sin(arg), g], axis=0)`** per
  scale (the matrix-form helper does this). Fingerprint dim grows
  `S * w → 4 * S * w` (~4×), which is the channel cost of trading
  Ricker's signed bandpass coefficient for the polar Morlet bandpass
  pair plus the Gaussian lowpass companion. **No DWT compression
  pinned** — left full-resolution by default; the existing
  `compression=Compression(kind='dwt', levels=L)` knob is still
  available per-call if a future eval shows the ~4× kNN cost
  dominates.

Validation gate:

- 🟡 **Per-strategy walk-forward A/B** (replaces the original "8-arm
  rerun on the existing DWT Modal entrypoint" plan — that script
  pre-dates the wavelet field and would have to be re-templated
  per-strategy anyway). The new pattern is one Modal entrypoint per
  strategy:
    - `apps/relational/scripts/modal/relational_morlet_phase2.py`
      runs `analog-ricker` vs `analog-morlet` on the Phase-2 21-
      ticker pool with the canonical knobs (top-10, rebal-20d, 10bps,
      lookback 120, fp_window 21, scales [5,7,10,12,21,26,50,90],
      k=50, h=20, min_sep=21, pool_mode=cross_ticker). Outputs land
      at `Output/relational-morlet-phase2-{equity.png, stats.txt,
      walkforward.csv, walkforward.txt}`. **Status: in flight as of
      2026-05-09.** Bar to clear: val Sharpe ≥ Ricker baseline of
      1.146.
    - When the next strategy is plumbed, copy the script and swap
      `analog_knn_scores_fast` for that strategy's score function;
      the segmentation + plot code is shared.
- After per-strategy validation,
  `apps/relational/scripts/build_canonical_checkpoints.py` gains a
  new entry pinning that strategy on Morlet (separate file from the
  existing Ricker JSON, named e.g.
  `Output/relational-{strategy}-morlet.json`, so live deploys can
  point at either explicitly). The Ricker JSON stays as the prior
  canonical until the operator chooses to swap.

Risk: this is six independent strategy revalidations. Almost certainly
some will regress — the Phase-2 wins are mega-cap-specific and
narrow; Morlet's narrowband response may erase whatever specific
spectral feature each scoring family was picking up. Plan for at
least one or two of the six to stay on Ricker permanently.

## `apps/notebook` scalogram visualizers — VIZ ONLY, low priority

Files:

- `apps/notebook/src/ss_notebook/scalogram.py:45` — `ss-scalogram`
  CLI's static composite figure.
- `apps/notebook/src/ss_notebook/scalogram_video.py` — `ss-scalogram-
  video` CLI's day-by-day mp4. (Imports `compute_scalogram_power`
  from the sibling module; no direct `causal_cwt` here.)

Migration:

- Add `--wavelet {ricker,morlet}` flag, default `morlet` once the
  regime/relational migrations land. For Morlet, plot `|c|^2` as the
  heatmap (matches the bandpass-power semantics Ricker plots use).
- Keep Ricker as a fallback so prior scalogram figures in
  `Output/` can be reproduced verbatim.

Risk: low. No live consumer; figures are diagnostic. Defer until
after regime + relational migrations confirm Morlet is the workspace
default.

## Out of scope for this migration

- `apps/v1/` — parked legacy app; its own `v1/util/indicators.py` is
  not the canonical implementation. Leave on Ricker.
- `apps/lie/` — research scaffolding; mostly indicator-shape based,
  doesn't use the CWT bundle.
- `apps/factor/src/factor/indicator_features.py` — uses the
  hand-crafted indicator path (`IndicatorGridConfig`), not CWT.
  Already a separate input pipeline; no migration needed.

## Decision order

1. ~~**Regime first**~~ — *deviated from*. The relational `analog`
   migration ran first (commit `c479ca3`); regime is still pending.
   The original rationale (regime is simpler, one checkpoint, one
   strategy) still holds for any *future* migration that hasn't been
   started yet.
2. **Relational, one strategy at a time.** Analog is the test case;
   the other five (`empirical`, `gmm`, `farthest`, `diversified`,
   `velocity`) follow once the analog A/B clears its gate. Expect
   heterogeneous results — Phase-2 wins are mega-cap-specific and
   narrow; Morlet's narrowband response may erase whatever specific
   spectral feature each scoring family was picking up.
3. **Regime** — pending the same per-strategy A/B template adapted
   to its Optuna walk-forward.
4. **Notebook scalograms last** — viz, not blocking anything else.

If a strategy fails its gate, that strategy stays on Ricker
permanently — the wavelet field on the checkpoint makes this trivial
to encode (one `'ricker'` and one `'morlet'` checkpoint coexist for
the same strategy). Treat polar Morlet as the *default for new
strategies*, not as a forced replacement for working ones.
