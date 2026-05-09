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
   relational scoreboard winners 1.07–1.13 on Phase-2) were tuned
   against Ricker-based scalograms. Switching to Morlet `|c|^2`
   shifts the per-scale frequency response (Morlet at `omega0=6` is
   narrowband at `1/scale`; Ricker is broadband around `1/scale`).
   **Every migrated checkpoint must re-pass walk-forward before
   live deploy.**

Public API the migrations should consume:

```python
from ss_features import (
    TickerData,                    # per-ticker container
    build_features_and_targets,    # CNN-input bundle (replay only)
    compute_scalogram_polar,       # 4-tuple (|c|, cos, sin, g)
    load_ticker,                   # one-shot loader
    CHANNELS_PER_SCALE,            # = 7
    channels_per_lag,              # n_scales -> 7 * n_scales
)
from ss_wavelets import (
    causal_cwt,                    # real Ricker (kept; do not delete)
    causal_cwt_morlet,             # complex Morlet, bandpass + phase
    causal_cwt_gaussian,           # real Gaussian, lowpass / trend
    DEFAULT_MORLET_OMEGA0,         # = 6
    KERNEL_HALF_EXTENT,            # = 3
)
```

Real Ricker `causal_cwt` stays in `ss_wavelets` as the legacy primitive
— don't delete it. Research scripts and the parked v1 app still
reference it.

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

Risk: the regime trainer's strongest signal is on long scales (126d
win 48% scale weight per the JAX-Adam finding in CLAUDE.md). Morlet
narrowband behaviour at long scales may either sharpen or noise that
signal — empirical question.

## `apps/relational` (live trading) — STRATEGY CHANGE, all six checkpoints

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

- Add `wavelet: str = "ricker"` to `RelationalCheckpoint` (in
  `apps/relational/src/relational/persist.py`) so live mirrors
  train-time choice.
- Plumb `wavelet` from `RelationalCheckpoint` through
  `scalogram_cache.compute_or_load` → all six `weights_*` builders.
- For Morlet: power = `np.abs(coeffs) ** 2` (signed `coeffs` is
  meaningless for Morlet — phase lives in `arg`, magnitude in `|c|`).
  Fingerprints currently use signed Ricker coefficients; the Morlet
  equivalent is `np.stack([|c|, cos(arg), sin(arg)], axis=...)` per
  scale, which roughly triples fingerprint dim. Decide if the
  fingerprint should compress this back via `Compression(kind='dwt')`
  or stay full-resolution.

Validation gate:

- Re-run the 8-arm walk-forward Modal entrypoint
  (`apps/relational/scripts/modal/relational_dwt_phase2.py`) on each
  of the six canonical strategies, both Ricker and Morlet arms, on
  the Phase-2 21-ticker pool. The val Sharpe must match or beat the
  current canonical (1.07–1.13).
- After validation, regenerate
  `apps/relational/scripts/build_canonical_checkpoints.py` outputs
  with the Morlet variants. Existing
  `Output/relational-{empirical,gmm,analog,farthest,diversified,velocity}.json`
  stay on Ricker until the per-strategy walk-forward signs off.

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

1. **Regime first** (simpler, one strategy, one checkpoint). If the
   Morlet arm clears the existing eval bar, we have evidence the
   migration won't kill production.
2. **Relational second**, but expect heterogeneous results across the
   six strategies. Migrate them one at a time; let each pass its own
   walk-forward before swapping the canonical JSON.
3. **Notebook scalograms last** — viz, not blocking anything else.

If regime fails the gate, halt the migration and treat polar Morlet
as a research-only primitive available via direct
`ss_wavelets.causal_cwt_morlet` import. The new bundle stays
canonical only for the SSL trainer in that scenario.
