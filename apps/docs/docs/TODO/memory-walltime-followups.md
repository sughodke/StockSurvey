# Memory and wall-time audit — follow-ups

Audit ran 2026-05-09 across the active apps + shared `numpy`/tinygrad
packages, motivated by the OOM cascade in
`apps/factor/scripts/modal/train_ssl_walkforward.py` (commits cac94e1
→ 3451900 → eef4e52: 128 GB → 192 GB → drop redundant copies → polar
Morlet pivot). The seven highest-confidence wins shipped in the same
session as the audit; the three architectural items required to scale
the supervised-`cnn` walkforward to 3000+ tickers (A1, A5, B1) shipped on the
follow-up. The items below are everything that did **not** ship.

Group A items are "likely wins, measure before fixing." Group B items
are architectural — flag for discussion, don't apply without a
design pass.

## Done

The 3000-ticker push (2026-05-09) shipped the three items required
to lift the OOM cliff:

* **A5 — thread-parallel CWT.** `causal_cwt` / `causal_cwt_morlet`
  / `causal_cwt_gaussian` in `packages/wavelets` now run their
  per-scale `fftconvolve` loop under a `ThreadPoolExecutor` (default
  `min(8, n_scales, cpu_count)`, override via `SS_WAVELETS_CWT_THREADS`).
  scipy releases the GIL inside the C kernels, so independent scales
  parallelize linearly. Hits every CWT site in the workspace.
* **B1 — rebal-subsampled alignment.** New
  `align_tickers_at_rebal(tickers, K, F, rebal_days)` in
  `apps/factor/src/factor/data.py` materializes features only at
  rebal positions (`(D', N, K, F)`) instead of the full daily panel
  (`(D, N, K, F)`). `precompute_inputs` switched to it. At 3000
  tickers / D=6500 / K=96 / F=105 this drops aligned-features
  residency from ~786 GB to ~39 GB and the encoder pass from
  ~470 GB to ~22 GB latent. `predict()` keeps the daily-features
  path for held-out evaluation; callers who want daily-frequency
  scores use `align_tickers` directly.
* **A1 — Modal worker disk-handoff.** `_load_one_ticker` in
  `train_ssl_walkforward.py` now writes each ticker's features
  to `/tmp/factor-features/<ticker>.npy` and returns a small stub
  with the path. Parent reconstitutes a `TickerData` with
  `features = np.load(path, mmap_mode='r')`; subsequent slicing in
  `align_tickers_at_rebal` only pages in the rebal rows
  (~13 MB / ticker, sparse). IPC residency drops from ~262 MB ×
  N tickers to a few KB × N. Container disk needs to hold the
  per-ticker `.npy` spool concurrently (~262 MB / ticker — ~78 GB
  at 297, ~786 GB at 3000); past a few hundred tickers configure
  a Modal Volume or larger ephemeral disk.

Combined working-set at 3000 tickers (estimated): aligned features
~39 GB + repr_rb ~22 GB + prices/valid <1 GB ≈ **65 GB resident**,
fits in the 192 GB envelope with margin for tinygrad compilation.

### A2. Optuna recomputes `causal_cwt` per trial in regime trainer

`apps/regime/src/regime/trainer.py:582-594` — the Optuna objective
rebuilds the full CWT cube on every trial. CWT depends only on
`(scales, lookback)` and the prices; everything else the search
varies (`top_n`, `n_tail`, `divergence`, `weights_kind`) reads off
the same cube. With 50 trials per window and a small set of
distinct `(scales, lookback)` combinations, the rebuild factor is
empirically >5×.

**Hypothesis to test:** wrap `causal_cwt(prices, scales, lookback)`
in a closure-scoped `lru_cache(maxsize=8)` keyed on
`(tuple(scales), lookback)`. `precompute_windows` similarly depends
only on `(power, lookback, n_tail)` and could share the same cache
discipline.

**Why deferred:** Optuna's sampler can drift the searched
`(scales, lookback)` enough that hit rate is lower than naive
counting suggests. Need to log the cache hit rate on a real run
before committing — if hit rate is <40%, the fix isn't worth the
closure-state complexity.

**Test design:** instrument with a `print(cache_info())` after each
window; wall-time delta should be ≥2× on a 50-trial sweep if hit
rate is ≥60%.

### A3. Aggregate-CWT cache key in empirical scorer is effectively unique-per-call

`apps/relational/src/relational/empirical_sectors.py:270,292` —
the aggregate-CWT path hashes `agg_prices` (a *derived* array — the
cluster aggregates change every refit), so the cache key is
guaranteed unique. The cache hit rate on `agg_coeffs` is ~0%
across runs that change anything in the empirical clustering, and
the SHA-256 over `agg_prices.tobytes()` is itself a measurable
overhead.

**Hypothesis to test:** disable caching of the aggregate CWT entirely
(it's almost-pure overhead) — or restructure so the per-ticker
scalogram is shared with the aggregate path, since both run on the
same dates / scales / lookback.

**Why deferred:** want to confirm with a `--profile` run that the
SHA-256 is actually visible in the wall-time profile. If `tobytes()`
+ hash is <2% of the call, just disable caching with a flag and
move on. If it is significant, the second restructure is worth a
day.

**Test design:** time a 10-call loop of `weights_empirical_sectors`
with caching on vs off on the canonical `Output/relational-empirical.json`
checkpoint. Positive result = ≥30% wall reduction with caching off.

### A4. Stage-2 `feat_rb` redundant `.astype(np.float32)`

`apps/factor/src/factor/train.py:381-388` — Stage-2 fine-tune does
`feat_rb = np.nan_to_num(aligned.features).astype(np.float32)`. After
the B1 refactor `aligned.features` is already f32 and pre-subsampled
to rebal positions (D'×N×K×F ≈ 4 GB at 297 tickers), so the cast is
a redundant copy — the only remaining waste is the `.astype` itself,
not the original `[rebal_idx]` indexing.

**Why deferred:** Stage 2 (`finetune_steps > 0`) is currently
disabled in the canonical supervised-`cnn` walkforward. No active runs hit the
path; fix when fine-tuning comes back into use. One-line fix when
it does: `.astype(np.float32, copy=False)`.

## B. Architectural — flag, don't fix

### B2. `apps/factor` and `apps/replay` reinvent data prep

Replay's `fit_cnn_multihead` (`decoders.py:572-684`) has its own
ownership-transfer trick (`del X_train` after in-place norm) that
factor's `precompute_inputs` doesn't share. Factor's
`align_tickers` doesn't know about replay's mask conventions
either. A unified `AlignedTickers`-with-explicit-ownership
convention would let both apps drop the per-app "is this safe to
mutate?" comments.

**Why architectural:** changing the shared abstraction touches every
walkforward script in factor and the multi-head training entrypoint
(`ss-replay --decoder cnn`) in replay. Worth a design doc, not a drive-by refactor.

### B3. Six relational scorers don't share a process-level CWT cache

The on-disk `.scalogram-cache` cache hit-rate is per-scorer. If the
canonical-checkpoint builder runs all six in one process, each
pays a fresh disk read of the same npz cache file. A process-level
`lru_cache` wrapper around `load_or_compute_cwt` would amortize
across the six scorers.

**Why architectural:** the scorers don't share a common entry point
today — `build_canonical_checkpoints.py` calls each `weights_*`
builder independently. Plumbing a shared cache through requires
either a context-manager pattern or a global, both of which need a
design choice.

## Where this came from

Rows in the leaderboard around the supervised-`cnn` walkforward
OOM arc (commits cac94e1, ed4043f, 3451900, eef4e52). The
`confirmed-null` on the supervised-`cnn` bundled backbone
(eef4e52) closed that specific arc, but the underlying memory math
survives any future retry on a wider universe — these follow-ups
were pre-emptive groundwork for the next walkforward at >300
tickers, and the A1 + A5 + B1 trio shipped on 2026-05-09 took the
supervised-`cnn` walkforward from "OOMs at 297 tickers in 192 GB"
to "fits 3000 tickers in ~65 GB."

Naming note: the Modal entrypoint is named `train_ssl_walkforward.py`
for historical reasons (the project once treated the `cnn`
indicator-reconstruction backbone as "SSL" in a loose sense). The
production backbone uses `--decoder cnn` (supervised reconstruction
with self-derived labels), not `--decoder masked-ae` (the strict-SSL
path, which is wired but never trained as a production npz). See
[replay-decoders](../findings/replay-decoders.md) for the canonical
terminology.

The seven high-confidence items shipped alongside the audit cover
`predict()`, `compute_input_stats`, `align_tickers`, the in-line
`forward_log_returns` / `daily_log_ret` f64 → f32, the missing
`cache_path=` on `build_canonical_checkpoints`, the f32 cast in
`_rolling_z_norm` before the per-scale FFT, and the dropped f64
upcast in `precompute_windows`. Together those restore the
headroom commit 3451900 was reaching for.
