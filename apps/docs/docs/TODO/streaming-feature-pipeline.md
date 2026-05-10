# Streaming feature pipeline (replace bulk pre-compute, lift the OOM ceiling)

!!! note "Superseded 2026-05-10 — OOM solved a different way"
    This page describes a JAX/Colab-era pipeline that no longer
    exists. The OOM problem it was designed to solve (12 GB Colab
    CPU runtime; `fit_and_evaluate` materializing
    `n_pool × K × F` rows; FiLM augmentation copies) was solved by
    moving training to Modal-T4 (`cpu=4 mem=192GB`) and shipping
    the A1 / A5 / B1 trio from
    [`memory-walltime-followups`](memory-walltime-followups.md) on
    2026-05-09 — those changes took the supervised-`cnn`
    walkforward from "OOMs at 297 tickers in 192 GB" to "fits 3000
    tickers in ~65 GB" without a streaming refactor. The
    `LazyTicker` / memmap design below references
    `decoders.py::fit_cnn_multihead` and the JAX device-copy path,
    neither of which exists in the current tinygrad+Modal pipeline.
    Kept as design archaeology in case streaming becomes
    load-bearing again at a wider universe / longer K.

Today's `fit_and_evaluate` concatenates every train ticker's full
feature matrix into one giant `(n_pool, K * F)` array before training.
At 19 tickers × ~3000 valid bars × K=96 × F=33 that's ~720 MB
(float32) on host RAM, plus a JAX device copy, plus FiLM-augmented
auxiliary arrays (Y_std, cond, train_pool_idx). Combined with a 12 GB
Colab CPU runtime this OOMs (2026-05-01 attempt killed at the supervised
sign-of-return run before training started).

The float32 cast (committed 2026-05-01) lifts the ceiling 2x but
doesn't change the asymptote — at K > 128 or train pool > 30 tickers
we'll hit it again. Streaming is the structural fix.

**Goal:** never materialize more than `batch_size` feature rows at
once. Per-step pseudocode:

```
for step in range(n_steps):
    batch = sample_batch(per_ticker_handles, batch_size, train_pool_idx)
    # `batch` is a fresh (batch_size, K, F) gather assembled on demand
    # from per-ticker on-disk / lazy feature arrays.
    loss, grads = value_and_grad(loss_fn)(params, batch, ...)
    params = update(params, grads)
```

**Two implementation candidates, ranked by effort:**

1. **`np.memmap` per-ticker features.** Save each ticker's
   `(n_dates, K * F)` feature matrix to a `.npy` on disk during
   `load_ticker`. The trainer's pool then holds an index of
   `(ticker_id, local_bar_idx)` tuples; the batch sampler maps each
   logical row to a memmap'd ticker file and reads the K-row slice on
   demand. JAX still materializes only `batch_size` worth of rows.
   Cost: ~50 lines (a `LazyTicker` class wrapping memmap + index, plus
   a batch sampler in `decoders.py::fit_cnn_multihead`). No change to
   the model or loss. Disk I/O per step is O(batch_size * K * F) —
   trivial on local SSD, slow on Colab's network disk but probably
   tolerable.

2. **Pure in-memory streaming via `tf.data` / `jax.experimental.shard`.**
   Heavier framework dependency; only worth it if (1) is also too slow.

**Why not just shrink the universe / K?** That works for individual
experiments but doesn't scale. We're going to want to push to
50+ tickers, K=128+, longer FiLM grids — all of which exceed Colab's
12 GB CPU budget AND a single A100's 80 GB if the float32 ceiling
keeps growing.

**Adjacent improvement worth folding in**: the FiLM augmentation
already uses `train_pool_idx` lazily (per `reconstruct.py:99`) but the
trainer still gathers the full augmented set into `Xj` on every step
in the full-batch path (`decoders.py:792`). The streaming refactor
should remove that path entirely — there's no scenario where holding
`n_pool * n_replicas` rows of features in memory is the right answer.

**Test for it:** a "scaling" smoke test in `apps/notebook/tests/`
that runs a 100-ticker / K=128 fit with cnn-batch-size=512 and
asserts peak RSS stays under 4 GB. Fails today; should pass after
streaming.
