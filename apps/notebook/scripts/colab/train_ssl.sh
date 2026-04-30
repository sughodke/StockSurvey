#!/usr/bin/env bash
# SSL pretrain — masked CWT autoencoding.
#
# Trains the conv backbone via `--decoder masked-ae`: ~`mask_ratio` of
# (lag, channel) cells are masked per training row; a small MLP decoder
# reconstructs the full bundle; loss = MSE on masked positions only.
# No per-target indicator heads — the gradient pulls the backbone
# toward broad encoding (every output cell of the decoder backprops
# through the backbone), not toward 4-indicator-shaped subspaces.
#
# This is the canonical "fix" for the diagnosis in
# `apps/notebook/src/ss_notebook/replay/README.md` "What the backbone
# actually learns": current supervised multi-head training only puts
# gradient on 4 directions of a 5632-d latent; the rest decays to noise.
# SSL puts gradient on every output cell -> backbone has to encode the
# full bundle structure.
#
# Output: `*-ssl-masked-ae-*.npz` containing only the backbone (no
# heads, no target standardizers). Compatible with `load_backbone` in
# `ss_notebook.scoring` — the scoring IC head reads it the same way it
# reads a supervised CNN backbone npz.
#
# Compute budget:
#   Supervised baseline ran 2000 steps. SSL needs ~5-10x more (per
#   NOTES.md / SSL plan). Currently set to 20000 steps.
#
# Tuning vs the first attempt (2026-04-30, ~30-60min on v5e-1):
#   First run reported train_mse_masked=0.891 in z-norm space — only
#   ~11% of input variance explained, with masked vs unmasked MSE
#   barely differing (0.891 vs 0.899). Symptom of decoder under-
#   capacity + insufficient compute, not future leakage (audit
#   confirmed strict causality wrt the downstream IC task).
#
#   Three changes vs that run:
#     * --ssl-decoder-hidden 1024 (was 256). Latent (~5632) -> bundle
#       (3168) through one ReLU at width 256 was a hard bottleneck.
#     * --mask-ratio 0.25 (was 0.4). 0.4 is the upper edge of the
#       time-series MAE range; backing off so the reconstruction
#       task is tractable enough to train cleanly first.
#     * --cnn-steps 20000 (was 10000). 2x compute for SSL convergence.
#
# Run order:
#   1. THIS SCRIPT          -> trains the SSL backbone
#   2. probe_ssl.sh         -> trains supervised heads on FROZEN SSL
#                              backbone, prints per-indicator R² to
#                              validate that the SSL latent preserves
#                              indicator-recoverable structure (the
#                              probe protocol from the SSL plan)
#   3. ssl_ic_scorer.py     -> IC head on the SSL backbone, the test
#                              that matters for forward-return alpha

set -euo pipefail

cd /content

JAX_PLATFORMS=tpu JAX_DEFAULT_MATMUL_PRECISION=highest \
ss-replay AAPL --yahoo \
    --train-tickers MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,GE,BA,XOM,KO,WMT,JNJ,UNH,T,NFLX,CRM,DIS \
    --val-ticker TSLA \
    --start 2013-01-29 --end 2025-12-11 \
    --window-cols 96 \
    --extra-high-freq-scales 1,2 \
    --include-zscore-stats \
    --include-returns \
    --decoder masked-ae \
    --mask-ratio 0.25 \
    --ssl-decoder-hidden 1024 \
    --ssl-decoder-layers 2 \
    --cnn-batch-size 8192 \
    --cnn-steps 20000 \
    --device auto \
    --output-dir /content/Output
