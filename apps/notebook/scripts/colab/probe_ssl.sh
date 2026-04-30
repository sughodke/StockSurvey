#!/usr/bin/env bash
# Frozen-backbone probe — supervised heads on the SSL backbone.
#
# Loads the SSL backbone from the npz produced by `train_ssl.sh`,
# freezes its conv weights via `--freeze-backbone`, and trains only
# the per-target heads (rsi/macd/price/vol, with FiLM-conditioned rsi
# over (n, w)). The per-indicator R² printed at the end is the
# diagnostic readout: how much of each classical indicator is linearly
# (or, for rsi, FiLM-conditionally) recoverable from the SSL latent?
#
# Interpretation:
#   * High R² across the board (>=0.85 for rsi/macd, >=0.95 for
#     price): SSL latent captures everything the supervised latent
#     did. Best case — SSL didn't lose indicator-recoverable structure
#     and probably encodes extra structure those indicators don't
#     summarize.
#   * Indicator R² drops modestly (0.7-0.85): the SSL latent
#     reallocated some capacity away from indicator-specific features.
#     Whether good depends on whether the reallocated capacity helps
#     downstream IC.
#   * Indicator R² collapses (<0.5): SSL pretrain didn't preserve
#     enough structure for known-good signals. Diagnostic, not
#     catastrophic — sweep mask_ratio (try 0.3, 0.5) or bump
#     ssl-decoder-hidden / cnn-steps.
#
# Edit the SSL_NPZ path below to match what train_ssl.sh produced.

set -euo pipefail

cd /content

# Pick the most recent SSL npz; edit if you want a specific one.
SSL_NPZ=$(ls -t /content/Output/*-ssl-masked-ae-*.npz | head -1)
echo "Probing SSL backbone: $SSL_NPZ"

# Accelerator notes (same as train_ssl.sh): batch 2048 instead of 8192,
# JAX_PLATFORMS=cuda, JAX_DEFAULT_MATMUL_PRECISION dropped. The
# FiLM rsi head DOES carry small per-sample latent contributions —
# but on GPU f32 matmul is the default so the bf16-quantization
# pathology from the TPU run does not apply.

# --- TPU PATH (inactive — uncomment + change runtime to v5e-1 to use) ---
# JAX_PLATFORMS=tpu JAX_DEFAULT_MATMUL_PRECISION=highest \

# --- GPU PATH (active — T4) ---
JAX_PLATFORMS=cuda \
ss-replay AAPL --yahoo \
    --train-tickers MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,GE,BA,XOM,KO,WMT,JNJ,UNH,T,NFLX,CRM,DIS \
    --val-ticker TSLA \
    --start 2013-01-29 --end 2025-12-11 \
    --window-cols 96 \
    --extra-high-freq-scales 1,2 \
    --include-zscore-stats \
    --include-returns \
    --decoder cnn --targets rsi,macd,price,vol \
    --rsi-n 7 --rsi-n-grid 5,7,9,13,17,21,25 \
    --rsi-w-grid 1,5,10,21 --rsi-anchor-w 1 \
    --vol-window 20 \
    --cnn-batch-size 2048 --cnn-steps 2000 \
    --freeze-backbone "$SSL_NPZ" \
    --device auto \
    --output-dir /content/Output
