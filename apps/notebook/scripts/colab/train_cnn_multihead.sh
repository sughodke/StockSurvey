#!/usr/bin/env bash
# Phase 2 / Experiment D: 2-D RSI conditioning over (w, n) cross-product.
# Extracted from cwt_vision_multihead.ipynb (Colab). Paths assume /content/
# layout — run from inside Colab or adapt for local use.
#
# TLDR results (story so far, prior single-head bundle runs):
#   RSI(7) CNN, no bundle, single-head:        train R² ~0.62, TSLA val 0.55
#   RSI(28) CNN, no bundle, single-head:       train ~0.83, val 0.77 (in-band)
#   RSI(7) CNN, BUNDLE (returns + zscore-stats + extra-scales), single-head:
#                                               train 0.977, TSLA val 0.954
#   CSCO zero-shot from that single-head bundle: 0.972
#   Bundle (returns + zscore-stats + extra HF scales) closes the val-R² gap.
# This Phase 2 / Exp D run (multi-head + FiLM (n, w) conditioning) heatmap
# numbers not archived here — see zeroshot_eval.py for the eval that
# produces the (w, n) R² grid.
#
# Targets:
#   rsi (cond_dim=2): RSI(n, w) via rsi_strided(prices, n, w). Δ_i =
#                     price[i] - price[i-w] then Wilder-smoothed over n
#                     strided observations. w=1 reduces to canonical
#                     daily RSI(n); w=5/10/21 are weekly/biweekly/monthly
#                     evaluated at every bar (dense supervision).
#   macd, price, vol: unconditioned (head_cond_dim=0).
#
# Grid: n ∈ {5,7,9,13,17,21,25} × w ∈ {1,5,10,21} → 28 logical replicas
# per pooled training row. Lazy augmentation keeps X_train at pool size.
#
# Notes:
#   * --window-cols 96 covers Wilder RSI(25) effective memory.
#   * --rsi-n-grid adds n=17,25 so the long end has right-neighbors
#     for linear conditioning interpolation.
#   * 18-ticker pool (apples-to-apples with the prior smoke test).
#
# JAX env: JAX_PLATFORMS=tpu picks the v5e-1 device (when --device auto).
# JAX_DEFAULT_MATMUL_PRECISION=highest forces f32 matmul on TPU; default
# bf16 quantizes the FiLM cond contribution to zero (head collapses to
# constant per (n, w) cell). Cost ~3× slower but trivial at this size.

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
    --decoder cnn --targets rsi,macd,price,vol \
    --rsi-n 7 --rsi-n-grid 5,7,9,13,17,21,25 \
    --rsi-w-grid 1,5,10,21 --rsi-anchor-w 1 \
    --vol-window 20 \
    --cnn-batch-size 8192 --cnn-steps 2000 \
    --device auto \
    --output-dir /content/Output
