#!/usr/bin/env bash
# Supervised CNN multi-head pretrain with SIGN-of-return instead of raw
# return as the support channel. Same FiLM rsi(n,w) head + linear macd /
# price / vol heads; only the input bundle changes.
#
# Hypothesis being tested:
#   The attention plot (2026-05-01) showed all four supervised heads
#   live ~entirely on the raw `return` channel — the 30 wavelet channels
#   contribute trace gradient. Replacing raw returns with sign(returns)
#   strips the magnitude shortcut. RSI / MACD / vol need both direction
#   AND magnitude; sign covers direction, so the wavelet channels MUST
#   carry magnitude info if the indicator R² is to stay high.
#
# Three predictable outcomes (record into NOTES.md after the run):
#   A. Indicator R² collapses (e.g. RSI 0.95 -> 0.30): wavelet band-limit
#      is the issue; CWT alone genuinely can't substitute for daily
#      magnitudes. Fix is structural — add finer scales (extra-high-freq
#      0.5, 0.7) or switch to a complex Morlet wavelet for phase info.
#   B. Indicator R² holds (RSI ~0.85+): the CWT does carry magnitude
#      info; the original `return` channel was a lazy shortcut, not a
#      structural necessity. Re-run the attention plot to confirm
#      wavelets are now lit up.
#   C. Indicator R² takes a moderate hit (RSI 0.70-0.85): wavelets are
#      mostly sufficient but with some loss. Channel-dropout training
#      (random mask of return during training, raw at eval) would
#      recover the rest.
#
# Accelerator notes: small CNN, 2000 steps, batch 2048 → expect a few
# minutes on T4, ~30-60 min on CPU. Channel count is unchanged vs the
# raw-returns baseline (sign occupies the same slot).

set -euo pipefail

# Hard-override any inherited JAX env vars (see train_ssl.sh for why).
unset JAX_PLATFORMS JAX_DEFAULT_MATMUL_PRECISION

cd /content

# --- TPU PATH (inactive — uncomment + change runtime to v5e-1 to use) ---
# JAX_PLATFORMS=tpu JAX_DEFAULT_MATMUL_PRECISION=highest \

# --- GPU PATH (inactive — uncomment + use a CUDA Colab runtime to use) ---
# JAX_PLATFORMS=cuda \

# --- CPU PATH (active) — JAX auto-detects, no platform pin needed.
ss-replay AAPL --yahoo \
    --train-tickers MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,GE,BA,XOM,KO,WMT,JNJ,UNH,T,NFLX,CRM,DIS \
    --val-ticker TSLA \
    --start 2013-01-29 --end 2025-12-11 \
    --window-cols 96 \
    --extra-high-freq-scales 1,2 \
    --include-zscore-stats \
    --include-return-sign \
    --decoder cnn --targets rsi,macd,price,vol \
    --rsi-n 7 --rsi-n-grid 5,7,9,13,17,21,25 \
    --rsi-w-grid 1,5,10,21 --rsi-anchor-w 1 \
    --vol-window 20 \
    --cnn-batch-size 2048 --cnn-steps 2000 \
    --device auto \
    --output-dir /content/Output
