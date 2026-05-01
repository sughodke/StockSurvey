"""Python wrapper for the supervised CNN multi-head pretrain with
sign(return) input channel — runs ss-replay's main() in-process via
sys.argv override so notebook stdout streams live (vs `!bash` where
the subprocess can buffer output until buffers fill or it exits).

Same args as `train_cnn_signreturn.sh`. To switch back to bash,
`!bash /content/StockSurvey/apps/notebook/scripts/colab/train_cnn_signreturn.sh`.

Hypothesis being tested (per attention finding 2026-05-01): the
supervised heads collapse onto the raw `return` channel because
returns expose direction AND magnitude — the lazy shortcut to
RSI/MACD/vol. Sign(return) keeps direction, strips magnitude → model
must extract magnitude from wavelets. Three predicted outcomes:
  A. Indicator R² collapses → wavelet band-limit is the issue,
     fix is structural (finer scales / Morlet).
  B. Indicator R² holds → wavelets carry magnitude info; raw return
     was just the lazy path.
  C. R² takes moderate hit → channel-dropout would close the gap.
"""
import os

# Hard-clear any inherited JAX env vars (the old supervised cell used
# to do `os.environ['JAX_PLATFORMS'] = 'tpu'` in this kernel; carries
# over to anything that imports jax via this script).
os.environ.pop('JAX_PLATFORMS', None)
os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION', None)

# --- TPU PATH (inactive — uncomment + change runtime to v5e-1 to use) ---
# os.environ['JAX_PLATFORMS'] = 'tpu'
# os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# --- GPU PATH (inactive — uncomment + use a CUDA Colab runtime to use) ---
# os.environ['JAX_PLATFORMS'] = 'cuda'

# --- CPU PATH (active) — JAX auto-detects, no platform pin needed.

import sys

os.chdir('/content')

sys.argv = [
    'ss-replay', 'AAPL', '--yahoo',
    '--train-tickers',
    'MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,GE,BA,XOM,KO,WMT,JNJ,UNH,T,NFLX,CRM,DIS',
    '--val-ticker', 'TSLA',
    '--start', '2013-01-29', '--end', '2025-12-11',
    '--window-cols', '96',
    '--extra-high-freq-scales', '1,2',
    '--include-zscore-stats',
    '--include-return-sign',
    '--decoder', 'cnn', '--targets', 'rsi,macd,price,vol',
    '--rsi-n', '7', '--rsi-n-grid', '5,7,9,13,17,21,25',
    '--rsi-w-grid', '1,5,10,21', '--rsi-anchor-w', '1',
    '--vol-window', '20',
    '--cnn-batch-size', '2048', '--cnn-steps', '2000',
    '--device', 'auto',
    '--output-dir', '/content/Output',
]

from ss_notebook.replay.cli import main
main()
