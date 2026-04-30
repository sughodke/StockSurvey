"""IC-trained linear scorer on the FROZEN SSL-pretrained backbone.

Sister script to `stage1_ic_scorer.py` — same protocol but loads the
SSL backbone (`*-ssl-masked-ae-*.npz`) instead of the supervised CNN
backbone (`*-cnn-*.npz`). The downstream `train_scorer` call is
identical because `scoring.backbone.load_backbone` reads SSL and
supervised npz files through the same per-target-prefix interface.

Run on Colab TPU: env-var pin must come BEFORE jax is imported.
JAX_DEFAULT_MATMUL_PRECISION=highest forces f32 matmul; default bf16
quantizes small per-sample latent contributions to zero (cost the
FiLM cell hit; safety belt here for the same reason).

What we're testing:
  Does the SSL backbone — broad, indicator-shape-unbiased encoding of
  the CWT bundle — yield better val IC than the supervised CNN
  backbone, which we already showed ties at the noise floor with raw
  features (NOTES.md "No-backbone IC baseline" 2026-04-30)?
  Hypothesis: yes, because supervised pretrain selects projections
  that are anti-correlated with return prediction (indicators don't
  predict returns), while SSL has no such bias.

Reference floor to beat (from NOTES.md):
  Encoder (supervised, K=96, 30-tk): val IC +0.0039, val Sharpe +0.554
  Raw / no-encoder, same setup    : val IC -0.0050, val Sharpe +0.628
  -> SSL needs to push val IC clearly off zero (e.g. +0.02..+0.04+)
     to count as real progress, not just shuffle within the noise.
"""
import os

# --- TPU PATH (inactive — uncomment + change runtime to v5e-1 to use) ---
# os.environ.setdefault('JAX_PLATFORMS', 'tpu')
# os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')

# --- GPU PATH (active — T4) ---
# CUDA matmul defaults to f32 so JAX_DEFAULT_MATMUL_PRECISION is unnecessary;
# JAX auto-detects the GPU backend so JAX_PLATFORMS is unnecessary.

import glob

import matplotlib.pyplot as plt

from ss_notebook.replay.features import load_ticker
from ss_notebook.scoring import load_backbone, train_scorer


# 1. Locate the SSL backbone (latest masked-ae npz).
candidates = sorted(glob.glob('/content/Output/*-ssl-masked-ae-*.npz'))
assert candidates, ('no *-ssl-masked-ae-*.npz under /content/Output; '
                    'run train_ssl.sh first')
BACKBONE_PATH = candidates[-1]
print('Loading SSL backbone:', BACKBONE_PATH)
backbone, meta = load_backbone(BACKBONE_PATH)
print(f'  hidden_flat={backbone.hidden_flat}  '
      f'(K={backbone.K} x F={backbone.F} -> K_post={backbone.K_post} '
      f'x hidden={backbone.hidden})')
print(f'  pretrain pool: {len(meta["train_tickers"])} tickers, '
      f'{meta["start"]}..{meta["end"]}')
print(f'  mask_ratio={meta.get("mask_ratio")}, '
      f'ssl_decoder_hidden={meta.get("ssl_decoder_hidden")}, '
      f'cnn_steps={meta.get("cnn_steps")}')

# 2. Cross-sectional universe — same 30-name pool used in
# `stage1_ic_scorer.py` so val IC numbers are directly comparable to
# the supervised-backbone result.
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'JPM',  'V',    'UNH',   'HD',   'BAC',  'PG',   'KO',   'PEP',
    'COST', 'WMT',  'XOM',   'CVX',  'PFE',  'MRK',  'ABBV', 'TMO',
    'CSCO', 'INTC', 'AMD',   'ORCL', 'CRM',  'ADBE',
]
load_kwargs = dict(
    stooq_dir=None, kaggle_dir=None, use_yahoo=True,
    start=meta['start'], end=meta['end'],
    scales=[int(s) for s in meta['scales']],
    lookback=int(meta['lookback']),
    window_cols=int(meta['window_cols']),
    include_zscore_stats=bool(meta['include_zscore_stats']),
    include_returns=bool(meta['include_returns']),
    decoder=meta['decoder'],
    rsi_n=int(meta['rsi_n']),
    macd_fast=int(meta['macd_fast']),
    macd_slow=int(meta['macd_slow']),
    macd_signal=int(meta['macd_signal']),
    vol_window=int(meta.get('vol_window', 20)),
    rsi_n_grid=(), rsi_w_grid=(),
)
tickers = []
for name in UNIVERSE:
    try:
        tickers.append(load_ticker(name, **load_kwargs))
    except Exception as e:
        print(f'  skip {name}: {e}')
print(f'Loaded {len(tickers)} tickers')

# 3. Stage 1 — frozen SSL backbone + linear head trained on rank IC.
result = train_scorer(
    tickers, backbone,
    rebal_days=5,
    train_frac=0.7,
    scorer='linear',
    n_steps=500,
    learning_rate=1e-3,
    finetune_steps=0,        # frozen backbone (no Stage 2 fine-tune)
    commission_bps=10.0,
    seed=0,
    verbose=True,
)

print(f'\n=== SSL-backbone IC scorer result ===')
print(f'train IC : {result.train_ic:+.4f}    val IC : {result.val_ic:+.4f}')
print(f'train Shp: {result.train_sharpe:+.3f}    val Shp: {result.val_sharpe:+.3f}')
print(f'rebal blocks: {result.n_train_bars} train / {result.n_val_bars} val')
print(f'\nReference floor (NOTES.md 2026-04-30):')
print(f'  Encoder (supervised): val IC +0.0039, val Sharpe +0.554')
print(f'  Raw / no-encoder    : val IC -0.0050, val Sharpe +0.628')

# 4. Curves.
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(result.train_history, label='train IC', alpha=0.5, lw=1)
if result.val_history:
    vs, vi, vsh = zip(*result.val_history)
    ax[0].plot(vs, vi, 'o-', label='val IC', lw=1.5)
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel('Adam step'); ax[0].set_ylabel('rank IC')
ax[0].set_title('SSL backbone, Stage 1: rank IC')
ax[0].legend(); ax[0].grid(alpha=0.3)

if result.val_history:
    ax[1].plot(vs, vsh, 'o-', color='tab:green', label='val Sharpe (eval-only)')
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('Adam step'); ax[1].set_ylabel('annualized Sharpe')
ax[1].set_title('SSL backbone, Stage 1: val Sharpe')
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/Output/ssl-stage1-ic-scorer.png', dpi=150)
plt.show()
