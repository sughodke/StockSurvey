"""Stage 1: IC-trained linear scorer on the FROZEN pretrained backbone.

Extracted from cwt_vision_multihead.ipynb (Colab cell). Paths assume
/content/ layout — run inside Colab or adapt for local use.

Run on Colab TPU: env-var pin must come BEFORE jax is imported.
JAX_DEFAULT_MATMUL_PRECISION=highest forces f32 matmul (default bf16
quantized the FiLM cond contribution to zero in earlier runs; safety
belt here for the same reason).

TLDR results (30-ticker universe, K=96/F=33, 2013-01-29..2025-12-11,
rebal_days=5, scorer='linear', n_steps=500, weight_decay=0,
452 train / 195 val rebalance blocks):

                              Encoder (5632-d head)   Raw / no encoder (3168-d)
  Final train IC              +0.7226 (overfit)       +0.3165
  Final val IC                +0.0039                 -0.0050
  Final val Sharpe            +0.554                  +0.628

Read: encoder and raw both tie at the noise floor. Cross-sectional IC
supervision at this scale is the binding constraint, not the encoder.
Floor at this setup: val IC ≈ 0, val Sharpe ≈ 0.55..0.63 with linear
scoring. New architecture has to clearly beat this floor to count.
See NOTES.md "No-backbone IC baseline" (2026-04-30).

Pipeline:
  1. Load the latest *-cnn-*.npz, strip per-target heads, keep the conv
     backbone (feat_mu/sd + conv layers).
  2. Build a cross-sectional universe of TickerData using the EXACT
     bundle params the backbone saw at pretrain (scales / window_cols /
     lookback / include_zscore_stats / include_returns / vol_window).
  3. train_scorer with finetune_steps=0 — Stage 1 only (backbone frozen,
     Adam updates only the linear head + log-temperature).

Objective: mean per-rebalance Pearson rank IC between scores and forward
log-returns. Sharpe (10bps + spread costs) is computed for eval only —
IC is the dense per-decision signal that converges fast; Sharpe is the
noisy single-number summary we ultimately care about.
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

from factor import load_backbone, train_scorer
from ss_notebook.replay.features import load_ticker


# 1. Locate the pretrained backbone npz (latest cnn run on disk).
candidates = sorted(glob.glob('/content/Output/*-cnn-*.npz'))
assert candidates, 'no *-cnn-*.npz under /content/Output; run training first'
BACKBONE_PATH = candidates[-1]
print('Loading backbone:', BACKBONE_PATH)
backbone, meta = load_backbone(BACKBONE_PATH)
print(f'  hidden_flat={backbone.hidden_flat}  '
      f'(K={backbone.K} x F={backbone.F} -> K_post={backbone.K_post} '
      f'x hidden={backbone.hidden})')
print(f'  pretrain pool: {len(meta["train_tickers"])} tickers, '
      f'{meta["start"]}..{meta["end"]}')

# 2. Cross-sectional universe — bigger than the pretrain pool. IC needs
# N tickers per rebalance to be meaningful; ~30 is a comfortable floor.
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
    # Conditioning grids only affect per-target heads (which we discard).
    # Leave empty so we don't pay the augmentation cost in load_ticker.
    rsi_n_grid=(), rsi_w_grid=(),
)
tickers = []
for name in UNIVERSE:
    try:
        tickers.append(load_ticker(name, **load_kwargs))
    except Exception as e:
        print(f'  skip {name}: {e}')
print(f'Loaded {len(tickers)} tickers')

# 3. Stage 1 only — frozen backbone + linear head trained on rank IC.
result = train_scorer(
    tickers, backbone,
    rebal_days=5,
    train_frac=0.7,
    scorer='linear',
    n_steps=500,
    learning_rate=1e-3,
    finetune_steps=0,        # Stage 2 disabled; backbone stays frozen
    commission_bps=10.0,
    seed=0,
    verbose=True,
)

print(f'\n=== Stage 1 result ===')
print(f'train IC : {result.train_ic:+.4f}    val IC : {result.val_ic:+.4f}')
print(f'train Shp: {result.train_sharpe:+.3f}    val Shp: {result.val_sharpe:+.3f}')
print(f'rebal blocks: {result.n_train_bars} train / {result.n_val_bars} val')

# 4. Curves.
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(result.train_history, label='train IC', alpha=0.5, lw=1)
if result.val_history:
    vs, vi, vsh = zip(*result.val_history)
    ax[0].plot(vs, vi, 'o-', label='val IC', lw=1.5)
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel('Adam step'); ax[0].set_ylabel('rank IC')
ax[0].set_title('Stage 1: rank IC')
ax[0].legend(); ax[0].grid(alpha=0.3)

if result.val_history:
    ax[1].plot(vs, vsh, 'o-', color='tab:green', label='val Sharpe (eval-only)')
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('Adam step'); ax[1].set_ylabel('annualized Sharpe')
ax[1].set_title('Stage 1: val Sharpe')
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/content/Output/stage1-ic-scorer.png', dpi=150)
plt.show()
