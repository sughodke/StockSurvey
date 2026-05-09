"""Linear scorer on flattened raw CWT bundle — no encoder.

Diagnostic baseline: tells us what val IC the linear span of the raw
input features can hit. If the SSL+linear path can't beat this, the
encoder isn't earning its keep.
"""
import time
import numpy as np

from factor import compute_input_stats, identity_backbone, train_scorer
from ss_features import load_ticker
from ss_wavelets import ALL_SCALES


# 30-ticker mega-cap pool. Yahoo source so it runs anywhere.
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO',
    'JPM',  'V',    'JNJ',   'WMT',  'PG',   'MA',   'HD',   'CVX',
    'ABBV', 'MRK',  'KO',    'PEP',  'ADBE', 'CRM',  'COST', 'BAC',
    'AMD',  'NFLX', 'CSCO',  'ACN',  'INTC', 'ORCL',
]
START, END = '2010-01-01', '2024-12-31'
WINDOW_COLS = 64
LOOKBACK = 252

LOAD_KW = dict(
    stooq_dir=None, kaggle_dir=None, use_yahoo=True,
    start=START, end=END,
    scales=list(ALL_SCALES), lookback=LOOKBACK, window_cols=WINDOW_COLS,
    rsi_n=7, macd_fast=12, macd_slow=26, macd_signal=9,
    vol_window=20,
)

t0 = time.time()
print(f'Loading {len(TICKERS)} tickers ({START} .. {END}) via Yahoo...')
tickers = []
for name in TICKERS:
    try:
        td = load_ticker(name, **LOAD_KW)
        tickers.append(td)
        print(f'  {name:6s} {td.valid.sum():5d} valid bars')
    except Exception as e:
        print(f'  {name:6s} FAILED: {e}')
print(f'Loaded {len(tickers)} tickers in {time.time() - t0:.1f}s')

K = WINDOW_COLS
F = tickers[0].features.shape[1] // K
print(f'\nK={K}, F={F}, hidden_flat={K * F} (= raw flat CWT bundle dim)')

mu, sd = compute_input_stats(tickers, K, F)
bb_identity = identity_backbone(K, F, feat_mu=mu, feat_sd=sd)

print('\n--- Run A: linear, weight_decay=1e-2, n_steps=500 ---')
result_a = train_scorer(
    tickers, bb_identity,
    rebal_days=5, scorer='linear',
    n_steps=500, weight_decay=1e-2,
    finetune_steps=0, verbose=True,
)
print(f'\nRun A FINAL: train_ic={result_a.train_ic:+.4f}  val_ic={result_a.val_ic:+.4f}  '
      f'val_sharpe={result_a.val_sharpe:+.3f}')
peak_val = max(result_a.val_history, key=lambda x: x[1])
print(f'Run A PEAK val: step={peak_val[0]} ic={peak_val[1]:+.4f} sharpe={peak_val[2]:+.3f}')

print('\n--- Run B: linear, weight_decay=1e-1, n_steps=500 (heavier ridge) ---')
result_b = train_scorer(
    tickers, bb_identity,
    rebal_days=5, scorer='linear',
    n_steps=500, weight_decay=1e-1,
    finetune_steps=0, verbose=True,
)
print(f'\nRun B FINAL: train_ic={result_b.train_ic:+.4f}  val_ic={result_b.val_ic:+.4f}  '
      f'val_sharpe={result_b.val_sharpe:+.3f}')
peak_val = max(result_b.val_history, key=lambda x: x[1])
print(f'Run B PEAK val: step={peak_val[0]} ic={peak_val[1]:+.4f} sharpe={peak_val[2]:+.3f}')

print('\n--- Run C: linear, weight_decay=0, n_steps=500 (no decay reference) ---')
result_c = train_scorer(
    tickers, bb_identity,
    rebal_days=5, scorer='linear',
    n_steps=500, weight_decay=0.0,
    finetune_steps=0, verbose=True,
)
print(f'\nRun C FINAL: train_ic={result_c.train_ic:+.4f}  val_ic={result_c.val_ic:+.4f}  '
      f'val_sharpe={result_c.val_sharpe:+.3f}')
peak_val = max(result_c.val_history, key=lambda x: x[1])
print(f'Run C PEAK val: step={peak_val[0]} ic={peak_val[1]:+.4f} sharpe={peak_val[2]:+.3f}')

print(f'\nTotal wall: {time.time() - t0:.1f}s')
