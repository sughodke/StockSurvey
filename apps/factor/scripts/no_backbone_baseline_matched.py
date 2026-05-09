"""Linear scorer on raw CWT bundle — apples-to-apples vs the Colab encoder run.

Matches the Colab pretrain bundle topology: K=96, F=33 (ALL_SCALES + extra
high-freq scales 1,2 + zscore stats + returns). Same date window 2013-01-29
to 2025-12-11.
"""
import time

from factor import compute_input_stats, identity_backbone, train_scorer
from ss_features import load_ticker
from ss_wavelets import ALL_SCALES


# Same 30-name mega-cap pool as the local run; pretrain pool was 19, but
# IC scorer pool may differ — comparison is "best-effort same window".
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO',
    'JPM',  'V',    'JNJ',   'WMT',  'PG',   'MA',   'HD',   'CVX',
    'ABBV', 'MRK',  'KO',    'PEP',  'ADBE', 'CRM',  'COST', 'BAC',
    'AMD',  'NFLX', 'CSCO',  'ACN',  'INTC', 'ORCL',
]
START, END = '2013-01-29', '2025-12-11'   # matches Colab pretrain window
WINDOW_COLS = 96                          # matches Colab K=96
EXTRA_SCALES = [1, 2]                     # ALL_SCALES + extras -> 15 scales -> F=33
LOOKBACK = 252

scales = sorted(set(EXTRA_SCALES) | set(ALL_SCALES))
LOAD_KW = dict(
    stooq_dir=None, kaggle_dir=None, use_yahoo=True,
    start=START, end=END,
    scales=scales, lookback=LOOKBACK, window_cols=WINDOW_COLS,
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
    except Exception as e:
        print(f'  {name:6s} FAILED: {e}')
print(f'Loaded {len(tickers)} tickers in {time.time() - t0:.1f}s')

K = WINDOW_COLS
F = tickers[0].features.shape[1] // K
print(f'K={K}, F={F}, hidden_flat={K * F}')
print(f'(matches Colab encoder K=96, F=33; encoder collapses to '
      f'K_post=88 x hidden=64 = 5632)')

mu, sd = compute_input_stats(tickers, K, F)
bb_identity = identity_backbone(K, F, feat_mu=mu, feat_sd=sd)

print('\n--- Run A: linear, weight_decay=0.0, n_steps=500 ---')
result_a = train_scorer(
    tickers, bb_identity,
    rebal_days=5, scorer='linear',
    n_steps=500, weight_decay=0.0,
    finetune_steps=0, verbose=True,
)
peak = max(result_a.val_history, key=lambda x: x[1])
print(f'\nFINAL: train_ic={result_a.train_ic:+.4f}  val_ic={result_a.val_ic:+.4f}  '
      f'val_sharpe={result_a.val_sharpe:+.3f}')
print(f'PEAK val: step={peak[0]} ic={peak[1]:+.4f} sharpe={peak[2]:+.3f}')

print(f'\nTotal wall: {time.time() - t0:.1f}s')
