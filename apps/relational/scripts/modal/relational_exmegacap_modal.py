"""Modal entrypoint: analog cross_ticker on the ex-Phase-2 universe.

Tests whether the Phase-2 OOS winner (analog cross_ticker
uncompressed, val Sharpe 1.146 on mega-caps 2021-2025) survives
moving off the 21 hand-picked mega-caps to the rest of the curated
long-history US universe (~296 names).

The point isn't to find a new winner — it's to disentangle "strategy
adds cross-sectional alpha" from "strategy was riding the mega-cap
macro tailwind". If the val Sharpe collapses on the wider universe
the way Phase-8 of the NO_OPTIONS arc did (Sharpe ~1.1 → ~0.4 on the
312-ticker universe), the Phase-2 win is regime-bound. If it holds
or only mildly degrades, the strategy has cross-sectional skill that
generalises.

Universe: `apps/notebook/data/stooq_us_long` minus PHASE2_TICKERS,
296 names. NOT true small caps — we lack market-cap data to filter
that precisely. This is "long-history US large/mid-cap survivors
ex-Phase-2", which is the cleanest non-mega-cap test we can run
with on-hand data.

Algorithm: `analog_knn_scores_fast` (vectorised matmul + argpartition
truncation; Pearson 0.994 vs slow path on Phase-2). The slow path
would be ~10-50× slower on N=296 and not finish inside reasonable
Modal wall.

Walk-forward: same Phase-2 split (train 2013-01-29 → 2020-12-31,
val 2021-01-01 → 2025-12-11) so the comparison is one-to-one with
the existing analog cross_ticker row in WALKFORWARD.md.

Usage:
    uv run python apps/relational/scripts/modal/prep_exmegacap_prices.py
    uvx modal run apps/relational/scripts/modal/relational_exmegacap_modal.py

Wall estimate: ~25-40 min on Modal `cpu=8` (BLAS-bound matmul
dominates; Python pick-walk is the secondary cost).
"""
from __future__ import annotations

from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'

DEFAULT_PRICES_PKL = Path('/tmp/exmegacap-prices.pkl')
CWT_CACHE_REMOTE = '/root/cwt-cache'

# Plain Python image — same template as relational_dwt_phase2.py. No
# GPU; the bottleneck is BLAS matmul + a Python pick walk. Previous
# 90-min run timed out with only 1/8 cores busy because (a) numpy
# BLAS wasn't multithreading and (b) the Python pick walk doesn't
# parallelize on its own. Fix in this revision: mp.Pool over the
# t-axis with fork()-shared cand arrays (n_workers=cpu count).
image = (
    modal.Image.debian_slim(python_version='3.13')
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            '.iv-cache/**',
            '.claude/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            'apps/factor/src/**',
            'apps/replay/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

# Persistent CWT cache so re-runs on the same universe + scales skip
# the 5-10 min causal_cwt precompute. The relational scalogram_cache
# keys by content hash, so only universe / scales / lookback changes
# invalidate.
cwt_volume = modal.Volume.from_name(
    'ss-relational-cwt-cache', create_if_missing=True)

app = modal.App('ss-relational-exmegacap', image=image)


@app.function(
    cpu=8, memory=32768, timeout=60 * 240,
    volumes={CWT_CACHE_REMOTE: cwt_volume},
)
def run_arm(prices_pkl: bytes) -> dict[str, bytes]:
    """Run the single arm + walk-forward segmentation; return artifacts."""
    import pickle
    import subprocess
    import warnings
    warnings.filterwarnings('ignore')

    print('=== Step 1/4: uv sync workspace deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'relational', '--extra', 'research',
         '--inexact'],
        cwd=REMOTE_REPO, check=True)

    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.13/site-packages')

    import bt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ss_portfolio import apply_nan_mask, select_top_n_matrix
    from ss_portfolio.bt_helpers import build_strategy

    from relational.analog_knn import analog_knn_scores_fast
    from relational.research.idea_b_analog_knn_dwt_walkforward import (
        TRAIN_END, TRAIN_START, VAL_END, VAL_START, segment_stats,
    )

    bundle = pickle.loads(prices_pkl)
    prices: pd.DataFrame = bundle['prices']
    start = bundle['start']
    end = bundle['end']
    print(f'\n=== Step 2/4: prices {prices.shape} '
          f'({start} → {end}); {len(prices.columns)}-ticker ex-Phase-2 universe ===',
          flush=True)

    lookback = 120
    top_n = 10
    fp_window = 21
    rebal_days = 20
    commission_bps = 10.0
    scales = [5, 7, 10, 12, 21, 26, 50, 90]

    import os
    n_workers = int(os.environ.get('SS_N_WORKERS', os.cpu_count() or 8))
    print(f'\n=== Step 3/4: analog cross_ticker scores '
          f'(N={prices.shape[1]}, T={prices.shape[0]}, '
          f'n_workers={n_workers}) ===', flush=True)
    import time
    t0 = time.time()
    scores = analog_knn_scores_fast(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, k_neighbors=50, forward_horizon=20,
        min_sep_days=21, pool_mode='cross_ticker',
        cache_dir=CWT_CACHE_REMOTE,
        n_workers=n_workers)
    print(f'  scores shape={scores.shape} in {time.time() - t0:.1f}s',
          flush=True)
    print(f'  finite cells: {np.isfinite(scores).sum()} / {scores.size}',
          flush=True)

    scores = apply_nan_mask(scores, prices.values, lookback)
    weights_arr = select_top_n_matrix(scores, top_n, ascending=False)
    weights = pd.DataFrame(
        weights_arr, index=prices.index[lookback:], columns=prices.columns)
    print(f'  weights shape={weights.shape}', flush=True)

    print(f'\n=== Step 4/4: bt + walk-forward segmentation ===', flush=True)
    strategy = build_strategy(
        'analog-exmegacap', prices, weights,
        rebal_days=rebal_days, commission_bps=commission_bps)
    result = bt.run(strategy)
    result.display()

    eq = result.prices['analog-exmegacap']
    seg_rows = []
    for window_label, w_start, w_end in (
        ('full',  TRAIN_START, VAL_END),
        ('train', TRAIN_START, TRAIN_END),
        ('val',   VAL_START,   VAL_END),
    ):
        stats = segment_stats(eq, w_start, w_end)
        stats.update({'arm': 'analog-exmegacap', 'window': window_label,
                      'start': w_start, 'end': w_end})
        seg_rows.append(stats)
    summary = pd.DataFrame(seg_rows)[
        ['arm', 'window', 'start', 'end', 'n_bars',
         'total_return', 'cagr', 'sharpe', 'sortino', 'max_dd']]
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print('\n=== Walk-forward segmented stats ===', flush=True)
    print(summary.to_string(index=False), flush=True)

    train_row = summary[summary.window == 'train'].iloc[0]
    val_row = summary[summary.window == 'val'].iloc[0]
    print(f'\nΔ (val − train): sharpe {val_row.sharpe - train_row.sharpe:+.4f}  '
          f'sortino {val_row.sortino - train_row.sortino:+.4f}  '
          f'cagr {val_row.cagr - train_row.cagr:+.4f}  '
          f'maxdd {val_row.max_dd - train_row.max_dd:+.4f}', flush=True)

    output = Path(REMOTE_REPO) / 'Output'
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    split_date = pd.Timestamp(VAL_START)
    ax.axvline(split_date, color='k', ls='--', alpha=0.5,
               label=f'train/val split ({VAL_START})')
    ax.legend(loc='upper left')
    ax.set_title(
        f'Analog cross_ticker on ex-Phase-2 universe '
        f'({prices.shape[1]} tickers, {start} → {end}, '
        f'top-{top_n}, rebal={rebal_days}d)')
    fig.tight_layout()
    fig_path = output / 'relational-exmegacap-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    stats_path = output / 'relational-exmegacap-stats.txt'
    stats_path.write_text(str(result.stats))

    seg_csv_path = output / 'relational-exmegacap-walkforward.csv'
    summary.to_csv(seg_csv_path, index=False)

    artifacts: dict[str, bytes] = {}
    for p in [fig_path, stats_path, seg_csv_path]:
        artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def main(prices_pkl_path: str = str(DEFAULT_PRICES_PKL)) -> None:
    src = Path(prices_pkl_path)
    if not src.exists():
        raise SystemExit(
            f'{src} not found — run `uv run python '
            f'apps/relational/scripts/modal/prep_exmegacap_prices.py` first.')
    pkl = src.read_bytes()
    print(f'>>> shipping {src} ({len(pkl):,} bytes) to Modal ...')
    artifacts = run_arm.remote(pkl)
    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f'\n=== Writing {len(artifacts)} artifacts to '
          f'{LOCAL_OUTPUT_DIR} ===')
    for name, blob in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(blob)
        print(f'  ← {out.name}  ({len(blob):,} bytes)')
    print('\nDone.')
