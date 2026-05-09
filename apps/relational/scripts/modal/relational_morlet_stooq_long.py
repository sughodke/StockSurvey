"""Modal entrypoint: 3-arm analog k-NN A/B on the full
`stooq_us_long` 312-ticker universe — gating experiment (2) from
`apps/docs/docs/findings/relational-morlet-failure.md`.

The Phase-2 21-ticker A/B (`relational_morlet_phase2.py`) showed the
raw polar Morlet bundle overfitting the 2013-2020 train slice (train
+0.22 / val −0.31 vs Ricker baseline). One mechanism hypothesis was
that the cross_ticker candidate pool at N=21 is too sparse to
constrain the bundle's 4× DOF. This entrypoint tests that directly
by running the same three arms — Ricker, raw Morlet, Morlet-DWT-L1 —
on the curated long-history US universe (~312 names), where the
candidate pool is ~15× larger.

Predictions if the small-N hypothesis is correct:

  * `analog-morlet` train>val gap shrinks substantially or reverses.
  * `analog-morlet-dwtL1` lands close to or above the Ricker baseline
    on val, and shouldn't widen the train>val gap (regularization
    + larger pool both pull in the same direction).

If the same train>val sign-flip happens at N=312, the universe-size
argument is wrong and the bundle has a deeper problem.

Algorithm: `analog_knn_scores_fast` with `n_workers=cpu_count`
(BLAS-bound matmul + a Python pick walk; the slow path would not
finish on N=312 inside reasonable Modal wall). Walk-forward
segmentation uses the same canonical Phase-2 dates so the table is
directly comparable to the existing Phase-2 row.

Walk-time estimate: ~60-90 min for the 3 arms cold; cached
re-runs hit the persistent Modal volume's panel npz and skip the
~5-10 min CWT precompute per wavelet.

Usage:
    uv run python apps/relational/scripts/modal/prep_stooq_long_prices.py
    uvx modal run apps/relational/scripts/modal/relational_morlet_stooq_long.py
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

DEFAULT_PRICES_PKL = Path('/tmp/stooq-long-prices.pkl')
CWT_CACHE_REMOTE = '/root/cwt-cache'

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

# Persistent CWT cache so Ricker + Morlet panels survive across runs.
# Same volume as `relational_exmegacap_modal.py` — the cache key includes
# both the wavelet name and the prices-bytes hash, so panels for the
# Phase-2 / ex-Phase-2 / full-stooq-long universes coexist without
# collision.
cwt_volume = modal.Volume.from_name(
    'ss-relational-cwt-cache', create_if_missing=True)

app = modal.App('ss-relational-morlet-stooq-long', image=image)


# Memory budget: at N=312 + Morlet fp_dim=672, the cand_fps tensor is
# ~900K candidates * 672 floats * 4 bytes ≈ 2.4 GB shared across
# fork()-spawned workers; per-worker matmul output is ~280 MB
# transient. 48 GB gives 2× headroom over peak; cpu=12 keeps BLAS
# threads isolated per worker (single-thread BLAS pinning lives in
# `_worker_init`). 4-hour timeout for the 3 arms.
@app.function(
    cpu=12, memory=49152, timeout=60 * 240,
    volumes={CWT_CACHE_REMOTE: cwt_volume},
)
def run_arms(prices_pkl: bytes) -> dict[str, bytes]:
    """Run the 3-arm Ricker / raw-Morlet / Morlet-DWT-L1 A/B on the
    full stooq_us_long universe and return artifacts."""
    import os
    import pickle
    import subprocess
    import time
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
    import pandas as _pd_local

    from ss_features import Compression
    from ss_portfolio import apply_nan_mask, select_top_n_matrix
    from ss_portfolio.bt_helpers import build_strategy

    from relational.analog_knn import analog_knn_scores_fast
    from relational.research.idea_b_analog_knn_dwt_walkforward import (
        TRAIN_END, TRAIN_START, VAL_END, VAL_START, segment_stats,
    )

    bundle = pickle.loads(prices_pkl)
    prices = bundle['prices']
    start = bundle['start']
    end = bundle['end']
    print(f'\n=== Step 2/4: prices {prices.shape} '
          f'({start} → {end}); {len(prices.columns)}-ticker '
          f'stooq_us_long universe ===', flush=True)

    lookback = 120
    top_n = 10
    fp_window = 21
    rebal_days = 20
    commission_bps = 10.0
    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    n_workers = int(os.environ.get('SS_N_WORKERS', os.cpu_count() or 8))

    comp_l1 = Compression(kind='dwt', levels=1, wavelet='haar',
                          pad_mode='periodization')

    arms: list[tuple[str, str, Compression | None]] = [
        ('analog-ricker',       'ricker', None),
        ('analog-morlet',       'morlet', None),
        ('analog-morlet-dwtL1', 'morlet', comp_l1),
    ]

    weights_by_arm: dict[str, _pd_local.DataFrame] = {}
    print(f'\n=== Step 3/4: 3-arm analog scores '
          f'(N={prices.shape[1]}, T={prices.shape[0]}, '
          f'n_workers={n_workers}) ===', flush=True)
    for name, wavelet, comp in arms:
        print(f'\n[{name}] computing weights (wavelet={wavelet}, '
              f'compression={comp!r}) ...', flush=True)
        t0 = time.time()
        scores = analog_knn_scores_fast(
            prices, lookback=lookback, scales=scales,
            fp_window=fp_window,
            k_neighbors=50, forward_horizon=20, min_sep_days=21,
            pool_mode='cross_ticker',
            wavelet=wavelet, compression=comp,
            cache_dir=CWT_CACHE_REMOTE,
            n_workers=n_workers)
        elapsed = time.time() - t0
        finite_frac = float(np.isfinite(scores).mean())
        print(f'[{name}] scores shape={scores.shape} '
              f'finite={finite_frac:.3f} in {elapsed:.1f}s', flush=True)
        scores = apply_nan_mask(scores, prices.values, lookback)
        weights = select_top_n_matrix(scores, top_n, ascending=False)
        weights_by_arm[name] = _pd_local.DataFrame(
            weights, index=prices.index[lookback:], columns=prices.columns)
        n_active = int((weights_by_arm[name].sum(axis=1) > 0).sum())
        print(f'[{name}] done — {n_active} rebalances rows', flush=True)

    print(f'\n=== Step 4/4: bt + walk-forward segmentation ===', flush=True)
    strategies = [
        build_strategy(name, prices, w,
                       rebal_days=rebal_days, commission_bps=commission_bps)
        for name, w in weights_by_arm.items()
    ]
    result = bt.run(*strategies)
    result.display()

    eq_panel = result.prices
    seg_rows: list[dict] = []
    for arm_name in weights_by_arm:
        equity = eq_panel[arm_name]
        for window_label, w_start, w_end in (
            ('full',  TRAIN_START, VAL_END),
            ('train', TRAIN_START, TRAIN_END),
            ('val',   VAL_START,   VAL_END),
        ):
            stats = segment_stats(equity, w_start, w_end)
            stats.update({'arm': arm_name, 'window': window_label,
                          'start': w_start, 'end': w_end})
            seg_rows.append(stats)

    summary = _pd_local.DataFrame(seg_rows)[
        ['arm', 'window', 'start', 'end', 'n_bars',
         'total_return', 'cagr', 'sharpe', 'sortino', 'max_dd']]

    print('\n=== Per-arm walk-forward segmented stats ===', flush=True)
    _pd_local.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(summary.to_string(index=False), flush=True)

    def _arm_row(arm_name: str, window_label: str):
        return summary[(summary.arm == arm_name)
                       & (summary.window == window_label)].iloc[0]

    delta_pairs = (
        ('morlet − ricker',          'analog-ricker', 'analog-morlet'),
        ('morlet-dwtL1 − ricker',    'analog-ricker', 'analog-morlet-dwtL1'),
        ('morlet-dwtL1 − morlet',    'analog-morlet', 'analog-morlet-dwtL1'),
    )
    for label, base_arm, comp_arm in delta_pairs:
        print(f'\n--- delta ({label}) ---', flush=True)
        for window_label in ('full', 'train', 'val'):
            base = _arm_row(base_arm, window_label)
            comp = _arm_row(comp_arm, window_label)
            print(f'  {window_label:6s}  '
                  f'Δsharpe={comp.sharpe - base.sharpe:+.4f}  '
                  f'Δsortino={comp.sortino - base.sortino:+.4f}  '
                  f'Δcagr={comp.cagr - base.cagr:+.4f}  '
                  f'Δmaxdd={comp.max_dd - base.max_dd:+.4f}', flush=True)

    output = Path(REMOTE_REPO) / 'Output'
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    split_date = _pd_local.Timestamp(VAL_START)
    ax.axvline(split_date, color='k', ls='--', alpha=0.5,
               label=f'train/val split ({VAL_START})')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(
        f'Relational analog k-NN — ricker / polar Morlet / Morlet-DWT-L1 '
        f'— stooq_us_long ({prices.shape[1]} tickers, {start} → {end}, '
        f'top-{top_n}, rebal={rebal_days}d)')
    fig.tight_layout()
    fig_path = output / 'relational-morlet-stooq-long-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    stats_path = output / 'relational-morlet-stooq-long-stats.txt'
    stats_path.write_text(str(result.stats))

    seg_csv_path = output / 'relational-morlet-stooq-long-walkforward.csv'
    summary.to_csv(seg_csv_path, index=False)

    seg_txt_path = output / 'relational-morlet-stooq-long-walkforward.txt'
    with seg_txt_path.open('w') as f:
        f.write(summary.to_string(index=False))
        for label, base_arm, comp_arm in delta_pairs:
            f.write(f'\n\n--- delta ({label}) ---\n')
            for window_label in ('full', 'train', 'val'):
                base = _arm_row(base_arm, window_label)
                comp = _arm_row(comp_arm, window_label)
                f.write(f'  {window_label:6s}  '
                        f'Δsharpe={comp.sharpe - base.sharpe:+.4f}  '
                        f'Δsortino={comp.sortino - base.sortino:+.4f}  '
                        f'Δcagr={comp.cagr - base.cagr:+.4f}  '
                        f'Δmaxdd={comp.max_dd - base.max_dd:+.4f}\n')

    artifacts: dict[str, bytes] = {}
    for p in [fig_path, stats_path, seg_csv_path, seg_txt_path]:
        artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def main(prices_pkl_path: str = str(DEFAULT_PRICES_PKL)) -> None:
    src = Path(prices_pkl_path)
    if not src.exists():
        raise SystemExit(
            f'{src} not found — run `uv run python '
            f'apps/relational/scripts/modal/prep_stooq_long_prices.py` first.')
    pkl = src.read_bytes()
    print(f'>>> shipping {src} ({len(pkl):,} bytes) to Modal ...')
    artifacts = run_arms.remote(pkl)
    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f'\n=== Writing {len(artifacts)} artifacts to '
          f'{LOCAL_OUTPUT_DIR} ===')
    for name, blob in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(blob)
        print(f'  ← {out.name}  ({len(blob):,} bytes)')
    print('\nDone.')
