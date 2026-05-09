"""Modal entrypoint: Ricker vs polar Morlet A/B for the analog k-NN
scorer on the Phase-2 21-ticker pool.

The two arms share every other knob — same scales, same lookback,
same fp_window, same k_neighbors / forward_horizon / min_sep_days /
pool_mode / rebal_days / commission. The only thing that changes is
how the per-(date, ticker) fingerprint is built:

  * `analog-ricker` — legacy real Ricker coefficients, flattened over
    the trailing `fp_window` bars. fp_dim = S * w.
  * `analog-morlet` — polar Morlet + Gaussian bundle from
    `ss_features.causal_polar_morlet_matrix`: 4 channels per scale
    `(|c|, cos(arg), sin(arg), g)` flattened over the same trailing
    `fp_window` bars. fp_dim = 4 * S * w.

Mirrors the layout of `relational_dwt_phase2.py` (price prep is local,
shipped over RPC as a pickle blob; arms run sequentially in one
container; CWT panels cache to a Modal volume so the second arm hits
its own cache miss only once). Cache key includes the wavelet name so
Ricker and Morlet panels coexist.

Plain CPU instance is fine — pure numpy + bt + an mp.Pool fork over
the kNN t-axis. The Phase-2 universe (N=21) is small enough that
total wall-time is dominated by the Morlet kNN compute (fp_dim 672 vs
Ricker's 168 → ~4x BLAS matmul cost per query).

Usage
-----
    uv run python apps/relational/scripts/modal/prep_phase2_prices.py
    uvx modal run apps/relational/scripts/modal/relational_morlet_phase2.py
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

DEFAULT_PRICES_PKL = Path('/tmp/phase2-prices.pkl')

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

app = modal.App('ss-relational-morlet-phase2', image=image)


@app.function(cpu=8, memory=16384, timeout=60 * 60)
def run_arms(prices_pkl: bytes) -> dict[str, bytes]:
    """Run the 2-arm Ricker-vs-Morlet head-to-head and return artifacts."""
    import pickle
    import subprocess
    import warnings
    warnings.filterwarnings('ignore')

    print('=== Step 1/3: uv sync workspace deps (one-time per cold start) ===',
          flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'relational', '--extra', 'research',
         '--inexact'],
        cwd=REMOTE_REPO, check=True)

    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.13/site-packages')

    import os
    import bt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as _pd_local

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
    print(f'\n=== Step 2/3: loaded prices {prices.shape} '
          f'({start} → {end}) ===', flush=True)

    lookback = 120
    top_n = 10
    fp_window = 21
    rebal_days = 20
    commission_bps = 10.0
    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    n_workers = max(1, (os.cpu_count() or 8) - 1)

    arms: list[tuple[str, str]] = [
        ('analog-ricker', 'ricker'),
        ('analog-morlet', 'morlet'),
    ]

    weights_by_arm: dict[str, _pd_local.DataFrame] = {}
    for name, wavelet in arms:
        print(f'\n[{name}] computing weights (wavelet={wavelet}, '
              f'n_workers={n_workers}) ...', flush=True)
        scores = analog_knn_scores_fast(
            prices, lookback=lookback, scales=scales,
            fp_window=fp_window,
            k_neighbors=50, forward_horizon=20, min_sep_days=21,
            pool_mode='cross_ticker',
            wavelet=wavelet, n_workers=n_workers)
        scores = apply_nan_mask(scores, prices.values, lookback)
        weights = select_top_n_matrix(scores, top_n, ascending=False)
        weights_by_arm[name] = _pd_local.DataFrame(
            weights, index=prices.index[lookback:], columns=prices.columns)
        n_active = int((weights_by_arm[name].sum(axis=1) > 0).sum())
        print(f'[{name}] done — {n_active} rebalances rows', flush=True)

    print('\n=== Step 3/3: bt backtests + walk-forward split ===', flush=True)
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

    print('\n--- delta (morlet − ricker) ---', flush=True)
    for window_label in ('full', 'train', 'val'):
        base = summary[(summary.arm == 'analog-ricker')
                       & (summary.window == window_label)].iloc[0]
        comp = summary[(summary.arm == 'analog-morlet')
                       & (summary.window == window_label)].iloc[0]
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
        f'Relational analog k-NN — ricker vs polar Morlet — Phase-2 '
        f'({start} → {end}, top-{top_n}, rebal={rebal_days}d)')
    fig.tight_layout()
    fig_path = output / 'relational-morlet-phase2-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    stats_path = output / 'relational-morlet-phase2-stats.txt'
    stats_path.write_text(str(result.stats))

    seg_csv_path = output / 'relational-morlet-phase2-walkforward.csv'
    summary.to_csv(seg_csv_path, index=False)

    seg_txt_path = output / 'relational-morlet-phase2-walkforward.txt'
    with seg_txt_path.open('w') as f:
        f.write(summary.to_string(index=False))
        f.write('\n\n--- delta (morlet − ricker) ---\n')
        for window_label in ('full', 'train', 'val'):
            base = summary[(summary.arm == 'analog-ricker')
                           & (summary.window == window_label)].iloc[0]
            comp = summary[(summary.arm == 'analog-morlet')
                           & (summary.window == window_label)].iloc[0]
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
    """Read the Phase-2 prices pickle (made by `prep_phase2_prices.py`)
    and ship it to the remote container."""
    src = Path(prices_pkl_path)
    if not src.exists():
        raise SystemExit(
            f'{src} not found — run `uv run python '
            f'apps/relational/scripts/modal/prep_phase2_prices.py` first.')
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
