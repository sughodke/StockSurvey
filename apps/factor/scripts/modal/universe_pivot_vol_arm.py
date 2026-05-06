"""Modal entrypoint to run the vol_innovation arm of the universe pivot.

Local computer crashed mid-arm during the 2073-ticker walkforward.
Recovery path: pre-prep the filtered close DataFrame locally (~110 MB
pickle), ship through Modal's RPC, run on T4 GPU, return per-window
results.

Mirrors the universe-pivot setup from
`apps/factor/scripts/universe_pivot_walkforward.py`:
  * full StooqData/ archive (filtered to 2162 raw → 2073 buildable
    tickers via the leading-NaN trim fix)
  * IndicatorGridConfig (74 channels), identity backbone (K=1)
  * train=63 / val=39 / step=39 walk-forward, rebal=20d
  * AdamW lr=1e-2 wd=1e-3 n_steps=200
  * vol_innovation target only — wide-return arm already saved at
    `Output/universe-pivot-wide-return-windows.npz` from the local
    run before the crash.

Usage:
    # 1. One-time local prep (uses project venv with pandas + ss_loaders):
    uv run python apps/factor/scripts/modal/prep_universe_pivot_data.py

    # 2. Ship to Modal (uvx, isolated env — no project-venv deps locally):
    uvx modal run apps/factor/scripts/modal/universe_pivot_vol_arm.py

Returns `Output/universe-pivot-wide-vol-windows.npz` (overwrites the
stale 95-ticker smoke version on disk).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'


image = (
    modal.Image.from_registry(
        'nvidia/cuda:12.4.0-devel-ubuntu22.04',
        add_python='3.12',
    )
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            'Output/**',
            'StooqData/**',         # not needed — we ship close DataFrame as RPC bytes
            'Nasdaq3347/**',
            'apps/relational/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            'apps/replay/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('factor-universe-pivot-vol', image=image)


def _build_one_ticker_modal(args):
    """Top-level for mp.Pool pickling. Same NaN-trim logic as the
    local universe_pivot driver."""
    ticker, prices_bytes, dates_bytes, dates_shape, cfg = args
    import numpy as np
    from factor import build_indicator_features
    from ss_features import TickerData
    prices = np.frombuffer(prices_bytes, dtype=np.float64).copy()
    dates = np.frombuffer(dates_bytes, dtype='datetime64[ns]').reshape(dates_shape).copy()
    finite = np.isfinite(prices)
    if not finite.any():
        return ticker, None, '(no finite prices)'
    first_valid = int(np.argmax(finite))
    prices_trimmed = prices[first_valid:]
    try:
        feats_trimmed, valid_trimmed = build_indicator_features(
            prices_trimmed, cfg)
    except Exception as e:
        return ticker, None, f'({type(e).__name__}: {e})'
    F = feats_trimmed.shape[1]
    feats = np.full((len(prices), F), np.nan, dtype=np.float32)
    feats[first_valid:] = feats_trimmed
    valid = np.zeros(len(prices), dtype=bool)
    valid[first_valid:] = valid_trimmed
    if not valid.any():
        return ticker, None, '(no valid bars)'
    return ticker, TickerData(
        name=ticker, prices=prices, dates=dates,
        features=feats, targets={}, valid=valid,
    ), None


@app.function(gpu='T4', cpu=8, memory=24576, timeout=2 * 60 * 60)
def vol_arm(
    close_pickle: bytes,
    *,
    rebal_days: int,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    n_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, bytes]:
    """Build TickerData + run vol_innovation walkforward on T4."""
    import os, subprocess
    os.environ['CUDA'] = '1'

    print('=== Step 1/4: uv sync workspace deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'factor', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(
            f'tinygrad picked Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    print('\n=== Step 2/4: deserialize close DataFrame from RPC ===', flush=True)
    import io
    import numpy as np
    import pandas as pd
    close: pd.DataFrame = pickle.loads(close_pickle)
    print(f'  close shape: {close.shape}  '
          f'date range: {close.index[0].date()} .. {close.index[-1].date()}',
          flush=True)

    print('\n=== Step 3/4: build TickerData per column (parallel, leading-NaN trim) ===',
          flush=True)
    from factor import IndicatorGridConfig
    from ss_features import TickerData
    import multiprocessing as mp

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'  cfg.feature_width() = {F}', flush=True)

    dates = np.asarray(close.index)
    dates_bytes = dates.tobytes()
    dates_shape = dates.shape
    n_total = close.shape[1]
    n_workers = max(1, os.cpu_count() or 4)
    print(f'  parallelizing across {n_workers} workers', flush=True)

    work_args = []
    for col in close.columns:
        prices = close[col].values.astype(np.float64)
        work_args.append((col, prices.tobytes(), dates_bytes, dates_shape, cfg))

    ticker_data: list[TickerData] = []
    failed: list[str] = []
    t0 = time.perf_counter()
    with mp.Pool(n_workers) as pool:
        for i, (ticker, td, err) in enumerate(
                pool.imap_unordered(_build_one_ticker_modal, work_args)):
            if td is not None:
                ticker_data.append(td)
            else:
                failed.append(f'{ticker} {err}')
            if (i + 1) % 200 == 0:
                print(f'  built {i + 1}/{n_total}  '
                      f'({time.perf_counter() - t0:.0f}s)', flush=True)
    ticker_data.sort(key=lambda td: td.name)
    print(f'  feature build done: {len(ticker_data)} usable / '
          f'{len(failed)} skipped  ({time.perf_counter() - t0:.0f}s)',
          flush=True)

    print('\n=== Step 4/4: walkforward (vol_innovation arm) ===', flush=True)
    from factor import train_scorer_indicators_walkforward
    t1 = time.perf_counter()
    wf = train_scorer_indicators_walkforward(
        ticker_data, cfg=cfg,
        rebal_days=rebal_days,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        scorer='linear',
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        forward_target_kind='vol_innovation',
    )
    print(f'  arm wall: {time.perf_counter() - t1:.0f}s', flush=True)

    # Per-window detail
    print(f'  per-window val IC: '
          f'{[round(w.val_ic, 4) for w in wf.windows]}', flush=True)
    print(f'  mean val IC: {wf.mean_val_ic:+.4f}', flush=True)
    print(f'  median val IC: {wf.median_val_ic:+.4f}', flush=True)
    print(f'  mean val Sharpe: {wf.mean_val_sharpe:+.3f}', flush=True)
    print(f'  positive-val-IC fraction: '
          f'{wf.positive_val_ic_fraction:.2f}  '
          f'({sum(1 for w in wf.windows if w.val_ic > 0)}/{wf.n_windows})',
          flush=True)

    # Pack windows.npz blob.
    blob = {
        'window_idx':   np.array([w.window_idx for w in wf.windows]),
        'train_ic':     np.array([w.train_ic for w in wf.windows]),
        'val_ic':       np.array([w.val_ic for w in wf.windows]),
        'train_sharpe': np.array([w.train_sharpe for w in wf.windows]),
        'val_sharpe':   np.array([w.val_sharpe for w in wf.windows]),
    }
    blob['_meta'] = np.array(json.dumps({
        'forward_target_kind': wf.forward_target_kind,
        'rebal_days': wf.rebal_days,
        'feature_width': wf.feature_width,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
        'n_universe': len(ticker_data),
    }))
    buf = io.BytesIO()
    np.savez(buf, **blob)
    return {'universe-pivot-wide-vol-windows.npz': buf.getvalue()}


@app.local_entrypoint()
def main(
    rebal_days: int = 20,
    train_window_blocks: int = 63,
    val_window_blocks: int = 39,
    step_window_blocks: int = 39,
    n_steps: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> None:
    """Read pre-prepped pickle as bytes (no pandas import locally) and ship
    to Modal. Run `prep_universe_pivot_data.py` first.
    """
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pickle_path = LOCAL_OUTPUT_DIR / 'universe_pivot_close.pkl'
    if not pickle_path.exists():
        raise SystemExit(
            f'pickle not found at {pickle_path}. Run prep first:\n'
            f'  uv run python apps/factor/scripts/modal/'
            f'prep_universe_pivot_data.py')

    print(f'[local] reading {pickle_path} '
          f'({pickle_path.stat().st_size / 1024 / 1024:.1f} MB)', flush=True)
    close_pickle = pickle_path.read_bytes()

    print(f'[local] launching Modal vol_arm.remote (T4) ...', flush=True)
    t0 = time.perf_counter()
    artifacts = vol_arm.remote(
        close_pickle,
        rebal_days=rebal_days,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    print(f'[local] remote done in {time.perf_counter() - t0:.0f}s',
          flush=True)

    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)',
              flush=True)
