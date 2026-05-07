"""Modal A/B: regime baseline vs augmented (volume + market CWT).

Two arms, single Modal job, identical seed / n_trials / data:

  * Arm A (baseline): current `weights_regime` — price CWT only.
  * Arm B (augmented): `weights_regime` with `volumes=` + `use_market_cwt=True`.

Both arms run the full Optuna+vectorbt walk-forward on the StooqData/
panel filtered to the relational scoreboard's 2013-01-29 → 2025-12-11
range. Arm B stacks (price-CWT, log-volume-CWT, EW-market-CWT) along
the scale axis so the divergence becomes a joint regime-shift score.

Usage:
    # 1. One-time local prep:
    uv run python apps/regime/scripts/modal/prep_regime_data.py

    # 2. Ship to Modal:
    uvx modal run apps/regime/scripts/modal/baseline_vs_augmented.py

Returns `Output/regime-baseline-vs-aug-{baseline,augmented}.json` with
per-window train/val Sharpe and best params for each arm, plus a
combined `regime-baseline-vs-aug-summary.json` with the headline
comparison.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
PICKLE_PATH = LOCAL_OUTPUT_DIR / 'regime_baseline_vs_aug.pkl'
REMOTE_REPO = '/root/StockSurvey'


# CPU-only image — Optuna+vectorbt is numba/numpy, no GPU needed.
image = (
    modal.Image.debian_slim(python_version='3.12')
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
            'StooqData/**',         # not needed — we ship the bundle as RPC bytes
            'Nasdaq3347/**',
            'apps/factor/src/**',
            'apps/replay/src/**',
            'apps/notebook/src/**',
            'apps/v1/src/**',
            'apps/relational/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('regime-baseline-vs-aug', image=image)


@app.function(cpu=8, memory=24576, timeout=4 * 60 * 60)
def run_arms(
    bundle_pickle: bytes,
    *,
    n_trials: int,
    seed: int,
    rebalance_days: int,
    commission_bps: float,
    train_years: int,
    val_years: int,
    step_years: int,
    n_jobs: int,
) -> dict[str, bytes]:
    """Build TickerData on Modal, run baseline + augmented arms back-to-back."""
    import os, subprocess

    print('=== Step 1/4: uv sync workspace deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'regime', '--extra', 'research', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    # vectorbt's `numba` dep is stripped by the workspace's
    # [[tool.uv.dependency-metadata]] override (PyPI has no Intel-macOS
    # wheel; nix provides it locally). On Modal Linux x86_64 we can
    # grab numba from PyPI directly — install into the venv post-sync.
    subprocess.run(
        ['uv', 'pip', 'install', 'numba', 'llvmlite',
         '--python', f'{REMOTE_REPO}/.venv/bin/python'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    print('\n=== Step 2/4: deserialize panel bundle from RPC ===', flush=True)
    import pandas as pd
    import numpy as np
    bundle = pickle.loads(bundle_pickle)
    close: pd.DataFrame = bundle['close']
    highs: pd.DataFrame = bundle['highs']
    lows: pd.DataFrame = bundle['lows']
    volumes: pd.DataFrame = bundle['volumes']
    print(f'  close shape: {close.shape}  '
          f'date range: {close.index[0].date()} .. {close.index[-1].date()}',
          flush=True)
    print(f'  volume coverage: '
          f'{(volumes.notna() & (volumes > 0)).mean().mean() * 100:.1f}%',
          flush=True)

    from ss_indicators import corwin_schultz_spread
    print('  computing Corwin-Schultz spreads ...', flush=True)
    t0 = time.perf_counter()
    spread_df = corwin_schultz_spread(highs, lows)
    print(f'  done ({time.perf_counter() - t0:.1f}s)', flush=True)

    from regime.trainer import train as regime_train

    common_kwargs = dict(
        strategy='regime',
        n_trials=n_trials,
        n_jobs=n_jobs,
        rebalance_days=rebalance_days,
        metric='sharpe',
        commission_bps=commission_bps,
        train_years=train_years,
        val_years=val_years,
        step_years=step_years,
        seed=seed,
        use_log_returns=False,  # raw close — established finding
    )

    def _serialize(result, label: str) -> dict:
        windows = []
        for w in result.windows:
            windows.append({
                'train_start': w.train_start.date().isoformat(),
                'train_end':   w.train_end.date().isoformat(),
                'val_end':     w.val_end.date().isoformat(),
                'best_params': dict(w.best_params),
                'train_sharpe': float(w.train_score),
                'val_sharpe':   float(w.val_score),
            })
        val_sharpes = [w['val_sharpe'] for w in windows
                       if w['val_sharpe'] == w['val_sharpe']]  # not nan
        train_sharpes = [w['train_sharpe'] for w in windows
                         if w['train_sharpe'] == w['train_sharpe']]
        return {
            'arm': label,
            'n_windows': len(windows),
            'mean_val_sharpe':   (sum(val_sharpes) / len(val_sharpes))
                                  if val_sharpes else float('nan'),
            'median_val_sharpe': (float(np.median(val_sharpes))
                                  if val_sharpes else float('nan')),
            'mean_train_sharpe': (sum(train_sharpes) / len(train_sharpes))
                                  if train_sharpes else float('nan'),
            'positive_val_fraction': (sum(1 for s in val_sharpes if s > 0)
                                      / len(val_sharpes))
                                     if val_sharpes else float('nan'),
            'windows': windows,
        }

    print('\n=== Step 3/4: Arm A — baseline (price CWT only) ===', flush=True)
    t0 = time.perf_counter()
    result_baseline = regime_train(close, spread_df, **common_kwargs)
    print(f'  arm A wall: {time.perf_counter() - t0:.0f}s', flush=True)
    summary_baseline = _serialize(result_baseline, 'baseline')
    print(f'  arm A mean val Sharpe:   {summary_baseline["mean_val_sharpe"]:+.4f}',
          flush=True)
    print(f'  arm A median val Sharpe: {summary_baseline["median_val_sharpe"]:+.4f}',
          flush=True)
    print(f'  arm A positive-val frac: {summary_baseline["positive_val_fraction"]:.2f}',
          flush=True)

    print('\n=== Step 4/4: Arm B — augmented (price + volume + market CWT) ===',
          flush=True)
    t0 = time.perf_counter()
    result_augmented = regime_train(
        close, spread_df, **common_kwargs,
        volumes=volumes, use_market_cwt=True)
    print(f'  arm B wall: {time.perf_counter() - t0:.0f}s', flush=True)
    summary_augmented = _serialize(result_augmented, 'augmented')
    print(f'  arm B mean val Sharpe:   {summary_augmented["mean_val_sharpe"]:+.4f}',
          flush=True)
    print(f'  arm B median val Sharpe: {summary_augmented["median_val_sharpe"]:+.4f}',
          flush=True)
    print(f'  arm B positive-val frac: {summary_augmented["positive_val_fraction"]:.2f}',
          flush=True)

    print('\n=== Comparison ===', flush=True)
    delta = (summary_augmented['mean_val_sharpe']
             - summary_baseline['mean_val_sharpe'])
    print(f'  delta mean val Sharpe (augmented - baseline): {delta:+.4f}',
          flush=True)

    overall = {
        'date_range': [close.index[0].date().isoformat(),
                       close.index[-1].date().isoformat()],
        'n_universe': int(close.shape[1]),
        'n_trials_per_window': n_trials,
        'rebalance_days': rebalance_days,
        'commission_bps': commission_bps,
        'seed': seed,
        'baseline':  summary_baseline,
        'augmented': summary_augmented,
        'delta_mean_val_sharpe': delta,
    }

    return {
        'regime-baseline-vs-aug-baseline.json':  json.dumps(summary_baseline,  indent=2).encode(),
        'regime-baseline-vs-aug-augmented.json': json.dumps(summary_augmented, indent=2).encode(),
        'regime-baseline-vs-aug-summary.json':   json.dumps(overall,           indent=2).encode(),
    }


@app.local_entrypoint()
def main(
    n_trials: int = 20,
    seed: int = 42,
    rebalance_days: int = 20,
    commission_bps: float = 10.0,
    train_years: int = 5,
    val_years: int = 3,
    step_years: int = 3,
    n_jobs: int = 4,
) -> None:
    """Read pre-prepped pickle as bytes (no pandas import locally) and ship
    to Modal. Run `prep_regime_data.py` first.
    """
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PICKLE_PATH.exists():
        raise SystemExit(
            f'pickle not found at {PICKLE_PATH}. Run prep first:\n'
            f'  uv run python apps/regime/scripts/modal/prep_regime_data.py')

    print(f'[local] reading {PICKLE_PATH} '
          f'({PICKLE_PATH.stat().st_size / 1024 / 1024:.1f} MB)', flush=True)
    bundle_bytes = PICKLE_PATH.read_bytes()

    print(f'[local] launching Modal run_arms.remote ...', flush=True)
    t0 = time.perf_counter()
    artifacts = run_arms.remote(
        bundle_bytes,
        n_trials=n_trials,
        seed=seed,
        rebalance_days=rebalance_days,
        commission_bps=commission_bps,
        train_years=train_years,
        val_years=val_years,
        step_years=step_years,
        n_jobs=n_jobs,
    )
    print(f'[local] remote done in {time.perf_counter() - t0:.0f}s',
          flush=True)

    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)',
              flush=True)
