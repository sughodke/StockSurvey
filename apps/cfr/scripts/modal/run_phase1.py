"""Modal entrypoint — Phase 1 CFR walk-forward on the full StooqData/
stooq_us_long universe (312 tickers, 2000-2025).

CPU-only — `apps/cfr` is pure numpy + pandas, no tinygrad path. The
runtime is bottlenecked by the StooqData/ load (avoided here by
shipping a pre-prepped pickle) and the per-window precompute /
walk; total wall on Modal CPU should be well under 5 minutes.

Usage:
    # 1. One-time local prep (uses project venv with pandas + ss_loaders):
    uv run python apps/cfr/scripts/modal/prep_phase1_data.py

    # 2. Ship to Modal:
    uvx modal run apps/cfr/scripts/modal/run_phase1.py

    # Optional overrides (all have defaults matching the apps/cfr TODO):
    uvx modal run apps/cfr/scripts/modal/run_phase1.py \
        --train-window-days 1260 --val-window-days 780 --step-window-days 780 \
        --rebal-days 20 --top-k 20 --n-training-passes 1

Returns `Output/cfr-phase1.json` with per-window + summary stats +
verdict against the pre-registered Phase 1 cuts.
"""
from __future__ import annotations

import json
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
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'curl', 'build-essential')
    .pip_install('uv')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            'Output/**',
            'StooqData/**',         # close panel arrives via RPC
            'Nasdaq3347/**',
            # Drop unrelated apps so the image stays small and we don't
            # uv-sync irrelevant dep graphs.
            'apps/relational/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            'apps/replay/src/**',
            'apps/factor/src/**',
            'apps/lie/src/**',
            'apps/notebook/data/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('cfr-phase1', image=image)


@app.function(cpu=8, memory=24576, timeout=2 * 60 * 60)
def run_walkforward_remote(
    close_pickle: bytes,
    *,
    train_window_days: int,
    val_window_days: int,
    step_window_days: int,
    rebal_days: int,
    top_k: int,
    n_training_passes: int,
    seed: int,
) -> dict[str, bytes]:
    """Deserialize close DataFrame, run CFRWalkForward, return summary JSON."""
    import os
    import subprocess

    print('=== Step 1/3: uv sync workspace deps (cfr only) ===', flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        ['uv', 'sync', '--package', 'cfr', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')
    print(f'  sync wall: {time.perf_counter() - t0:.1f}s', flush=True)

    print('\n=== Step 2/3: deserialize close DataFrame ===', flush=True)
    import pickle
    import pandas as pd
    close: pd.DataFrame = pickle.loads(close_pickle)
    print(f'  close shape: {close.shape}  '
          f'date range: {close.index[0].date()} .. {close.index[-1].date()}',
          flush=True)

    print('\n=== Step 3/3: CFR walk-forward ===', flush=True)
    from cfr.menu import default_phase1_menu
    from cfr.state import default_infoset_builder
    from cfr.walkforward import CFRWalkForward

    driver = CFRWalkForward(
        menu_builder=lambda: default_phase1_menu(top_k=top_k),
        infoset_builder_factory=default_infoset_builder,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        step_window_days=step_window_days,
        rebal_days=rebal_days,
        commission_bps=10.0,
        n_training_passes=n_training_passes,
        rng_seed=seed,
    )
    menu = driver.menu_builder()
    print(f'  menu ({menu.n_actions} actions): {menu.action_keys}', flush=True)

    t1 = time.perf_counter()
    per_window, summary = driver.run(close)
    print(f'  walk wall: {time.perf_counter() - t1:.1f}s '
          f'({len(per_window)} windows)', flush=True)

    print(f'\n{"win":>3s} {"val_dates":>23s} {"cfr_sh":>7s} {"pas_sh":>7s} '
          f'{"trl_sh":>7s} {"nai_sh":>7s} {"alpha":>7s} {"vs_trl":>7s}',
          flush=True)
    print('-' * 95, flush=True)
    for w in per_window:
        print(f'{w.window_idx:>3d} {w.val_start}→{w.val_end} '
              f'{w.cfr_sharpe:>+7.3f} {w.passive_ew_sharpe:>+7.3f} '
              f'{w.trailing_best_sharpe:>+7.3f} '
              f'{w.naive_uniform_sharpe:>+7.3f} '
              f'{w.cfr_alpha:>+7.3f} '
              f'{w.cfr_sharpe - w.trailing_best_sharpe:>+7.3f}',
              flush=True)
    print(f'\nmean CFR Sharpe        = {summary["mean_cfr_sharpe"]:+.3f}',
          flush=True)
    print(f'mean passive EW Sharpe = {summary["mean_passive_sharpe"]:+.3f}',
          flush=True)
    print(f'mean trailing-best     = {summary["mean_trailing_sharpe"]:+.3f}',
          flush=True)
    print(f'mean CFR vs trailing   = '
          f'{summary["mean_cfr_minus_trailing_best"]:+.3f}', flush=True)
    print(f'CFR>trailing fraction  = '
          f'{summary["cfr_beats_trailing_fraction"]:.2f}', flush=True)
    print(f'\nverdict: {summary["verdict"]}', flush=True)

    payload = {
        'config': {
            'train_window_days': train_window_days,
            'val_window_days': val_window_days,
            'step_window_days': step_window_days,
            'rebal_days': rebal_days,
            'top_k': top_k,
            'n_training_passes': n_training_passes,
            'seed': seed,
            'universe_n': close.shape[1],
            'date_range': [str(close.index[0].date()),
                           str(close.index[-1].date())],
        },
        'summary': summary,
        'per_window': [w.__dict__ for w in per_window],
    }
    return {'cfr-phase1.json': json.dumps(payload, indent=2, default=str).encode()}


@app.local_entrypoint()
def main(
    train_window_days: int = 1260,   # ~5y
    val_window_days:   int = 780,    # ~3y
    step_window_days:  int = 780,
    rebal_days:        int = 20,
    top_k:             int = 20,
    n_training_passes: int = 1,
    seed:              int = 0,
) -> None:
    """Ship the pre-prepped pickle to Modal and write the result JSON."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pickle_path = LOCAL_OUTPUT_DIR / 'cfr_phase1_close.pkl'
    if not pickle_path.exists():
        raise SystemExit(
            f'pickle not found at {pickle_path}. Run prep first:\n'
            f'  uv run python apps/cfr/scripts/modal/prep_phase1_data.py')

    print(f'[local] reading {pickle_path} '
          f'({pickle_path.stat().st_size / 1024 / 1024:.1f} MB)', flush=True)
    close_pickle = pickle_path.read_bytes()

    print(f'[local] launching Modal run_walkforward_remote (CPU 8c) ...',
          flush=True)
    t0 = time.perf_counter()
    artifacts = run_walkforward_remote.remote(
        close_pickle,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        step_window_days=step_window_days,
        rebal_days=rebal_days,
        top_k=top_k,
        n_training_passes=n_training_passes,
        seed=seed,
    )
    print(f'[local] remote done in {time.perf_counter() - t0:.0f}s',
          flush=True)

    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)',
              flush=True)
