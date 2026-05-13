"""Modal entrypoint — Phase 2a CFR walk-forward with the enriched
deterministic action menu (Phase 1 + 4 documented-alpha modes).

Same architecture as `run_phase1.py`; the only difference is the
menu builder. Pre-prep pickle from
`apps/cfr/scripts/modal/prep_phase1_data.py` (no separate prep
needed — same close panel).

Usage:
    uv run python apps/cfr/scripts/modal/prep_phase1_data.py
    uvx modal run apps/cfr/scripts/modal/run_phase2a.py

Returns `Output/cfr-phase2a.json`.
"""
from __future__ import annotations

import json
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
            'StooqData/**',
            'Nasdaq3347/**',
            'apps/relational/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            'apps/replay/src/**',
            'apps/factor/src/**',
            'apps/lie/src/**',
            'apps/notebook/data/**',
            'apps/docs/**',         # docs aren't needed in Modal
            'packages/edgar/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('cfr-phase2a', image=image)


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

    print('\n=== Step 3/3: CFR Phase 2a walk-forward ===', flush=True)
    from cfr.menu import default_phase2a_menu
    from cfr.state import default_infoset_builder
    from cfr.walkforward import CFRWalkForward

    driver = CFRWalkForward(
        menu_builder=lambda: default_phase2a_menu(top_k=top_k),
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
    print(f'  Phase 2a menu ({menu.n_actions} actions):', flush=True)
    for k in menu.action_keys:
        print(f'    {k}', flush=True)

    t1 = time.perf_counter()
    per_window, summary = driver.run(close)
    print(f'\n  walk wall: {time.perf_counter() - t1:.1f}s '
          f'({len(per_window)} windows)', flush=True)

    print(f'\n{"win":>3s} {"val_dates":>23s} {"cfr_sh":>7s} {"pas_sh":>7s} '
          f'{"trl_sh":>7s} {"nai_sh":>7s} {"alpha":>7s} {"vs_trl":>7s} '
          f'{"vs_nai":>7s}', flush=True)
    print('-' * 110, flush=True)
    for w in per_window:
        print(f'{w.window_idx:>3d} {w.val_start}→{w.val_end} '
              f'{w.cfr_sharpe:>+7.3f} {w.passive_ew_sharpe:>+7.3f} '
              f'{w.trailing_best_sharpe:>+7.3f} '
              f'{w.naive_uniform_sharpe:>+7.3f} '
              f'{w.cfr_alpha:>+7.3f} '
              f'{w.cfr_sharpe - w.trailing_best_sharpe:>+7.3f} '
              f'{w.cfr_sharpe - w.naive_uniform_sharpe:>+7.3f}',
              flush=True)
    print(f'\nmean CFR Sharpe        = {summary["mean_cfr_sharpe"]:+.3f}',
          flush=True)
    print(f'mean passive EW Sharpe = {summary["mean_passive_sharpe"]:+.3f}',
          flush=True)
    print(f'mean trailing-best     = {summary["mean_trailing_sharpe"]:+.3f}',
          flush=True)
    print(f'mean naive uniform     = {summary["mean_naive_sharpe"]:+.3f}',
          flush=True)
    print(f'mean CFR vs trailing   = '
          f'{summary["mean_cfr_minus_trailing_best"]:+.3f}', flush=True)
    cfr_minus_naive = (summary['mean_cfr_sharpe']
                       - summary['mean_naive_sharpe'])
    print(f'mean CFR vs naive      = {cfr_minus_naive:+.3f}', flush=True)
    print(f'CFR>trailing fraction  = '
          f'{summary["cfr_beats_trailing_fraction"]:.2f}', flush=True)
    print(f'\nverdict: {summary["verdict"]}', flush=True)

    payload = {
        'config': {
            'phase': '2a',
            'menu': 'default_phase2a_menu',
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
    return {'cfr-phase2a.json': json.dumps(payload, indent=2, default=str).encode()}


@app.local_entrypoint()
def main(
    train_window_days: int = 1260,
    val_window_days:   int = 780,
    step_window_days:  int = 780,
    rebal_days:        int = 20,
    top_k:             int = 20,
    n_training_passes: int = 1,
    seed:              int = 0,
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pickle_path = LOCAL_OUTPUT_DIR / 'cfr_phase1_close.pkl'
    if not pickle_path.exists():
        raise SystemExit(
            f'pickle not found at {pickle_path}. Run prep first:\n'
            f'  uv run python apps/cfr/scripts/modal/prep_phase1_data.py')

    print(f'[local] reading {pickle_path} '
          f'({pickle_path.stat().st_size / 1024 / 1024:.1f} MB)', flush=True)
    close_pickle = pickle_path.read_bytes()

    print(f'[local] launching Modal Phase 2a (CPU 8c) ...', flush=True)
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
