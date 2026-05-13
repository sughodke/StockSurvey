"""Modal entrypoint — Phase 2b CFR walk-forward with the Phase 2a
deterministic menu PLUS a `Top13FConsensusMode` derived from real
SEC 13F-HR filings across 15 curated institutional managers.

Pipeline:
    apps/cfr/scripts/modal/prep_phase1_data.py    → close panel pickle
    apps/cfr/scripts/modal/prep_phase2b_data.py   → 13F consensus panel pickle
    uvx modal run apps/cfr/scripts/modal/run_phase2b.py

Returns `Output/cfr-phase2b.json`.

The consensus panel is built from 13F-HR filings since 2013-01-01;
walk-forward windows starting before 2013 (the early w0, w1) will
have the `top13f` mode reduced to cash for most of their span. The
algorithm should learn this and not concentrate on 13F in those
windows; in late windows (2013+) the 13F mode has real signal
content.
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
            '.edgar-cache/**',     # cache is large; consensus panel ships via RPC
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
            'apps/docs/**',         # in-flight doc edits not needed in Modal
            '**/__pycache__/**',
            '**/.pytest_cache/**',
            '**/.ruff_cache/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('cfr-phase2b', image=image)


@app.function(cpu=8, memory=24576, timeout=2 * 60 * 60)
def run_walkforward_remote(
    close_pickle: bytes,
    consensus_pickle: bytes,
    *,
    train_window_days: int,
    val_window_days: int,
    step_window_days: int,
    rebal_days: int,
    top_k: int,
    n_training_passes: int,
    seed: int,
    filing_lag_days: int,
) -> dict[str, bytes]:
    import os
    import subprocess

    print('=== Step 1/4: uv sync workspace deps ===', flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        ['uv', 'sync', '--package', 'cfr', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')
    print(f'  sync wall: {time.perf_counter() - t0:.1f}s', flush=True)

    print('\n=== Step 2/4: deserialize close + consensus panels ===', flush=True)
    import pickle
    import pandas as pd
    close: pd.DataFrame = pickle.loads(close_pickle)
    consensus: pd.DataFrame = pickle.loads(consensus_pickle)
    print(f'  close shape: {close.shape}  '
          f'date range: {close.index[0].date()} .. {close.index[-1].date()}',
          flush=True)
    print(f'  consensus shape: {consensus.shape}  '
          f'period range: {consensus.index[0].date()} .. {consensus.index[-1].date()}',
          flush=True)
    print(f'  in-universe consensus tickers: '
          f'{len([c for c in consensus.columns if c in close.columns])}',
          flush=True)

    print('\n=== Step 3/4: build Phase 2b menu (Phase 2a + top13f) ===', flush=True)
    from cfr.menu import (
        ActionMenu, EqualWeightMode, TopKMode, default_phase2a_menu,
    )
    from cfr.modes_13f import Top13FConsensusMode
    from cfr.state import default_infoset_builder
    from cfr.walkforward import CFRWalkForward

    def build_menu() -> ActionMenu:
        modes = [
            EqualWeightMode(name='ew'),
            TopKMode(name='mom', score_kind='momentum',
                     score_window=21, top_k=top_k),
            TopKMode(name='rev', score_kind='reversal',
                     score_window=5, top_k=top_k),
            TopKMode(name='lowv', score_kind='low_vol',
                     score_window=21, top_k=top_k),
            TopKMode(name='highv', score_kind='high_vol',
                     score_window=21, top_k=top_k),
            TopKMode(name='mom121', score_kind='mom_12_1',
                     score_window=252, top_k=top_k, min_lookback=252),
            TopKMode(name='lowv252', score_kind='low_vol',
                     score_window=252, top_k=top_k, min_lookback=252),
            TopKMode(name='shtop', score_kind='sharpe_top',
                     score_window=252, top_k=top_k, min_lookback=252),
            TopKMode(name='trend', score_kind='trend_str',
                     score_window=252, top_k=top_k, min_lookback=252),
            Top13FConsensusMode(
                name='top13f', consensus_panel=consensus,
                filing_lag_days=filing_lag_days,
            ),
        ]
        return ActionMenu(modes=modes, gross_levels=(0.0, 0.5, 1.0, 2.0))

    driver = CFRWalkForward(
        menu_builder=build_menu,
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
    print(f'  Phase 2b menu ({menu.n_actions} actions):', flush=True)
    for k in menu.action_keys:
        print(f'    {k}', flush=True)

    print('\n=== Step 4/4: walk-forward ===', flush=True)
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
            'phase': '2b',
            'menu': 'phase2a + top13f (15 funds, 13F-HR since 2013)',
            'train_window_days': train_window_days,
            'val_window_days': val_window_days,
            'step_window_days': step_window_days,
            'rebal_days': rebal_days,
            'top_k': top_k,
            'n_training_passes': n_training_passes,
            'filing_lag_days': filing_lag_days,
            'seed': seed,
            'universe_n': close.shape[1],
            'consensus_n_quarters': consensus.shape[0],
            'consensus_n_tickers_in_universe': len(
                [c for c in consensus.columns if c in close.columns]),
            'date_range': [str(close.index[0].date()),
                           str(close.index[-1].date())],
        },
        'summary': summary,
        'per_window': [w.__dict__ for w in per_window],
    }
    return {'cfr-phase2b.json': json.dumps(payload, indent=2, default=str).encode()}


@app.local_entrypoint()
def main(
    train_window_days: int = 1260,
    val_window_days:   int = 780,
    step_window_days:  int = 780,
    rebal_days:        int = 20,
    top_k:             int = 20,
    n_training_passes: int = 1,
    seed:              int = 0,
    filing_lag_days:   int = 45,
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close_path = LOCAL_OUTPUT_DIR / 'cfr_phase1_close.pkl'
    consensus_path = LOCAL_OUTPUT_DIR / 'cfr_phase2b_consensus_panel.pkl'
    for p in (close_path, consensus_path):
        if not p.exists():
            raise SystemExit(
                f'pickle not found at {p}. Run prep first:\n'
                f'  uv run python apps/cfr/scripts/modal/prep_phase1_data.py\n'
                f'  uv run python apps/cfr/scripts/modal/prep_phase2b_data.py')

    close_pickle = close_path.read_bytes()
    consensus_pickle = consensus_path.read_bytes()
    print(f'[local] close pickle:     {len(close_pickle) / 1024 / 1024:.1f} MB',
          flush=True)
    print(f'[local] consensus pickle: {len(consensus_pickle) / 1024:.1f} KB',
          flush=True)

    print(f'[local] launching Modal Phase 2b (CPU 8c) ...', flush=True)
    t0 = time.perf_counter()
    artifacts = run_walkforward_remote.remote(
        close_pickle, consensus_pickle,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        step_window_days=step_window_days,
        rebal_days=rebal_days,
        top_k=top_k,
        n_training_passes=n_training_passes,
        seed=seed,
        filing_lag_days=filing_lag_days,
    )
    print(f'[local] remote done in {time.perf_counter() - t0:.0f}s',
          flush=True)

    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)',
              flush=True)
