"""Modal entrypoint — Phase 3 Deep CFR walk-forward.

Replaces tabular CFR's `(infoset, action) → cumulative regret` with
a tinygrad MLP regret_net `(state_vec → predicted regret per action)`
over a continuous 10-feature state vector (6 universe-internal + 4
macro). Same Phase 2b 31-action menu (Phase 2a deterministic +
top13f), same per-bar action availability (Phase 2b bugfix),
same Phase 2b friction.

Pipeline:
    apps/cfr/scripts/modal/prep_phase1_data.py     → close panel pickle
    apps/cfr/scripts/modal/prep_phase2b_data.py    → 13F consensus pickle
    apps/cfr/scripts/modal/prep_phase3_data.py     → macro panel pickle
    uvx modal run apps/cfr/scripts/modal/run_phase3.py

Returns `Output/cfr-phase3.json` with per-window + summary stats +
verdict against the pre-registered Phase 3 cut: deep CFR mean
Sharpe ≥ Phase 1 + 0.15 (i.e., ≥ +0.74) AND CFR > naive uniform
on Phase 2b menu by ≥ +0.10.
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
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            '.edgar-cache/**',
            '.macro-cache/**',
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
            'apps/docs/**',
            '**/__pycache__/**',
            '**/.pytest_cache/**',
            '**/.ruff_cache/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('cfr-phase3', image=image)


@app.function(cpu=8, memory=24576, timeout=2 * 60 * 60)
def run_walkforward_remote(
    close_pickle: bytes,
    consensus_pickle: bytes,
    macro_pickle: bytes,
    *,
    train_window_days: int,
    val_window_days: int,
    step_window_days: int,
    rebal_days: int,
    top_k: int,
    hidden: int,
    learning_rate: float,
    batch_size: int,
    train_every: int,
    n_sgd_per_batch: int,
    seed: int,
    filing_lag_days: int,
    output_name: str,
) -> dict[str, bytes]:
    import os
    import subprocess

    print('=== Step 1/4: uv sync workspace deps (cfr) ===', flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        ['uv', 'sync', '--package', 'cfr', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    # Tinygrad needed for Phase 3; install separately to keep base
    # cfr deps light.
    subprocess.run(
        ['uv', 'pip', 'install', 'tinygrad'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')
    print(f'  sync wall: {time.perf_counter() - t0:.1f}s', flush=True)

    print('\n=== Step 2/4: deserialize panels ===', flush=True)
    import pickle
    import pandas as pd
    close: pd.DataFrame = pickle.loads(close_pickle)
    consensus: pd.DataFrame = pickle.loads(consensus_pickle)
    macro: pd.DataFrame = pickle.loads(macro_pickle)
    print(f'  close: {close.shape}  '
          f'{close.index[0].date()} → {close.index[-1].date()}', flush=True)
    print(f'  consensus: {consensus.shape}  '
          f'{consensus.index[0].date()} → {consensus.index[-1].date()}',
          flush=True)
    print(f'  macro: {macro.shape}  cols {list(macro.columns)}',
          flush=True)

    print('\n=== Step 3/4: build menu + state-vec builder ===', flush=True)
    from cfr.menu import (
        ActionMenu, EqualWeightMode, TopKMode,
    )
    from cfr.modes_13f import Top13FConsensusMode
    from cfr.state_vec import StateVecBuilder
    from cfr.deep_walkforward import DeepCFRWalkForward

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

    driver = DeepCFRWalkForward(
        menu_builder=build_menu,
        state_vec_builder_factory=StateVecBuilder,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        step_window_days=step_window_days,
        rebal_days=rebal_days,
        commission_bps=10.0,
        rng_seed=seed,
        hidden=hidden,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_every=train_every,
        n_sgd_per_batch=n_sgd_per_batch,
    )
    menu = driver.menu_builder()
    print(f'  Phase 3 menu ({menu.n_actions} actions)', flush=True)

    print('\n=== Step 4/4: deep CFR walk-forward ===', flush=True)
    t1 = time.perf_counter()
    per_window, summary = driver.run(close, macro)
    print(f'\n  walk wall: {time.perf_counter() - t1:.1f}s '
          f'({len(per_window)} windows)', flush=True)

    print(f'\n{"win":>3s} {"val_dates":>23s} {"cfr_sh":>7s} {"pas_sh":>7s} '
          f'{"trl_sh":>7s} {"nai_sh":>7s} {"alpha":>7s} {"vs_trl":>7s} '
          f'{"vs_nai":>7s} {"loss":>10s}', flush=True)
    print('-' * 122, flush=True)
    for w in per_window:
        print(f'{w.window_idx:>3d} {w.val_start}→{w.val_end} '
              f'{w.cfr_sharpe:>+7.3f} {w.passive_ew_sharpe:>+7.3f} '
              f'{w.trailing_best_sharpe:>+7.3f} '
              f'{w.naive_uniform_sharpe:>+7.3f} '
              f'{w.cfr_alpha:>+7.3f} '
              f'{w.cfr_sharpe - w.trailing_best_sharpe:>+7.3f} '
              f'{w.cfr_sharpe - w.naive_uniform_sharpe:>+7.3f} '
              f'{w.final_train_loss:>10.3e}',
              flush=True)
    print(f'\nmean CFR Sharpe        = {summary["mean_cfr_sharpe"]:+.3f}',
          flush=True)
    print(f'mean passive EW Sharpe = {summary["mean_passive_sharpe"]:+.3f}',
          flush=True)
    print(f'mean trailing-best     = {summary["mean_trailing_sharpe"]:+.3f}',
          flush=True)
    print(f'mean naive uniform     = {summary["mean_naive_sharpe"]:+.3f}',
          flush=True)
    print(f'mean CFR vs trailing   = {summary["mean_cfr_minus_trailing"]:+.3f}',
          flush=True)
    print(f'mean CFR vs naive      = {summary["mean_cfr_minus_naive"]:+.3f}',
          flush=True)
    print(f'\nverdict: {summary["verdict"]}', flush=True)

    payload = {
        'config': {
            'phase': '3',
            'menu': 'phase2a + top13f (deep CFR replaces tabular)',
            'state_dim': per_window[0].state_dim if per_window else 0,
            'train_window_days': train_window_days,
            'val_window_days': val_window_days,
            'step_window_days': step_window_days,
            'rebal_days': rebal_days,
            'top_k': top_k,
            'hidden': hidden,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'train_every': train_every,
            'n_sgd_per_batch': n_sgd_per_batch,
            'seed': seed,
            'filing_lag_days': filing_lag_days,
            'universe_n': close.shape[1],
        },
        'summary': summary,
        'per_window': [w.__dict__ for w in per_window],
    }
    return {output_name: json.dumps(payload, indent=2, default=str).encode()}


@app.local_entrypoint()
def main(
    train_window_days: int = 1260,
    val_window_days:   int = 780,
    step_window_days:  int = 780,
    rebal_days:        int = 20,
    top_k:             int = 20,
    hidden:            int = 64,
    learning_rate:     float = 1e-3,
    batch_size:        int = 64,
    train_every:       int = 5,
    n_sgd_per_batch:   int = 5,
    seed:              int = 0,
    filing_lag_days:   int = 45,
    output_name:       str = 'cfr-phase3.json',
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close_path = LOCAL_OUTPUT_DIR / 'cfr_phase1_close.pkl'
    consensus_path = LOCAL_OUTPUT_DIR / 'cfr_phase2b_consensus_panel.pkl'
    macro_path = LOCAL_OUTPUT_DIR / 'cfr_phase3_macro.pkl'
    for p in (close_path, consensus_path, macro_path):
        if not p.exists():
            raise SystemExit(
                f'pickle not found at {p}. Run prep first:\n'
                f'  uv run python apps/cfr/scripts/modal/prep_phase1_data.py\n'
                f'  uv run python apps/cfr/scripts/modal/prep_phase2b_data.py\n'
                f'  uv run python apps/cfr/scripts/modal/prep_phase3_data.py')

    close_pickle = close_path.read_bytes()
    consensus_pickle = consensus_path.read_bytes()
    macro_pickle = macro_path.read_bytes()
    print(f'[local] close pickle:     {len(close_pickle) / 1024 / 1024:.1f} MB',
          flush=True)
    print(f'[local] consensus pickle: {len(consensus_pickle) / 1024:.1f} KB',
          flush=True)
    print(f'[local] macro pickle:     {len(macro_pickle) / 1024:.1f} KB',
          flush=True)

    print(f'[local] launching Modal Phase 3 (CPU 8c) ...', flush=True)
    t0 = time.perf_counter()
    artifacts = run_walkforward_remote.remote(
        close_pickle, consensus_pickle, macro_pickle,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        step_window_days=step_window_days,
        rebal_days=rebal_days,
        top_k=top_k,
        hidden=hidden,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_every=train_every,
        n_sgd_per_batch=n_sgd_per_batch,
        seed=seed,
        filing_lag_days=filing_lag_days,
        output_name=output_name,
    )
    print(f'[local] remote done in {time.perf_counter() - t0:.0f}s',
          flush=True)

    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)',
              flush=True)
