"""Modal entrypoint — Phase 4a hybrid: Phase 3 deep CFR + macro VIX gate.

Same as Phase 3 but adds a per-rebal pre-gate that suspends
deployment when VIX < 1y trailing rolling median (the macro v1b
gate from `findings/macro-regime-diagnostic.md`). On rebal bars
where the gate is closed, target portfolio = cash (no
exposure); on bars where it's open, deep CFR's policy decides.

Pre-registered cut: gated mean Sharpe ≥ Phase 3 + 0.15 AND alpha
vs gated-EW baseline ≥ +0.10.
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
            '.git/**', '.venv/**', '.edgar-cache/**', '.macro-cache/**',
            'Output/**', 'StooqData/**', 'Nasdaq3347/**',
            'apps/relational/src/**', 'apps/regime/src/**',
            'apps/v1/src/**', 'apps/replay/src/**',
            'apps/factor/src/**', 'apps/lie/src/**',
            'apps/notebook/data/**', 'apps/docs/**',
            '**/__pycache__/**', '**/.pytest_cache/**',
            '**/.ruff_cache/**', '**/*.pyc',
        ],
    )
)

app = modal.App('cfr-phase4a', image=image)


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
    vix_lookback_days: int,
    output_name: str,
) -> dict[str, bytes]:
    import os, subprocess
    print('=== Step 1/4: uv sync workspace deps (cfr) ===', flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        ['uv', 'sync', '--package', 'cfr', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    subprocess.run(
        ['uv', 'pip', 'install', 'tinygrad'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')
    print(f'  sync wall: {time.perf_counter() - t0:.1f}s', flush=True)

    print('\n=== Step 2/4: deserialize panels + build VIX gate ===', flush=True)
    import pickle
    import numpy as np
    import pandas as pd
    close: pd.DataFrame = pickle.loads(close_pickle)
    consensus: pd.DataFrame = pickle.loads(consensus_pickle)
    macro: pd.DataFrame = pickle.loads(macro_pickle)
    print(f'  close: {close.shape}  '
          f'{close.index[0].date()} → {close.index[-1].date()}', flush=True)

    # Pre-compute VIX-above-1y-rolling-median per bar in close.index.
    vix_aligned = macro['vix'].reindex(close.index, method='ffill')
    vix_median = vix_aligned.rolling(window=vix_lookback_days,
                                     min_periods=vix_lookback_days // 2).median()
    vix_gate = (vix_aligned > vix_median).fillna(False)
    n_open = int(vix_gate.sum())
    print(f'  VIX gate: {n_open}/{len(vix_gate)} bars open '
          f'({100*n_open/len(vix_gate):.1f}%)', flush=True)

    # Build a closure that maps bar_date → bool.
    gate_lookup = vix_gate.to_dict()
    def vix_pre_rebal_gate(bar_date: pd.Timestamp) -> bool:
        return bool(gate_lookup.get(bar_date, False))

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
        pre_rebal_gate=vix_pre_rebal_gate,
    )

    print('\n=== Step 4/4: walk-forward (Phase 4a, gated) ===', flush=True)
    t1 = time.perf_counter()
    per_window, summary = driver.run(close, macro)
    print(f'\n  walk wall: {time.perf_counter() - t1:.1f}s '
          f'({len(per_window)} windows)', flush=True)

    print(f'\n{"win":>3s} {"val_dates":>23s} {"cfr_sh":>7s} {"pas_sh":>7s} '
          f'{"trl_sh":>7s} {"nai_sh":>7s} {"alpha":>7s} {"vs_nai":>7s}',
          flush=True)
    print('-' * 100, flush=True)
    for w in per_window:
        print(f'{w.window_idx:>3d} {w.val_start}→{w.val_end} '
              f'{w.cfr_sharpe:>+7.3f} {w.passive_ew_sharpe:>+7.3f} '
              f'{w.trailing_best_sharpe:>+7.3f} '
              f'{w.naive_uniform_sharpe:>+7.3f} '
              f'{w.cfr_alpha:>+7.3f} '
              f'{w.cfr_sharpe - w.naive_uniform_sharpe:>+7.3f}',
              flush=True)
    print(f'\nmean CFR Sharpe        = {summary["mean_cfr_sharpe"]:+.3f}',
          flush=True)
    print(f'mean passive EW Sharpe = {summary["mean_passive_sharpe"]:+.3f}',
          flush=True)
    print(f'mean naive uniform     = {summary["mean_naive_sharpe"]:+.3f}',
          flush=True)
    print(f'mean CFR vs naive      = {summary["mean_cfr_minus_naive"]:+.3f}',
          flush=True)
    print(f'\nverdict: {summary["verdict"]}', flush=True)

    payload = {
        'config': {
            'phase': '4a',
            'menu': 'phase2a + top13f (Phase 3 menu) + VIX gate',
            'vix_lookback_days': vix_lookback_days,
            'vix_gate_open_fraction': n_open / len(vix_gate),
            'train_window_days': train_window_days,
            'val_window_days': val_window_days,
            'step_window_days': step_window_days,
            'rebal_days': rebal_days,
            'top_k': top_k,
            'hidden': hidden,
            'learning_rate': learning_rate,
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
    learning_rate:     float = 5e-4,
    batch_size:        int = 64,
    train_every:       int = 5,
    n_sgd_per_batch:   int = 5,
    seed:              int = 0,
    filing_lag_days:   int = 45,
    vix_lookback_days: int = 252,
    output_name:       str = 'cfr-phase4a.json',
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close_path = LOCAL_OUTPUT_DIR / 'cfr_phase1_close.pkl'
    consensus_path = LOCAL_OUTPUT_DIR / 'cfr_phase2b_consensus_panel.pkl'
    macro_path = LOCAL_OUTPUT_DIR / 'cfr_phase3_macro.pkl'
    for p in (close_path, consensus_path, macro_path):
        if not p.exists():
            raise SystemExit(f'pickle missing: {p}; run prep scripts first')

    artifacts = run_walkforward_remote.remote(
        close_path.read_bytes(),
        consensus_path.read_bytes(),
        macro_path.read_bytes(),
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
        vix_lookback_days=vix_lookback_days,
        output_name=output_name,
    )
    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)',
              flush=True)
