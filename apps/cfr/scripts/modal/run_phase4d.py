"""Modal entrypoint — Phase 4d deep CFR on multi-asset universe.

13 assets: 9 SPDR sector ETFs + 2 bond ETFs (TLT, IEF) + 2
commodity ETFs (GLD, DBC). Cross-asset has documented regime-
switching alpha (60/40 → barbell during high inflation, etc.).
Same Phase 3 deep CFR architecture; small `top_k` for the
modes (4 instead of 20) since the universe is 13 names.

`Top13FConsensusMode` is dropped (ETFs don't appear in 13F
filings).

Pre-registered cut: mean CFR Sharpe ≥ Phase 3 + 0.20 AND alpha
vs EW-of-multi-asset ≥ +0.15.
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

app = modal.App('cfr-phase4d', image=image)


@app.function(cpu=8, memory=16384, timeout=2 * 60 * 60)
def run_walkforward_remote(
    close_pickle: bytes,
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
    output_name: str,
) -> dict[str, bytes]:
    import os, subprocess
    print('=== Step 1/4: uv sync ===', flush=True)
    t0 = time.perf_counter()
    subprocess.run(['uv', 'sync', '--package', 'cfr', '--inexact'],
                   cwd=REMOTE_REPO, check=True)
    subprocess.run(['uv', 'pip', 'install', 'tinygrad'],
                   cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')
    print(f'  sync wall: {time.perf_counter() - t0:.1f}s', flush=True)

    print('\n=== Step 2/4: deserialize ===', flush=True)
    import pickle
    import pandas as pd
    close: pd.DataFrame = pickle.loads(close_pickle)
    macro: pd.DataFrame = pickle.loads(macro_pickle)
    print(f'  close: {close.shape}  cols {list(close.columns)}', flush=True)

    print('\n=== Step 3/4: build menu (no 13F) ===', flush=True)
    from cfr.menu import (
        ActionMenu, EqualWeightMode, TopKMode,
    )
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
    print(f'  Phase 4d menu ({driver.menu_builder().n_actions} actions) on '
          f'{close.shape[1]}-asset multi-asset universe', flush=True)

    print('\n=== Step 4/4: walk-forward ===', flush=True)
    t1 = time.perf_counter()
    per_window, summary = driver.run(close, macro)
    print(f'\n  walk wall: {time.perf_counter() - t1:.1f}s', flush=True)

    print(f'\n{"win":>3s} {"val_dates":>23s} {"cfr_sh":>7s} {"pas_sh":>7s} '
          f'{"trl_sh":>7s} {"nai_sh":>7s} {"alpha":>7s} {"vs_nai":>7s}', flush=True)
    for w in per_window:
        print(f'{w.window_idx:>3d} {w.val_start}→{w.val_end} '
              f'{w.cfr_sharpe:>+7.3f} {w.passive_ew_sharpe:>+7.3f} '
              f'{w.trailing_best_sharpe:>+7.3f} '
              f'{w.naive_uniform_sharpe:>+7.3f} '
              f'{w.cfr_alpha:>+7.3f} '
              f'{w.cfr_sharpe - w.naive_uniform_sharpe:>+7.3f}', flush=True)
    print(f'\nmean CFR Sharpe={summary["mean_cfr_sharpe"]:+.3f}  '
          f'pas={summary["mean_passive_sharpe"]:+.3f}  '
          f'naive={summary["mean_naive_sharpe"]:+.3f}', flush=True)
    print(f'CFR vs naive={summary["mean_cfr_minus_naive"]:+.3f}', flush=True)
    print(f'\nverdict: {summary["verdict"]}', flush=True)

    payload = {
        'config': {
            'phase': '4d',
            'menu': 'phase2a (no 13F) on multi-asset 13-ETF universe',
            'train_window_days': train_window_days,
            'val_window_days': val_window_days,
            'step_window_days': step_window_days,
            'rebal_days': rebal_days,
            'top_k': top_k, 'hidden': hidden,
            'learning_rate': learning_rate, 'seed': seed,
            'universe_n': close.shape[1],
            'tickers': list(close.columns),
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
    top_k:             int = 4,    # 13 assets → small top-K
    hidden:            int = 64,
    learning_rate:     float = 5e-4,
    batch_size:        int = 64,
    train_every:       int = 5,
    n_sgd_per_batch:   int = 5,
    seed:              int = 0,
    output_name:       str = 'cfr-phase4d.json',
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close_path = LOCAL_OUTPUT_DIR / 'cfr_phase4d_multiasset_close.pkl'
    macro_path = LOCAL_OUTPUT_DIR / 'cfr_phase3_macro.pkl'
    for p in (close_path, macro_path):
        if not p.exists():
            raise SystemExit(f'pickle missing: {p}; run prep scripts first')

    artifacts = run_walkforward_remote.remote(
        close_path.read_bytes(),
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
        output_name=output_name,
    )
    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)', flush=True)
