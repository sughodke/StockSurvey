"""Modal entrypoint for the e2e-portfolio walk-forward (Phase 4d 13-ETF).

Per CLAUDE.md compute-placement: tinygrad training >2k steps MUST run
on Modal. The local entrypoint runs in the uvx ephemeral env (no
project-venv deps), so we ship a pre-built pickle as raw bytes.

Setup
-----
  # one-time (assumed done)
  uvx modal token new

  # Prep the pickle locally first (project venv has ss_indicators, ss_macro):
  uv run python apps/e2e_portfolio/scripts/prep_data.py

  # Smoke (~5 min)
  uvx modal run apps/e2e_portfolio/scripts/modal/train_walkforward.py \
      --n-steps 200

  # Full 3-fold walk-forward (~30 min on T4)
  uvx modal run apps/e2e_portfolio/scripts/modal/train_walkforward.py \
      --n-steps 5000

Returns: per-fold checkpoint npz + daily return npz + pooled npz +
results json, all written into local Output/.
"""
from __future__ import annotations

from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
LOCAL_PREP_PKL = LOCAL_OUTPUT_DIR / 'e2e-portfolio-prep.pkl'
LOCAL_VOL_NPZ = LOCAL_OUTPUT_DIR / 'vol-v3-dolthub-oos-c200-returns.npz'
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
            '.claude/**',
            '.iv-cache/**',
            '.hl-cache/**',
            '.congress-cache/**',
            '.macro-cache/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            'apps/regime/src/**',
            'apps/relational/src/**',
            'apps/replay/src/**',
            'apps/v1/src/**',
            'apps/notebook/src/**',
            'apps/notebook/data/**',
            'apps/docs/docs/**',
            'apps/factor/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('e2e-portfolio-walkforward', image=image)


@app.function(gpu='T4', cpu=8, memory=32768, timeout=2 * 60 * 60)
def train_walkforward_remote(
    prep_pkl: bytes,
    vol_npz: bytes,
    n_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, bytes]:
    import os
    import subprocess
    import time
    os.environ['CUDA'] = '1'

    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')
    # Drop input artifacts onto the remote disk so the project code
    # can load them by path.
    (output / 'e2e-portfolio-prep.pkl').write_bytes(prep_pkl)
    (output / 'vol-v3-dolthub-oos-c200-returns.npz').write_bytes(vol_npz)
    print(f'  staged prep ({len(prep_pkl) // (1 << 20)} MB) + '
          f'vol-v3 ({len(vol_npz) // 1024} KB)', flush=True)

    print('=== Step 1/3: uv sync e2e-portfolio deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'e2e-portfolio', '--inexact'],
        cwd=REMOTE_REPO, check=True)

    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device, Tensor
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(
            f'tinygrad picked Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    # ---------- Step 2: load panel from pickle, run walk-forward ----------
    print('\n=== Step 2/3: load panel + train per-fold ===', flush=True)
    import pickle
    import json
    import numpy as np
    import pandas as pd

    payload = pickle.loads(prep_pkl)
    tickers = list(payload['tickers'])
    close_arr = payload['close']
    close_index = pd.DatetimeIndex(payload['close_dates'])
    close = pd.DataFrame(close_arr, index=close_index, columns=tickers)
    print(f'  panel: X_assets={payload["X_assets"].shape}  '
          f'X_macro={payload["X_macro"].shape}  fwd_ret={payload["fwd_ret"].shape}',
          flush=True)
    print(f'  dates: {pd.Timestamp(payload["dates"][0]).date()} -> '
          f'{pd.Timestamp(payload["dates"][-1]).date()}', flush=True)

    from e2e_portfolio.data import K_FORWARD, Panel
    from e2e_portfolio.model import Hparams
    from e2e_portfolio.train import TrainConfig
    from e2e_portfolio.eval import FOLDS, run_fold, pool_and_report

    full_panel = Panel(
        X_assets=payload['X_assets'],
        X_macro=payload['X_macro'],
        fwd_ret=payload['fwd_ret'],
        dates=pd.DatetimeIndex(payload['dates']),
        tickers=tickers,
    )

    cfg = TrainConfig(n_steps=n_steps, batch_size=batch_size,
                      lr=lr, weight_decay=weight_decay, seed=seed)
    hp = Hparams()

    t0 = time.perf_counter()
    per_fold = []
    for fold_cfg in FOLDS:
        res = run_fold(full_panel, close, fold_cfg, cfg, hp,
                       save_prefix='e2e-portfolio')
        per_fold.append(res)
    print(f'  3 folds done in {time.perf_counter() - t0:.0f}s', flush=True)

    print('\n=== Step 3/3: pool + baseline comparisons ===', flush=True)
    summary = pool_and_report(per_fold, close, save_prefix='e2e-portfolio')
    results_path = output / 'e2e-portfolio-results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  wrote {results_path}', flush=True)
    print(json.dumps({
        'pooled_n': summary['pooled_n'],
        'pooled_sharpe_ann': summary['pooled_sharpe_ann'],
        'pooled_max_dd': summary['pooled_max_dd'],
        'vs_dca': summary['baseline_comparisons']['dca'],
        'vs_ew': summary['baseline_comparisons']['ew'],
        'vs_det_2leg': summary['baseline_comparisons']['deterministic_2leg'],
        'vs_learned_2leg': summary['baseline_comparisons']['learned_2leg'],
    }, indent=2), flush=True)

    # Bundle all e2e-portfolio-* artifacts back.
    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if (p.is_file() and p.name.startswith('e2e-portfolio-')
                and p.name != 'e2e-portfolio-prep.pkl'):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def walkforward(
    n_steps: int = 5000,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> None:
    """Local entrypoint — ships the prep pickle to Modal, drops artifacts
    back into local Output/.

    The uvx env has only `modal` + stdlib (no `pickle`-incompatible
    ext deps), so we treat prep + vol-v3 as raw bytes and let the
    remote function unpickle inside the CUDA image.
    """
    if not LOCAL_PREP_PKL.exists():
        raise SystemExit(
            f'prep pickle not found at {LOCAL_PREP_PKL}\n'
            f'run first: uv run python apps/e2e_portfolio/scripts/prep_data.py')
    if not LOCAL_VOL_NPZ.exists():
        raise SystemExit(
            f'vol-v3 baseline npz not found at {LOCAL_VOL_NPZ}\n'
            f'expected for the deterministic-2leg + learned-2leg baselines')
    prep_bytes = LOCAL_PREP_PKL.read_bytes()
    vol_bytes = LOCAL_VOL_NPZ.read_bytes()
    print(f'launching e2e-portfolio walkforward on Modal '
          f'(n_steps={n_steps}, batch_size={batch_size}, lr={lr}, '
          f'weight_decay={weight_decay}, prep={len(prep_bytes) // (1<<20)} MB)')
    artifacts = train_walkforward_remote.remote(
        prep_pkl=prep_bytes,
        vol_npz=vol_bytes,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
    )
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
