"""Modal entrypoint for e2e-portfolio v3 walk-forward.

Volumes:
  - ss-e2e-iv-data   -> /root/iv-data    (prep pickle cache for reuse)
  - ss-e2e-artifacts -> /root/artifacts  (per-fold ckpts + daily streams)

Run:
  uv run python apps/e2e_portfolio/scripts/prep_data_v3.py
  uvx modal run apps/e2e_portfolio/scripts/modal/train_v3_walkforward.py \\
      --n-steps 5000
"""
from __future__ import annotations

from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
LOCAL_PREP_PKL = LOCAL_OUTPUT_DIR / 'e2e-portfolio-v3-prep.pkl'
LOCAL_VOL_NPZ = LOCAL_OUTPUT_DIR / 'vol-v3-dolthub-oos-c200-returns.npz'
LOCAL_PHASE4D_PKL = LOCAL_OUTPUT_DIR / 'cfr_phase4d_multiasset_close.pkl'
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
            '.git/**', '.venv/**', '.claude/**',
            '.iv-cache/**', '.hl-cache/**', '.congress-cache/**',
            '.macro-cache/**',
            'Output/**', 'StooqData/**', 'Nasdaq3347/**',
            'apps/regime/src/**', 'apps/relational/src/**',
            'apps/replay/src/**', 'apps/v1/src/**',
            'apps/notebook/src/**', 'apps/notebook/data/**',
            'apps/docs/docs/**', 'apps/factor/src/**',
            '**/__pycache__/**', '**/*.pyc',
        ],
    )
)

app = modal.App('e2e-portfolio-v3-walkforward', image=image)

iv_volume = modal.Volume.from_name('ss-e2e-iv-data', create_if_missing=True)
artifacts_volume = modal.Volume.from_name('ss-e2e-artifacts', create_if_missing=True)


@app.function(
    gpu='T4', cpu=8, memory=49152, timeout=2 * 60 * 60,
    volumes={'/root/iv-data': iv_volume, '/root/artifacts': artifacts_volume},
)
def train_v3_walkforward_remote(
    prep_pkl: bytes,
    vol_npz: bytes,
    phase4d_pkl: bytes,
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
    iv_vol = Path('/root/iv-data')
    iv_vol.mkdir(exist_ok=True, parents=True)
    iv_vol_prep = iv_vol / 'e2e-portfolio-v3-prep.pkl'

    if len(prep_pkl) == 0:
        if not iv_vol_prep.exists():
            raise RuntimeError(
                'empty prep_pkl arg AND no prep cached on ss-e2e-iv-data volume; '
                'upload via `uvx modal volume put ss-e2e-iv-data Output/e2e-portfolio-v3-prep.pkl /e2e-portfolio-v3-prep.pkl`'
            )
        prep_pkl = iv_vol_prep.read_bytes()
        print(f'  loaded prep from ss-e2e-iv-data volume ({len(prep_pkl) // (1 << 20)} MB)', flush=True)
    else:
        iv_vol_prep.write_bytes(prep_pkl)
        print(f'  cached prep into ss-e2e-iv-data volume ({len(prep_pkl) // (1 << 20)} MB)', flush=True)

    (output / 'e2e-portfolio-v3-prep.pkl').write_bytes(prep_pkl)
    (output / 'vol-v3-dolthub-oos-c200-returns.npz').write_bytes(vol_npz)
    (output / 'cfr_phase4d_multiasset_close.pkl').write_bytes(phase4d_pkl)
    print(f'  staged vol-v3 ({len(vol_npz) // 1024} KB) + '
          f'phase4d ({len(phase4d_pkl) // 1024} KB)', flush=True)

    print('=== Step 1/3: uv sync e2e-portfolio deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'e2e-portfolio', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(f'tinygrad Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    print('\n=== Step 2/3: load v3 panel + train per-fold ===', flush=True)
    import pickle, json
    import numpy as np
    import pandas as pd

    payload = pickle.loads(prep_pkl)
    # Cast X_assets back to float32 (was shipped as float16 to fit RPC).
    if payload['X_assets'].dtype != np.float32:
        payload['X_assets'] = payload['X_assets'].astype(np.float32)
    tickers = list(payload['tickers'])
    close = pd.DataFrame(
        payload['close'],
        index=pd.DatetimeIndex(payload['close_dates']),
        columns=tickers,
    )

    from e2e_portfolio.data_v3 import PanelV3, DEFAULT_K_ACTIVE
    from e2e_portfolio.model_v3 import HparamsV3
    from e2e_portfolio.train_v3 import TrainConfigV3
    from e2e_portfolio.eval_v3 import FOLDS, run_fold, pool_and_report

    full_panel = PanelV3(
        X_assets=payload['X_assets'],
        X_macro=payload['X_macro'],
        valid_mask=payload['valid_mask'],
        fwd_ret=payload['fwd_ret'],
        fwd_vol_pnl=payload['fwd_vol_pnl'],
        dates=pd.DatetimeIndex(payload['dates']),
        tickers=tickers,
    )
    K = full_panel.X_assets.shape[1]
    print(f'  panel: X_assets={full_panel.X_assets.shape} K={K}', flush=True)

    cfg = TrainConfigV3(n_steps=n_steps, batch_size=batch_size,
                        lr=lr, weight_decay=weight_decay, seed=seed)
    hp = HparamsV3(n_assets=K, k_active=DEFAULT_K_ACTIVE)

    t0 = time.perf_counter()
    per_fold = []
    for fold_cfg in FOLDS:
        res = run_fold(full_panel, close, fold_cfg, cfg, hp,
                       save_prefix='e2e-portfolio-v3')
        per_fold.append(res)
    print(f'  3 folds done in {time.perf_counter() - t0:.0f}s', flush=True)

    print('\n=== Step 3/3: pool + baseline comparisons ===', flush=True)
    summary = pool_and_report(per_fold, save_prefix='e2e-portfolio-v3')
    results_path = output / 'e2e-portfolio-v3-results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  wrote {results_path}', flush=True)
    print(json.dumps({
        'pooled_n': summary['pooled_n'],
        'pooled_sharpe_ann': summary['pooled_sharpe_ann'],
        'pooled_max_dd': summary['pooled_max_dd'],
        'vol_scale': {
            'mean': summary.get('pooled_vol_scale_mean'),
            'std': summary.get('pooled_vol_scale_std'),
            'min': summary.get('pooled_vol_scale_min'),
            'max': summary.get('pooled_vol_scale_max'),
        },
        'vs_dca': summary['baseline_comparisons']['dca'],
        'vs_vol_v3': summary['baseline_comparisons']['vol_v3'],
        'vs_det_2leg': summary['baseline_comparisons']['deterministic_2leg'],
        'vs_learned_2leg': summary['baseline_comparisons']['learned_2leg'],
    }, indent=2), flush=True)

    # Bundle v3 artifacts back; also copy into artifacts volume.
    art_vol = Path('/root/artifacts')
    art_vol.mkdir(exist_ok=True, parents=True)
    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if (p.is_file() and p.name.startswith('e2e-portfolio-v3-')
                and p.name != 'e2e-portfolio-v3-prep.pkl'):
            data = p.read_bytes()
            artifacts[p.name] = data
            (art_vol / p.name).write_bytes(data)
    artifacts_volume.commit()
    iv_volume.commit()
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
    if not LOCAL_VOL_NPZ.exists():
        raise SystemExit(f'vol-v3 baseline npz not found at {LOCAL_VOL_NPZ}')
    if not LOCAL_PHASE4D_PKL.exists():
        raise SystemExit(f'phase4d close pkl not found at {LOCAL_PHASE4D_PKL}')
    prep_bytes = b''  # 1GB+ pickle is read from ss-e2e-iv-data Volume in-remote
    vol_bytes = LOCAL_VOL_NPZ.read_bytes()
    phase4d_bytes = LOCAL_PHASE4D_PKL.read_bytes()
    print(f'launching e2e-portfolio v3 walkforward on Modal '
          f'(n_steps={n_steps}, prep_from_volume=ss-e2e-iv-data)')
    artifacts = train_v3_walkforward_remote.remote(
        prep_pkl=prep_bytes, vol_npz=vol_bytes, phase4d_pkl=phase4d_bytes,
        n_steps=n_steps, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, seed=seed,
    )
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
