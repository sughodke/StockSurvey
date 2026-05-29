"""Modal entrypoint for e2e-portfolio v3.5 walk-forward.

Extends v3 by adding a long-vol output head over Stooq VIXY returns.
Same Volume layout as v3:
  - ss-e2e-iv-data: prep pickle (reads from /root/iv-data/ if not shipped)
  - ss-e2e-artifacts: per-fold ckpts + daily streams

Run:
  uv run python apps/e2e_portfolio/scripts/prep_data_v3p5.py
  uvx modal volume put ss-e2e-iv-data Output/e2e-portfolio-v3p5-prep.pkl \\
      /e2e-portfolio-v3p5-prep.pkl --force
  uvx modal run --detach apps/e2e_portfolio/scripts/modal/train_v3p5_walkforward.py \\
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
LOCAL_PREP_PKL = LOCAL_OUTPUT_DIR / 'e2e-portfolio-v3p5-prep.pkl'
LOCAL_VOL_NPZ = LOCAL_OUTPUT_DIR / 'vol-v3-dolthub-oos-c200-returns.npz'
LOCAL_PHASE4D_PKL = LOCAL_OUTPUT_DIR / 'cfr_phase4d_multiasset_close.pkl'
LOCAL_V3_POOLED = LOCAL_OUTPUT_DIR / 'e2e-portfolio-v3-pooled-daily.npz'
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

app = modal.App('e2e-portfolio-v3p5-walkforward', image=image)

iv_volume = modal.Volume.from_name('ss-e2e-iv-data', create_if_missing=True)
artifacts_volume = modal.Volume.from_name('ss-e2e-artifacts', create_if_missing=True)


@app.function(
    gpu='T4', cpu=8, memory=49152, timeout=2 * 60 * 60,
    volumes={'/root/iv-data': iv_volume, '/root/artifacts': artifacts_volume},
)
def train_v3p5_walkforward_remote(
    prep_pkl: bytes,
    vol_npz: bytes,
    phase4d_pkl: bytes,
    v3_pooled_npz: bytes,
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
    iv_vol_prep = iv_vol / 'e2e-portfolio-v3p5-prep.pkl'

    if len(prep_pkl) == 0:
        if not iv_vol_prep.exists():
            raise RuntimeError(
                'empty prep_pkl arg AND no prep cached on ss-e2e-iv-data; '
                'upload via `uvx modal volume put ss-e2e-iv-data '
                'Output/e2e-portfolio-v3p5-prep.pkl /e2e-portfolio-v3p5-prep.pkl`'
            )
        prep_pkl = iv_vol_prep.read_bytes()
        print(f'  loaded v3.5 prep from ss-e2e-iv-data volume '
              f'({len(prep_pkl) // (1 << 20)} MB)', flush=True)
    else:
        iv_vol_prep.write_bytes(prep_pkl)
        print(f'  cached v3.5 prep into ss-e2e-iv-data volume '
              f'({len(prep_pkl) // (1 << 20)} MB)', flush=True)

    (output / 'e2e-portfolio-v3p5-prep.pkl').write_bytes(prep_pkl)
    (output / 'vol-v3-dolthub-oos-c200-returns.npz').write_bytes(vol_npz)
    (output / 'cfr_phase4d_multiasset_close.pkl').write_bytes(phase4d_pkl)
    if len(v3_pooled_npz) > 0:
        (output / 'e2e-portfolio-v3-pooled-daily.npz').write_bytes(v3_pooled_npz)
    print(f'  staged vol-v3 ({len(vol_npz) // 1024} KB) + '
          f'phase4d ({len(phase4d_pkl) // 1024} KB) + '
          f'v3-pooled ({len(v3_pooled_npz) // 1024} KB)', flush=True)

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

    print('\n=== Step 2/3: load v3.5 panel + train per-fold ===', flush=True)
    import pickle, json
    import numpy as np
    import pandas as pd

    payload = pickle.loads(prep_pkl)
    if payload['X_assets'].dtype != np.float32:
        payload['X_assets'] = payload['X_assets'].astype(np.float32)
    tickers = list(payload['tickers'])
    close = pd.DataFrame(
        payload['close'],
        index=pd.DatetimeIndex(payload['close_dates']),
        columns=tickers,
    )
    vixy_close = payload['vixy_close']  # pd.Series

    from e2e_portfolio.data_v3p5 import PanelV3p5
    from e2e_portfolio.data_v3 import DEFAULT_K_ACTIVE
    from e2e_portfolio.model_v3p5 import HparamsV3p5
    from e2e_portfolio.train_v3p5 import TrainConfigV3p5
    from e2e_portfolio.eval_v3p5 import FOLDS, run_fold, pool_and_report

    full_panel = PanelV3p5(
        X_assets=payload['X_assets'],
        X_macro=payload['X_macro'],
        valid_mask=payload['valid_mask'],
        fwd_ret=payload['fwd_ret'],
        fwd_vol_pnl=payload['fwd_vol_pnl'],
        fwd_long_vol_ret=payload['fwd_long_vol_ret'],
        dates=pd.DatetimeIndex(payload['dates']),
        tickers=tickers,
    )
    K = full_panel.X_assets.shape[1]
    fwd_lv = full_panel.fwd_long_vol_ret
    print(f'  panel: X_assets={full_panel.X_assets.shape} K={K}', flush=True)
    print(f'  fwd_long_vol_ret: shape={fwd_lv.shape} '
          f'mean={fwd_lv.mean():.5f} std={fwd_lv.std():.5f} '
          f'nonzero_frac={(fwd_lv != 0).mean():.3f}', flush=True)

    cfg = TrainConfigV3p5(n_steps=n_steps, batch_size=batch_size,
                          lr=lr, weight_decay=weight_decay, seed=seed)
    hp = HparamsV3p5(n_assets=K, k_active=DEFAULT_K_ACTIVE)

    t0 = time.perf_counter()
    per_fold = []
    for fold_cfg in FOLDS:
        res = run_fold(full_panel, close, vixy_close, fold_cfg, cfg, hp,
                       save_prefix='e2e-portfolio-v3p5')
        per_fold.append(res)
    print(f'  3 folds done in {time.perf_counter() - t0:.0f}s', flush=True)

    print('\n=== Step 3/3: pool + baseline comparisons ===', flush=True)
    summary = pool_and_report(per_fold, save_prefix='e2e-portfolio-v3p5')
    results_path = output / 'e2e-portfolio-v3p5-results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  wrote {results_path}', flush=True)

    headline = {
        'pooled_n': summary['pooled_n'],
        'pooled_sharpe_ann': summary['pooled_sharpe_ann'],
        'pooled_max_dd': summary['pooled_max_dd'],
        'pooled_short_vol_scale_mean': summary.get('pooled_short_vol_scale_mean'),
        'pooled_long_vol_position_mean': summary.get('pooled_long_vol_position_mean'),
        'fold2_2020q1_long_vol_mean': next(
            (f.get('long_vol_position_2020q1_mean') for f in summary['per_fold']
             if f.get('name') == 'fold2'), None),
    }
    headline.update({
        f'vs_{k}': summary['baseline_comparisons'].get(k, {})
        for k in ['dca', 'vol_v3', 'deterministic_2leg', 'learned_2leg', 'v3']
    })
    print(json.dumps(headline, indent=2, default=str), flush=True)

    # Bundle artifacts.
    art_vol = Path('/root/artifacts')
    art_vol.mkdir(exist_ok=True, parents=True)
    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if (p.is_file() and p.name.startswith('e2e-portfolio-v3p5-')
                and p.name != 'e2e-portfolio-v3p5-prep.pkl'):
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
    prep_bytes = b''  # 1GB+ pickle read from Volume in-remote
    vol_bytes = LOCAL_VOL_NPZ.read_bytes()
    phase4d_bytes = LOCAL_PHASE4D_PKL.read_bytes()
    v3_pooled_bytes = (LOCAL_V3_POOLED.read_bytes()
                       if LOCAL_V3_POOLED.exists() else b'')
    print(f'launching e2e-portfolio v3.5 walkforward on Modal '
          f'(n_steps={n_steps}, prep_from_volume=ss-e2e-iv-data, '
          f'v3_pooled={len(v3_pooled_bytes) // 1024} KB)')
    artifacts = train_v3p5_walkforward_remote.remote(
        prep_pkl=prep_bytes, vol_npz=vol_bytes, phase4d_pkl=phase4d_bytes,
        v3_pooled_npz=v3_pooled_bytes,
        n_steps=n_steps, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, seed=seed,
    )
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
