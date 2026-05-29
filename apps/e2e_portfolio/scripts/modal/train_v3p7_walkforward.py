"""Modal entrypoint for e2e-portfolio v3.7 walk-forward.

v3.7 = v3.5 architecture + imitation-prior bias init + structural floor on
long-vol head, run with v3.6's continuous rolling-retrain walk pattern.

Initialization emits the deterministic recipe at step 0:
  - Equity weights ~ DCA basket (~1/9 on each of 9 Phase 4d ETFs in
    universe; ~0 elsewhere; ~0 cash)
  - short_vol_scale ~ 2.0 (canonical vol_v3 vega)
  - long_vol_position ~ 0.31 (just above floor of 0.3, ZZR analog)

Direct-Sharpe loss can only tilt the policy *away* from deterministic
where signal justifies it. Optimizer cannot drive long_vol_position
below 0.3 — closing the v3.5 fold-2 COVID failure mode.

Reuses v3.5 prep pickle on ss-e2e-iv-data Volume (same panel + VIXY).
"""
from __future__ import annotations

from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
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

app = modal.App('e2e-portfolio-v3p7-walkforward', image=image)
iv_volume = modal.Volume.from_name('ss-e2e-iv-data', create_if_missing=True)
artifacts_volume = modal.Volume.from_name('ss-e2e-artifacts', create_if_missing=True)


@app.function(
    gpu='T4', cpu=8, memory=49152, timeout=3 * 60 * 60,
    volumes={'/root/iv-data': iv_volume, '/root/artifacts': artifacts_volume},
)
def train_v3p7_walkforward_remote(
    vol_npz: bytes, phase4d_pkl: bytes, v3_pooled_npz: bytes,
    initial_steps: int, refine_steps: int, retrain_every: int,
    batch_size: int, lr: float, weight_decay: float, seed: int,
) -> dict[str, bytes]:
    import os, subprocess, time
    os.environ['CUDA'] = '1'
    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')

    iv_vol_prep = Path('/root/iv-data/e2e-portfolio-v3p5-prep.pkl')
    if not iv_vol_prep.exists():
        raise RuntimeError('v3.5 prep not found on ss-e2e-iv-data')
    prep_pkl = iv_vol_prep.read_bytes()
    print(f'  loaded v3.5 prep from volume ({len(prep_pkl) // (1 << 20)} MB)',
          flush=True)

    (output / 'vol-v3-dolthub-oos-c200-returns.npz').write_bytes(vol_npz)
    (output / 'cfr_phase4d_multiasset_close.pkl').write_bytes(phase4d_pkl)
    if len(v3_pooled_npz) > 0:
        (output / 'e2e-portfolio-v3-pooled-daily.npz').write_bytes(v3_pooled_npz)

    print('=== Step 1/3: uv sync ===', flush=True)
    subprocess.run(['uv', 'sync', '--package', 'e2e-portfolio', '--inexact'],
                    cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(f'Device={Device.DEFAULT!r}')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    print('\n=== Step 2/3: load v3.5 panel + run v3.7 continuous walk ===',
          flush=True)
    import pickle, json
    import numpy as np
    import pandas as pd

    payload = pickle.loads(prep_pkl)
    if payload['X_assets'].dtype != np.float32:
        payload['X_assets'] = payload['X_assets'].astype(np.float32)
    tickers = list(payload['tickers'])
    close = pd.DataFrame(payload['close'],
                         index=pd.DatetimeIndex(payload['close_dates']),
                         columns=tickers)
    vixy_close = payload['vixy_close']

    from e2e_portfolio.data_v3p5 import PanelV3p5
    from e2e_portfolio.data_v3 import DEFAULT_K_ACTIVE
    from e2e_portfolio.model_v3p7 import HparamsV3p7, PHASE4D_TICKERS
    from e2e_portfolio.train_v3p5 import TrainConfigV3p5
    from e2e_portfolio.eval_v3p6 import FOLDS
    from e2e_portfolio.eval_v3p7 import (
        run_walkforward_continuous_v3p7, pool_and_report,
    )

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
    n_dca = sum(t in PHASE4D_TICKERS for t in tickers)
    print(f'  panel: X_assets={full_panel.X_assets.shape} K={K}', flush=True)
    print(f'  Phase 4d tickers in universe: {n_dca}/13', flush=True)
    print(f'  initial_steps={initial_steps} refine_steps={refine_steps} '
          f'retrain_every={retrain_every}d', flush=True)

    cfg = TrainConfigV3p5(n_steps=initial_steps, batch_size=batch_size,
                          lr=lr, weight_decay=weight_decay, seed=seed)
    hp = HparamsV3p7(n_assets=K, k_active=DEFAULT_K_ACTIVE, use_bf16=False)

    t0 = time.perf_counter()
    per_fold = run_walkforward_continuous_v3p7(
        full_panel, close, vixy_close, FOLDS, cfg, hp,
        initial_steps=initial_steps, refine_steps=refine_steps,
        retrain_every=retrain_every,
        save_prefix='e2e-portfolio-v3p7')
    print(f'  walk done in {time.perf_counter() - t0:.0f}s', flush=True)

    print('\n=== Step 3/3: pool + baselines ===', flush=True)
    summary = pool_and_report(per_fold, save_prefix='e2e-portfolio-v3p7')
    results_path = output / 'e2e-portfolio-v3p7-results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

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

    art_vol = Path('/root/artifacts')
    art_vol.mkdir(exist_ok=True, parents=True)
    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name.startswith('e2e-portfolio-v3p7'):
            data = p.read_bytes()
            artifacts[p.name] = data
            (art_vol / p.name).write_bytes(data)
    artifacts_volume.commit()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def walkforward(
    initial_steps: int = 5000,
    refine_steps: int = 500,
    retrain_every: int = 63,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> None:
    if not LOCAL_VOL_NPZ.exists():
        raise SystemExit(f'vol-v3 baseline npz not found at {LOCAL_VOL_NPZ}')
    if not LOCAL_PHASE4D_PKL.exists():
        raise SystemExit(f'phase4d close pkl not found at {LOCAL_PHASE4D_PKL}')
    vol_bytes = LOCAL_VOL_NPZ.read_bytes()
    phase4d_bytes = LOCAL_PHASE4D_PKL.read_bytes()
    v3_pooled_bytes = (LOCAL_V3_POOLED.read_bytes()
                       if LOCAL_V3_POOLED.exists() else b'')
    print(f'launching e2e-portfolio v3.7 [imitation prior + lvp floor] on Modal '
          f'(initial_steps={initial_steps} refine_steps={refine_steps} '
          f'retrain_every={retrain_every}d)')
    artifacts = train_v3p7_walkforward_remote.remote(
        vol_npz=vol_bytes, phase4d_pkl=phase4d_bytes,
        v3_pooled_npz=v3_pooled_bytes,
        initial_steps=initial_steps, refine_steps=refine_steps,
        retrain_every=retrain_every, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, seed=seed,
    )
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
