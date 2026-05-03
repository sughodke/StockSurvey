"""Modal entrypoint: ss-replay multi-head CNN training + zero-shot eval suite.

Wraps the Phase-2 / Exp-D recipe (mirrors colab/train_cnn_multihead.sh):
trains AAPL + 17-ticker pool, val on TSLA, with FiLM heads on rsi
(n × w grid), cci (n × w grid), vol (n grid), macd (fast grid). After
training, runs the zero-shot suite on CSCO + the FiLM/uncond input-
attention saliency on AAPL, and bundles all artifacts back to the
caller's local Output/.

The eval helpers were extracted to `replay.eval` so this script is now
pure orchestration: image setup, train cmd dispatch, eval-helper
invocations, artifact bundling. ~250 lines (was ~1150).

Usage
-----
One-time setup (local):
    pip install --user modal      # or `uv tool install modal`
    modal token new               # browser flow

Smoke (~5-15 min wall, ~$0.05-0.15):
    modal run apps/replay/scripts/modal/train_cnn_multihead.py --steps 500

Full canonical run (~30-60 min wall, ~$0.30-0.60):
    modal run apps/replay/scripts/modal/train_cnn_multihead.py --steps 2000

Loss is printed by the tinygrad trainer to stdout; Modal streams it to
your terminal in real time. No W&B wiring (ask if you want it added).
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import modal


# REPO_ROOT only matters on the local side (image build + artifact write).
# Inside the Modal container this script is dropped at /root/<basename> with
# only 2 parents, so parents[4] would IndexError at import time.
try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')   # remote fallback (unused there)
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'

# Phase-2/Exp-D pool (mirrors colab/train_cnn_multihead.sh).
TRAIN_POOL_EXTRA = ('MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,GE,BA,XOM,KO,WMT,'
                    'JNJ,UNH,T,NFLX,CRM,DIS')

# Stooq subset baked into the image (built via apps/replay/data/stooq_phase2/).
# 21 tickers (AAPL + 18 train + TSLA val + CSCO eval) totaling ~15 MB,
# preserving the daily/us/<exchange>/<bucket>/ layout that load_stooq_matrix
# walks. Replaces the per-cold-start yahoo fetch (~30-60 s) with a zero-cost
# read from the local FS, and gives bit-identical inputs across runs.
STOOQ_SUBSET_REL = 'apps/replay/data/stooq_phase2'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'

# Input-bundle configurations. Maps a single named experimental cell to the
# ss-replay flags + target subset + artifact prefix. The 4 cells span the
# experimentally meaningful subset of the (zscore on/off) x (returns mode
# in {none, raw, sign}) cross-product:
#
#   full       — zscore + raw returns + CWT (winning documented recipe)
#   full-sign  — zscore + sign returns + CWT (magnitude-shortcut diagnostic)
#   cwt-only   — CWT only (purest "scalogram alone" test)
#   cwt-sign   — sign returns + CWT (direction anchor, no price level)
#
# `price` head is dropped for the no-zscore cells: without rolling mu/sd,
# level info is gone from the input, so the price head would train to a
# useless ~0-R² constant.
BUNDLE_CONFIGS: dict[str, dict] = {
    'full': {
        'flags': ['--include-zscore-stats', '--include-returns'],
        'targets': 'rsi,macd,price,vol,cci',
        'prefix': '',
    },
    'full-sign': {
        'flags': ['--include-zscore-stats', '--include-return-sign'],
        'targets': 'rsi,macd,price,vol,cci',
        'prefix': 'fullsign-',
    },
    'cwt-only': {
        'flags': [],
        'targets': 'rsi,macd,vol,cci',
        'prefix': 'cwtonly-',
    },
    'cwt-sign': {
        'flags': ['--include-return-sign'],
        'targets': 'rsi,macd,vol,cci',
        'prefix': 'cwtsign-',
    },
}

# Image: NVIDIA CUDA devel (provides nvcc, which tinygrad's CUDA backend
# needs to JIT-compile kernels — debian_slim only has the runtime libs)
# + uv + the repo source. uv sync runs at function cold start (cached for
# the container's lifetime) so this layer doesn't re-build when source
# changes.
image = (
    modal.Image.from_registry(
        'nvidia/cuda:12.4.0-devel-ubuntu22.04',
        add_python='3.12',
    )
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('ss-replay-multihead', image=image)


@app.function(gpu='T4', timeout=60 * 75)
def train_and_eval(
    steps: int,
    cnn_batch_size: int,
    val_ticker: str,
    eval_ticker: str,
    primary: str,
    train_extra: str,
    start: str,
    end: str,
    bundle: str = 'full',
) -> dict[str, bytes]:
    """Run multi-head CNN training, then zero-shot eval + attention.

    Train via subprocess (`uv run ss-replay ...`); call
    `replay.eval.zeroshot_eval` / `film_attention` / `uncond_attention`
    for the post-training analysis suite. Returns every file under
    Output/ as a dict of `{filename: bytes}` so the local entrypoint
    can mirror them back to the caller's disk.
    """
    if bundle not in BUNDLE_CONFIGS:
        raise ValueError(
            f'unknown bundle {bundle!r}; valid: {list(BUNDLE_CONFIGS)}')
    cfg = BUNDLE_CONFIGS[bundle]
    import os
    os.environ['CUDA'] = '1'   # tinygrad: pin CUDA backend on Modal T4

    output = f'{REMOTE_REPO}/Output'
    os.makedirs(output, exist_ok=True)

    print('=== Step 1/4: uv sync workspace deps (one-time per cold start) ===',
          flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'replay', '--inexact'],
        cwd=REMOTE_REPO, check=True)

    print(f'\n=== Step 2/4: ss-replay multi-head CNN '
          f'(steps={steps}, batch={cnn_batch_size}) ===', flush=True)
    # `--package replay` scopes the env to just this app's deps. Without
    # it, `uv run` defaults to a full-workspace sync which pulls in
    # regime[research] -> bt 1.1.5 -> sdist build (slow + needs clang
    # for bt's vectorbt-style C extensions).
    cmd = [
        'uv', 'run', '--package', 'replay',
        'ss-replay', primary,
        '--stooq-dir', STOOQ_SUBSET,
        '--train-tickers', train_extra,
        '--val-ticker', val_ticker,
        '--start', start, '--end', end,
        '--window-cols', '96',
        '--extra-high-freq-scales', '1,2',
        *cfg['flags'],
        '--decoder', 'cnn', '--targets', cfg['targets'],
        '--rsi-n', '7',
        '--rsi-n-grid', '5,7,9,13,17,21,25',
        '--rsi-w-grid', '1,5,10,21',
        '--rsi-anchor-w', '1',
        '--cci-n', '20',
        '--cci-n-grid', '10,14,20,28,40',
        '--cci-w-grid', '1,5,10,21',
        '--cci-anchor-w', '1',
        '--vol-window', '20',
        '--vol-n-grid', '5,10,20,30,60',
        '--macd-fast-grid', '8,12,16,24',
        '--cnn-batch-size', str(cnn_batch_size),
        '--cnn-steps', str(steps),
        '--cnn-no-bf16',           # T4 (sm_75) has no native bf16; PTX
                                   # emitted for bf16 fails to link
                                   # against NVRTC's default include set.
        '--device', 'auto',
        '--output-dir', output,
    ]
    print('+ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)

    # Activate the uv venv's site-packages so the in-process eval helpers
    # can import editable-installed workspace packages. site.addsitedir
    # processes .pth files and PEP 660 editable hooks; bare sys.path.insert
    # does not.
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')
    from replay.eval import film_attention, uncond_attention, zeroshot_eval

    print(f'\n=== Step 3/4: zero-shot eval on {eval_ticker} ===', flush=True)
    npz_paths = sorted(Path(output).glob(f'{primary}+*-cnn-*.npz'))
    if not npz_paths:
        raise RuntimeError(
            f'no {primary}+*-cnn-*.npz produced under {output}; '
            f'training likely failed silently')
    npz_path = npz_paths[-1]
    print(f'eval source: {npz_path.name}', flush=True)
    eval_stats = zeroshot_eval(
        npz_path=npz_path, ticker=eval_ticker, output_dir=Path(output),
        stooq_dir=STOOQ_SUBSET)
    (Path(output) / f'{eval_ticker}-zeroshot-stats.json').write_text(
        json.dumps(eval_stats, indent=2, default=float))

    print(f'\n=== Step 4/4: input-attention saliency on {primary} ===',
          flush=True)
    print('  -- FiLM rsi head (cond_a vs cond_b) --', flush=True)
    film_stats = film_attention(
        npz_path=npz_path, ticker=primary, output_dir=Path(output),
        stooq_dir=STOOQ_SUBSET)
    (Path(output) / f'{primary}-film-attention-stats.json').write_text(
        json.dumps(film_stats, indent=2, default=float))
    print('  -- unconditioned heads (macd vs vol) --', flush=True)
    uncond_stats = uncond_attention(
        npz_path=npz_path, ticker=primary, output_dir=Path(output),
        stooq_dir=STOOQ_SUBSET)
    (Path(output) / f'{primary}-uncond-attention-stats.json').write_text(
        json.dumps(uncond_stats, indent=2, default=float))

    # Tag returned filenames so different bundles' artifacts coexist in
    # the caller's local Output/ (per BUNDLE_CONFIGS prefix).
    name_prefix = cfg['prefix']
    artifacts: dict[str, bytes] = {}
    for p in sorted(Path(output).iterdir()):
        if p.is_file():
            artifacts[f'{name_prefix}{p.name}'] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts (prefix={name_prefix!r})',
          flush=True)
    return artifacts


@app.local_entrypoint()
def main(
    steps: int = 500,
    cnn_batch_size: int = 8192,
    val_ticker: str = 'TSLA',
    eval_ticker: str = 'CSCO',
    primary: str = 'AAPL',
    train_extra: str = TRAIN_POOL_EXTRA,
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    bundle: str = 'full',
):
    """Fire the remote training run; write returned artifacts to Output/.

    `--bundle` selects the input channel mix and target set:

      full       (default) zscore + raw-returns + CWT
      full-sign            zscore + sign-returns + CWT
      cwt-only             CWT only (purest scalogram-alone test)
      cwt-sign             CWT + sign-returns (no price-level info)

    Each non-`full` bundle's artifacts are written under a prefix
    (`fullsign-`, `cwtonly-`, `cwtsign-`) so multiple bundles' results
    coexist in Output/ without overwriting.
    """
    if bundle not in BUNDLE_CONFIGS:
        raise SystemExit(
            f'unknown --bundle {bundle!r}; valid: {list(BUNDLE_CONFIGS)}')
    cfg = BUNDLE_CONFIGS[bundle]
    flags_summary = ' '.join(cfg['flags']) or '(no zscore, no returns)'
    print(f'>>> ss-replay multi-head CNN on Modal T4  (bundle={bundle})')
    print(f'    channels: {flags_summary}')
    print(f'    targets:  {cfg["targets"]}')
    print(f'    steps={steps}  batch={cnn_batch_size}  '
          f'primary={primary}  val={val_ticker}  eval={eval_ticker}')
    print(f'    pool: {primary},{train_extra}')
    print(f'    span: {start} → {end}\n')
    artifacts = train_and_eval.remote(
        steps=steps,
        cnn_batch_size=cnn_batch_size,
        val_ticker=val_ticker,
        eval_ticker=eval_ticker,
        primary=primary,
        train_extra=train_extra,
        start=start, end=end,
        bundle=bundle,
    )
    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f'\n=== Writing {len(artifacts)} artifacts to {LOCAL_OUTPUT_DIR} ===')
    for name, blob in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(blob)
        print(f'  ← {out.name}  ({len(blob):,} bytes)')
    print('\nDone.')
