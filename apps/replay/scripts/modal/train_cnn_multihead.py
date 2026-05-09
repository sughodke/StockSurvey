"""Modal entrypoint: ss-replay multi-head CNN training + zero-shot eval suite.

Trains the multi-head FiLM CNN on a wide Stooq pool (default ~290
tickers from `apps/notebook/data/stooq_us_long`), with FiLM heads on
rsi (n × w grid), cci (n × w grid), vol (n grid), macd (fast grid).
After training, runs the zero-shot suite on CSCO + the FiLM/uncond
input-attention saliency on AAPL, and bundles all artifacts back to
the caller's local Output/.

**Universe.** Reads `apps/notebook/data/stooq_us_long/manifest.json`
(312 tickers with >=22y of history) and picks the train pool by
filtering to `min_history_bars` (default 6500, matching the
`apps/factor` walk-forward baseline) and excluding the primary, val,
and eval tickers. This produces an SSL backbone whose train universe
overlaps the factor walk-forward universe ≈1:1 — the necessary
condition for an apples-to-apples backbone-vs-deterministic-indicator
comparison.

**Val ticker.** Default is `NVDA` (full 26y history, sector-diverse,
high-vol). The original Phase-2 default `TSLA` is not in
`stooq_us_long` — its 2010 IPO post-dates the 2000-01-01 cutoff used
when building the subset. Override with `--val-ticker` if you want a
different held-out generalization probe.

**Memory.** Each ticker's feature tensor is ~75 MB (≈30 scales × 96
window_cols × 6500 timesteps × 4 bytes). Loading 290 tickers up-front
is ~22 GB host RAM, well over Modal's 16 GB T4 default — the
function is decorated `memory=32768` to compensate.

The eval helpers were extracted to `replay.eval` so this script is
pure orchestration: image setup, train cmd dispatch, eval-helper
invocations, artifact bundling.

Usage
-----
One-time setup (local):
    pip install --user modal      # or `uv tool install modal`
    modal token new               # browser flow

Smoke (~5-15 min wall, ~$0.05-0.15):
    modal run apps/replay/scripts/modal/train_cnn_multihead.py \\
        --steps 200 --max-train-tickers 30

Full canonical run (~45-90 min wall, ~$0.50-1.00 at T4 prices with
the bumped memory):
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

# Stooq subset baked into the image. The 312-ticker `stooq_us_long` subset
# (built via `apps/notebook/data/build_stooq_us_long.py` from the user's
# StooqData/ archive) is the same on-disk pool the apps/factor walk-forward
# baseline uses — same `daily/<country>/<exchange>/<bucket>/*.txt` Stooq
# layout, same manifest.json. Sharing it here means the SSL backbone is
# trained on the same universe the factor scorer evaluates on.
STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_us_long'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'

# Input-bundle configurations. After the 2026-05-09 polar-Morlet +
# Gaussian + log-L2-amp rewrite the channel stack is fixed (no
# include-* flags), so this is now a single cell — kept as a dict so
# the harness's per-bundle artifact-prefix machinery still works.
#
#   cwt-only   — Canonical SSL recipe. 7 channels per scale: polar
#                Morlet `(|c|, |c|^2, cos(arg), sin(arg))` over rolling-
#                z-normed prices, Gaussian `(g, g^2)` over cumulative
#                log-returns (lowpass / trend), and per-scale log-L2
#                amplitude over the trailing K-bar `|c|` slice.
BUNDLE_CONFIGS: dict[str, dict] = {
    'cwt-only': {
        'flags': [],
        'targets': 'rsi,macd,vol,cci',
        'prefix': 'cwtonly-',
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
            '.iv-cache/**',         # dolt-backed cache writes mid-build
                                    # → "modified during build" abort.
            '.claude/**',           # claude-code scheduled-task lock
                                    # also writes mid-build → same abort.
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            # `uv sync --package replay` walks every workspace member's
            # pyproject.toml so we keep those, but skip the `src/` trees
            # of apps that aren't deps of replay (factor / regime /
            # relational / v1) — concurrent edits there have raced
            # Modal's directory hash before.
            'apps/factor/src/**',
            'apps/regime/src/**',
            'apps/relational/src/**',
            'apps/v1/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('ss-replay-multihead', image=image)


def _resolve_train_pool(
    train_extra: str,
    primary: str,
    val_ticker: str,
    eval_ticker: str,
    min_history_bars: int,
    max_train_tickers: int,
) -> str:
    """Pick the train pool from the baked-in stooq_us_long manifest.

    If the caller passes `train_extra` explicitly (non-empty), respect
    that — they are pinning the pool. Otherwise read the manifest, drop
    tickers below `min_history_bars`, exclude the primary/val/eval
    tickers (so the held-out probes really are held out), and return a
    comma-joined string ready for ss-replay's `--train-tickers` flag.
    """
    if train_extra:
        return train_extra
    manifest_path = Path(STOOQ_SUBSET) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    exclude = {primary.upper(), val_ticker.upper(), eval_ticker.upper()}
    entries = [t for t in manifest['tickers']
               if t['ticker'].upper() not in exclude]
    if min_history_bars > 0:
        before = len(entries)
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
        dropped = before - len(entries)
        if dropped:
            print(f'  min_history_bars={min_history_bars}: '
                  f'dropped {dropped} short-history tickers',
                  flush=True)
    names = [t['ticker'] for t in entries]
    if max_train_tickers > 0:
        names = names[:max_train_tickers]
    return ','.join(names)


@app.function(gpu='T4', cpu=4, memory=196608, timeout=60 * 90)
def train_and_eval(
    steps: int,
    cnn_batch_size: int,
    val_ticker: str,
    eval_ticker: str,
    primary: str,
    train_extra: str,
    start: str,
    end: str,
    min_history_bars: int,
    max_train_tickers: int,
    bundle: str = 'cwt-only',
    compress: str = 'none',
    compress_levels: int = 1,
    compress_wavelet: str = 'haar',
    compress_pad_mode: str = 'periodization',
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
    train_extra_resolved = _resolve_train_pool(
        train_extra, primary, val_ticker, eval_ticker,
        min_history_bars, max_train_tickers,
    )
    n_extra = len([t for t in train_extra_resolved.split(',') if t.strip()])
    print(f'  train pool: {n_extra} extra tickers '
          f'(+ primary={primary}; val={val_ticker}; eval={eval_ticker})',
          flush=True)
    # `--package replay` scopes the env to just this app's deps. Without
    # it, `uv run` defaults to a full-workspace sync which pulls in
    # regime[research] -> bt 1.1.5 -> sdist build (slow + needs clang
    # for bt's vectorbt-style C extensions).
    compress_flags: list[str] = []
    if compress != 'none':
        compress_flags = [
            '--compress', compress,
            '--compress-levels', str(compress_levels),
            '--compress-wavelet', compress_wavelet,
            '--compress-pad-mode', compress_pad_mode,
        ]
    cmd = [
        'uv', 'run', '--package', 'replay',
        'ss-replay', primary,
        '--stooq-dir', STOOQ_SUBSET,
        '--train-tickers', train_extra_resolved,
        '--val-ticker', val_ticker,
        '--start', start, '--end', end,
        '--window-cols', '96',
        '--extra-high-freq-scales', '1,2',
        *cfg['flags'],
        *compress_flags,
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
    # the caller's local Output/ (per BUNDLE_CONFIGS prefix). Compressed
    # variants also get a `dwtL{N}-` infix so the A/B runs against the
    # uncompressed baseline don't clobber each other.
    name_prefix = cfg['prefix']
    if compress != 'none':
        name_prefix = f'{name_prefix}{compress}L{compress_levels}-'
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
    val_ticker: str = 'NVDA',
    eval_ticker: str = 'CSCO',
    primary: str = 'AAPL',
    train_extra: str = '',
    start: str = '2000-01-03',
    end: str = '2026-04-01',
    bundle: str = 'cwt-only',
    min_history_bars: int = 6500,
    # Full pool by default. After the streaming-predict + ownership-
    # transfer refactor (2026-05-09) peak memory at K=96, 7-channel-
    # per-scale stack, ~300-ticker pool is ~155 GB — fits comfortably
    # in the 192 GB Modal allocation. Set explicit cap only for smoke
    # runs.
    max_train_tickers: int = 0,
    compress: str = 'none',
    compress_levels: int = 1,
    compress_wavelet: str = 'haar',
    compress_pad_mode: str = 'periodization',
):
    """Fire the remote training run; write returned artifacts to Output/.

    `--bundle` selects the input channel mix and target set:

      cwt-only    (default) Polar Morlet + Gaussian + log-L2 amplitude —
                  the canonical SSL recipe after the 2026-05-09 rewrite.
                  No optional channels; bundle list is single-cell now.

    Artifacts go under prefix `cwtonly-` so this bundle's results are
    distinguishable from any future per-bundle additions in Output/.

    `--train-extra` is empty by default → the train pool is built from
    `apps/notebook/data/stooq_us_long/manifest.json`, dropping tickers
    with `n_bars < min_history_bars` and excluding the primary/val/eval
    tickers. Pass an explicit comma-separated list to override.
    `--max-train-tickers > 0` caps the resolved pool (smoke runs).
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
    if train_extra:
        print(f'    pool (explicit): {primary},{train_extra}')
    else:
        print(f'    pool: from stooq_us_long manifest, '
              f'min_history_bars={min_history_bars}, '
              f'max_train_tickers={max_train_tickers or "all"}')
    print(f'    span: {start} → {end}')
    if compress != 'none':
        print(f'    compress: {compress} L={compress_levels} '
              f'wavelet={compress_wavelet} pad={compress_pad_mode}\n')
    else:
        print(f'    compress: none\n')
    artifacts = train_and_eval.remote(
        steps=steps,
        cnn_batch_size=cnn_batch_size,
        val_ticker=val_ticker,
        eval_ticker=eval_ticker,
        primary=primary,
        train_extra=train_extra,
        start=start, end=end,
        min_history_bars=min_history_bars,
        max_train_tickers=max_train_tickers,
        bundle=bundle,
        compress=compress,
        compress_levels=compress_levels,
        compress_wavelet=compress_wavelet,
        compress_pad_mode=compress_pad_mode,
    )
    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f'\n=== Writing {len(artifacts)} artifacts to {LOCAL_OUTPUT_DIR} ===')
    for name, blob in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(blob)
        print(f'  ← {out.name}  ({len(blob):,} bytes)')
    print('\nDone.')
