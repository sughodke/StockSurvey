"""Modal entrypoint for SSL-backbone walk-forward eval.

Parallel to `train_indicator.py::walkforward` but feeds the
SSL-pretrained CWT backbone (produced by ss-replay --decoder cnn) into
the same `train_scorer_walkforward` head + objective. The point of
this comparison: does the encoder add information beyond what the
deterministic indicator stack already encodes in closed form?

Setup
-----
  uvx modal token new          # one-time

  # Smoke (~10 min, ~$0.15-0.20)
  uvx modal run apps/factor/scripts/modal/train_ssl_walkforward.py \\
      --max-tickers 30 --n-steps 100 --scorers linear

  # Full apples-to-apples vs the deterministic baseline (~25-35 min)
  uvx modal run apps/factor/scripts/modal/train_ssl_walkforward.py \\
      --scorers linear,mlp --n-steps 200 --weight-decay 1e-3

The default backbone path
(`Output/cwtonly-AAPL+294tickers-h631e9d47-rsi+macd+vol+cci-cnn-nogit.npz`)
is the artifact from the May 3 cwt-only pretrain on the matching 297-
ticker stooq_us_long universe. Pass `--backbone-npz` to use a
different one.

Returns one `ssl-walkforward-{scorer}-...windows.npz` per scorer plus
a summary json + comparison plot, mirrored back to local Output/.
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

STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_us_long'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'

# Default backbone npz produced by the matching cwt-only SSL pretrain.
DEFAULT_BACKBONE_NPZ = (
    'Output/cwtonly-AAPL+294tickers-h631e9d47-rsi+macd+vol+cci-cnn-nogit.npz')
REMOTE_BACKBONE_DIR = '/root/backbone'

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
            '.iv-cache/**',
            'Output/**',                # backbone npz mounted separately
            'StooqData/**',
            'Nasdaq3347/**',
            # apps not in the factor dep tree — exclude src/ to avoid
            # concurrent-edit races during the upload hash.
            'apps/regime/src/**',
            'apps/relational/src/**',
            'apps/replay/src/**',
            'apps/v1/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
    # Mount the local Output/ separately, filtered to just .npz backbones.
    # Keeps image build fast (other Output/ artifacts are big plots/jsons
    # we don't need on Modal).
    .add_local_dir(
        (REPO_ROOT / 'Output').as_posix(),
        remote_path=REMOTE_BACKBONE_DIR,
        ignore=['*.png', '*.json', '*.txt', '*.log'],
    )
)

app = modal.App('factor-ssl-walkforward', image=image)


def _resolve_ticker_list(
    tickers: str, max_tickers: int, min_history_bars: int = 0,
) -> list[str]:
    """Same selection rule as train_indicator.py: read manifest, drop
    short-history tickers, optionally cap. Kept duplicated here rather
    than imported so this entrypoint script stays self-contained inside
    Modal's add_local_dir tree."""
    manifest_path = Path(STOOQ_SUBSET) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if tickers:
        requested = {t.strip().upper() for t in tickers.split(',') if t.strip()}
        entries = [t for t in manifest['tickers'] if t['ticker'].upper() in requested]
    else:
        entries = list(manifest['tickers'])
    if min_history_bars > 0:
        before = len(entries)
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
        dropped = before - len(entries)
        if dropped:
            print(f'  min_history_bars={min_history_bars}: '
                  f'dropped {dropped} short-history tickers', flush=True)
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
    return names


def _load_one_ticker(args):
    """Worker: build a TickerData via load_ticker() with the supervised-`cnn`
    backbone's training-time meta, persist its features to a per-ticker
    `.npy` on container disk, and return a small stub (no big array
    through pickle).

    Returns `(ticker, stub_dict)` on success or `(ticker, error_str)` on
    failure. The stub carries the small daily arrays (`prices`, `dates`,
    `valid`, ~50KB total) plus the `.npy` path; the parent reconstitutes
    a TickerData with `features` mmap'd from disk so subsequent slicing
    by `align_tickers_at_rebal` only pages in the rebal rows.

    Why disk-handoff instead of pickling features: at the supervised-`cnn`
    pretrain panel size (D=6500, K=96, F=105) each `.features` array is
    ~262 MB.
    The original `imap_unordered` pickle path streamed ~78 GB through
    the parent at 297 tickers (~786 GB at 3000), driving the peak RSS
    that triggered the 192 GB OOM cliff. Disk-handoff drops parent IPC
    residency to a few KB per ticker; per-ticker disk peak is the
    `.features` size (transient — parent unlinks after assembling the
    aligned panel).
    """
    import os
    import numpy as np

    ticker, stooq_subset, start, end, load_kwargs, tmp_dir = args
    from ss_features import load_ticker
    try:
        td = load_ticker(
            ticker, stooq_dir=stooq_subset, kaggle_dir=None,
            start=start, end=end, **load_kwargs,
        )
        if not td.valid.any():
            return ticker, '(no valid bars)'
        # `np.save` (single-array .npy) is mmap-able; `np.savez` (zipped
        # archive) is not. `allow_pickle=False` is a safety knob — features
        # are dense numeric, never object arrays.
        npy_path = os.path.join(tmp_dir, f'{ticker}.features.npy')
        np.save(npy_path, td.features, allow_pickle=False)
        return ticker, {
            'name':           td.name,
            'dates':          td.dates,
            'prices':         td.prices,
            'valid':          td.valid,
            'features_path':  npy_path,
            'features_shape': tuple(td.features.shape),
            'features_dtype': str(td.features.dtype),
        }
    except Exception as e:
        return ticker, f'({type(e).__name__}: {e})'


# Memory profile (after the 2026-05-09 audit fixes — disk-handoff
# workers + `align_tickers_at_rebal` + streaming compute_input_stats):
#   - Workers persist per-ticker .features (~262 MB each at K=96, F=105,
#     D=6500) to /tmp/factor-features/<ticker>.npy and return a small
#     stub. Parent IPC residency is a few KB per ticker.
#   - Parent reconstitutes TickerData with `features = np.load(..., mmap)`,
#     so the 262 MB-per-ticker arrays live on container disk; only the
#     pages aligned with rebal positions get paged in.
#   - `align_tickers_at_rebal` allocates `(D', N, K, F)` where D' = D /
#     rebal_days ≈ 325 — that's ~4 GB at 297 tickers, ~39 GB at 3000.
#     Encoder forward yields the same shape latent, so peak adds another
#     ~3 GB / ~22 GB.
#   - Per-ticker .npy files are unlinked after the panel is built.
# Container disk (ephemeral) needs to hold N × per-ticker .features bytes
# concurrently — ~78 GB at 297 tickers, ~786 GB at 3000. Configure a
# Modal Volume or larger ephemeral disk if scaling past what the host's
# /tmp + rootfs can hold.
@app.function(gpu='T4', cpu=4, memory=196608, timeout=60 * 90)
def train_ssl_walkforward(
    backbone_npz_name: str,
    scorers: str,
    n_steps: int,
    weight_decay: float,
    tickers: str,
    start: str,
    end: str,
    rebal_days: int,
    learning_rate: float,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    max_tickers: int,
    min_history_bars: int,
    mlp_hidden: int,
    mlp_layers: int,
    aux_weight: float,
    aux_winsor_lo: float,
    aux_winsor_hi: float,
) -> dict[str, bytes]:
    import os
    import subprocess
    import multiprocessing as mp
    import shutil

    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')
    # Per-ticker `.features` npy spool. Ephemeral; cleaned up at end.
    features_tmp_dir = '/tmp/factor-features'
    if os.path.exists(features_tmp_dir):
        shutil.rmtree(features_tmp_dir)
    os.makedirs(features_tmp_dir, exist_ok=True)
    os.environ['CUDA'] = '1'

    print('=== Step 1/4: uv sync workspace deps (one-time per cold start) ===',
          flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'factor', '--inexact'],
        cwd=REMOTE_REPO, check=True)

    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(
            f'tinygrad picked Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    # ---------- Step 2: load backbone + ticker pool ----------
    print('\n=== Step 2/4: load SSL backbone + build features ===', flush=True)
    from factor import (
        load_backbone, train_scorer_walkforward,
    )
    from ss_features import TickerData

    backbone_path = Path(REMOTE_BACKBONE_DIR) / backbone_npz_name
    if not backbone_path.exists():
        raise FileNotFoundError(
            f'backbone npz not found at {backbone_path} — check that '
            f'{backbone_npz_name} is under local Output/ before launch')
    backbone, meta = load_backbone(backbone_path)
    print(f'  backbone: K={backbone.K} F={backbone.F} hidden={backbone.hidden} '
          f'K_post={backbone.K_post} n_layers={backbone.n_layers}  '
          f'flat_dim={backbone.hidden_flat}', flush=True)
    print(f'  backbone npz: {backbone_npz_name}  '
          f'(trained on {len(meta.get("train_tickers", []))} tickers, '
          f'targets={meta.get("targets")})', flush=True)

    # Mirror the load_ticker meta from the backbone's training config so
    # the encoder sees the same input layout it was pretrained on.
    load_kwargs = {
        'scales':               meta['scales'],
        'lookback':             meta['lookback'],
        'window_cols':          meta['window_cols'],
        'rsi_n':                meta.get('rsi_n', 7),
        'macd_fast':            meta.get('macd_fast', 12),
        'macd_slow':            meta.get('macd_slow', 26),
        'macd_signal':          meta.get('macd_signal', 9),
        'vol_window':           meta.get('vol_window', 20),
        'cci_n':                meta.get('cci_n', 20),
    }

    ticker_list = _resolve_ticker_list(tickers, max_tickers, min_history_bars)
    print(f'  universe: {len(ticker_list)} tickers '
          f'(first 5: {ticker_list[:5]} ...)', flush=True)

    n_workers = max(1, int(os.environ.get('FACTOR_FEATURE_WORKERS',
                                          os.cpu_count() or 4)))
    print(f'  parallelizing feature build across {n_workers} workers',
          flush=True)

    import numpy as np
    t0 = time.perf_counter()
    ticker_data: list[TickerData] = []
    skipped: list[str] = []
    work_args = [(t, STOOQ_SUBSET, start, end, load_kwargs, features_tmp_dir)
                 for t in ticker_list]
    with mp.Pool(n_workers) as pool:
        for i, (ticker, result) in enumerate(
                pool.imap_unordered(_load_one_ticker, work_args)):
            if isinstance(result, dict):
                # `np.load(..., mmap_mode='r')` opens the .npy as a
                # memory-mapped array; downstream slices in
                # `align_tickers_at_rebal` only fault in the rebal rows.
                features_mmap = np.load(result['features_path'], mmap_mode='r')
                ticker_data.append(TickerData(
                    name=result['name'],
                    prices=result['prices'],
                    dates=result['dates'],
                    features=features_mmap,
                    targets={},
                    valid=result['valid'],
                ))
            else:
                skipped.append(f'{ticker} {result}')
            if (i + 1) % 25 == 0:
                print(f'  built {i + 1}/{len(ticker_list)}  '
                      f'({time.perf_counter()-t0:.0f}s)', flush=True)
    ticker_data.sort(key=lambda td: td.name)
    print(f'  feature build done: {len(ticker_data)} usable / '
          f'{len(skipped)} skipped  ({time.perf_counter()-t0:.0f}s)', flush=True)
    # Disk usage check — at full pool the spool can be hundreds of GB.
    spool_bytes = sum(p.stat().st_size for p in Path(features_tmp_dir).iterdir())
    print(f'  features spool: {spool_bytes / (1 << 30):.1f} GB on '
          f'{features_tmp_dir}', flush=True)
    if skipped[:5]:
        print(f'  first 5 skipped: {skipped[:5]}', flush=True)
    if len(ticker_data) < 4:
        raise RuntimeError(
            f'only {len(ticker_data)} tickers built — too few for IC training')

    # ---------- Step 3: walk-forward per scorer ----------
    print('\n=== Step 3/4: walk-forward eval per scorer ===', flush=True)
    s_list = [s.strip() for s in scorers.split(',') if s.strip()]
    print(f'  scorers: {s_list}  '
          f'(train={train_window_blocks} val={val_window_blocks} '
          f'step={step_window_blocks} blocks)', flush=True)

    aux_winsor = (aux_winsor_lo, aux_winsor_hi)
    summary: list[dict] = []
    for scorer in s_list:
        # Tag artifact filenames with the aux config so an A/B against
        # a baseline run does not overwrite — only multitask scorers
        # get the aux suffix; baseline `linear` / `mlp` keep their
        # existing filename pattern.
        aux_tag = (
            f'-aux{aux_weight:g}' if scorer == 'mlp_multitask' else '')
        prefix = (
            f'ssl-walkforward-{scorer}-s{n_steps}-wd{weight_decay:g}{aux_tag}')
        print(f'\n  >>> {prefix}', flush=True)
        t1 = time.perf_counter()
        try:
            wf = train_scorer_walkforward(
                ticker_data, backbone,
                rebal_days=rebal_days,
                train_window_blocks=train_window_blocks,
                val_window_blocks=val_window_blocks,
                step_window_blocks=step_window_blocks,
                scorer=scorer,
                mlp_hidden=mlp_hidden, mlp_layers=mlp_layers,
                n_steps=n_steps, learning_rate=learning_rate,
                weight_decay=weight_decay,
                aux_weight=(aux_weight if scorer == 'mlp_multitask' else 0.0),
                aux_winsor=aux_winsor,
                verbose=True,
            )
        except Exception as e:
            print(f'    FAILED: {type(e).__name__}: {e}', flush=True)
            summary.append({'scorer': scorer, 'failed': True,
                            'error': f'{type(e).__name__}: {e}'})
            continue
        wall = time.perf_counter() - t1
        npz_path = output / f'{prefix}-windows.npz'
        _save_walkforward_npz(npz_path, wf, backbone)
        per_window = [
            {
                'window_idx': w.window_idx,
                'train_block_start': w.train_block_start,
                'train_block_end': w.train_block_end,
                'val_block_start': w.val_block_start,
                'val_block_end': w.val_block_end,
                'train_ic': w.train_ic, 'val_ic': w.val_ic,
                'train_sharpe': w.train_sharpe, 'val_sharpe': w.val_sharpe,
                'train_aux_mse': w.train_aux_mse,
                'val_aux_mse': w.val_aux_mse,
                'n_train_bars': w.n_train_bars, 'n_val_bars': w.n_val_bars,
            } for w in wf.windows
        ]
        summary.append({
            'scorer': scorer, 'n_steps': n_steps, 'weight_decay': weight_decay,
            'aux_weight': (aux_weight if scorer == 'mlp_multitask' else 0.0),
            'aux_winsor': list(aux_winsor),
            'n_windows': wf.n_windows,
            'mean_val_ic': wf.mean_val_ic,
            'median_val_ic': wf.median_val_ic,
            'mean_val_sharpe': wf.mean_val_sharpe,
            'positive_val_ic_fraction': wf.positive_val_ic_fraction,
            'wall_seconds': round(wall, 1),
            'windows': per_window,
            'failed': False,
        })
        print(f'    {wf.n_windows} windows  mean val IC={wf.mean_val_ic:+.4f}  '
              f'median val IC={wf.median_val_ic:+.4f}  '
              f'pos-val-IC frac={wf.positive_val_ic_fraction:.2f}  '
              f'wall={wall:.1f}s', flush=True)

    (output / 'ssl-walkforward-summary.json').write_text(
        json.dumps({
            'backbone_npz': backbone_npz_name,
            'backbone_K':   backbone.K,
            'backbone_F':   backbone.F,
            'backbone_hidden_flat': backbone.hidden_flat,
            'universe_size': len(ticker_data),
            'rebal_days': rebal_days,
            'learning_rate': learning_rate,
            'train_window_blocks': train_window_blocks,
            'val_window_blocks':   val_window_blocks,
            'step_window_blocks':  step_window_blocks,
            'start': start, 'end': end,
            'scorers': summary,
        }, indent=2))

    print('\n=== Step 4/4: per-window comparison plot ===', flush=True)
    _plot_walkforward(summary, output / 'ssl-walkforward-comparison.png')

    # Free the per-ticker .features npy spool — by this point the
    # aligned panel has been built and per-window training is done.
    # Keeping the spool around would just sit on container disk until
    # teardown.
    if os.path.exists(features_tmp_dir):
        shutil.rmtree(features_tmp_dir)

    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name.startswith('ssl-walkforward-'):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


def _save_walkforward_npz(path: Path, wf, backbone) -> None:
    import numpy as np
    blob: dict[str, 'np.ndarray'] = {}
    blob['window_idx'] = np.array([w.window_idx for w in wf.windows], dtype=np.int32)
    blob['train_block_start'] = np.array(
        [w.train_block_start for w in wf.windows], dtype=np.int32)
    blob['train_block_end'] = np.array(
        [w.train_block_end for w in wf.windows], dtype=np.int32)
    blob['val_block_start'] = np.array(
        [w.val_block_start for w in wf.windows], dtype=np.int32)
    blob['val_block_end'] = np.array(
        [w.val_block_end for w in wf.windows], dtype=np.int32)
    blob['train_ic'] = np.array([w.train_ic for w in wf.windows], dtype=np.float32)
    blob['val_ic']   = np.array([w.val_ic   for w in wf.windows], dtype=np.float32)
    blob['train_sharpe'] = np.array(
        [w.train_sharpe for w in wf.windows], dtype=np.float32)
    blob['val_sharpe'] = np.array(
        [w.val_sharpe for w in wf.windows], dtype=np.float32)
    blob['train_aux_mse'] = np.array(
        [w.train_aux_mse for w in wf.windows], dtype=np.float32)
    blob['val_aux_mse'] = np.array(
        [w.val_aux_mse for w in wf.windows], dtype=np.float32)
    if wf.windows:
        for k in wf.windows[0].head_params:
            blob[f'head_{k}'] = np.stack(
                [np.asarray(w.head_params[k], dtype=np.float32)
                 for w in wf.windows])
    blob['_summary'] = np.array(json.dumps({
        'scorer': wf.scorer, 'n_steps': wf.n_steps,
        'learning_rate': wf.learning_rate, 'weight_decay': wf.weight_decay,
        'rebal_days': wf.rebal_days,
        'train_window_blocks': wf.train_window_blocks,
        'val_window_blocks': wf.val_window_blocks,
        'step_window_blocks': wf.step_window_blocks,
        'feature_width': backbone.hidden_flat,
        'n_windows': wf.n_windows,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
    }))
    np.savez(path, **blob)
    print(f'    -> {path.name} ({path.stat().st_size // 1024}KB)', flush=True)


def _plot_walkforward(summary: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ok = [s for s in summary if not s.get('failed')]
    if not ok:
        print(f'  no successful scorers — skipping {out_path.name}', flush=True)
        return
    n_scorers = len(ok)
    fig, axes = plt.subplots(n_scorers, 1, figsize=(10, 3.2 * n_scorers),
                             sharex=True)
    if n_scorers == 1:
        axes = [axes]
    for ax, s in zip(axes, ok):
        wins = s['windows']
        idx = np.array([w['window_idx'] for w in wins])
        tr  = np.array([w['train_ic'] for w in wins])
        va  = np.array([w['val_ic']   for w in wins])
        x = np.arange(len(idx))
        ax.bar(x - 0.2, tr, width=0.4, label='train IC', color='steelblue')
        ax.bar(x + 0.2, va, width=0.4, label='val IC',   color='darkorange')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'w{i}' for i in idx], fontsize=8)
        ax.set_title(
            f"SSL backbone + {s['scorer']} "
            f"(n_steps={s['n_steps']}, wd={s['weight_decay']:g}) "
            f"— mean val IC={s['mean_val_ic']:+.4f}  "
            f"pos-val frac={s['positive_val_ic_fraction']:.2f}")
        ax.set_ylabel('IC')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  -> {out_path.name}', flush=True)


@app.local_entrypoint()
def walkforward(
    backbone_npz: str = DEFAULT_BACKBONE_NPZ,
    scorers: str = 'linear,mlp',
    n_steps: int = 200,
    weight_decay: float = 1e-3,
    tickers: str = '',
    start: str = '2000-01-03',
    end: str = '2026-04-01',
    rebal_days: int = 20,
    learning_rate: float = 1e-2,
    train_window_blocks: int = 63,
    val_window_blocks:   int = 39,
    step_window_blocks:  int = 39,
    max_tickers: int = 0,
    min_history_bars: int = 6500,
    mlp_hidden: int = 64,
    mlp_layers: int = 1,
    aux_weight: float = 0.0,
    aux_winsor_lo: float = 0.01,
    aux_winsor_hi: float = 0.99,
) -> None:
    """Local entrypoint. The backbone npz path is interpreted relative to
    the workspace root, e.g. `Output/cwtonly-AAPL+...npz`. The bytes get
    mirrored into the Modal container under /root/backbone/<basename>.

    Multi-task aux head: pass `--scorers mlp_multitask --aux-weight 0.1`
    (or any positive value) to add a magnitude-aware auxiliary head
    sharing the trunk with the rank-IC primary head. The aux target is
    cross-sectionally winsorized + z-scored forward log returns at
    `[--aux-winsor-lo, --aux-winsor-hi]` quantile clip. Aux head is
    regularization-only — `mean_val_ic` reports the primary head."""
    backbone_local = REPO_ROOT / backbone_npz
    if not backbone_local.exists():
        raise SystemExit(
            f'backbone npz not found at {backbone_local}\n'
            f'(passed --backbone-npz {backbone_npz!r})')
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aux_tag = (
        f' aux_weight={aux_weight} winsor=({aux_winsor_lo}, {aux_winsor_hi})'
        if aux_weight > 0.0 else '')
    print(f'launching factor SSL walkforward on Modal '
          f'(backbone={backbone_local.name}, scorers={scorers}, '
          f'n_steps={n_steps}, wd={weight_decay}, '
          f'train={train_window_blocks}/val={val_window_blocks}/step={step_window_blocks} blocks, '
          f'max_tickers={max_tickers}, min_history_bars={min_history_bars}'
          f'{aux_tag})')
    artifacts = train_ssl_walkforward.remote(
        backbone_npz_name=backbone_local.name,
        scorers=scorers,
        n_steps=n_steps,
        weight_decay=weight_decay,
        tickers=tickers,
        start=start,
        end=end,
        rebal_days=rebal_days,
        learning_rate=learning_rate,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        max_tickers=max_tickers,
        min_history_bars=min_history_bars,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
        aux_weight=aux_weight,
        aux_winsor_lo=aux_winsor_lo,
        aux_winsor_hi=aux_winsor_hi,
    )
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
