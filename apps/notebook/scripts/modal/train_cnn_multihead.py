"""Modal entrypoint for ss-replay multi-head CNN training + (n, w) RSI eval.

Wraps the Phase-2 / Exp-D recipe (mirrors colab/train_cnn_multihead.sh):
trains AAPL + 17-ticker pool, val on TSLA, RSI head FiLM-conditioned
over n grid {5,7,9,13,17,21,25} x w grid {1,5,10,21}; macd / price /
vol heads unconditioned. After training, runs the (n, w) zero-shot
generalization eval on CSCO and bundles all artifacts back to the
caller's local Output/.

Usage
-----
One-time setup (local):
    pip install --user modal      # or `uv tool install modal`
    modal token new               # browser flow

Smoke test (~5-15 min wall, ~$0.05-0.15):
    modal run apps/notebook/scripts/modal/train_cnn_multihead.py --steps 500

Full canonical run (~30-60 min wall, ~$0.30-0.60):
    modal run apps/notebook/scripts/modal/train_cnn_multihead.py --steps 2000

Loss is printed by the tinygrad trainer to stdout; Modal streams it to
your terminal in real time. No W&B wiring (ask if you want it added).
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
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

# Stooq subset baked into the image (built via apps/notebook/data/stooq_phase2/).
# 21 tickers (AAPL + 18 train + TSLA val + CSCO eval) totaling ~15 MB,
# preserving the daily/us/<exchange>/<bucket>/ layout that load_stooq_matrix
# walks. Replaces the per-cold-start yahoo fetch (~30-60 s) with a zero-cost
# read from the local FS, and gives bit-identical inputs across runs.
STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_phase2'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'

# Input-bundle configurations. Maps a single named experimental cell to the
# ss-replay flags + target subset + artifact prefix. The 4 cells span the
# experimentally meaningful subset of the (zscore on/off) x (returns mode in
# {none, raw, sign}) cross-product:
#
#   full       — zscore + raw returns + CWT (winning documented recipe)
#   full-sign  — zscore + sign returns + CWT (magnitude-shortcut diagnostic)
#   cwt-only   — CWT only (purest "scalogram alone" test)
#   cwt-sign   — sign returns + CWT (direction anchor, no price level)
#
# `price` head is dropped for the no-zscore cells: without rolling mu/sd,
# level info is genuinely gone from the input, so the price head would
# train to a useless ~0-R² constant and just waste the head's gradient.
BUNDLE_CONFIGS: dict[str, dict] = {
    'full': {
        'flags': ['--include-zscore-stats', '--include-returns'],
        'targets': 'rsi,macd,price,vol',
        'prefix': '',
    },
    'full-sign': {
        'flags': ['--include-zscore-stats', '--include-return-sign'],
        'targets': 'rsi,macd,price,vol',
        'prefix': 'fullsign-',
    },
    'cwt-only': {
        'flags': [],
        'targets': 'rsi,macd,vol',
        'prefix': 'cwtonly-',
    },
    'cwt-sign': {
        'flags': ['--include-return-sign'],
        'targets': 'rsi,macd,vol',
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
    """Run multi-head CNN training, then (n, w) RSI eval on `eval_ticker`.

    `bundle` selects the input channel mix and target set — see
    BUNDLE_CONFIGS at module top for the 4 cells. Each non-`full` bundle
    tags returned artifacts with its prefix so multiple bundles can
    coexist in the caller's Output/.

    Returns every file under Output/ as a dict of {filename: bytes} so
    the local entrypoint can mirror them back to the caller's disk.
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
        ['uv', 'sync', '--package', 'stocksurvey-notebook', '--inexact'],
        cwd=REMOTE_REPO, check=True)

    print(f'\n=== Step 2/4: ss-replay multi-head CNN '
          f'(steps={steps}, batch={cnn_batch_size}) ===', flush=True)
    # `--package stocksurvey-notebook` scopes the env to just this app's
    # deps. Without it, `uv run` defaults to a full-workspace sync, which
    # pulls in regime[research] -> bt 1.1.5 -> sdist build (slow + needs
    # clang for bt's vectorbt-style C extensions).
    cmd = [
        'uv', 'run', '--package', 'stocksurvey-notebook',
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
        '--vol-window', '20',
        '--cnn-batch-size', str(cnn_batch_size),
        '--cnn-steps', str(steps),
        '--cnn-no-bf16',           # T4 (sm_75) has no native bf16; PTX
                                   # emitted for bf16 fails to link against
                                   # NVRTC's default include set. fp32
                                   # works fine within T4's 16 GB.
        '--device', 'auto',
        '--output-dir', output,
    ]
    print('+ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)

    print(f'\n=== Step 3/4: zero-shot (n, w) RSI eval on {eval_ticker} ===',
          flush=True)
    npz_paths = sorted(Path(output).glob(f'{primary}+*-cnn-*.npz'))
    if not npz_paths:
        raise RuntimeError(
            f'no {primary}+*-cnn-*.npz produced under {output}; '
            f'training likely failed silently')
    npz_path = npz_paths[-1]
    print(f'eval source: {npz_path.name}', flush=True)

    # Activate the uv venv's site-packages so the eval (which runs in the
    # system Python, not via subprocess) can import the editable-installed
    # workspace packages. site.addsitedir processes .pth files and PEP 660
    # editable hooks; bare sys.path.insert does not.
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')
    eval_stats = _zeroshot_eval(
        npz_path=npz_path, ticker=eval_ticker, output_dir=Path(output))
    (Path(output) / f'{eval_ticker}-zeroshot-stats.json').write_text(
        json.dumps(eval_stats, indent=2, default=float))

    print(f'\n=== Step 4/4: input-attention saliency on {primary} ===',
          flush=True)
    print('  -- FiLM rsi head (cond_a vs cond_b) --', flush=True)
    film_stats = _film_attention(
        npz_path=npz_path, ticker=primary, output_dir=Path(output))
    (Path(output) / f'{primary}-film-attention-stats.json').write_text(
        json.dumps(film_stats, indent=2, default=float))
    print('  -- unconditioned heads (macd vs vol) --', flush=True)
    uncond_stats = _uncond_attention(
        npz_path=npz_path, ticker=primary, output_dir=Path(output))
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


def _load_npz_meta(npz_path: Path):
    """Shared npz + meta loader for downstream eval/attention helpers.

    Returns (data, meta, K, scales, rsi_n_grid, rsi_w_grid, n_max, w_max).
    """
    import numpy as np
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(data['_meta'].item())
    K = int(meta['window_cols'])
    scales = [int(s) for s in meta['scales']]
    rsi_n_grid = tuple(meta.get('rsi_n_grid') or ())
    rsi_w_grid = tuple(meta.get('rsi_w_grid') or ())
    n_max = (float(max(rsi_n_grid)) if rsi_n_grid
             else float(meta['rsi_n']))
    w_max = float(max(rsi_w_grid)) if rsi_w_grid else 1.0
    return data, meta, K, scales, rsi_n_grid, rsi_w_grid, n_max, w_max


def _load_eval_ticker(ticker: str, meta: dict, scales: list[int]):
    """Wrap `load_ticker` mirroring the training config from `meta`.

    Always passes empty rsi_n_grid/rsi_w_grid — both downstream consumers
    (zeroshot and attention) compute RSI(n, w) themselves via
    `rsi_strided`, so the per-cell ground-truth grid in `td` is unused.
    """
    from ss_notebook.replay.features import load_ticker
    return load_ticker(
        ticker,
        stooq_dir=STOOQ_SUBSET, kaggle_dir=None, use_yahoo=False,
        start=meta['start'], end=meta['end'],
        scales=scales, lookback=int(meta['lookback']),
        window_cols=int(meta['window_cols']),
        include_zscore_stats=bool(meta.get('include_zscore_stats')),
        include_returns=bool(meta.get('include_returns')),
        include_return_sign=bool(meta.get('include_return_sign', False)),
        decoder=meta['decoder'],
        rsi_n=int(meta['rsi_n']),
        macd_fast=int(meta['macd_fast']),
        macd_slow=int(meta['macd_slow']),
        macd_signal=int(meta['macd_signal']),
        vol_window=int(meta.get('vol_window', 20)),
        rsi_n_grid=(), rsi_w_grid=(),
    )


def _channel_labels(meta: dict, scales: list[int]) -> list[str]:
    """Build the per-channel label list matching the trainer's input stack."""
    return_label = ('return-sign' if meta.get('include_return_sign')
                    else 'return' if meta.get('include_returns')
                    else None)
    return (
        [f'coeff s={s}' for s in scales]
        + [f'power s={s}' for s in scales]
        + (['z-mu', 'z-std'] if meta.get('include_zscore_stats') else [])
        + ([return_label] if return_label else [])
    )


def _tinygrad_backbone(data, ref: str = 'rsi'):
    """Load shared backbone tensors (identical across per-target prefixes)
    as frozen tinygrad Tensors. Returns (feat_mu, feat_sd, conv_params).

    All returned tensors have `requires_grad=False`; the only autograd
    leaf in the attention computation is the input `X`, which is what we
    take gradients with respect to.
    """
    import numpy as np
    from tinygrad.tensor import Tensor
    feat_mu = Tensor(np.asarray(data[f'{ref}__feat_mu'], dtype=np.float32),
                     requires_grad=False)
    feat_sd = Tensor(np.asarray(data[f'{ref}__feat_sd'], dtype=np.float32),
                     requires_grad=False)
    n_layers = sum(1 for k in data.files
                   if k.startswith(f'{ref}__conv') and k.endswith('_W'))
    conv_params = [
        (Tensor(np.asarray(data[f'{ref}__conv{i}_W'], dtype=np.float32),
                requires_grad=False),
         Tensor(np.asarray(data[f'{ref}__conv{i}_b'], dtype=np.float32),
                requires_grad=False))
        for i in range(n_layers)
    ]
    return feat_mu, feat_sd, conv_params


def _conv1d_tg(x, W, b):
    """Canonical NHC/HIO/NHC stride-1 valid convolution mirroring
    `ss_notebook.replay.decoders._conv1d`. Tinygrad's `conv2d` is NCHW
    so we permute in/out to keep the npz tensor layout identical to the
    legacy JAX path (and to what the trainer wrote)."""
    x_bcl = x.permute(0, 2, 1)
    W_oik = W.permute(2, 1, 0)
    y_bcl = x_bcl.conv2d(W_oik)
    return y_bcl.permute(0, 2, 1) + b


def _batched_saliency(forward_fn, X_batch_np, K: int, F: int):
    """Run a single batched forward + backward through `forward_fn`,
    return the mean of `|d output / d X|` across the batch as `(K, F)`.

    `forward_fn` takes a Tensor of shape `(B, K, F)` and returns a
    `(B,)`-shaped scalar-per-row tensor. We sum and backward once: the
    gradient of `sum(output)` wrt `X[i]` equals `d output[i] / d X[i]`
    because the per-row forwards have no cross-batch interaction.
    Replaces the JAX/jit-grad per-bar Python loop with one tinygrad
    kernel — much faster on CPU, same numerics."""
    import numpy as np
    from tinygrad.tensor import Tensor
    X = Tensor(X_batch_np.astype(np.float32), requires_grad=True)
    output = forward_fn(X)            # (B,)
    output.sum().backward()
    grads = X.grad.numpy()            # (B, K, F)
    return np.mean(np.abs(grads), axis=0)


def _topk_stats(sal, chan_labels: list[str], F: int, k: int = 8) -> dict:
    """Top-k (lag, channel) cells + per-channel sum-of-|grad| from a
    saliency map of shape (K, F). Returns a JSON-serializable dict."""
    import numpy as np
    flat = sal.flatten()
    top_idx = np.argsort(flat)[::-1][:k]
    cells = [
        {'lag': int(fi // F), 'ch': int(fi % F),
         'ch_label': chan_labels[int(fi % F)],
         'grad': float(flat[fi])}
        for fi in top_idx
    ]
    per_chan = sal.sum(axis=0)
    chan_top = np.argsort(per_chan)[::-1][:k]
    chans = [
        {'ch': int(ci), 'ch_label': chan_labels[int(ci)],
         'sum_grad': float(per_chan[ci])}
        for ci in chan_top
    ]
    return {'top_cells': cells, 'top_channels': chans}


def _plot_3panel_attention(
    *, sal_a, sal_b, label_a: str, label_b: str,
    chan_labels: list[str], F: int, suptitle: str, out_path: Path,
    color_a: str = 'Blues', color_b: str = 'Reds',
    grad_label: str = '|d head / d X| avg',
    diff_title: str,
    shared_vmax: bool = True, normalize_diff: bool = False,
) -> None:
    """3-panel saliency figure: |sal_a|, |sal_b|, signed diff.

    `shared_vmax=True` uses max(sal_a, sal_b) as the shared color scale
    (FiLM rsi case where the two cond vectors are commensurate).
    `shared_vmax=False` uses each panel's own max (macd/vol case where
    the heads are on different scales).

    `normalize_diff=True` normalizes each map by its own max before
    differencing so the diff reflects relative pattern, not magnitude
    (used for cross-head comparisons).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)
    if shared_vmax:
        vmax = float(max(sal_a.max(), sal_b.max()))
        vmaxes = (vmax, vmax)
    else:
        vmaxes = (float(sal_a.max()), float(sal_b.max()))
    for ax, sal, color, title, vmax in [
        (axes[0], sal_a, color_a, label_a, vmaxes[0]),
        (axes[1], sal_b, color_b, label_b, vmaxes[1]),
    ]:
        im = ax.imshow(sal.T, aspect='auto', origin='lower',
                       cmap=color, vmin=0, vmax=vmax)
        ax.set_xlabel('Lag (0 = most recent bar)')
        ax.set_ylabel('Channel')
        ax.set_yticks(range(F))
        ax.set_yticklabels(chan_labels, fontsize=6)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=grad_label, fraction=0.025)

    if normalize_diff:
        a_n = sal_a / max(sal_a.max(), 1e-12)
        b_n = sal_b / max(sal_b.max(), 1e-12)
        diff = a_n - b_n
        diff_clabel = 'Δ saliency (per-head normalized)'
    else:
        diff = sal_a - sal_b
        diff_clabel = 'Δ saliency'
    vlim = float(np.abs(diff).max()) or 1.0
    im = axes[2].imshow(
        diff.T, aspect='auto', origin='lower', cmap='seismic_r',
        norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim))
    axes[2].set_xlabel('Lag (0 = most recent bar)')
    axes[2].set_ylabel('Channel')
    axes[2].set_yticks(range(F))
    axes[2].set_yticklabels(chan_labels, fontsize=6)
    axes[2].set_title(diff_title)
    fig.colorbar(im, ax=axes[2], label=diff_clabel, fraction=0.025)

    fig.suptitle(suptitle, fontsize=11, fontweight='bold')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _zeroshot_eval(
    *, npz_path: Path, ticker: str, output_dir: Path,
) -> dict:
    """Programmatic port of colab/zeroshot_eval.py.

    Loads the trained npz, builds features for `ticker` matching the
    training config, runs the backbone forward (numpy conv1d — the
    model is small enough that JAX is overkill here), applies each
    head, and produces:
      - {ticker}-replay-zeroshot-uncond.png    (price/macd/vol)
      - {ticker}-replay-zeroshot-rsi-wn-sweep.png  (RSI heatmap + ts)
    Returns a stats dict (per unconditioned head + (n, w) R² grid).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from ss_notebook.replay.features import rsi_strided
    from ss_notebook.replay.metrics import fit_stats

    (data, meta, K, scales, rsi_n_grid, rsi_w_grid,
     n_max_grid, w_max_grid) = _load_npz_meta(npz_path)
    print(f'  trained on {len(meta["train_tickers"])} tickers, '
          f'K={K}, scales={scales}, targets={meta["targets"]}, '
          f'rsi_n_grid={rsi_n_grid}, rsi_w_grid={rsi_w_grid}')

    td = _load_eval_ticker(ticker, meta, scales)
    F = td.features.shape[1] // K
    print(f'  {ticker}: {len(td.prices)} bars, {td.valid.sum()} valid, '
          f'(K, F) = ({K}, {F})')

    ref = meta['targets'][0]
    feat_mu = data[f'{ref}__feat_mu']
    feat_sd = data[f'{ref}__feat_sd']
    X = td.features.reshape(-1, K, F).astype(np.float32)
    Xn = (X - feat_mu) / feat_sd

    n_layers = sum(1 for k in data.files
                   if k.startswith(f'{ref}__conv') and k.endswith('_W'))
    convs = [
        (data[f'{ref}__conv{i}_W'], data[f'{ref}__conv{i}_b'])
        for i in range(n_layers)
    ]

    # numpy conv1d (NLC layout -> stride-1 valid conv via stride tricks).
    def conv1d_relu(x: 'np.ndarray', W: 'np.ndarray',
                    b: 'np.ndarray') -> 'np.ndarray':
        # x: (N, L, C_in)  W: (kW, C_in, C_out)  b: (C_out,)
        kW, C_in, C_out = W.shape
        N, L, _ = x.shape
        L_out = L - kW + 1
        # build (N, L_out, kW, C_in) view via stride tricks
        s_n, s_l, s_c = x.strides
        windows = np.lib.stride_tricks.as_strided(
            x, shape=(N, L_out, kW, C_in),
            strides=(s_n, s_l, s_l, s_c), writeable=False)
        # contract over (kW, C_in) with W reshaped to (kW * C_in, C_out)
        W_flat = W.reshape(kW * C_in, C_out)
        Wx = windows.reshape(N, L_out, kW * C_in) @ W_flat + b
        return np.maximum(Wx, 0.0)

    chunk = 16_384
    H_chunks = []
    for s in range(0, Xn.shape[0], chunk):
        h = Xn[s:s + chunk]
        for W, b in convs:
            h = conv1d_relu(h, W, b)
        H_chunks.append(h.reshape(h.shape[0], -1))
    H = np.concatenate(H_chunks)
    print(f'  backbone latent: {H.shape}')

    def _film_mlp(W0, b0, W1, b1, c):
        return np.maximum(0.0, c @ W0 + b0) @ W1 + b1

    def apply_head(target: str, cond_vec=None) -> 'np.ndarray':
        head_W = data[f'{target}__head_W']
        head_b = data[f'{target}__head_b']
        target_mu = float(data[f'{target}__target_mu'][0])
        target_sd = float(data[f'{target}__target_sd'][0])
        cond_dim_key = f'{target}__head_cond_dim'
        cond_dim = (int(data[cond_dim_key][0])
                    if cond_dim_key in data.files else 0)
        has_film = f'{target}__head_film_gamma_W0' in data.files

        if cond_dim == 0:
            latent = H
        elif has_film:
            cond_arr = np.asarray(cond_vec, dtype=np.float32)
            cb = np.broadcast_to(cond_arr[None, :], (H.shape[0], cond_dim))
            gamma = _film_mlp(
                data[f'{target}__head_film_gamma_W0'],
                data[f'{target}__head_film_gamma_b0'],
                data[f'{target}__head_film_gamma_W1'],
                data[f'{target}__head_film_gamma_b1'], cb) + 1.0
            beta = _film_mlp(
                data[f'{target}__head_film_beta_W0'],
                data[f'{target}__head_film_beta_b0'],
                data[f'{target}__head_film_beta_W1'],
                data[f'{target}__head_film_beta_b1'], cb)
            latent = gamma * H + beta
        else:
            cond_arr = np.asarray(cond_vec, dtype=np.float32)
            cb = np.broadcast_to(cond_arr[None, :], (H.shape[0], cond_dim))
            latent = np.concatenate([H, cb], axis=-1)
        yhat_std = (latent @ head_W + head_b).squeeze(-1)
        return yhat_std.astype(np.float64) * target_sd + target_mu

    out_stats: dict = {'unconditioned': {}, 'rsi_wn_grid': {}}

    panel_specs = {
        'price': ('Close', None),
        'macd':  (f'MACD({meta["macd_fast"]},{meta["macd_slow"]},'
                  f'{meta["macd_signal"]}) line', (0,)),
        'vol':   (f'RealizedVol({meta.get("vol_window", 20)})', None),
    }
    uncond = [t for t in ('price', 'macd', 'vol') if t in meta['targets']]
    fig, axes = plt.subplots(
        len(uncond), 1, figsize=(13, 3.2 * len(uncond)),
        sharex=True, squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f'{ticker} zero-shot — unconditioned heads '
                 f'(K={K}, scales={len(meta["scales"])}, n_features={K*F})',
                 fontsize=12, fontweight='bold')
    for ax, target in zip(axes, uncond):
        yhat = apply_head(target)
        gt = td.targets[target]
        v = td.valid
        stats = fit_stats(yhat[v], gt[v])
        out_stats['unconditioned'][target] = stats
        print(f'  {ticker} zero-shot {target:>5s}: '
              f'R²={stats["r2"]:.4f}  RMSE={stats["rmse"]:.3e}  '
              f'max|Δ|={stats["max_abs"]:.3e}')
        yhat_full = np.full_like(gt, np.nan)
        yhat_full[v] = yhat[v]
        label, hlines = panel_specs[target]
        ax.plot(td.dates, gt, color='black', linewidth=0.7, alpha=0.6,
                label=f'{label} ground truth')
        ax.plot(td.dates, yhat_full, color='crimson', linewidth=0.9,
                linestyle='--',
                label=f'{label} reconstructed (zero-shot)')
        if hlines:
            for y in hlines:
                ax.axhline(y, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylabel(label)
        ax.set_title(f'R²={stats["r2"]:.4f}  RMSE={stats["rmse"]:.3e}',
                     fontsize=9, loc='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(td.dates[0], td.dates[-1])
    plt.tight_layout()
    fig.savefig(output_dir / f'{ticker}-replay-zeroshot-uncond.png', dpi=150)
    plt.close(fig)

    if 'rsi' not in meta['targets'] or not rsi_n_grid:
        print('  (no rsi grid in this run — skipping (n, w) sweep)')
        return out_stats

    n_sweep = sorted({*rsi_n_grid, 6, 8, 11, 15, 19, 28, 35})
    w_sweep = sorted({*rsi_w_grid, 3, 7, 15, 25}) if rsi_w_grid else [1]
    print(f'\n  RSI (n, w) sweep:  n={n_sweep}  w={w_sweep}')
    header = '  w \\ n  |  ' + '  '.join(f'{n:>6d}' for n in n_sweep)
    print(header)
    print('-' * len(header))
    r2_grid = np.full((len(w_sweep), len(n_sweep)), np.nan)
    cell_records: dict[tuple[int, int], dict] = {}
    for wi, w in enumerate(w_sweep):
        row = []
        for ni, n in enumerate(n_sweep):
            gt_n = rsi_strided(td.prices, n=int(n), w=int(w))
            cond_vec = np.array([n / n_max_grid, w / w_max_grid],
                                dtype=np.float32)
            yhat_n = apply_head('rsi', cond_vec=cond_vec)
            v = td.valid & np.isfinite(gt_n) & np.isfinite(yhat_n)
            stats = fit_stats(yhat_n[v], gt_n[v])
            r2_grid[wi, ni] = stats['r2']
            cell_records[(w, n)] = dict(
                stats=stats, gt=gt_n, yhat=yhat_n, valid=v,
                in_n_grid=(n in rsi_n_grid),
                in_w_grid=(w in rsi_w_grid),
            )
            out_stats['rsi_wn_grid'][f'w={w},n={n}'] = stats
            in_grid = n in rsi_n_grid and w in rsi_w_grid
            tag = '*' if in_grid else ' '
            row.append(f'{tag}{stats["r2"]:>5.2f}')
        print(f'  w={w:>3d}  |  ' + '  '.join(row))

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.0])
    fig.suptitle(
        f'{ticker} zero-shot RSI(n, w) — FiLM head trained on '
        f'n ∈ {sorted(rsi_n_grid)}  ×  w ∈ {sorted(rsi_w_grid)}',
        fontsize=12, fontweight='bold')
    ax_hm = fig.add_subplot(gs[0, :])
    im = ax_hm.imshow(r2_grid, aspect='auto', origin='lower',
                      vmin=-0.5, vmax=1.0, cmap='RdYlGn')
    ax_hm.set_xticks(range(len(n_sweep)))
    ax_hm.set_xticklabels([str(n) for n in n_sweep])
    ax_hm.set_yticks(range(len(w_sweep)))
    ax_hm.set_yticklabels([str(w) for w in w_sweep])
    ax_hm.set_xlabel('RSI period n')
    ax_hm.set_ylabel('Stride w (1=daily, 5≈weekly, 21≈monthly)')
    ax_hm.set_title('R² across the (w, n) grid — boxes mark training cells',
                    fontsize=10)
    for wi, w in enumerate(w_sweep):
        for ni, n in enumerate(n_sweep):
            r2 = r2_grid[wi, ni]
            ax_hm.text(ni, wi, f'{r2:.2f}', ha='center', va='center',
                       fontsize=8,
                       color='white' if r2 < 0.5 or r2 > 0.95 else 'black')
            if (n in rsi_n_grid) and (w in rsi_w_grid):
                ax_hm.add_patch(plt.Rectangle(
                    (ni - 0.5, wi - 0.5), 1, 1, fill=False,
                    edgecolor='black', linewidth=1.6))
    fig.colorbar(im, ax=ax_hm, label='R²', fraction=0.025)

    ts_picks: list[tuple[int, int, str]] = []
    in_grid_n_mid = sorted(rsi_n_grid)[len(rsi_n_grid) // 2]
    if 1 in rsi_w_grid:
        ts_picks.append((1, in_grid_n_mid, 'daily / in-grid n'))
    if 21 in rsi_w_grid:
        ts_picks.append((21, in_grid_n_mid, 'monthly / in-grid n'))
    ax_ts = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (w, n, label) in zip(ax_ts, ts_picks):
        rec = cell_records[(w, n)]
        gt_n, yhat_n, v, st = (rec['gt'], rec['yhat'], rec['valid'],
                               rec['stats'])
        yhat_full = np.full_like(gt_n, np.nan)
        yhat_full[v] = yhat_n[v]
        ax.plot(td.dates, gt_n, color='black', linewidth=0.7, alpha=0.6,
                label=f'true RSI(n={n}, w={w})')
        ax.plot(td.dates, yhat_full, color='crimson', linewidth=0.9,
                linestyle='--', label=f'pred RSI(n={n}, w={w})')
        ax.axhline(30, color='gray', linestyle=':', alpha=0.4)
        ax.axhline(70, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylabel(f'RSI({n}, w={w})')
        ax.set_title(f'{label}  R²={st["r2"]:.4f}', fontsize=9, loc='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(td.dates[0], td.dates[-1])

    off_cells = [(w, n) for (w, n), rec in cell_records.items()
                 if not (rec['in_n_grid'] and rec['in_w_grid'])]
    off_cells.sort(key=lambda wn: cell_records[wn]['stats']['r2'])
    ax_off = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    picks_off = (off_cells[:1] + off_cells[-1:]) if off_cells else []
    for ax, (w, n) in zip(ax_off, picks_off):
        rec = cell_records[(w, n)]
        gt_n, yhat_n, v, st = (rec['gt'], rec['yhat'], rec['valid'],
                               rec['stats'])
        yhat_full = np.full_like(gt_n, np.nan)
        yhat_full[v] = yhat_n[v]
        kind = (('OFF-GRID-n' if not rec['in_n_grid'] else 'in-grid-n') +
                ' / ' +
                ('OFF-GRID-w' if not rec['in_w_grid'] else 'in-grid-w'))
        ax.plot(td.dates, gt_n, color='black', linewidth=0.7, alpha=0.6,
                label=f'true RSI(n={n}, w={w})')
        ax.plot(td.dates, yhat_full, color='crimson', linewidth=0.9,
                linestyle='--', label=f'pred RSI(n={n}, w={w})')
        ax.axhline(30, color='gray', linestyle=':', alpha=0.4)
        ax.axhline(70, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylabel(f'RSI({n}, w={w})')
        ax.set_title(f'{kind}  R²={st["r2"]:.4f}', fontsize=9, loc='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(td.dates[0], td.dates[-1])

    plt.tight_layout()
    fig.savefig(output_dir / f'{ticker}-replay-zeroshot-rsi-wn-sweep.png',
                dpi=150)
    plt.close(fig)
    return out_stats


def _film_attention(
    *, npz_path: Path, ticker: str, output_dir: Path,
    cond_a: tuple[int, int] = (7, 1),
    cond_b: tuple[int, int] = (17, 10),
    n_bars: int = 200,
) -> dict:
    """Programmatic port of colab/film_attention.py.

    For the FiLM-conditioned rsi head, computes |d rsi / d X| averaged
    over `n_bars` random bars at two cond vectors and renders a 3-panel
    figure (cond_a saliency, cond_b saliency, signed diff). Diagnoses
    the FiLM machinery's wavelength selectivity: short-period RSI cond
    should attend to recent lags + high-freq scales; long-period RSI
    cond should attend to longer lags + low-freq scales.

    Output: `{ticker}-film-attention.png` + returned dict of top-k
    cells per cond. tinygrad is used for autograd (matches the trainer
    framework; the npz weights become frozen Tensors and only the input
    is a leaf with requires_grad=True).
    """
    import numpy as np
    from tinygrad.tensor import Tensor

    (data, meta, K, scales, rsi_n_grid, rsi_w_grid,
     n_max_grid, w_max_grid) = _load_npz_meta(npz_path)

    if 'rsi__head_film_gamma_W0' not in data.files:
        print('  WARN: no FiLM keys in npz; skipping attention plot '
              '(saliency would be cond-invariant under additive concat)')
        return {'skipped': 'no FiLM keys'}

    print(f'  backbone: K={K}, scales={scales}, '
          f'rsi_n_grid={rsi_n_grid}, rsi_w_grid={rsi_w_grid}')

    feat_mu, feat_sd, conv_params = _tinygrad_backbone(data, ref='rsi')
    head_W = Tensor(np.asarray(data['rsi__head_W'], dtype=np.float32),
                    requires_grad=False)
    head_b = Tensor(np.asarray(data['rsi__head_b'], dtype=np.float32),
                    requires_grad=False)
    target_mu = float(data['rsi__target_mu'][0])
    target_sd = float(data['rsi__target_sd'][0])
    film = {k: Tensor(np.asarray(data[f'rsi__head_film_{k}'],
                                  dtype=np.float32), requires_grad=False)
            for k in ('gamma_W0', 'gamma_b0', 'gamma_W1', 'gamma_b1',
                      'beta_W0', 'beta_b0', 'beta_W1', 'beta_b1')}

    def film_mlp(W0, b0, W1, b1, c):
        return (c @ W0 + b0).relu() @ W1 + b1

    def make_forward(cond_n: int, cond_w: int):
        cv_np = np.array([cond_n / n_max_grid, cond_w / w_max_grid],
                         dtype=np.float32)
        cv = Tensor(cv_np, requires_grad=False)

        def forward(X):
            # X: (B, K, F) — leaf with requires_grad=True
            Xn = (X - feat_mu) / feat_sd                 # broadcast (1,K,F)
            h = Xn
            for W, b in conv_params:
                h = _conv1d_tg(h, W, b).relu()
            latent = h.reshape(h.shape[0], -1)           # (B, latent_dim)
            gamma = film_mlp(film['gamma_W0'], film['gamma_b0'],
                             film['gamma_W1'], film['gamma_b1'],
                             cv) + 1.0                   # (latent_dim,)
            beta = film_mlp(film['beta_W0'], film['beta_b0'],
                            film['beta_W1'], film['beta_b1'], cv)
            latent_mod = gamma * latent + beta            # broadcast
            yhat_std = (latent_mod @ head_W + head_b).squeeze(-1)  # (B,)
            return yhat_std * target_sd + target_mu

        return forward

    td = _load_eval_ticker(ticker, meta, scales)
    F = td.features.shape[1] // K
    X_all = td.features.reshape(-1, K, F).astype(np.float32)
    valid_idx = np.where(td.valid)[0]
    n_use = min(n_bars, len(valid_idx))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(valid_idx, size=n_use, replace=False)
    X_batch = X_all[sample_idx]
    print(f'  {ticker}: batched |grad| over {n_use} bars '
          f'(out of {len(valid_idx)} valid)')

    print(f'  cond_a = (n={cond_a[0]}, w={cond_a[1]})')
    sal_a = _batched_saliency(make_forward(*cond_a), X_batch, K, F)
    print(f'  cond_b = (n={cond_b[0]}, w={cond_b[1]})')
    sal_b = _batched_saliency(make_forward(*cond_b), X_batch, K, F)

    chan_labels = _channel_labels(meta, scales)
    if len(chan_labels) != F:
        raise RuntimeError(
            f'channel-label count {len(chan_labels)} != F={F}; '
            f'meta channel config likely drifted from npz weights')

    stats = {
        'cond_a': list(cond_a),
        'cond_b': list(cond_b),
        'n_bars': int(n_use),
        'cond_a_topk': _topk_stats(sal_a, chan_labels, F),
        'cond_b_topk': _topk_stats(sal_b, chan_labels, F),
    }
    print('  top channels @ cond_a (sum |grad| over lags):')
    for r in stats['cond_a_topk']['top_channels'][:5]:
        print(f"    ch {r['ch']:>2d} ({r['ch_label']:<14s})  "
              f"sum |grad|={r['sum_grad']:.3e}")
    print('  top channels @ cond_b (sum |grad| over lags):')
    for r in stats['cond_b_topk']['top_channels'][:5]:
        print(f"    ch {r['ch']:>2d} ({r['ch_label']:<14s})  "
              f"sum |grad|={r['sum_grad']:.3e}")

    _plot_3panel_attention(
        sal_a=sal_a, sal_b=sal_b,
        label_a=f'RSI(n={cond_a[0]}, w={cond_a[1]}) — short period',
        label_b=f'RSI(n={cond_b[0]}, w={cond_b[1]}) — long period',
        chan_labels=chan_labels, F=F,
        suptitle=(f'FiLM rsi-head input attention — {ticker}, K={K}, '
                  f'{n_use} bars averaged\nbackbone: {npz_path.name}'),
        out_path=output_dir / f'{ticker}-film-attention.png',
        grad_label='|d rsi / d X| avg',
        diff_title=(f'sal[(n={cond_a[0]},w={cond_a[1]})] − '
                    f'sal[(n={cond_b[0]},w={cond_b[1]})]\n'
                    f'blue = short dominates; red = long dominates'),
        shared_vmax=True, normalize_diff=False,
    )
    return stats


def _uncond_attention(
    *, npz_path: Path, ticker: str, output_dir: Path,
    head_a: str = 'macd', head_b: str = 'vol',
    n_bars: int = 200,
) -> dict:
    """Programmatic port of colab/attention_macd_vol.py.

    Computes |d head / d X| averaged over `n_bars` bars for two
    unconditioned heads (default macd vs vol) and renders a 3-panel
    figure (head_a saliency, head_b saliency, normalized diff).

    Question this answers: do macd and vol live on the same input
    channels (e.g. both shortcut through `return` like rsi did), or do
    they pull attention onto different parts of the bundle? vol is
    naturally a 2nd-moment-of-returns quantity that the wavelet `power`
    channels carry — if it's reading them, the indicator-shape bias
    isn't universal.

    Output: `{ticker}-uncond-attention.png` + JSON-serializable top-k
    dict per head. tinygrad autograd matches the trainer framework.
    """
    import numpy as np
    from tinygrad.tensor import Tensor

    (data, meta, K, scales, _rsi_n_grid, _rsi_w_grid,
     _n_max_grid, _w_max_grid) = _load_npz_meta(npz_path)

    for h in (head_a, head_b):
        if f'{h}__head_W' not in data.files:
            print(f'  WARN: head {h!r} not in npz (targets={meta["targets"]});'
                  f' skipping uncond attention plot')
            return {'skipped': f'missing head {h!r}'}
        cdk = f'{h}__head_cond_dim'
        cd = int(data[cdk][0]) if cdk in data.files else 0
        if cd != 0:
            print(f'  WARN: head {h!r} has cond_dim={cd}; uncond attention '
                  f'expects 0. Skipping.')
            return {'skipped': f'head {h!r} is conditioned'}

    feat_mu, feat_sd, conv_params = _tinygrad_backbone(data, ref=head_a)

    def make_forward(target: str):
        head_W = Tensor(np.asarray(data[f'{target}__head_W'],
                                    dtype=np.float32), requires_grad=False)
        head_b = Tensor(np.asarray(data[f'{target}__head_b'],
                                    dtype=np.float32), requires_grad=False)
        target_mu = float(data[f'{target}__target_mu'][0])
        target_sd = float(data[f'{target}__target_sd'][0])

        def forward(X):
            Xn = (X - feat_mu) / feat_sd
            h = Xn
            for W, b in conv_params:
                h = _conv1d_tg(h, W, b).relu()
            latent = h.reshape(h.shape[0], -1)
            yhat_std = (latent @ head_W + head_b).squeeze(-1)
            return yhat_std * target_sd + target_mu

        return forward

    td = _load_eval_ticker(ticker, meta, scales)
    F = td.features.shape[1] // K
    X_all = td.features.reshape(-1, K, F).astype(np.float32)
    valid_idx = np.where(td.valid)[0]
    n_use = min(n_bars, len(valid_idx))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(valid_idx, size=n_use, replace=False)
    X_batch = X_all[sample_idx]
    print(f'  {ticker}: batched |grad| over {n_use} bars per head')

    print(f'  computing {head_a} saliency...')
    sal_a = _batched_saliency(make_forward(head_a), X_batch, K, F)
    print(f'  computing {head_b} saliency...')
    sal_b = _batched_saliency(make_forward(head_b), X_batch, K, F)

    chan_labels = _channel_labels(meta, scales)
    if len(chan_labels) != F:
        raise RuntimeError(
            f'channel-label count {len(chan_labels)} != F={F}')

    stats = {
        'head_a': head_a, 'head_b': head_b,
        'n_bars': int(n_use),
        f'{head_a}_topk': _topk_stats(sal_a, chan_labels, F),
        f'{head_b}_topk': _topk_stats(sal_b, chan_labels, F),
    }
    for h, key in ((head_a, f'{head_a}_topk'), (head_b, f'{head_b}_topk')):
        print(f'  top channels @ {h} (sum |grad| over lags):')
        for r in stats[key]['top_channels'][:5]:
            print(f"    ch {r['ch']:>2d} ({r['ch_label']:<14s})  "
                  f"sum |grad|={r['sum_grad']:.3e}")

    label_a = (f'{head_a} ({meta["macd_fast"]}, {meta["macd_slow"]}, '
               f'{meta["macd_signal"]})') if head_a == 'macd' else head_a
    label_b = (f'{head_b} ({meta.get("vol_window", 20)}-bar)'
               if head_b == 'vol' else head_b)

    _plot_3panel_attention(
        sal_a=sal_a, sal_b=sal_b,
        label_a=f'{label_a} — unconditioned head',
        label_b=f'{label_b} — unconditioned head',
        chan_labels=chan_labels, F=F,
        suptitle=(f'{head_a} vs {head_b} input attention — {ticker}, K={K}, '
                  f'{n_use} bars averaged\nbackbone: {npz_path.name}'),
        out_path=output_dir / f'{ticker}-uncond-attention.png',
        grad_label='|d head / d X| avg',
        diff_title=(f'Normalized diff ({head_a} − {head_b})\n'
                    f'blue = {head_a} dominates; red = {head_b} dominates'),
        shared_vmax=False, normalize_diff=True,
    )
    return stats


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
