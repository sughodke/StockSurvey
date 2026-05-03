"""Post-training evaluation for the multi-head CNN backbone.

Four public entry points, all consuming a saved npz produced by
`ss-replay --decoder cnn`:

  * `zeroshot_eval(npz_path, ticker, output_dir, ...)` — full
    package: per-head reconstruction stats on `ticker`, single-row
    "uncond" PNG (price/macd/vol/cci heads that aren't FiLM-
    conditioned), and one (n, w) heatmap PNG per FiLM-conditioned
    head (rsi/cci 2-D, vol/macd 1-D). Mutates and returns a stats
    dict. Wraps `grid_sweep_eval` internally.
  * `grid_sweep_eval(...)` — single-head (n, w) sweep; called from
    `zeroshot_eval` per FiLM-conditioned target. Useful standalone
    if you only want one head's heatmap.
  * `film_attention(npz_path, ticker, output_dir, ...)` — input-
    gradient saliency for the FiLM-conditioned rsi head at two cond
    vectors, with shared color scale and a signed-diff panel.
  * `uncond_attention(npz_path, ticker, output_dir, ...)` — input-
    gradient saliency for two unconditioned heads (default macd vs
    vol), normalized-diff panel.

These were extracted from `apps/replay/scripts/modal/train_cnn_multihead.py`
(which had grown to ~800 lines, 60% of which was post-training eval).
The Modal entrypoint now imports from here; local callers and
notebooks can use the same code without firing Modal. Colab scripts
under `scripts/colab/` are now duplicates and can be deleted.

Data source for ticker loading is parameterized — pass `stooq_dir`,
`kaggle_dir`, or `use_yahoo=True`. The Modal entrypoint passes its
baked-in Stooq subset path; local invocations typically use
`use_yahoo=True`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- helpers


def _load_npz_meta(npz_path: Path):
    """Shared npz + meta loader.

    Returns (data, meta, K, scales, rsi_n_grid, rsi_w_grid, n_max, w_max).
    """
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


def _load_eval_ticker(
    ticker: str, meta: dict, scales: list[int],
    *, stooq_dir: str | None = None, kaggle_dir: str | None = None,
    use_yahoo: bool = False,
):
    """Wrap `load_ticker` mirroring the training config from `meta`.

    Always passes empty grid kwargs — both downstream consumers
    (zeroshot and attention) compute strided indicators themselves
    (rsi_strided / cci_strided / etc.), so the per-cell ground-truth
    grid in `td` is unused.

    If none of stooq_dir/kaggle_dir is set and use_yahoo=False,
    raises — caller must specify a data source.
    """
    from replay.features import load_ticker
    if stooq_dir is None and kaggle_dir is None and not use_yahoo:
        raise ValueError(
            'specify one of stooq_dir, kaggle_dir, or use_yahoo=True')
    return load_ticker(
        ticker,
        stooq_dir=stooq_dir, kaggle_dir=kaggle_dir, use_yahoo=use_yahoo,
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
        cci_n=int(meta.get('cci_n', 20)),
        rsi_n_grid=(), rsi_w_grid=(),
        cci_n_grid=(), cci_w_grid=(),
    )


def _channel_labels(meta: dict, scales: list[int]) -> list[str]:
    """Per-channel label list matching the trainer's input stack."""
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
    """Load shared backbone tensors as frozen tinygrad Tensors.

    All returned tensors have `requires_grad=False`; the only autograd
    leaf in the attention computation is the input `X`.
    """
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
    """NHC/HIO/NHC stride-1 valid conv mirroring `replay.decoders._conv1d`."""
    x_bcl = x.permute(0, 2, 1)
    W_oik = W.permute(2, 1, 0)
    y_bcl = x_bcl.conv2d(W_oik)
    return y_bcl.permute(0, 2, 1) + b


def _batched_saliency(forward_fn, X_batch_np, K: int, F: int):
    """Single batched forward + backward; returns mean |grad| as (K, F)."""
    from tinygrad.tensor import Tensor
    X = Tensor(X_batch_np.astype(np.float32), requires_grad=True)
    output = forward_fn(X)
    output.sum().backward()
    grads = X.grad.numpy()
    return np.mean(np.abs(grads), axis=0)


def _topk_stats(sal, chan_labels: list[str], F: int, k: int = 8) -> dict:
    """Top-k (lag, channel) cells + per-channel sum-of-|grad|."""
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
    """3-panel saliency figure: |sal_a|, |sal_b|, signed diff."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

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


def _draw_uncond_panels(axes, uncond, panel_specs, apply_head, td,
                        out_stats, ticker, spearmanr) -> None:
    """Render per-head reconstruction panels for unconditioned heads.

    Mutates `out_stats['unconditioned']` with each head's stats dict
    (R²/RMSE/max-|Δ|/rank_IC/sign_acc) and prints the per-target line.
    """
    from ss_features import fit_stats
    for ax, target in zip(axes, uncond):
        yhat = apply_head(target)
        gt = td.targets[target]
        v = td.valid
        stats = dict(fit_stats(yhat[v], gt[v]))
        # Scale-invariant companions to R² (R² collapses on scale errors
        # but rank_IC/sign_acc reveal whether the shape was learned).
        rho, _ = spearmanr(yhat[v], gt[v])
        stats['rank_ic'] = float(rho) if np.isfinite(rho) else 0.0
        stats['sign_acc'] = float(
            (np.sign(yhat[v]) == np.sign(gt[v])).mean())
        out_stats['unconditioned'][target] = stats
        print(f'  {ticker} zero-shot {target:>5s}: '
              f'R²={stats["r2"]:>7.4f}  rank_IC={stats["rank_ic"]:>+6.3f}  '
              f'sign_acc={stats["sign_acc"]:.3f}  '
              f'RMSE={stats["rmse"]:.3e}  max|Δ|={stats["max_abs"]:.3e}')
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


# --------------------------------------------------------------- public API


def grid_sweep_eval(
    *, target: str, strided_fn, n_grid: tuple[int, ...],
    w_grid: tuple[int, ...], n_max: float, w_max: float,
    off_grid_n_extras: tuple[int, ...],
    hlines: tuple[int, ...], n_axis_label: str,
    apply_head, td, ticker: str, output_dir: Path,
    out_stats: dict, stats_key: str,
) -> None:
    """Run the (n, w) sweep + heatmap PNG for one FiLM-conditioned head.

    `strided_fn(prices, n, w)` is the per-cell ground-truth function
    (rsi_strided / cci_strided / vol-at-n / macd-at-fast). `apply_head
    (target, cond_vec=...)` is the closure built in `zeroshot_eval`
    that runs the head's forward pass with the supplied cond. Mutates
    `out_stats[stats_key]` in place; saves
    `{ticker}-replay-zeroshot-{target}-wn-sweep.png`.

    1-D conditioning (empty `w_grid`) renders as a single-row heatmap;
    cond_vec width matches the head's stored cond_dim.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from ss_features import fit_stats

    out_stats[stats_key] = {}
    n_sweep = sorted({*n_grid, *off_grid_n_extras})
    is_2d = bool(w_grid)
    w_sweep = sorted({*w_grid, 3, 7, 15, 25}) if is_2d else [1]
    upper_label = target.upper()
    sweep_label = '(n, w) sweep' if is_2d else '(n) sweep'
    print(f'\n  {upper_label} {sweep_label}:  n={n_sweep}  w={w_sweep}')
    header = '  w \\ n  |  ' + '  '.join(f'{n:>6d}' for n in n_sweep)
    print(header)
    print('-' * len(header))
    r2_grid = np.full((len(w_sweep), len(n_sweep)), np.nan)
    cell_records: dict[tuple[int, int], dict] = {}
    for wi, w in enumerate(w_sweep):
        row = []
        for ni, n in enumerate(n_sweep):
            gt_n = strided_fn(td.prices, n=int(n), w=int(w))
            if is_2d:
                cond_vec = np.array([n / n_max, w / w_max], dtype=np.float32)
            else:
                cond_vec = np.array([n / n_max], dtype=np.float32)
            yhat_n = apply_head(target, cond_vec=cond_vec)
            v = td.valid & np.isfinite(gt_n) & np.isfinite(yhat_n)
            stats = dict(fit_stats(yhat_n[v], gt_n[v]))
            rho, _ = spearmanr(yhat_n[v], gt_n[v])
            stats['rank_ic'] = float(rho) if np.isfinite(rho) else 0.0
            stats['sign_acc'] = float(
                (np.sign(yhat_n[v]) == np.sign(gt_n[v])).mean())
            r2_grid[wi, ni] = stats['r2']
            cell_records[(w, n)] = dict(
                stats=stats, gt=gt_n, yhat=yhat_n, valid=v,
                in_n_grid=(n in n_grid),
                in_w_grid=(not is_2d) or (w in w_grid),
            )
            cell_key = f'n={n}' if not is_2d else f'w={w},n={n}'
            out_stats[stats_key][cell_key] = stats
            in_grid = (n in n_grid) and ((not is_2d) or (w in w_grid))
            tag = '*' if in_grid else ' '
            row.append(f'{tag}{stats["r2"]:>5.2f}')
        print(f'  w={w:>3d}  |  ' + '  '.join(row))

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.0])
    grid_caption = (f'n ∈ {sorted(n_grid)}  ×  w ∈ {sorted(w_grid)}'
                    if is_2d else f'n ∈ {sorted(n_grid)}')
    fig.suptitle(
        f'{ticker} zero-shot {upper_label} — FiLM head trained on '
        f'{grid_caption}',
        fontsize=12, fontweight='bold')
    ax_hm = fig.add_subplot(gs[0, :])
    im = ax_hm.imshow(r2_grid, aspect='auto', origin='lower',
                      vmin=-0.5, vmax=1.0, cmap='RdYlGn')
    ax_hm.set_xticks(range(len(n_sweep)))
    ax_hm.set_xticklabels([str(n) for n in n_sweep])
    ax_hm.set_yticks(range(len(w_sweep)))
    ax_hm.set_yticklabels([str(w) for w in w_sweep])
    ax_hm.set_xlabel(n_axis_label)
    ax_hm.set_ylabel('Stride w (1=daily, 5≈weekly, 21≈monthly)')
    ax_hm.set_title('R² across the (w, n) grid — boxes mark training cells',
                    fontsize=10)
    for wi, w in enumerate(w_sweep):
        for ni, n in enumerate(n_sweep):
            r2 = r2_grid[wi, ni]
            ax_hm.text(ni, wi, f'{r2:.2f}', ha='center', va='center',
                       fontsize=8,
                       color='white' if r2 < 0.5 or r2 > 0.95 else 'black')
            if (n in n_grid) and (w in w_grid):
                ax_hm.add_patch(plt.Rectangle(
                    (ni - 0.5, wi - 0.5), 1, 1, fill=False,
                    edgecolor='black', linewidth=1.6))
    fig.colorbar(im, ax=ax_hm, label='R²', fraction=0.025)

    def _draw_ts(ax, w, n, prefix):
        rec = cell_records[(w, n)]
        gt_n, yhat_n, v, st = (rec['gt'], rec['yhat'], rec['valid'],
                               rec['stats'])
        yhat_full = np.full_like(gt_n, np.nan)
        yhat_full[v] = yhat_n[v]
        ax.plot(td.dates, gt_n, color='black', linewidth=0.7, alpha=0.6,
                label=f'true {upper_label}(n={n}, w={w})')
        ax.plot(td.dates, yhat_full, color='crimson', linewidth=0.9,
                linestyle='--',
                label=f'pred {upper_label}(n={n}, w={w})')
        for y in hlines:
            ax.axhline(y, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylabel(f'{upper_label}({n}, w={w})')
        ax.set_title(f'{prefix}  R²={st["r2"]:.4f}', fontsize=9, loc='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(td.dates[0], td.dates[-1])

    ts_picks: list[tuple[int, int, str]] = []
    sorted_n = sorted(n_grid)
    in_grid_n_mid = sorted_n[len(sorted_n) // 2]
    if is_2d:
        if 1 in w_grid:
            ts_picks.append((1, in_grid_n_mid, 'daily / in-grid n'))
        if 21 in w_grid:
            ts_picks.append((21, in_grid_n_mid, 'monthly / in-grid n'))
    else:
        ts_picks.append((1, in_grid_n_mid, f'in-grid n={in_grid_n_mid}'))
        if len(sorted_n) > 1:
            ts_picks.append(
                (1, sorted_n[-1], f'in-grid n={sorted_n[-1]} (largest)'))
    ax_ts = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (w, n, label) in zip(ax_ts, ts_picks):
        _draw_ts(ax, w, n, label)

    off_cells = [(w, n) for (w, n), rec in cell_records.items()
                 if not (rec['in_n_grid'] and rec['in_w_grid'])]
    off_cells.sort(key=lambda wn: cell_records[wn]['stats']['r2'])
    ax_off = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    picks_off = (off_cells[:1] + off_cells[-1:]) if off_cells else []
    for ax, (w, n) in zip(ax_off, picks_off):
        rec = cell_records[(w, n)]
        kind = (('OFF-GRID-n' if not rec['in_n_grid'] else 'in-grid-n') +
                ' / ' +
                ('OFF-GRID-w' if not rec['in_w_grid'] else 'in-grid-w'))
        _draw_ts(ax, w, n, kind)

    plt.tight_layout()
    fig.savefig(output_dir / f'{ticker}-replay-zeroshot-{target}-wn-sweep.png',
                dpi=150)
    plt.close(fig)


def zeroshot_eval(
    *, npz_path: Path, ticker: str, output_dir: Path,
    stooq_dir: str | None = None, kaggle_dir: str | None = None,
    use_yahoo: bool = False,
) -> dict:
    """Full zero-shot eval suite for a multi-head CNN backbone.

    Loads the trained npz, builds `ticker`'s feature stack matching
    the training config, runs the backbone forward (numpy conv1d —
    the model is small enough that JAX/tinygrad are overkill for
    inference), applies each head, and produces:
      - `{ticker}-replay-zeroshot-uncond.png` — price/macd/vol/cci
        unconditioned-head reconstruction panels (skipped if every
        head is FiLM-conditioned).
      - `{ticker}-replay-zeroshot-{target}-wn-sweep.png` per FiLM-
        conditioned head — heatmap + selected time-series.

    Returns a stats dict with `unconditioned` per-head stats and
    `{target}_grid` per-cell stats for each FiLM-conditioned target.

    Data source (one of) must be set: `stooq_dir`, `kaggle_dir`, or
    `use_yahoo=True`.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from ss_indicators import cci_strided, macd as macd_fn, rsi_strided
    from replay.features import realized_vol

    (data, meta, K, scales, rsi_n_grid, rsi_w_grid,
     n_max_grid, w_max_grid) = _load_npz_meta(npz_path)
    print(f'  trained on {len(meta["train_tickers"])} tickers, '
          f'K={K}, scales={scales}, targets={meta["targets"]}, '
          f'rsi_n_grid={rsi_n_grid}, rsi_w_grid={rsi_w_grid}')

    td = _load_eval_ticker(
        ticker, meta, scales,
        stooq_dir=stooq_dir, kaggle_dir=kaggle_dir, use_yahoo=use_yahoo)
    F = td.features.shape[1] // K
    print(f'  {ticker}: {len(td.prices)} bars, {td.valid.sum()} valid, '
          f'(K, F) = ({K}, {F})')

    # Numpy conv1d backbone forward (no autograd needed for inference).
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

    def conv1d_relu(x, W, b):
        # x: (N, L, C_in)  W: (kW, C_in, C_out)  b: (C_out,)
        kW, C_in, C_out = W.shape
        N, L, _ = x.shape
        L_out = L - kW + 1
        s_n, s_l, s_c = x.strides
        windows = np.lib.stride_tricks.as_strided(
            x, shape=(N, L_out, kW, C_in),
            strides=(s_n, s_l, s_l, s_c), writeable=False)
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

    def apply_head(target: str, cond_vec=None):
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

    out_stats: dict = {'unconditioned': {}}
    panel_specs = {
        'price': ('Close', None),
        'macd':  (f'MACD({meta["macd_fast"]},{meta["macd_slow"]},'
                  f'{meta["macd_signal"]}) line', (0,)),
        'vol':   (f'RealizedVol({meta.get("vol_window", 20)})', None),
        'cci':   (f'CCI({meta.get("cci_n", 20)})', (-100, 0, 100)),
    }

    def _is_unconditioned(target: str) -> bool:
        key = f'{target}__head_cond_dim'
        if key not in data.files:
            return True
        return int(data[key][0]) == 0

    uncond = [t for t in ('price', 'macd', 'vol', 'cci')
              if t in meta['targets'] and _is_unconditioned(t)]
    if not uncond:
        # All heads FiLM-conditioned (e.g. the full 4-head bundle).
        # Skip the uncond figure cleanly (plt.subplots would otherwise
        # trip on nrows=0).
        print(f'  (no unconditioned heads in this run — skipping '
              f'{ticker}-replay-zeroshot-uncond.png)')
    else:
        fig, axes = plt.subplots(
            len(uncond), 1, figsize=(13, 3.2 * len(uncond)),
            sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f'{ticker} zero-shot — unconditioned heads '
                     f'(K={K}, scales={len(meta["scales"])}, '
                     f'n_features={K*F})',
                     fontsize=12, fontweight='bold')
        _draw_uncond_panels(axes, uncond, panel_specs, apply_head, td,
                            out_stats, ticker, spearmanr)
        plt.tight_layout()
        fig.savefig(output_dir / f'{ticker}-replay-zeroshot-uncond.png',
                    dpi=150)
        plt.close(fig)

    # FiLM-conditioned heads — (n, w) or 1-D grid eval.
    def _vol_at_n(prices, n, w=1):
        return realized_vol(prices, window=int(n))

    def _macd_at_fast(prices, n, w=1):
        f = int(n)
        line, _, _ = macd_fn(prices, fast=f, slow=2 * f,
                             signal=max(2, (f * 3) // 4))
        return line.astype(np.float64)

    def _resolve(grid_key: str, anchor_key: str, anchor_default):
        g = tuple(meta.get(grid_key) or ())
        n_max_v = (float(max(g)) if g
                   else float(meta.get(anchor_key, anchor_default)))
        return g, n_max_v

    cci_g, cci_n_max = _resolve('cci_n_grid', 'cci_n', 20)
    cci_wg = tuple(meta.get('cci_w_grid') or ())
    cci_w_max = float(max(cci_wg)) if cci_wg else 1.0
    vol_g, vol_n_max = _resolve('vol_n_grid', 'vol_window', 20)
    macd_g, macd_n_max = _resolve('macd_fast_grid', 'macd_fast', 12)

    grid_specs = [
        ('rsi',  rsi_strided,   rsi_n_grid,    rsi_w_grid,
         n_max_grid, w_max_grid,
         (30, 70), 'RSI period n', (6, 8, 11, 15, 19, 28, 35)),
        ('cci',  cci_strided,   cci_g,         cci_wg,
         cci_n_max, cci_w_max,
         (-100, 0, 100), 'CCI period n', (6, 8, 12, 16, 24, 35, 50)),
        ('vol',  _vol_at_n,     vol_g,         (),
         vol_n_max, 1.0,
         (), 'vol window n', (3, 8, 15, 45, 90)),
        ('macd', _macd_at_fast, macd_g,        (),
         macd_n_max, 1.0,
         (0,), 'macd fast n', (5, 10, 20, 32)),
    ]
    for (name, strided_fn, n_grid_m, w_grid_m, n_max_m, w_max_m,
         hlines, n_label, off_grid_n) in grid_specs:
        if name not in meta['targets'] or not n_grid_m:
            print(f'  (no {name} grid in this run — skipping sweep)')
            continue
        grid_sweep_eval(
            target=name, strided_fn=strided_fn,
            n_grid=n_grid_m, w_grid=w_grid_m,
            n_max=n_max_m, w_max=w_max_m,
            off_grid_n_extras=off_grid_n,
            hlines=hlines, n_axis_label=n_label,
            apply_head=apply_head, td=td, ticker=ticker,
            output_dir=output_dir, out_stats=out_stats,
            stats_key=f'{name}_grid')
    return out_stats


def film_attention(
    *, npz_path: Path, ticker: str, output_dir: Path,
    cond_a: tuple[int, int] = (7, 1),
    cond_b: tuple[int, int] = (17, 10),
    n_bars: int = 200,
    stooq_dir: str | None = None, kaggle_dir: str | None = None,
    use_yahoo: bool = False,
) -> dict:
    """FiLM rsi-head input-gradient saliency at two cond vectors.

    Computes |d rsi / d X| averaged over `n_bars` random bars at
    `cond_a` and `cond_b`, renders a 3-panel figure (cond_a, cond_b,
    signed diff). tinygrad autograd matches the trainer framework;
    npz weights become frozen Tensors and only the input is a leaf.

    Returns top-k cells/channels per cond. Output PNG:
    `{ticker}-film-attention.png`.

    Data source (one of) must be set: stooq_dir, kaggle_dir, use_yahoo.
    """
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
            Xn = (X - feat_mu) / feat_sd
            h = Xn
            for W, b in conv_params:
                h = _conv1d_tg(h, W, b).relu()
            latent = h.reshape(h.shape[0], -1)
            gamma = film_mlp(film['gamma_W0'], film['gamma_b0'],
                             film['gamma_W1'], film['gamma_b1'],
                             cv) + 1.0
            beta = film_mlp(film['beta_W0'], film['beta_b0'],
                            film['beta_W1'], film['beta_b1'], cv)
            latent_mod = gamma * latent + beta
            yhat_std = (latent_mod @ head_W + head_b).squeeze(-1)
            return yhat_std * target_sd + target_mu

        return forward

    td = _load_eval_ticker(
        ticker, meta, scales,
        stooq_dir=stooq_dir, kaggle_dir=kaggle_dir, use_yahoo=use_yahoo)
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


def uncond_attention(
    *, npz_path: Path, ticker: str, output_dir: Path,
    head_a: str = 'macd', head_b: str = 'vol',
    n_bars: int = 200,
    stooq_dir: str | None = None, kaggle_dir: str | None = None,
    use_yahoo: bool = False,
) -> dict:
    """Two-head input-gradient saliency for unconditioned heads.

    Default `head_a='macd'` vs `head_b='vol'`. Renders a 3-panel
    figure (head_a saliency, head_b saliency, normalized diff). The
    diff is per-head normalized so different output-magnitude scales
    don't dominate the comparison.

    Skips cleanly if either head is missing from the npz or is
    FiLM-conditioned (cond_dim != 0). Output PNG:
    `{ticker}-uncond-attention.png`.

    Data source (one of) must be set: stooq_dir, kaggle_dir, use_yahoo.
    """
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
        head_b_ = Tensor(np.asarray(data[f'{target}__head_b'],
                                     dtype=np.float32), requires_grad=False)
        target_mu = float(data[f'{target}__target_mu'][0])
        target_sd = float(data[f'{target}__target_sd'][0])

        def forward(X):
            Xn = (X - feat_mu) / feat_sd
            h = Xn
            for W, b in conv_params:
                h = _conv1d_tg(h, W, b).relu()
            latent = h.reshape(h.shape[0], -1)
            yhat_std = (latent @ head_W + head_b_).squeeze(-1)
            return yhat_std * target_sd + target_mu

        return forward

    td = _load_eval_ticker(
        ticker, meta, scales,
        stooq_dir=stooq_dir, kaggle_dir=kaggle_dir, use_yahoo=use_yahoo)
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
