"""Zero-shot CSCO from the multi-head CNN, with FiLM-conditioned RSI head.

Extracted from cwt_vision_multihead.ipynb (Colab cell). Paths assume
/content/ layout — run inside Colab or adapt for local use.

TLDR results:
  Single-head bundle CNN (RSI(7), no FiLM), CSCO zero-shot: R² 0.972.
  Bundle generalizes — once the channel set has returns + lag-windowed
  zscore stats + extra HF scales, the trained backbone transfers cleanly
  to an unseen ticker (within sector/cap class).

  Phase 2 / Exp D FiLM (n, w) heatmap numbers not archived here. The
  cell prints a R² grid spanning trained n ∈ {5,7,9,13,17,21,25} ×
  trained w ∈ {1,5,10,21}, plus interior + extrapolation off-grid
  cells (n ∈ {6,8,11,15,19,28,35}, w ∈ {3,7,15,25}). Use the Colab
  output of this cell to populate.

apply_head detects whether the npz has FiLM keys (head_film_gamma_W0,
etc.). If yes, runs cond -> {gamma, beta} MLPs over the conditioning
vector to produce per-latent scale/shift, modulates H -> gamma*H + beta,
then applies the linear head. If no FiLM keys (older npz files), falls
back to legacy additive-concat path: cond is appended to H, head_W
absorbs the cond contribution.

The FiLM cond MLPs only ever see the cond vector (no per-sample
latent), so they can't memorize per-ticker patterns. The latent->output
projection stays linear (head_W shape = (latent_dim, 1) under FiLM),
preserving the same risk profile as the original linear head.

The (n, w) sweep spans the full cross-product plus a few off-grid n
values and four off-grid w values:
  - w=3      interior interpolation between trained w=1 and w=5
  - w=7      interior interpolation between trained w=5 and w=10
  - w=15     interior interpolation between trained w=10 and w=21
  - w=25     extrapolation past trained w=21
Ground truth comes from rsi_strided(prices, n, w) so it matches the
trainer's target definition exactly.

Unconditioned heads (price, macd, vol) are unchanged.
"""
import glob
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ss_indicators import rsi_strided
from ss_notebook.replay.features import load_ticker
from ss_notebook.replay.metrics import fit_stats


# --- 1. Load weights + meta. ------------------------------------------------
candidates = sorted(glob.glob('/content/Output/AAPL+*-cnn-*.npz'))
assert candidates, 'no AAPL+...-cnn-*.npz found in /content/Output'
WEIGHTS = candidates[-1]
data = np.load(WEIGHTS, allow_pickle=False)
meta = json.loads(data['_meta'].item())
print(f'loaded: {WEIGHTS}')
print(f'  trained on {len(meta["train_tickers"])} tickers, decoder={meta["decoder"]}, '
      f'K={meta["window_cols"]}, scales={meta["scales"]}')
print(f'  zscore_stats={meta["include_zscore_stats"]}, returns={meta["include_returns"]}, '
      f'targets={meta["targets"]}, rsi_n={meta["rsi_n"]}, '
      f'rsi_n_grid={meta.get("rsi_n_grid")}, rsi_w_grid={meta.get("rsi_w_grid")}, '
      f'vol_window={meta.get("vol_window")}, '
      f'film_hidden={meta.get("cnn_film_hidden")}')

rsi_n_grid = tuple(meta.get('rsi_n_grid') or ())
rsi_w_grid = tuple(meta.get('rsi_w_grid') or ())
n_max_grid = float(max(rsi_n_grid)) if rsi_n_grid else float(meta['rsi_n'])
w_max_grid = float(max(rsi_w_grid)) if rsi_w_grid else 1.0

# --- 2. Build CSCO features matching the training config exactly. -----------
ticker = 'CSCO'
td = load_ticker(
    ticker,
    stooq_dir=None, kaggle_dir=None, use_yahoo=True,
    start=meta['start'], end=meta['end'],
    scales=[int(s) for s in meta['scales']],
    lookback=int(meta['lookback']),
    window_cols=int(meta['window_cols']),
    include_zscore_stats=bool(meta['include_zscore_stats']),
    include_returns=bool(meta['include_returns']),
    decoder=meta['decoder'],
    rsi_n=int(meta['rsi_n']),
    macd_fast=int(meta['macd_fast']),
    macd_slow=int(meta['macd_slow']),
    macd_signal=int(meta['macd_signal']),
    vol_window=int(meta.get('vol_window', 20)),
    rsi_n_grid=rsi_n_grid,
    rsi_w_grid=rsi_w_grid,
)
K = int(meta['window_cols'])
F = td.features.shape[1] // K
print(f'\n{ticker}: {len(td.prices)} bars total, {td.valid.sum()} valid after warm-up, '
      f'shape (n, K, F) = (n, {K}, {F})')

# --- 3. Shared backbone forward pass. ---------------------------------------
ref_target = meta['targets'][0]
feat_mu = data[f'{ref_target}__feat_mu']
feat_sd = data[f'{ref_target}__feat_sd']
X = td.features.reshape(-1, K, F).astype(np.float32)
Xn = (X - feat_mu) / feat_sd

n_layers = sum(1 for k in data.files
               if k.startswith(f'{ref_target}__conv') and k.endswith('_W'))
conv_params = [
    (jnp.asarray(data[f'{ref_target}__conv{i}_W']),
     jnp.asarray(data[f'{ref_target}__conv{i}_b']))
    for i in range(n_layers)
]


def conv1d(x, W, b):
    return jax.lax.conv_general_dilated(
        x, W, window_strides=(1,), padding='VALID',
        dimension_numbers=('NHC', 'HIO', 'NHC')) + b


@jax.jit
def backbone(x):
    h = x
    for W, b in conv_params:
        h = jax.nn.relu(conv1d(h, W, b))
    return h.reshape(h.shape[0], -1)


CHUNK = 32_768
H_chunks = []
for s in range(0, Xn.shape[0], CHUNK):
    H_chunks.append(np.asarray(backbone(jnp.asarray(Xn[s:s + CHUNK]))))
H = np.concatenate(H_chunks)  # shape (n, latent_dim)
print(f'backbone latent shape: {H.shape}  (latent_dim = K_post * hidden)')


def _film_mlp(W0, b0, W1, b1, c):
    """2-layer ReLU MLP for FiLM gamma/beta. `c` is (n, cond_dim)."""
    return np.maximum(0.0, c @ W0 + b0) @ W1 + b1


def apply_head(target: str, cond_vec: np.ndarray | None = None) -> np.ndarray:
    """Apply a target's head to the cached backbone latent H.

    Three modes, picked from the npz's keys:
      1. FiLM (head_film_gamma_W0 present): MLPs map cond_vec to
         gamma/beta over latent_dim, modulate H -> gamma * H + beta,
         then apply the linear head. head_W shape = (latent_dim, 1).
      2. Legacy additive concat (head_cond_dim > 0, no film keys):
         broadcast cond_vec across rows, append to H, apply linear
         head with shape (latent_dim + cond_dim, 1).
      3. Unconditioned (head_cond_dim == 0): plain linear head.
    """
    head_W = data[f'{target}__head_W']
    head_b = data[f'{target}__head_b']
    target_mu = float(data[f'{target}__target_mu'][0])
    target_sd = float(data[f'{target}__target_sd'][0])
    cond_dim_key = f'{target}__head_cond_dim'
    cond_dim = int(data[cond_dim_key][0]) if cond_dim_key in data.files else 0
    has_film = f'{target}__head_film_gamma_W0' in data.files

    if cond_dim == 0:
        latent = H
    elif has_film:
        if cond_vec is None or len(cond_vec) != cond_dim:
            raise ValueError(f'{target} head_cond_dim={cond_dim} but '
                             f'cond_vec={cond_vec!r}')
        cond_arr = np.asarray(cond_vec, dtype=np.float32)
        cond_block = np.broadcast_to(cond_arr[None, :],
                                     (H.shape[0], cond_dim))
        gamma = _film_mlp(
            data[f'{target}__head_film_gamma_W0'],
            data[f'{target}__head_film_gamma_b0'],
            data[f'{target}__head_film_gamma_W1'],
            data[f'{target}__head_film_gamma_b1'],
            cond_block) + 1.0
        beta = _film_mlp(
            data[f'{target}__head_film_beta_W0'],
            data[f'{target}__head_film_beta_b0'],
            data[f'{target}__head_film_beta_W1'],
            data[f'{target}__head_film_beta_b1'],
            cond_block)
        latent = gamma * H + beta
    else:
        # Legacy additive concat path.
        if cond_vec is None or len(cond_vec) != cond_dim:
            raise ValueError(f'{target} head_cond_dim={cond_dim} but '
                             f'cond_vec={cond_vec!r}')
        cond_arr = np.asarray(cond_vec, dtype=np.float32)
        cond_block = np.broadcast_to(cond_arr[None, :],
                                     (H.shape[0], cond_dim))
        latent = np.concatenate([H, cond_block], axis=-1)

    yhat_std = (latent @ head_W + head_b).squeeze(-1)
    return yhat_std.astype(np.float64) * target_sd + target_mu


# --- 4. Score the unconditioned heads (price, macd, vol). -------------------
panel_specs = {
    'price': ('Close', None),
    'macd':  (f'MACD({meta["macd_fast"]},{meta["macd_slow"]},'
              f'{meta["macd_signal"]}) line', (0,)),
    'vol':   (f'RealizedVol({meta.get("vol_window", 20)})', None),
}
unconditioned = [t for t in ('price', 'macd', 'vol') if t in meta['targets']]

fig_static, axes_static = plt.subplots(
    len(unconditioned), 1, figsize=(13, 3.2 * len(unconditioned)),
    sharex=True, squeeze=False)
axes_static = axes_static.flatten()
fig_static.suptitle(f'{ticker} zero-shot — unconditioned heads '
                    f'(K={K}, scales={len(meta["scales"])}, n_features={K*F})',
                    fontsize=12, fontweight='bold')

for ax, target in zip(axes_static, unconditioned):
    yhat = apply_head(target)
    gt = td.targets[target]
    v = td.valid
    stats = fit_stats(yhat[v], gt[v])
    print(f'{ticker} zero-shot {target:>5s}: '
          f'R²={stats["r2"]:.4f}  RMSE={stats["rmse"]:.3e}  max|Δ|={stats["max_abs"]:.3e}')
    yhat_full = np.full_like(gt, np.nan)
    yhat_full[v] = yhat[v]
    label, hlines = panel_specs[target]
    ax.plot(td.dates, gt, color='black', linewidth=0.7, alpha=0.6,
            label=f'{label} ground truth')
    ax.plot(td.dates, yhat_full, color='crimson', linewidth=0.9, linestyle='--',
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
plt.savefig(f'/content/Output/{ticker}-replay-zeroshot-uncond.png', dpi=150)
plt.show()


# --- 5. RSI (n, w) generalization sweep. ------------------------------------
# n axis: training grid + handful of in-interior off-grid + far extrapolation
# (n=35). w axis: training grid + four off-grid strides:
#   w=3, w=7   interior interpolation (between trained 1<->5 and 5<->10)
#   w=15       interior interpolation (between trained 10<->21)
#   w=25       extrapolation past trained max (21)
if 'rsi' not in meta['targets']:
    print('\n(no rsi target in this run — skipping RSI generalization sweep)')
elif not rsi_n_grid:
    print('\n(rsi was trained without --rsi-n-grid — skipping conditioned eval)')
else:
    n_sweep = sorted({*rsi_n_grid, 6, 8, 11, 15, 19, 28, 35})
    if rsi_w_grid:
        w_sweep = sorted({*rsi_w_grid, 3, 7, 15, 25})
    else:
        w_sweep = [1]
    print(f'\nRSI (n, w) generalization sweep:')
    print(f'  trained n_grid = {sorted(rsi_n_grid)}')
    print(f'  trained w_grid = {sorted(rsi_w_grid)}')
    print(f'  sweep n = {n_sweep}')
    print(f'  sweep w = {w_sweep}')
    print()
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
            in_grid = n in rsi_n_grid and w in rsi_w_grid
            tag = '*' if in_grid else ' '
            row.append(f'{tag}{stats["r2"]:>5.2f}')
        print(f'  w={w:>3d}  |  ' + '  '.join(row))
    print('\n(* = both n and w are in the training grid)')

    # --- 6. Render figure: heatmap + selected (w, n) time-series. -----------
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.0])
    fig.suptitle(
        f'{ticker} zero-shot RSI(n, w) — FiLM-conditioned head trained on '
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
    ax_hm.set_title('R² across the (w, n) grid — boxes mark training-grid cells',
                    fontsize=10)
    for wi, w in enumerate(w_sweep):
        for ni, n in enumerate(n_sweep):
            r2 = r2_grid[wi, ni]
            ax_hm.text(ni, wi, f'{r2:.2f}', ha='center', va='center',
                       fontsize=8,
                       color='white' if r2 < 0.5 or r2 > 0.95 else 'black')
            if (n in rsi_n_grid) and (w in rsi_w_grid):
                ax_hm.add_patch(plt.Rectangle((ni - 0.5, wi - 0.5), 1, 1,
                                              fill=False, edgecolor='black',
                                              linewidth=1.6))
    fig.colorbar(im, ax=ax_hm, label='R²', fraction=0.025)

    ts_picks = []
    in_grid_n_mid = sorted(rsi_n_grid)[len(rsi_n_grid) // 2]
    if 1 in rsi_w_grid:
        ts_picks.append((1, in_grid_n_mid, 'daily / in-grid n'))
    if 21 in rsi_w_grid:
        ts_picks.append((21, in_grid_n_mid, 'monthly / in-grid n'))

    ax_ts = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, (w, n, label) in zip(ax_ts, ts_picks):
        rec = cell_records[(w, n)]
        gt_n, yhat_n, v, stats = (rec['gt'], rec['yhat'], rec['valid'],
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
        ax.set_title(f'{label}  R²={stats["r2"]:.4f}', fontsize=9, loc='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(td.dates[0], td.dates[-1])

    off_cells = [(w, n) for (w, n), rec in cell_records.items()
                 if not (rec['in_n_grid'] and rec['in_w_grid'])]
    off_cells.sort(key=lambda wn: cell_records[wn]['stats']['r2'])
    ax_off = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    picks_off = (off_cells[:1] + off_cells[-1:]) if off_cells else []
    for ax, (w, n) in zip(ax_off, picks_off):
        rec = cell_records[(w, n)]
        gt_n, yhat_n, v, stats = (rec['gt'], rec['yhat'], rec['valid'],
                                  rec['stats'])
        yhat_full = np.full_like(gt_n, np.nan)
        yhat_full[v] = yhat_n[v]
        kind = ('OFF-GRID-n' if not rec['in_n_grid'] else 'in-grid-n') + ' / ' + \
               ('OFF-GRID-w' if not rec['in_w_grid'] else 'in-grid-w')
        ax.plot(td.dates, gt_n, color='black', linewidth=0.7, alpha=0.6,
                label=f'true RSI(n={n}, w={w})')
        ax.plot(td.dates, yhat_full, color='crimson', linewidth=0.9,
                linestyle='--', label=f'pred RSI(n={n}, w={w})')
        ax.axhline(30, color='gray', linestyle=':', alpha=0.4)
        ax.axhline(70, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylabel(f'RSI({n}, w={w})')
        ax.set_title(f'{kind}  R²={stats["r2"]:.4f}', fontsize=9, loc='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(td.dates[0], td.dates[-1])

    plt.tight_layout()
    out = f'/content/Output/{ticker}-replay-zeroshot-rsi-wn-sweep.png'
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved {out}')
