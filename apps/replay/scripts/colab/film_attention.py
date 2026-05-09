"""FiLM attention plot — input-gradient saliency over CWT bundle.

For the supervised CNN backbone with FiLM-conditioned rsi head, computes
`|dy_rsi/dX|` averaged across many bars at a chosen `(n, w)` cond vector,
then renders a heatmap over the (lag, channel) input grid. Comparing
two cond vectors side-by-side shows the FiLM machinery's wavelength
selectivity: short-period rsi should attend to recent lags + high-freq
scales; long-period rsi should attend to longer lags + low-freq scales.

Three panels:
  - Blue heatmap: |sal| at cond_a (default n=7, w=1; canonical daily RSI)
  - Red heatmap:  |sal| at cond_b (default n=17, w=10; biweekly RSI)
  - Diff:  blue where cond_a saliency > cond_b; red where reversed.

Reads the latest *-cnn-*.npz under /content/Output/ (Colab) or
/Users/sidghodke/Downloads/ (local). Prefers a fully FiLM-flavored npz
(checks for head_film_gamma_W0). Falls back with a clear error otherwise.

Channel layout reference (F=33 for default config; verify against meta):
  ch 0..14:  coeffs at scales [1,2,3,5,7,10,12,15,21,26,42,50,63,90,126]
  ch 15..29: power at the same scales
  ch 30:     rolling z-norm mu
  ch 31:     rolling z-norm std
  ch 32:     log return
Lag layout: lag 0 = most recent bar; lag K-1 = oldest bar in window.
"""
import glob
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from replay.features import load_ticker


# --- 1. Load weights + meta. ------------------------------------------------
GLOB_DIRS = ['/content/Output', '/Users/sidghodke/Downloads']
candidates: list[str] = []
for d in GLOB_DIRS:
    if os.path.isdir(d):
        candidates.extend(sorted(glob.glob(f'{d}/*-cnn-*.npz')))
assert candidates, 'no *-cnn-*.npz under /content/Output or ~/Downloads'

# Prefer a FiLM-flavored npz (the comparison only makes sense if rsi is FiLM-conditioned).
def _has_film(p: str) -> bool:
    z = np.load(p, allow_pickle=False)
    out = 'rsi__head_film_gamma_W0' in z.files
    z.close()
    return out

film_npzs = [c for c in candidates if _has_film(c)]
if film_npzs:
    WEIGHTS = sorted(film_npzs, key=os.path.getmtime)[-1]
    print(f'Using FiLM-flavored npz: {WEIGHTS}')
else:
    WEIGHTS = candidates[-1]
    print(f'WARN: no FiLM-flavored npz found; falling back to {WEIGHTS}')
    print('      (this script needs FiLM keys — saliency will be the same '
          'at any cond vec since the head will be additive-concat)')

data = np.load(WEIGHTS, allow_pickle=False)
meta = json.loads(data['_meta'].item())

K = int(meta['window_cols'])
scales = [int(s) for s in meta['scales']]
# Polar Morlet (4) + Gaussian (2) + log-L2-amp (1) per scale.
F_meta = 7 * len(scales)
print(f'Backbone: K={K}, F={F_meta}, scales={scales}')

rsi_n_grid = tuple(meta.get('rsi_n_grid') or ())
rsi_w_grid = tuple(meta.get('rsi_w_grid') or ())
n_max_grid = float(max(rsi_n_grid)) if rsi_n_grid else float(meta['rsi_n'])
w_max_grid = float(max(rsi_w_grid)) if rsi_w_grid else 1.0
print(f'rsi_n_grid={rsi_n_grid}, rsi_w_grid={rsi_w_grid}')


# --- 2. Pull backbone + FiLM rsi head into pytrees. -------------------------
ref_target = 'rsi'
feat_mu = jnp.asarray(data[f'{ref_target}__feat_mu'])  # (1, K, F)
feat_sd = jnp.asarray(data[f'{ref_target}__feat_sd'])
n_layers = sum(1 for k in data.files
               if k.startswith(f'{ref_target}__conv') and k.endswith('_W'))
conv_params = [
    (jnp.asarray(data[f'{ref_target}__conv{i}_W']),
     jnp.asarray(data[f'{ref_target}__conv{i}_b']))
    for i in range(n_layers)
]
head_W = jnp.asarray(data['rsi__head_W'])      # (5632, 1) under FiLM
head_b = jnp.asarray(data['rsi__head_b'])
target_mu = float(data['rsi__target_mu'][0])
target_sd = float(data['rsi__target_sd'][0])
film_gamma_W0 = jnp.asarray(data['rsi__head_film_gamma_W0'])
film_gamma_b0 = jnp.asarray(data['rsi__head_film_gamma_b0'])
film_gamma_W1 = jnp.asarray(data['rsi__head_film_gamma_W1'])
film_gamma_b1 = jnp.asarray(data['rsi__head_film_gamma_b1'])
film_beta_W0  = jnp.asarray(data['rsi__head_film_beta_W0'])
film_beta_b0  = jnp.asarray(data['rsi__head_film_beta_b0'])
film_beta_W1  = jnp.asarray(data['rsi__head_film_beta_W1'])
film_beta_b1  = jnp.asarray(data['rsi__head_film_beta_b1'])


def conv1d(x, W, b):
    return jax.lax.conv_general_dilated(
        x, W, window_strides=(1,), padding='VALID',
        dimension_numbers=('NHC', 'HIO', 'NHC')) + b


def film_mlp(W0, b0, W1, b1, c):
    """2-layer ReLU MLP. `c` shape (cond_dim,)."""
    return jnp.maximum(0.0, c @ W0 + b0) @ W1 + b1


def rsi_head_output(X_raw_kf, cond_vec):
    """Forward pass: raw input (K, F) + cond -> rsi prediction scalar.

    Z-norms input internally (so the gradient lands on the raw input
    domain, not z-normed — matches what's actually plotted as the bundle).
    """
    X = (X_raw_kf - feat_mu[0]) / feat_sd[0]      # (K, F), z-normed
    h = X[None]                                    # (1, K, F)
    for W, b in conv_params:
        h = jax.nn.relu(conv1d(h, W, b))
    latent = h.reshape(1, -1)                      # (1, latent_dim)
    gamma = film_mlp(film_gamma_W0, film_gamma_b0,
                     film_gamma_W1, film_gamma_b1, cond_vec) + 1.0
    beta  = film_mlp(film_beta_W0, film_beta_b0,
                     film_beta_W1, film_beta_b1, cond_vec)
    latent_mod = gamma * latent + beta
    yhat_std = (latent_mod @ head_W + head_b).squeeze()
    return yhat_std * target_sd + target_mu


saliency_fn = jax.jit(jax.grad(rsi_head_output))


# --- 3. Build a ticker's input bundle and pick representative bars. ---------
TICKER = 'AAPL'
load_kwargs = dict(
    stooq_dir=None, kaggle_dir=None, use_yahoo=True,
    start=meta['start'], end=meta['end'],
    scales=scales,
    lookback=int(meta['lookback']),
    window_cols=K,
    rsi_n=int(meta['rsi_n']),
    macd_fast=int(meta['macd_fast']),
    macd_slow=int(meta['macd_slow']),
    macd_signal=int(meta['macd_signal']),
    vol_window=int(meta.get('vol_window', 20)),
    rsi_n_grid=(), rsi_w_grid=(),
)
td = load_ticker(TICKER, **load_kwargs)
F = td.features.shape[1] // K
assert F == F_meta, f'F mismatch: meta says {F_meta}, got {F} from features'
X_all = td.features.reshape(-1, K, F).astype(np.float32)
valid_idx = np.where(td.valid)[0]
print(f'{TICKER}: {len(valid_idx)} valid bars (out of {len(td.prices)})')


# --- 4. Compute saliency averaged over N random bars per cond vec. ----------
# Cond vec ordering matches replay/decoders.py: [n / n_max, w / w_max].
COND_A = (7,  1)   # canonical daily RSI(7)
COND_B = (17, 10)  # biweekly RSI(17)
N_BARS = min(200, len(valid_idx))

rng = np.random.default_rng(0)
sample_idx = rng.choice(valid_idx, size=N_BARS, replace=False)


def average_saliency(cond_n: int, cond_w: int) -> np.ndarray:
    """|grad of rsi(n, w) head output wrt raw input X|, averaged across
    sample bars. Returns shape (K, F)."""
    cond_vec = jnp.array([cond_n / n_max_grid, cond_w / w_max_grid],
                         dtype=jnp.float32)
    sal_sum = np.zeros((K, F), dtype=np.float64)
    for i, bar_idx in enumerate(sample_idx):
        X_raw = jnp.asarray(X_all[bar_idx])  # (K, F)
        g = saliency_fn(X_raw, cond_vec)     # (K, F), signed grad
        sal_sum += np.abs(np.asarray(g))
        if (i + 1) % 50 == 0:
            print(f'  cond=({cond_n},{cond_w}) bar {i+1}/{N_BARS}')
    return sal_sum / N_BARS


print(f'\nComputing saliency for cond_a = (n={COND_A[0]}, w={COND_A[1]})...')
sal_a = average_saliency(*COND_A)
print(f'Computing saliency for cond_b = (n={COND_B[0]}, w={COND_B[1]})...')
sal_b = average_saliency(*COND_B)


# --- 4b. Textual top-N breakdown — what the colors are missing. -------------
def report_topk(sal: np.ndarray, label: str, k: int = 8) -> None:
    flat = sal.flatten()
    top_idx = np.argsort(flat)[::-1][:k]
    print(f'\nTop {k} (lag, channel) cells for {label}:')
    for rank, fi in enumerate(top_idx, 1):
        lag = fi // F
        ch = fi % F
        print(f'  {rank:2d}. lag={lag:>3d}  ch={ch:>2d} ({chan_labels[ch]:<14s})  '
              f'|grad|={flat[fi]:.4e}')
    # Per-channel-aggregated saliency (sum across lags).
    per_chan = sal.sum(axis=0)
    chan_top = np.argsort(per_chan)[::-1][:k]
    print(f'  Per-channel total |grad| (top {k}):')
    for rank, ci in enumerate(chan_top, 1):
        print(f'    {rank:2d}. ch={ci:>2d} ({chan_labels[ci]:<14s})  '
              f'sum |grad|={per_chan[ci]:.4e}')

report_topk(sal_a, f'cond_a = RSI(n={COND_A[0]}, w={COND_A[1]})')
report_topk(sal_b, f'cond_b = RSI(n={COND_B[0]}, w={COND_B[1]})')


# --- 5. Plot. ---------------------------------------------------------------
chan_labels = (
    [f'|c| s={s}' for s in scales] +
    [f'|c|^2 s={s}' for s in scales] +
    [f'cos(arg) s={s}' for s in scales] +
    [f'sin(arg) s={s}' for s in scales] +
    [f'g s={s}' for s in scales] +
    [f'g^2 s={s}' for s in scales] +
    [f'logL2 s={s}' for s in scales]
)
assert len(chan_labels) == F, f'channel label count {len(chan_labels)} != F={F}'

fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)

# Shared color scale for blue / red panels so they're directly comparable.
vmax_ab = max(sal_a.max(), sal_b.max())

for ax, sal, color, title in [
    (axes[0], sal_a, 'Blues', f'RSI(n={COND_A[0]}, w={COND_A[1]}) — daily, short period'),
    (axes[1], sal_b, 'Reds',  f'RSI(n={COND_B[0]}, w={COND_B[1]}) — biweekly, long period'),
]:
    im = ax.imshow(sal.T, aspect='auto', origin='lower',
                   cmap=color, vmin=0, vmax=vmax_ab)
    ax.set_xlabel('Lag (0 = most recent bar)')
    ax.set_ylabel('Channel')
    ax.set_yticks(range(F))
    ax.set_yticklabels(chan_labels, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='|d rsi / d X| avg', fraction=0.025)

# Diff panel: positive (blue) where cond_a dominates, negative (red) where cond_b dominates.
diff = sal_a - sal_b
vlim = float(np.abs(diff).max())
im = axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='seismic_r',
                    norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim))
axes[2].set_xlabel('Lag (0 = most recent bar)')
axes[2].set_ylabel('Channel')
axes[2].set_yticks(range(F))
axes[2].set_yticklabels(chan_labels, fontsize=6)
axes[2].set_title(f'sal[{COND_A}] − sal[{COND_B}]\nblue = short dominates; red = long dominates')
fig.colorbar(im, ax=axes[2], label='Δ saliency', fraction=0.025)

fig.suptitle(
    f'FiLM rsi-head input attention — {TICKER}, K={K}, '
    f'{N_BARS} bars averaged\nbackbone: {os.path.basename(WEIGHTS)}',
    fontsize=11, fontweight='bold')

out_dir = '/content/Output' if os.path.isdir('/content/Output') else '/tmp'
out = os.path.join(out_dir, 'film-attention.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved {out}')
