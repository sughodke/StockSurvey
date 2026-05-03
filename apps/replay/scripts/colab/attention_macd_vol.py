"""Input-gradient attention plot — MACD vs realized vol.

Both heads are unconditioned linear heads on top of the shared CNN
backbone (no FiLM, no conditioning vec). The question this plot answers:
do MACD and vol live on the same input channels (just `return` again, like
RSI did), or do they pull attention onto different parts of the bundle?

Reasoning behind the choice:
  - We already established RSI lives ~entirely on the `return` channel
    (film_attention.py result).
  - MACD is fast=12 EMA - slow=26 EMA of price → long effective memory.
  - vol is rolling std of log returns over 20 bars → similar long memory
    but operates on the 2nd moment of returns, not the level.
  - If both also collapse onto `return`, the indicator-bias problem is
    universal across our 4 heads. If they differ (e.g., vol uses CWT
    power channels because std is naturally a power-of-signal quantity),
    that's a meaningful difference.

BLUE = MACD, RED = vol. Diff panel: blue where MACD dominates, red where
vol dominates. Same structure as film_attention.py.
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


# --- 1. Load weights + meta. (Same locator pattern as film_attention.py) ----
GLOB_DIRS = ['/content/Output', '/Users/sidghodke/Downloads']
candidates: list[str] = []
for d in GLOB_DIRS:
    if os.path.isdir(d):
        candidates.extend(sorted(glob.glob(f'{d}/*-cnn-*.npz')))
assert candidates, 'no *-cnn-*.npz under /content/Output or ~/Downloads'

# Pick latest. We don't require FiLM here since macd + vol are both
# unconditioned. Either FiLM-flavored or legacy npz works.
WEIGHTS = sorted(candidates, key=os.path.getmtime)[-1]
print(f'Using npz: {WEIGHTS}')
data = np.load(WEIGHTS, allow_pickle=False)
meta = json.loads(data['_meta'].item())

K = int(meta['window_cols'])
scales = [int(s) for s in meta['scales']]
F_meta = (2 * len(scales)
          + (2 if meta.get('include_zscore_stats') else 0)
          + (1 if meta.get('include_returns') else 0))
print(f'Backbone: K={K}, F={F_meta}, scales={scales}')

# Sanity: both target heads must exist and be unconditioned in this npz.
for tgt in ('macd', 'vol'):
    assert f'{tgt}__head_W' in data.files, f'{tgt}__head_W missing in npz'
    cdk = f'{tgt}__head_cond_dim'
    cd = int(data[cdk][0]) if cdk in data.files else 0
    assert cd == 0, f'{tgt} head has cond_dim={cd}; this script assumes 0'


# --- 2. Pull backbone (same for both heads) and per-head linear weights. ----
ref_target = 'macd'   # backbone tensors are identical across prefixes
feat_mu = jnp.asarray(data[f'{ref_target}__feat_mu'])
feat_sd = jnp.asarray(data[f'{ref_target}__feat_sd'])
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


def make_head_output_fn(target: str):
    """Return a `(X_raw_kf,) -> scalar` function for the named target.

    The function z-norms the raw input internally so the gradient lives
    in the raw input domain (matches the channel labels in the plot).
    """
    head_W = jnp.asarray(data[f'{target}__head_W'])
    head_b = jnp.asarray(data[f'{target}__head_b'])
    target_mu = float(data[f'{target}__target_mu'][0])
    target_sd = float(data[f'{target}__target_sd'][0])

    def head_out(X_raw_kf):
        X = (X_raw_kf - feat_mu[0]) / feat_sd[0]      # (K, F), z-normed
        h = X[None]                                    # (1, K, F)
        for W, b in conv_params:
            h = jax.nn.relu(conv1d(h, W, b))
        latent = h.reshape(1, -1)                      # (1, latent_dim)
        yhat_std = (latent @ head_W + head_b).squeeze()
        return yhat_std * target_sd + target_mu

    return head_out


macd_grad = jax.jit(jax.grad(make_head_output_fn('macd')))
vol_grad  = jax.jit(jax.grad(make_head_output_fn('vol')))


# --- 3. Build a ticker's input bundle and pick representative bars. ---------
TICKER = 'AAPL'
load_kwargs = dict(
    stooq_dir=None, kaggle_dir=None, use_yahoo=True,
    start=meta['start'], end=meta['end'],
    scales=scales,
    lookback=int(meta['lookback']),
    window_cols=K,
    include_zscore_stats=bool(meta.get('include_zscore_stats')),
    include_returns=bool(meta.get('include_returns')),
    decoder=meta['decoder'],
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


# --- 4. Compute |grad| averaged over N random bars per head. ----------------
N_BARS = min(200, len(valid_idx))
rng = np.random.default_rng(0)
sample_idx = rng.choice(valid_idx, size=N_BARS, replace=False)


def average_saliency(grad_fn, label: str) -> np.ndarray:
    sal_sum = np.zeros((K, F), dtype=np.float64)
    for i, bar_idx in enumerate(sample_idx):
        X_raw = jnp.asarray(X_all[bar_idx])
        g = grad_fn(X_raw)
        sal_sum += np.abs(np.asarray(g))
        if (i + 1) % 50 == 0:
            print(f'  {label} bar {i+1}/{N_BARS}')
    return sal_sum / N_BARS


print(f'\nComputing MACD saliency...')
sal_macd = average_saliency(macd_grad, 'macd')
print(f'Computing vol saliency...')
sal_vol = average_saliency(vol_grad, 'vol')


# --- 4b. Textual top-N breakdown. -------------------------------------------
chan_labels = (
    [f'coeff s={s}' for s in scales] +
    [f'power s={s}' for s in scales] +
    (['z-mu', 'z-std'] if meta.get('include_zscore_stats') else []) +
    (['return'] if meta.get('include_returns') else [])
)
assert len(chan_labels) == F


def report_topk(sal: np.ndarray, label: str, k: int = 8) -> None:
    flat = sal.flatten()
    top_idx = np.argsort(flat)[::-1][:k]
    print(f'\nTop {k} (lag, channel) cells for {label}:')
    for rank, fi in enumerate(top_idx, 1):
        lag = fi // F
        ch = fi % F
        print(f'  {rank:2d}. lag={lag:>3d}  ch={ch:>2d} ({chan_labels[ch]:<14s})  '
              f'|grad|={flat[fi]:.4e}')
    per_chan = sal.sum(axis=0)
    chan_top = np.argsort(per_chan)[::-1][:k]
    print(f'  Per-channel total |grad| (top {k}):')
    for rank, ci in enumerate(chan_top, 1):
        print(f'    {rank:2d}. ch={ci:>2d} ({chan_labels[ci]:<14s})  '
              f'sum |grad|={per_chan[ci]:.4e}')


report_topk(sal_macd, 'MACD')
report_topk(sal_vol, 'vol')


# --- 5. Plot — same structure as film_attention.py. -------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)

# Each abs panel uses its own max so within-head pattern is visible.
# (Earlier shared-vmax made vol invisible if MACD was much larger.)
for ax, sal, color, title in [
    (axes[0], sal_macd, 'Blues',
     f'MACD ({meta["macd_fast"]}, {meta["macd_slow"]}, {meta["macd_signal"]}) — '
     f'unconditioned head'),
    (axes[1], sal_vol, 'Reds',
     f'realized vol ({meta.get("vol_window", 20)}-bar) — unconditioned head'),
]:
    im = ax.imshow(sal.T, aspect='auto', origin='lower',
                   cmap=color, vmin=0, vmax=float(sal.max()))
    ax.set_xlabel('Lag (0 = most recent bar)')
    ax.set_ylabel('Channel')
    ax.set_yticks(range(F))
    ax.set_yticklabels(chan_labels, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='|d head / d X| avg', fraction=0.025)

# Diff panel: positive (blue) = MACD-dominant, negative (red) = vol-dominant.
# Normalize each map by its own max before differencing so the diff isn't
# dominated by absolute-magnitude differences between the two heads.
sal_macd_n = sal_macd / max(sal_macd.max(), 1e-12)
sal_vol_n  = sal_vol  / max(sal_vol.max(),  1e-12)
diff = sal_macd_n - sal_vol_n
vlim = float(np.abs(diff).max())
im = axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='seismic_r',
                    norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim))
axes[2].set_xlabel('Lag (0 = most recent bar)')
axes[2].set_ylabel('Channel')
axes[2].set_yticks(range(F))
axes[2].set_yticklabels(chan_labels, fontsize=6)
axes[2].set_title('Normalized diff (MACD − vol)\nblue = MACD dominates; red = vol dominates')
fig.colorbar(im, ax=axes[2], label='Δ saliency (per-head normalized)', fraction=0.025)

fig.suptitle(
    f'MACD vs vol input attention — {TICKER}, K={K}, '
    f'{N_BARS} bars averaged\nbackbone: {os.path.basename(WEIGHTS)}',
    fontsize=11, fontweight='bold')

out_dir = '/content/Output' if os.path.isdir('/content/Output') else '/tmp'
out = os.path.join(out_dir, 'attention-macd-vol.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved {out}')
