"""Side-by-side PCA-2 of Phase-2 fingerprints at fixed snapshot dates.

Mirrors the existing `cwt_2d_projection_video.py` per-frame layout
(scatter of all tickers, colored + labeled per ticker, square aspect)
but as a static composite so we can compare *what the latent space
looks like* across compression modes:

  - CWT raw          — full-resolution `(S, w)` flatten, fp_dim=168
  - DWT-L1 (Haar)    — 2D wavelet keep-LL, fp_dim=44
  - DCT-k44          — 2D DCT zigzag keep top-44 (matches DWT-L1
                       compression ratio)
  - DCT-k12          — 2D DCT zigzag keep top-12 (matches DWT-L2's
                       12-dim fingerprint, the over-compressed arm
                       that lost in the bt backtest)

PCA is fit *per arm* on a stratified random subsample of the same
panel, so each panel shows what that compression mode considers the
two largest sources of variance — apples-to-apples for "is the
geometry similar?".

Output: `Output/relational-latent-compare.png` — one composite PNG
with rows = snapshot dates and columns = compression modes.

Run:
    uv run python -m relational.research.relational_latent_compare \\
        --data-dir ./StooqData
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_features import Compression
from ss_loaders import load_stooq_matrix

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt
from relational.sectors import PHASE2_TICKERS

warnings.filterwarnings('ignore')


def _fit_pca2(
    fps: np.ndarray, *, sample_size: int = 50_000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-2 right singular vectors of a centered subsample. Returns
    `(mean (fp_dim,), basis (fp_dim, 2))`."""
    flat = fps.reshape(-1, fps.shape[-1])
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.shape[0] == 0:
        raise ValueError('no finite fingerprints')
    rng = np.random.default_rng(seed)
    if flat.shape[0] > sample_size:
        idx = rng.choice(flat.shape[0], size=sample_size, replace=False)
        sample = flat[idx]
    else:
        sample = flat
    mean = sample.mean(axis=0).astype(np.float32)
    centered = sample - mean
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    return mean, Vt[:2].T.astype(np.float32, copy=False)


def _ticker_colors(tickers: list[str]) -> np.ndarray:
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(tickers))
    cmap = matplotlib.colormaps['hsv']
    return cmap(perm / max(len(tickers) - 1, 1))


def _project(fps_at_t: np.ndarray, mean: np.ndarray,
             basis: np.ndarray) -> np.ndarray:
    """`(n_tickers, fp_dim) → (n_tickers, 2)` PCA projection."""
    return (fps_at_t.astype(np.float32) - mean) @ basis


def _draw_panel(ax, coords, finite, tickers, colors,
                title: str, label_size: float = 7.0) -> None:
    finite_pts = coords[finite]
    if finite_pts.shape[0] >= 2:
        x_lo, x_hi = np.quantile(finite_pts[:, 0], [0.0, 1.0])
        y_lo, y_hi = np.quantile(finite_pts[:, 1], [0.0, 1.0])
    else:
        x_lo, x_hi, y_lo, y_hi = -1, 1, -1, 1
    pad_x = 0.10 * max(x_hi - x_lo, 1e-3)
    pad_y = 0.10 * max(y_hi - y_lo, 1e-3)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
    ax.scatter(coords[finite, 0], coords[finite, 1],
               s=22, c=colors[finite], alpha=0.85, edgecolors='none')
    for j, ok in enumerate(finite):
        if ok:
            ax.text(coords[j, 0], coords[j, 1], tickers[j],
                    fontsize=label_size, color=colors[j],
                    ha='center', va='center', alpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def run(
    *, data_dir: str,
    snapshot_dates: list[str] | None = None,
    fp_window: int = 21,
    lookback: int = 120,
    output_dir: str = 'Output',
    output_name: str = 'relational-latent-compare.png',
) -> None:
    snapshot_dates = snapshot_dates or [
        '2015-01-02',
        '2020-03-23',
        '2025-12-11',
    ]

    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _highs, _lows, _vol = load_stooq_matrix(
        data_dir,
        min_history=lookback + fp_window + 10,
        start_date='2013-01-29', end_date='2025-12-11',
        tickers=list(PHASE2_TICKERS))
    print(f'  {prices.shape[0]} dates x {prices.shape[1]} tickers')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'\nComputing causal CWT (scales={scales}, lookback={lookback})...')
    coeffs = load_or_compute_cwt(prices, scales, lookback)

    arms = [
        ('CWT raw (fp_dim=168)', None),
        ('DWT-L1 Haar (fp_dim=44)',
         Compression(kind='dwt', levels=1, wavelet='haar',
                     pad_mode='periodization')),
        ('DCT k=44 (matches DWT-L1)',
         Compression(kind='dct', keep_top_k=44)),
        ('DCT k=12 (matches DWT-L2)',
         Compression(kind='dct', keep_top_k=12)),
    ]

    print('\nExtracting fingerprints + fitting PCA per arm ...')
    fp_panels: dict[str, np.ndarray] = {}
    pca_per_arm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, comp in arms:
        fps = extract_fingerprints(coeffs, w=fp_window, znorm=True,
                                   compression=comp)
        fp_panels[label] = fps
        pca_per_arm[label] = _fit_pca2(fps)
        print(f'  {label:32s}  fps shape={fps.shape}  '
              f'fp_dim={fps.shape[-1]}')

    tickers = list(prices.columns)
    colors = _ticker_colors(tickers)

    n_rows = len(snapshot_dates)
    n_cols = len(arms)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 4.2 * n_rows), dpi=140)
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r, snap_str in enumerate(snapshot_dates):
        snap = pd.Timestamp(snap_str)
        if snap not in prices.index:
            # Fall back to nearest trading day on or before
            valid = prices.index[prices.index <= snap]
            if len(valid) == 0:
                raise ValueError(
                    f'snapshot date {snap_str} is before any data')
            snap = valid[-1]
        t = int(prices.index.get_loc(snap))

        for c, (label, _comp) in enumerate(arms):
            mean, basis = pca_per_arm[label]
            fps = fp_panels[label]
            fps_at_t = fps[t]
            coords = _project(fps_at_t, mean, basis)
            finite = np.isfinite(coords).all(axis=1)
            ax = axes[r, c]
            _draw_panel(
                ax, coords, finite, tickers, colors,
                title=(f'{label}\n{snap.date()}' if r == 0
                       else f'{snap.date()}'))
            if c == 0:
                ax.set_ylabel(f'{snap.date()}', fontsize=11)
            else:
                ax.set_ylabel('')

    suptitle = (f'Phase-2 fingerprint latent space — CWT raw vs '
                f'compressed (Haar DWT-L1 / DCT zigzag-top-k); PCA-2 per '
                f'arm; w={fp_window}, scales={scales}')
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / output_name
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--snapshot-dates', default='2015-01-02,2020-03-23,2025-12-11',
                   help='Comma-separated YYYY-MM-DD dates.')
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--output-dir', default='Output')
    p.add_argument('--output-name', default='relational-latent-compare.png')
    args = p.parse_args()
    snapshots = [s.strip() for s in args.snapshot_dates.split(',') if s.strip()]
    run(data_dir=args.data_dir, snapshot_dates=snapshots,
        fp_window=args.fp_window, lookback=args.lookback,
        output_dir=args.output_dir, output_name=args.output_name)
