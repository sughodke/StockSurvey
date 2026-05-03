"""2D PCA projection of per-(ticker, date) CWT fingerprints, animated.

For every trading day since 2000, each ticker has a scalogram window
fingerprint (flattened `(scales × w)` slice of the causal CWT). PCA-2
collapses each fingerprint to an `(x, y)` point; the video shows all
tickers as a moving cloud over time.

Three things make this efficient:

  1. CWT is computed once via `relational.scalogram_cache` (≈ 1 minute
     cold, instant warm) — the cache key includes the full price
     content hash so re-runs against the same data are free.
  2. PCA basis is fit once on a stratified random sample of all
     `(date, ticker)` fingerprints. Projection of the entire panel is a
     single `(N_total, fp_dim) @ (fp_dim, 2)` matmul.
  3. Per-frame work is `scatter.set_offsets(coords[t])` plus a title
     update. No recomputation, no axis re-render.

Universe defaults to the bundled `apps/notebook/data/stooq_us_long`
(312 tickers, 2000-01-03 → 2026-04-24) — keeps the run self-contained
without scanning the full 12K-file Stooq archive. Pass `--data-dir`
to override, or `--tickers AAPL,MSFT,...` to scope further.

Output: `Output/cwt-2d-projection.mp4`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless rendering — no GUI dependency
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt

warnings.filterwarnings('ignore')


_DEFAULT_DATA_DIR = (Path(__file__).resolve().parents[4]
                     / 'notebook' / 'data' / 'stooq_us_long')


def _configure_ffmpeg() -> None:
    """Mirror `ss_notebook.scalogram_video._configure_ffmpeg` — prefer
    the system `ffmpeg` if available, fall back to `imageio-ffmpeg`'s
    bundled binary so the script works on Intel macOS where
    nix `ffmpeg_7+` crashes (CLAUDE.md note).
    """
    import shutil
    sys_ffmpeg = shutil.which('ffmpeg')
    if sys_ffmpeg:
        matplotlib.rcParams['animation.ffmpeg_path'] = sys_ffmpeg
        return
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = (
        imageio_ffmpeg.get_ffmpeg_exe())


def _fit_pca2(
    fps: np.ndarray,
    *,
    sample_size: int = 50_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2-component PCA on a random subsample of fingerprints.

    Returns (mean, basis) where basis is `(fp_dim, 2)` — the top-2
    right singular vectors of the centered sample. Deterministic for
    a given seed.
    """
    flat = fps.reshape(-1, fps.shape[-1])
    finite_mask = np.isfinite(flat).all(axis=1)
    flat = flat[finite_mask]
    if flat.shape[0] == 0:
        raise ValueError(
            'no finite fingerprints to fit PCA — check input data')

    rng = np.random.default_rng(seed)
    if flat.shape[0] > sample_size:
        idx = rng.choice(flat.shape[0], size=sample_size, replace=False)
        sample = flat[idx]
    else:
        sample = flat

    mean = sample.mean(axis=0)
    centered = sample - mean
    # SVD over the wider dimension (samples) of the centered matrix.
    # `full_matrices=False` keeps it cheap when fp_dim < sample_size.
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:2].T.astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), basis


def _ticker_colors(tickers: list[str], cmap_name: str = 'hsv') -> np.ndarray:
    """Stable per-ticker RGBA color seeded by ticker name hash. Index
    into a periodic colormap so adjacent ticker indices don't end up
    sharing a hue.
    """
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(tickers))
    cmap = matplotlib.colormaps[cmap_name]
    return cmap(perm / max(len(tickers) - 1, 1))


def run(
    *,
    data_dir: str | None = None,
    tickers: list[str] | None = None,
    start: str = '2000-01-03',
    end: str = '2026-04-24',
    scales: list[int] | None = None,
    lookback: int = 252,
    fp_window: int = 21,
    frame_stride: int = 5,
    fps: int = 30,
    dpi: int = 110,
    sample_size: int = 50_000,
    point_size: float = 14.0,
    label_size: float | None = None,
    output_dir: str = 'Output',
    output_name: str = 'cwt-2d-projection.mp4',
) -> None:
    if scales is None:
        # 8-scale grid; same family the regime trainer uses but biased
        # toward shorter scales to keep visual variation high.
        scales = [3, 5, 7, 10, 21, 42, 63, 126]

    data_dir = data_dir or str(_DEFAULT_DATA_DIR)
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _highs, _lows, _vol = load_stooq_matrix(
        data_dir,
        min_history=lookback + fp_window + 10,
        start_date=start, end_date=end,
        tickers=list(tickers) if tickers else None)
    print(f'  {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'({prices.index[0].date()} → {prices.index[-1].date()})')

    print(f'\nComputing causal CWT (scales={scales}, lookback={lookback})...')
    coeffs = load_or_compute_cwt(prices, scales, lookback)

    print(f'Extracting fingerprints (w={fp_window}, znorm=True)...')
    fps_arr = extract_fingerprints(coeffs, w=fp_window, znorm=True)
    print(f'  fingerprint panel: {fps_arr.shape} '
          f'({fps_arr.dtype}, {fps_arr.nbytes / 1e6:.0f} MB)')

    print(f'Fitting PCA(2) on a {sample_size:,}-point subsample...')
    mean, basis = _fit_pca2(fps_arr, sample_size=sample_size)
    print(f'  basis shape = {basis.shape}')

    # Project: (T, N, fp_dim) → (T, N, 2). Single matmul.
    coords = ((fps_arr.reshape(-1, fps_arr.shape[-1]) - mean) @ basis
              ).reshape(fps_arr.shape[0], fps_arr.shape[1], 2)

    # Viewport limits based on robust quantiles so outlier frames
    # don't shrink the visible cloud.
    finite = coords[np.isfinite(coords).all(axis=-1)]
    x_lo, x_hi = np.quantile(finite[:, 0], [0.005, 0.995])
    y_lo, y_hi = np.quantile(finite[:, 1], [0.005, 0.995])
    pad_x = 0.05 * (x_hi - x_lo)
    pad_y = 0.05 * (y_hi - y_lo)

    # Drop the warm-up region: dates before `lookback + fp_window` have
    # incomplete causal histories (the CWT z-norm ramps up and the
    # fingerprint window is partly zero-padded).
    warmup = lookback + fp_window
    frame_indices = np.arange(warmup, coords.shape[0], frame_stride)
    print(f'\nAnimating {len(frame_indices)} frames '
          f'(stride={frame_stride}, fps={fps}) — '
          f'≈ {len(frame_indices) / fps:.0f}s of video...')

    tickers_list = list(prices.columns)
    colors = _ticker_colors(tickers_list)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)

    # Initial scatter at the first valid frame so artists have correct
    # type from the start; per-frame updates only call set_offsets.
    init_pts = coords[frame_indices[0]]
    scatter = ax.scatter(
        init_pts[:, 0], init_pts[:, 1],
        s=point_size, c=colors, alpha=0.75, edgecolors='none')

    # Auto-scale label font for readability vs density. With 312
    # tickers anything above ~5pt becomes a wall of text; below ~30
    # tickers we want labels you can actually read at video size.
    n_tickers = prices.shape[1]
    if label_size is None:
        if n_tickers <= 30:
            label_size_eff = 8.0
        elif n_tickers <= 100:
            label_size_eff = 5.5
        else:
            label_size_eff = 4.0
    else:
        label_size_eff = float(label_size)

    # One persistent Text artist per ticker. Per-frame work is just
    # `set_position` + visibility flip on NaN — no artist creation,
    # no font shaping (matplotlib caches the layout per text).
    labels = [
        ax.text(init_pts[i, 0], init_pts[i, 1], tickers_list[i],
                fontsize=label_size_eff, color=colors[i],
                ha='center', va='center', alpha=0.9,
                clip_on=True)
        for i in range(n_tickers)
    ]
    title = ax.set_title('')

    def update(i: int):
        t = frame_indices[i]
        pts = coords[t]   # (n_tickers, 2) — NaN tickers naturally hidden
        scatter.set_offsets(pts)
        finite_mask = np.isfinite(pts).all(axis=1)
        for j, lbl in enumerate(labels):
            if finite_mask[j]:
                lbl.set_position((pts[j, 0], pts[j, 1]))
                if not lbl.get_visible():
                    lbl.set_visible(True)
            elif lbl.get_visible():
                lbl.set_visible(False)
        title.set_text(
            f'CWT-fingerprint PCA projection — '
            f'{prices.index[t].date()}    '
            f'(scales={len(scales)}, w={fp_window}, '
            f'{prices.shape[1]} tickers)')
        return (scatter, title, *labels)

    _configure_ffmpeg()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / output_name

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, codec='libx264', bitrate=6000,
        extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f'\nSaved {out_path}')


def _parse_tickers(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [t.strip().upper() for t in s.split(',') if t.strip()]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--data-dir', default=None,
                   help='Stooq archive root (default: '
                        'apps/notebook/data/stooq_us_long).')
    p.add_argument('--tickers', default=None,
                   help='Comma-separated whitelist (default: all in archive).')
    p.add_argument('--start', default='2000-01-03')
    p.add_argument('--end', default='2026-04-24')
    p.add_argument('--lookback', type=int, default=252,
                   help='Causal-CWT z-norm window in bars.')
    p.add_argument('--fp-window', type=int, default=21,
                   help='Fingerprint window in bars (default 21 = 1 month).')
    p.add_argument('--frame-stride', type=int, default=5,
                   help='Bars between rendered frames. 1=daily (slow), '
                        '5=weekly (default), 21=monthly (fast).')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--dpi', type=int, default=110)
    p.add_argument('--sample-size', type=int, default=50_000,
                   help='Points sampled for PCA basis fit.')
    p.add_argument('--point-size', type=float, default=14.0)
    p.add_argument('--label-size', type=float, default=None,
                   help='Override auto-scaled ticker label font size '
                        '(default scales with universe size: 8 / 5.5 / 4 '
                        'for ≤30 / ≤100 / >100 tickers).')
    p.add_argument('--output-dir', default='Output')
    p.add_argument('--output-name', default='cwt-2d-projection.mp4')
    args = p.parse_args()
    run(
        data_dir=args.data_dir,
        tickers=_parse_tickers(args.tickers),
        start=args.start, end=args.end,
        lookback=args.lookback, fp_window=args.fp_window,
        frame_stride=args.frame_stride,
        fps=args.fps, dpi=args.dpi,
        sample_size=args.sample_size, point_size=args.point_size,
        label_size=args.label_size,
        output_dir=args.output_dir, output_name=args.output_name,
    )
