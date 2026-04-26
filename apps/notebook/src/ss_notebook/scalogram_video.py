"""Day-by-day animation of the causal scalogram across a window.

Why this script exists
----------------------
The regime trainer (`apps/regime/`) ranks tickers from `causal_cwt`
power. At each rebalance bar `t`, the trainer's view of ticker `i` is:

    coeffs[:, :t+1, i]   ← strictly causal: depends only on prices[:t+1]

and the divergence-based score it produces compares the **recent**
window `[t - n_tail + 1, t]` against the **historical** window
`[t - lookback + 1, t - n_tail]` of CWT power across scales (see
`ss_wavelets.precompute_windows` and `regime.trainer.weights_regime`).

A static scalogram shows the final state. This animation rolls `t`
forward one trading day at a time so you can watch:

  * the right edge grow as new wavelet evidence arrives;
  * the rolling z-norm in `causal_cwt` keep older coefficients visually
    stable (no look-ahead, no in-place rewrites of past values);
  * the moving (recent / historical) split that the regime divergence
    is computed across.

Performance note
----------------
Because `causal_cwt` is strictly causal, the columns of
`causal_cwt(prices_full)[:, :t+1]` equal `causal_cwt(prices[:t+1])`
exactly. We compute the full scalogram **once** and per-frame mask the
columns `> t` with NaN — no recomputation in the animation loop.

Writers
-------
`--writer ffmpeg` (default) writes MP4 — needs ffmpeg in PATH.
`--writer pillow` writes a GIF with no external deps but is slower
and produces much larger files.

Usage
-----
    uv run ss-scalogram-video --start 2018-01-01 --end 2020-01-01 TSLA
    uv run ss-scalogram-video --stride 5 --fps 30 NVDA
    uv run ss-scalogram-video --lookback 252 --n-tail 21 AAPL
"""

from __future__ import annotations

import argparse
import os

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def _configure_ffmpeg() -> str:
    """Prefer system `ffmpeg`; fall back to the `imageio-ffmpeg` bundle.

    nix-built ffmpeg 7+ links AVFoundation symbols introduced in the
    macOS 14 SDK; on older Intel-macOS hosts it dies with a dyld
    `_AVCaptureDeviceTypeContinuityCamera` not-found error before
    matplotlib gets to write a single frame. ffmpeg 6.x doesn't
    reference that symbol and works.

    Returns 'system', 'bundled', or 'none' for diagnostics.
    """
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], check=True,
                       capture_output=True, timeout=3)
        return 'system'
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, OSError):
        pass
    try:
        import imageio_ffmpeg
        mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        return 'bundled'
    except ImportError:
        return 'none'

from ss_plotting.scalogram import INDICATOR_SCALES

from ss_notebook.scalogram import (
    _scalogram_scales,
    compute_scalogram_power,
    load_prices,
)


def _setup_figure(
    ticker: str,
    dates: np.ndarray,
    prices: np.ndarray,
    log_power: np.ndarray,
    scales: np.ndarray,
    lookback: int,
    n_tail: int,
) -> tuple[plt.Figure, dict]:
    """Build the figure once and return artists the animator updates each frame.

    Returns the figure and a dict of mutable artists keyed by name so
    `update()` can call `.set_array(...)` / `.set_data(...)` without
    re-creating the canvas.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f'{ticker} — Causal Scalogram (day-by-day)',
        fontsize=13, fontweight='bold', y=0.98)

    ax_price = fig.add_axes([0.08, 0.78, 0.86, 0.16])
    ax_price.plot(dates, prices, color='lightgray', linewidth=0.6, zorder=1)
    line_seen, = ax_price.plot([], [], color='black', linewidth=0.9,
                               zorder=2, animated=True)
    marker_now, = ax_price.plot([], [], 'o', color='red',
                                markersize=4, zorder=3, animated=True)
    ax_price.set_ylabel('Price', fontsize=9)
    ax_price.set_xlim(dates[0], dates[-1])
    ax_price.set_ylim(prices.min() * 0.95, prices.max() * 1.05)
    ax_price.tick_params(labelbottom=False)
    title_txt = ax_price.set_title('', fontsize=10, animated=True)

    ax_sg = fig.add_axes([0.08, 0.10, 0.86, 0.62])
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad(color='#1a1a1a')
    # Render the full heatmap ONCE — never updated per-frame. Future
    # cells get masked by the moving fog rectangle below, so the
    # expensive `pcolormesh.set_array` call is gone from the hot loop.
    mesh = ax_sg.pcolormesh(
        dates, scales, log_power,
        cmap=cmap, shading='nearest',
        vmin=np.nanpercentile(log_power, 2),
        vmax=np.nanpercentile(log_power, 98))
    ax_sg.set_yscale('log')
    ax_sg.set_ylabel('Scale (trading days)', fontsize=10)
    ax_sg.set_xlim(dates[0], dates[-1])
    ax_sg.invert_yaxis()
    cbar = fig.colorbar(mesh, ax=ax_sg, fraction=0.02, pad=0.01)
    cbar.set_label('log10 power', fontsize=9)

    for label, scale in INDICATOR_SCALES.items():
        if scales[0] <= scale <= scales[-1]:
            ax_sg.axhline(scale, color='white', linewidth=0.4,
                          alpha=0.4, linestyle='--')

    # Fog-of-war: an opaque rectangle covering [t+1 .. n_dates-1] that
    # hides future scalogram cells. Colored to match `cmap.set_bad` so
    # the visual matches the older "NaN-mask the future" approach.
    # Initially covers the whole heatmap (so the price line at frame 0
    # sees no spoilers).
    fog = ax_sg.axvspan(
        dates[0], dates[-1], facecolor='#1a1a1a',
        edgecolor='none', zorder=2.5, animated=True)

    # Three vertical guides that move with t:
    # red   = current bar (t)          — what the trainer is scoring
    # cyan  = recent window left edge  (t - n_tail + 1)
    # blue  = historical left edge     (t - lookback + 1)
    vline_now = ax_sg.axvline(dates[0], color='red', linewidth=1.0,
                              alpha=0.9, animated=True)
    vline_recent = ax_sg.axvline(
        dates[0], color='cyan', linewidth=0.8, alpha=0.7,
        linestyle='--', animated=True)
    vline_hist = ax_sg.axvline(
        dates[0], color='deepskyblue', linewidth=0.8, alpha=0.7,
        linestyle=':', animated=True)

    ax_sg.text(0.99, 1.02,
               f'recent={n_tail}d (cyan)   historical={lookback}d (blue)',
               transform=ax_sg.transAxes, ha='right', fontsize=8,
               color='black')
    for tick in ax_sg.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment('right')

    return fig, {
        'fog': fog,
        'line_seen': line_seen,
        'marker_now': marker_now,
        'title': title_txt,
        'vline_now': vline_now,
        'vline_recent': vline_recent,
        'vline_hist': vline_hist,
        'dates': dates,
        'prices': prices,
        'lookback': lookback,
        'n_tail': n_tail,
        'n_dates': log_power.shape[1],
    }


def _update_frame(t: int, art: dict):
    """Per-frame: shift the fog rectangle, the guides, and the price line.
    No mesh redraw — that was the slow path."""
    dates = art['dates']
    prices = art['prices']
    n_dates = art['n_dates']

    # `axvspan` returns a `Rectangle` with bbox geometry: x is in data
    # space (matplotlib date ordinals) and y is in axes-fraction. We
    # move its left edge to dates[t+1] and rescale its width to reach
    # dates[-1]. On the final frame the rectangle collapses to zero
    # width — still a valid artist, blitting handles it fine.
    next_idx = t + 1 if t + 1 < n_dates else n_dates - 1
    x_left = mpl.dates.date2num(dates[next_idx].astype('datetime64[D]'))
    x_right = mpl.dates.date2num(dates[-1].astype('datetime64[D]'))
    art['fog'].set_x(x_left)
    art['fog'].set_width(max(x_right - x_left, 0.0))

    art['line_seen'].set_data(dates[:t + 1], prices[:t + 1])
    art['marker_now'].set_data([dates[t]], [prices[t]])

    art['vline_now'].set_xdata([dates[t], dates[t]])
    rt = max(0, t - art['n_tail'] + 1)
    ht = max(0, t - art['lookback'] + 1)
    art['vline_recent'].set_xdata([dates[rt], dates[rt]])
    art['vline_hist'].set_xdata([dates[ht], dates[ht]])

    date_str = np.datetime_as_string(dates[t], unit='D')
    art['title'].set_text(
        f'{date_str}   bar {t + 1}/{n_dates}   '
        f'price={prices[t]:.2f}')
    return (art['fog'], art['line_seen'], art['marker_now'],
            art['vline_now'], art['vline_recent'], art['vline_hist'],
            art['title'])


def render_video(
    ticker: str,
    prices: np.ndarray,
    dates: np.ndarray,
    *,
    lookback: int,
    n_tail: int,
    output: str,
    fps: int,
    stride: int,
    writer_name: str,
    start_frame: int = 0,
) -> str:
    """Compute the scalogram once, then animate columns 0..t day-by-day.

    Returns the path the video was written to. `start_frame` skips the
    early warm-up bars where the rolling z-norm hasn't filled (default 0
    = start at bar 0; useful values: `lookback` to start once the
    historical window is fully populated).
    """
    scales = _scalogram_scales()
    power, _ = compute_scalogram_power(prices, scales, lookback=lookback)
    log_power = np.log10(power + 1e-12)

    fig, art = _setup_figure(
        ticker, dates, prices, log_power, scales, lookback, n_tail)

    n_dates = log_power.shape[1]
    frames = list(range(start_frame, n_dates, stride))

    print(f'Rendering {len(frames)} frames @ {fps} fps '
          f'(={len(frames) / fps:.1f}s) via {writer_name}...')

    anim = animation.FuncAnimation(
        fig, _update_frame, frames=frames, fargs=(art,),
        interval=1000 / fps, blit=True, repeat=False)

    if writer_name == 'ffmpeg':
        source = _configure_ffmpeg()
        if source == 'none':
            raise RuntimeError(
                'no ffmpeg available — install ffmpeg (nix `ffmpeg_6` works '
                'on Intel macOS; ffmpeg 7+ from nix hits a dyld AVFoundation '
                'symbol missing) or `uv add imageio-ffmpeg` for a bundled '
                'fallback. Or pass `--writer pillow` for GIF.')
        print(f'Using {source} ffmpeg.')
        # `-pix_fmt yuv420p` keeps QuickTime/Safari/most embedded players
        # happy; without it H.264 defaults to yuv444p which several
        # macOS players refuse to decode.
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2400, codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p'])
    elif writer_name == 'pillow':
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f'unknown writer {writer_name!r}')

    anim.save(output, writer=writer, dpi=110)
    plt.close(fig)
    print(f'Saved {output}')
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Day-by-day animation of the causal scalogram.')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols')
    parser.add_argument('--stooq-dir', default=None,
                        help='Stooq archive root (default ./StooqData).')
    parser.add_argument('--kaggle-dir', default=None,
                        help='Use Nasdaq3347-style CSV matrix instead.')
    parser.add_argument('--start', default=None, help='YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='YYYY-MM-DD')
    parser.add_argument('--lookback', type=int, default=90,
                        help='Causal z-norm + historical window length '
                             '(default 90, sized for log-returns input — '
                             'see ss_notebook.scalogram.compute_scalogram_power).')
    parser.add_argument('--n-tail', type=int, default=21,
                        help='Recent window length for the moving split '
                             '(default 21).')
    parser.add_argument('--stride', type=int, default=1,
                        help='Days per frame (1 = every trading day, '
                             '5 = weekly). Higher = shorter video.')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--writer', choices=['ffmpeg', 'pillow'],
                        default='ffmpeg',
                        help='`ffmpeg` (default, MP4, needs ffmpeg binary) or '
                             '`pillow` (GIF, no external deps but slower).')
    parser.add_argument('--output-dir', default='Output')
    parser.add_argument('--start-after-lookback', action='store_true',
                        help='Skip the first `--lookback` warm-up frames where '
                             'the rolling z-norm has not yet filled.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ext = 'mp4' if args.writer == 'ffmpeg' else 'gif'

    for ticker in args.tickers:
        try:
            series = load_prices(
                ticker,
                stooq_dir=args.stooq_dir,
                kaggle_dir=args.kaggle_dir,
                start=args.start, end=args.end,
            )
        except (KeyError, RuntimeError) as exc:
            print(f'Skipping {ticker}: {exc}')
            continue

        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        if len(prices) < args.lookback + args.n_tail + 5:
            print(f'Skipping {ticker}: not enough bars '
                  f'({len(prices)}) for lookback={args.lookback}, '
                  f'n_tail={args.n_tail}')
            continue

        output = os.path.join(
            args.output_dir, f'{ticker}-scalogram-video.{ext}')
        start_frame = args.lookback if args.start_after_lookback else 0
        render_video(
            ticker, prices, dates,
            lookback=args.lookback, n_tail=args.n_tail,
            output=output, fps=args.fps, stride=args.stride,
            writer_name=args.writer, start_frame=start_frame)


if __name__ == '__main__':
    main()
