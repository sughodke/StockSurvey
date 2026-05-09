"""CLI for `ss-replay`: train one+ ticker decoders, eval zero-shot on a held-out
ticker, save reconstruction figures.

Examples
--------
    # Single-ticker fit, no held-out eval (in-sample expressivity probe):
    uv run ss-replay AAPL --window-cols 64 --include-zscore-stats \\
        --decoder mlp

    # Train on AAPL, zero-shot eval on TSLA:
    uv run ss-replay AAPL --val-ticker TSLA --window-cols 64 \\
        --include-zscore-stats --decoder mlp

    # Multi-ticker train pool, held-out eval on TSLA:
    uv run ss-replay AAPL --train-tickers MSFT,GOOGL,AMZN \\
        --val-ticker TSLA --window-cols 64 --include-zscore-stats \\
        --decoder mlp
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from replay.features import TARGET_NAMES, load_ticker
from replay.plot import plot_reconstruction
from replay.reconstruct import fit_and_evaluate, fit_and_evaluate_ssl
from ss_features import DEFAULT_STOOQ_DIR, Compression
from ss_wavelets import ALL_SCALES


def _log_tinygrad_device(decoder: str) -> None:
    """Surface which tinygrad device backend is active. Only relevant for
    the tinygrad decoders. Prints once before the fit so a Colab runtime
    that silently fell back to CPU is visible immediately rather than
    after a multi-minute wait."""
    if decoder not in ('mlp', 'cnn', 'masked-ae'):
        return
    try:
        from tinygrad import Device
        print(f'tinygrad device: {Device.DEFAULT}')
    except Exception as e:
        print(f'tinygrad device probe failed: {e}')


def _git_sha_and_dirty() -> tuple[str, bool]:
    """Short git SHA + dirty-tree flag for the current working dir.
    Returns `('nogit', False)` outside a git repo."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'nogit', False
    try:
        dirty = subprocess.run(
            ['git', 'diff', '--quiet', 'HEAD'],
            stderr=subprocess.DEVNULL).returncode != 0
    except Exception:
        dirty = False
    return sha, dirty


def _split_tickers(value: str | None) -> list[str]:
    """Split a comma-separated ticker list, ignoring empty entries."""
    if not value:
        return []
    return [t.strip() for t in value.split(',') if t.strip()]


def _short_train_tag(primary_name: str, extra_names: list[str],
                     max_inline: int = 4) -> str:
    """Filesystem-safe variant of the verbose `PRIMARY+T1+T2+...` tag.

    Joining all train ticker names with '+' produces filenames > NAME_MAX
    (255 bytes on most filesystems) once the pool is wider than ~30
    tickers. For ≤max_inline extras keep the verbose form (still useful
    for small Phase-2-style runs); for larger pools collapse to
    `PRIMARY+Ntickers-h<8hex>`, where the hash deterministically
    identifies the specific pool so two different 294-ticker subsets get
    distinct filenames.
    """
    if len(extra_names) <= max_inline:
        if not extra_names:
            return primary_name
        return primary_name + '+' + '+'.join(extra_names)
    import hashlib
    pool_hash = hashlib.sha1(
        '+'.join(extra_names).encode()).hexdigest()[:8]
    return f'{primary_name}+{len(extra_names)}tickers-h{pool_hash}'


def _print_target_stats(label: str, stats: dict[str, dict[str, float]]) -> None:
    for name, s in stats.items():
        print(f'  {label} {name:5s}  R²={s["r2"]:.6f}  '
              f'RMSE={s["rmse"]:.3e}  max|Δ|={s["max_abs"]:.3e}')


def _run_ssl(args, train_data, val_data, *,
             cnn_channels_per_lag: int, scales: list[int],
             extra_scales: list[int]) -> None:
    """Self-supervised pretrain entry point — masked CWT autoencoding.

    Skips per-target plotting/stats since the SSL fit has no targets.
    Saves an npz with `ssl__feat_mu/sd`, `ssl__conv{i}_W/b`, and the
    decoder under `ssl__head_dec*` (skipped by `load_backbone`'s `head_`
    filter so the file works as a frozen-backbone source for the probe).
    """
    n_features = train_data[0].features.shape[1]
    train_names = ', '.join(d.name for d in train_data)
    if args.compress != 'none':
        compress_str = (f'compress={args.compress}/L{args.compress_levels}/'
                        f'{args.compress_wavelet}, ')
    else:
        compress_str = ''
    print(f'train pool: {train_names}  |  '
          f'{sum(d.valid.sum() for d in train_data)} valid rows  |  '
          f'{len(scales)} scales, lookback={args.lookback}, '
          f'window_cols={args.window_cols}, '
          f'{compress_str}'
          f'decoder=masked-ae, mask_ratio={args.mask_ratio}, '
          f'n_features={n_features} '
          f'(channels_per_lag={cnn_channels_per_lag})')

    params_per_target, ssl_stats = fit_and_evaluate_ssl(
        train_data, val_data,
        cnn_channels_per_lag=cnn_channels_per_lag,
        cnn_hidden=args.cnn_hidden, cnn_kernel=args.cnn_kernel,
        cnn_layers=args.cnn_layers, cnn_steps=args.cnn_steps,
        cnn_batch_size=args.cnn_batch_size,
        cnn_microbatch_size=args.cnn_microbatch_size,
        ssl_decoder_hidden=args.ssl_decoder_hidden,
        ssl_decoder_layers=args.ssl_decoder_layers,
        mask_ratio=args.mask_ratio,
        use_bf16=not args.cnn_no_bf16)

    print('\nSSL pretrain stats (z-norm space):')
    for k, v in ssl_stats.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.6f}')
        else:
            print(f'  {k}: {v}')

    os.makedirs(args.output_dir, exist_ok=True)
    train_tag = _short_train_tag(
        train_data[0].name, [t.name for t in train_data[1:]])

    sha, dirty = _git_sha_and_dirty()
    sha_tag = sha + ('-dirty' if dirty else '')
    weights_arrays: dict[str, np.ndarray] = {}
    for target_name, params in params_per_target.items():
        for key, arr in params.items():
            weights_arrays[f'{target_name}__{key}'] = arr
    metadata = {
        'train_tickers': [d.name for d in train_data],
        'val_tickers': [d.name for d in val_data],
        'targets': [],            # SSL has no per-target supervision
        'decoder': 'masked-ae',
        'window_cols': args.window_cols,
        'compress': args.compress,
        'compress_levels': args.compress_levels,
        'compress_wavelet': args.compress_wavelet,
        'compress_pad_mode': args.compress_pad_mode,
        'lookback': args.lookback,
        'scales': scales,
        'extra_high_freq_scales': sorted(extra_scales),
        'n_features': n_features,
        'rsi_n': args.rsi_n,
        'macd_fast': args.macd_fast,
        'macd_slow': args.macd_slow,
        'macd_signal': args.macd_signal,
        'vol_window': args.vol_window,
        'cnn_hidden': args.cnn_hidden,
        'cnn_kernel': args.cnn_kernel,
        'cnn_layers': args.cnn_layers,
        'cnn_steps': args.cnn_steps,
        'cnn_batch_size': args.cnn_batch_size,
        'ssl_decoder_hidden': args.ssl_decoder_hidden,
        'ssl_decoder_layers': args.ssl_decoder_layers,
        'mask_ratio': args.mask_ratio,
        'start': args.start,
        'end': args.end,
        'git_sha': sha,
        'git_dirty': dirty,
        'ssl_stats': ssl_stats,
    }
    weights_arrays['_meta'] = np.array(json.dumps(metadata, default=float))
    weights_fname = (Path(args.output_dir) /
                     f'{train_tag}-ssl-masked-ae-{sha_tag}.npz')
    np.savez(weights_fname, **weights_arrays)
    print(f'\nSaved {weights_fname}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reconstruct RSI / MACD / price from per-bar CWT '
                    'slices via a trained decoder; plot vs ground truth '
                    'with R² + error.')
    parser.add_argument('ticker',
                        help='Primary train ticker. Always saves a figure '
                             'at <output-dir>/<ticker>-replay.png.')
    parser.add_argument('--train-tickers', default=None,
                        help='Comma-separated additional train tickers '
                             '(e.g. MSFT,GOOGL,AMZN). Their features are '
                             'pooled with `ticker` into a single decoder '
                             'fit. No figures are saved for them.')
    parser.add_argument('--val-ticker', default=None,
                        help='Optional held-out ticker. The pooled-train '
                             'decoder is applied zero-shot here. Reports '
                             'val stats and saves '
                             '<output-dir>/<val-ticker>-replay-zeroshot-'
                             'from-<ticker>.png.')
    parser.add_argument('--stooq-dir', default=None,
                        help=f'Stooq archive root. Default: {DEFAULT_STOOQ_DIR}.')
    parser.add_argument('--kaggle-dir', default=None,
                        help='Slice columns from a Nasdaq3347-style CSV '
                             'matrix instead of using Stooq.')
    parser.add_argument('--yahoo', action='store_true',
                        help='Fetch each ticker via yfinance instead of '
                             'Stooq/Kaggle. No on-disk archive needed; '
                             'use on Colab or any environment without '
                             'the Stooq dump.')
    parser.add_argument('--start', default=None,
                        help='Trim every loaded ticker to start at this date.')
    parser.add_argument('--end', default=None,
                        help='Trim every loaded ticker to end at this date.')
    parser.add_argument('--lookback', type=int, default=252,
                        help='Causal z-norm window for the CWT (default 252, '
                             'matches the regime trainer\'s raw-close path).')
    parser.add_argument('--window-cols', type=int, default=1,
                        help='Trailing-window size K — features at date t are '
                             'the (coeffs, power) columns over [t-K+1 ... t]. '
                             'K=1 (default) is single-column; K large is '
                             'closer to the full CWT matrix and approaches the '
                             'invertibility ceiling. Cost is O(K) extra '
                             'features per bar.')
    parser.add_argument('--compress', choices=['none', 'dwt'], default='none',
                        help='Per-bar 2D compression of each `(K, n_scales)` '
                             'channel tile before it reaches the CNN. dwt = '
                             'L levels of 2D wavelet decomposition, keep LL '
                             'approximation only — output tile per channel '
                             'is `(ceil(K/2^L), ceil(S/2^L))`. Causality '
                             'preserved because each tile contains only past '
                             'bars. Applied uniformly across all 7 per-scale '
                             'channels (Morlet polar + Gaussian + log-L2-amp).')
    parser.add_argument('--compress-levels', type=int, default=1,
                        help='DWT levels for --compress dwt. Output K and '
                             'scale axes shrink by 2^L each. Default 1.')
    parser.add_argument('--compress-wavelet', default='haar',
                        help='Wavelet family for --compress dwt (any name '
                             'accepted by pywt, e.g. haar / db2 / sym4). '
                             'Default haar — orthonormal, shortest filter, '
                             'cleanest LL = block average interpretation.')
    parser.add_argument('--compress-pad-mode', default='periodization',
                        help='pywt boundary mode for --compress dwt. '
                             'Default periodization — output sizes are '
                             'predictable ceil(N/2^L) per axis.')
    parser.add_argument('--extra-high-freq-scales', default='',
                        help='Comma-separated extra scales to prepend to '
                             '`ALL_SCALES` (e.g. "1,2"). Adds finer-grained '
                             'wavelets at the high-frequency end. Default '
                             'empty (use ALL_SCALES as-is). Each added scale '
                             'is +2 channels per lag.')
    parser.add_argument('--decoder',
                        choices=['linear', 'mlp', 'cnn', 'masked-ae'],
                        default='linear',
                        help='`linear` = OLS via `np.linalg.lstsq`. `mlp` = '
                             'tiny JAX MLP (Adam) — captures the RSI sigmoid '
                             'nonlinearity that OLS can\'t. `cnn` = 1-D '
                             'Conv1D over the trailing-K window with '
                             'shared weights across lags — the right '
                             'inductive bias for fixed linear filters like '
                             'Wilder/EMA. Requires --window-cols > 1. '
                             '`masked-ae` = self-supervised pretrain via '
                             'masked CWT autoencoding (no per-target '
                             'supervision). Encoder shape matches `cnn` so '
                             'the saved npz is loadable by '
                             '`scoring.backbone.load_backbone`. Validate '
                             'with `--decoder cnn --freeze-backbone <npz>`.')
    parser.add_argument('--mlp-hidden', type=int, default=128)
    parser.add_argument('--mlp-layers', type=int, default=2)
    parser.add_argument('--mlp-steps', type=int, default=2000)
    parser.add_argument('--mlp-batch-size', type=int, default=None,
                        help='Stochastic Adam batch size. Default = '
                             'full-batch GD. Set when the train pool '
                             'is too large for full-batch device memory '
                             '(rule of thumb: needed past ~30k pooled '
                             'rows at K=64). 8192 is a good starting '
                             'point.')
    parser.add_argument('--cnn-hidden', type=int, default=64,
                        help='CNN channel width per conv layer. Default 64.')
    parser.add_argument('--cnn-kernel', type=int, default=5,
                        help='CNN kernel size in lags. Default 5.')
    parser.add_argument('--cnn-layers', type=int, default=2,
                        help='Number of stacked Conv1D + ReLU blocks. '
                             'Default 2.')
    parser.add_argument('--cnn-steps', type=int, default=2000,
                        help='Adam steps for the CNN fit. Default 2000.')
    parser.add_argument('--cnn-batch-size', type=int, default=None,
                        help='Stochastic Adam batch size for CNN. Default = '
                             'full-batch GD. Set when the train pool is too '
                             'large for full-batch device memory; CNN is '
                             'more activation-heavy than MLP, so this kicks '
                             'in at smaller pool sizes. 8192 works at K=64.')
    parser.add_argument('--cnn-microbatch-size', type=int, default=None,
                        help='Per-step gradient-accumulation chunk size. '
                             'Default = `--cnn-batch-size` (no accumulation). '
                             'Set smaller (e.g. 256, 128) to keep the '
                             '*effective* batch high while shrinking VRAM — '
                             'each Adam step does ceil(batch/microbatch) '
                             'forward+backward passes whose grads are '
                             'averaged before optimizer.step().')
    parser.add_argument('--cnn-no-bf16', action='store_true',
                        help='Disable bf16 mixed precision for the CNN '
                             'forward (default: enabled). Use on backends '
                             'whose shader compiler lacks bf16 (Metal on '
                             'Intel macOS) or for fp32 reproducibility.')
    parser.add_argument('--cnn-film-hidden', type=int, default=32,
                        help='Hidden width for the FiLM gamma/beta MLPs that '
                             'modulate the latent for conditioned heads. '
                             'Each conditioned head gets two MLPs (cond_dim '
                             '-> film_hidden -> latent_dim); the modulated '
                             'latent then feeds the linear head. 32 (default) '
                             'gives true latent x cond interaction without '
                             'memorization risk (the latent -> output map '
                             'stays linear; only the cond -> {gamma, beta} '
                             'maps are non-linear, and they never see the '
                             'latent). 0 disables FiLM and falls back to the '
                             'legacy additive-concat path (cond appended to '
                             'latent, absorbed by linear head weights — only '
                             'works when grid targets are highly correlated '
                             'across cond, which fails for the (w, n) RSI '
                             'cross-product).')
    parser.add_argument('--mask-ratio', type=float, default=0.4,
                        help='Fraction of (lag, channel) input cells masked '
                             'per training row in `--decoder masked-ae`. '
                             'Default 0.4. Lower (~0.2) is too easy — the '
                             'encoder doesn\'t have to compress; higher '
                             '(~0.7) is too hard — gradient is noisy. The '
                             'MAE literature sweet spot for vision is '
                             '0.4–0.75; for time series 0.3–0.5 is more '
                             'common.')
    parser.add_argument('--ssl-decoder-hidden', type=int, default=256,
                        help='MLP decoder hidden width for `--decoder '
                             'masked-ae`. Default 256. Decoder maps the '
                             'flattened backbone latent (K_post * hidden) '
                             'back to the K * F z-normed input. Symmetric '
                             'to encoder is the standard default; weaker '
                             'starves the encoder of useful gradient, '
                             'stronger lets the decoder do the work and '
                             'the encoder learns less.')
    parser.add_argument('--ssl-decoder-layers', type=int, default=2,
                        help='Number of MLP layers in the masked-AE '
                             'decoder. Default 2.')
    parser.add_argument('--freeze-backbone', default=None,
                        help='Path to a previously-trained npz (typically '
                             'from `--decoder masked-ae` or another '
                             '`--decoder cnn` run). Loads the conv '
                             'backbone weights and holds them fixed; only '
                             'per-target heads + FiLM train. This is the '
                             'probe protocol — read off per-target R² to '
                             'see what the SSL latent encoded. Requires '
                             '`--decoder cnn`.')
    parser.add_argument('--rsi-n', type=int, default=7,
                        help='Anchor RSI period. Used for the 1-D ground-'
                             'truth target (what plotting/stats compare '
                             'against) and as the conditioning value at '
                             'prediction time when --rsi-n-grid is empty.')
    parser.add_argument('--rsi-n-grid', default='',
                        help='Comma-separated periods (e.g. "5,7,9,13,21") '
                             'enabling RSI head period conditioning. Each '
                             'pooled training row is replicated per grid '
                             'value with the matching RSI(n) target and the '
                             'normalized n is concatenated to the latent '
                             'before the linear head. Backbone latent stays '
                             'parameter-agnostic so the frozen features keep '
                             'the same shape downstream. Empty (default) = '
                             'fixed-period RSI from --rsi-n. Only --decoder '
                             'cnn supports this.')
    parser.add_argument('--rsi-w-grid', default='',
                        help='Comma-separated resampling strides (e.g. '
                             '"1,5,10,21") extending RSI head conditioning '
                             'to the (w, n) cross-product. RSI is computed '
                             'on stride-w price changes (w=1 reduces to '
                             'standard daily RSI; w=5 ~ rolling weekly RSI; '
                             'w=21 ~ monthly). Conditioning becomes p_dim=2 '
                             '= (n_norm, w_norm). Each pool row is then '
                             'replicated len(n_grid) * len(w_grid) times. '
                             'Requires --rsi-n-grid to also be set.')
    parser.add_argument('--rsi-anchor-w', type=int, default=1,
                        help='Stride used when applying the conditioned '
                             'RSI head at prediction time. Default 1 '
                             '(daily RSI), matching the 1-D ground-truth '
                             'panel. Set to a value in --rsi-w-grid to '
                             'render reconstructions at a longer horizon.')
    parser.add_argument('--macd-fast', type=int, default=12)
    parser.add_argument('--macd-slow', type=int, default=26)
    parser.add_argument('--macd-signal', type=int, default=9)
    parser.add_argument('--vol-window', type=int, default=20,
                        help='Window size for the realized-volatility target '
                             '(rolling std of daily log returns). Default 20 '
                             'matches the regime trainer rebalance horizon.')
    parser.add_argument('--cci-n', type=int, default=20,
                        help='Anchor CCI period (Lambert 0.015, close-only). '
                             'Used for the 1-D ground-truth target and as the '
                             'conditioning value at prediction time when '
                             '--cci-n-grid is empty. Default 20.')
    parser.add_argument('--cci-n-grid', default='',
                        help='Comma-separated CCI periods (e.g. '
                             '"10,14,20,28,40") enabling CCI head period '
                             'conditioning. Same plumbing as --rsi-n-grid: '
                             'pool replicas tile across the (n, w) grid, '
                             'normalized n is concatenated to the latent '
                             'before the FiLM-modulated head. Only --decoder '
                             'cnn supports this.')
    parser.add_argument('--cci-w-grid', default='',
                        help='Comma-separated CCI strides extending '
                             'conditioning to the (w, n) cross-product. '
                             'CCI(n) is computed on stride-w price history. '
                             'Same convention as --rsi-w-grid; w=1 reduces '
                             'to canonical daily CCI(n), w>1 evaluates the '
                             'longer-horizon CCI at every bar. Requires '
                             '--cci-n-grid.')
    parser.add_argument('--cci-anchor-w', type=int, default=1,
                        help='Stride used when applying the conditioned CCI '
                             'head at prediction time. Default 1 matches the '
                             '1-D ground-truth panel.')
    parser.add_argument('--vol-n-grid', default='',
                        help='Comma-separated realized-vol windows (e.g. '
                             '"5,10,20,30,60") enabling FiLM conditioning on '
                             'the vol head. cond_dim=1 (n / max_n). Anchor '
                             'comes from --vol-window. CNN-only.')
    parser.add_argument('--macd-fast-grid', default='',
                        help='Comma-separated MACD fast-EMA periods (e.g. '
                             '"8,12,16,24") enabling FiLM conditioning on '
                             'the macd head. slow / signal are derived from '
                             'fast via `ss_indicators.macd_from_fast` '
                             '(slow ≈ 2.167*fast, signal ≈ 0.75*fast — the '
                             'canonical textbook (12, 26, 9) ratio scaled), '
                             'so the f=12 cell collapses exactly onto the '
                             'canonical anchor. cond_dim=1. Anchor comes '
                             'from --macd-fast.')
    # ----- Lie-shape heads (added 2026-05-09) -----
    # Each is FiLM-conditioned over a single horizon `n`. Defaults match
    # `apps/lie.ticker_features.TickerFeatureConfig`. CNN-only.
    parser.add_argument('--momentum-n', type=int, default=21,
                        help='Anchor horizon for the vol-norm-momentum head. '
                             'Default 21 (one trading month). The 1-D ground-'
                             'truth target uses this; the FiLM head conditions '
                             'on it at predict time when --momentum-n-grid is '
                             'empty.')
    parser.add_argument('--momentum-n-grid', default='',
                        help='Comma-separated horizons (e.g. "5,10,21,42,63") '
                             'enabling FiLM conditioning on the momentum head. '
                             'cond_dim=1. CNN-only.')
    parser.add_argument('--drawdown-n', type=int, default=63,
                        help='Anchor horizon for the drawdown head. Default '
                             '63 (one trading quarter).')
    parser.add_argument('--drawdown-n-grid', default='',
                        help='Comma-separated horizons (e.g. "21,42,63,126") '
                             'enabling FiLM conditioning on the drawdown head. '
                             'cond_dim=1. CNN-only.')
    parser.add_argument('--skew-n', type=int, default=63,
                        help='Anchor horizon for the trailing-return-skew head. '
                             'Default 63.')
    parser.add_argument('--skew-n-grid', default='',
                        help='Comma-separated horizons enabling FiLM '
                             'conditioning on the skew head. cond_dim=1. '
                             'CNN-only.')
    parser.add_argument('--kurt-n', type=int, default=63,
                        help='Anchor horizon for the trailing-return-kurt '
                             'head. Default 63.')
    parser.add_argument('--kurt-n-grid', default='',
                        help='Comma-separated horizons enabling FiLM '
                             'conditioning on the kurt head. cond_dim=1. '
                             'CNN-only.')
    parser.add_argument('--output-dir', default='Output',
                        help='Where reconstruction figures are saved. '
                             'matplotlib never opens an interactive window.')
    parser.add_argument('--targets', default=','.join(TARGET_NAMES),
                        help='Comma-separated subset of '
                             f'{",".join(TARGET_NAMES)}. Each target = one '
                             'extra decoder fit, so dropping unused ones '
                             'is a real compute lever (each fit is a full '
                             'Adam pass).')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'],
                        default='auto',
                        help='`auto` (default) lets tinygrad pick its '
                             'default backend (Metal on macOS, CUDA on '
                             'NVIDIA, AMD on Linux+KFD, else CPU). `cpu` '
                             'forces CPU via CPU=1. `gpu` is a no-op hint '
                             '(tinygrad selects the highest-priority '
                             'available accelerator automatically).')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CPU'] = '1'
    targets = tuple(_split_tickers(args.targets))
    unknown = set(targets) - set(TARGET_NAMES)
    if unknown:
        parser.error(f'unknown targets {sorted(unknown)!r}; '
                     f'valid: {TARGET_NAMES}')
    if not targets:
        parser.error('--targets must list at least one target')

    extra_scales = [int(s) for s in _split_tickers(args.extra_high_freq_scales)]
    if any(s < 1 for s in extra_scales):
        parser.error('--extra-high-freq-scales must be positive integers')
    scales = sorted(set(extra_scales) | set(ALL_SCALES))

    if args.compress != 'none':
        compression = Compression(
            kind=args.compress, levels=args.compress_levels,
            wavelet=args.compress_wavelet, pad_mode=args.compress_pad_mode)
        try:
            K_post, S_post = compression.output_shape(
                args.window_cols, len(scales))
        except Exception as exc:
            parser.error(f'--compress configuration invalid: {exc}')
        if K_post < 1 or S_post < 1:
            parser.error(
                f'--compress-levels={args.compress_levels} on '
                f'--window-cols={args.window_cols}, n_scales={len(scales)} '
                f'produces empty tile (K_post={K_post}, S_post={S_post}); '
                f'reduce --compress-levels')
        effective_window_cols = K_post
    else:
        compression = None
        effective_window_cols = args.window_cols

    rsi_n_grid = tuple(int(s) for s in _split_tickers(args.rsi_n_grid))
    if rsi_n_grid and any(n < 2 for n in rsi_n_grid):
        parser.error('--rsi-n-grid values must be >= 2')
    if rsi_n_grid and args.decoder != 'cnn':
        parser.error('--rsi-n-grid requires --decoder cnn '
                     '(head conditioning is only wired into the CNN trainer)')
    rsi_w_grid = tuple(int(s) for s in _split_tickers(args.rsi_w_grid))
    if rsi_w_grid and any(w < 1 for w in rsi_w_grid):
        parser.error('--rsi-w-grid values must be >= 1')
    if rsi_w_grid and not rsi_n_grid:
        parser.error('--rsi-w-grid requires --rsi-n-grid (w-conditioning '
                     'extends n-conditioning to the (w, n) cross-product)')
    if rsi_w_grid and args.rsi_anchor_w not in rsi_w_grid:
        parser.error(f'--rsi-anchor-w={args.rsi_anchor_w} not in '
                     f'--rsi-w-grid={list(rsi_w_grid)}')

    vol_n_grid = tuple(int(s) for s in _split_tickers(args.vol_n_grid))
    if vol_n_grid and any(n < 2 for n in vol_n_grid):
        parser.error('--vol-n-grid values must be >= 2')
    if vol_n_grid and args.decoder != 'cnn':
        parser.error('--vol-n-grid requires --decoder cnn')
    macd_fast_grid = tuple(int(s) for s in _split_tickers(args.macd_fast_grid))
    if macd_fast_grid and any(f < 2 for f in macd_fast_grid):
        parser.error('--macd-fast-grid values must be >= 2')
    if macd_fast_grid and args.decoder != 'cnn':
        parser.error('--macd-fast-grid requires --decoder cnn')

    cci_n_grid = tuple(int(s) for s in _split_tickers(args.cci_n_grid))
    if cci_n_grid and any(n < 2 for n in cci_n_grid):
        parser.error('--cci-n-grid values must be >= 2')
    if cci_n_grid and args.decoder != 'cnn':
        parser.error('--cci-n-grid requires --decoder cnn '
                     '(head conditioning is only wired into the CNN trainer)')
    cci_w_grid = tuple(int(s) for s in _split_tickers(args.cci_w_grid))
    if cci_w_grid and any(w < 1 for w in cci_w_grid):
        parser.error('--cci-w-grid values must be >= 1')
    if cci_w_grid and not cci_n_grid:
        parser.error('--cci-w-grid requires --cci-n-grid (w-conditioning '
                     'extends n-conditioning to the (w, n) cross-product)')
    if cci_w_grid and args.cci_anchor_w not in cci_w_grid:
        parser.error(f'--cci-anchor-w={args.cci_anchor_w} not in '
                     f'--cci-w-grid={list(cci_w_grid)}')

    # Lie-shape head grids: each is 1-axis (`n` only). Same min-length and
    # cnn-only rules the other 1-D heads (vol, macd) follow.
    momentum_n_grid = tuple(int(s) for s in _split_tickers(args.momentum_n_grid))
    if momentum_n_grid and any(n < 2 for n in momentum_n_grid):
        parser.error('--momentum-n-grid values must be >= 2')
    if momentum_n_grid and args.decoder != 'cnn':
        parser.error('--momentum-n-grid requires --decoder cnn')
    drawdown_n_grid = tuple(int(s) for s in _split_tickers(args.drawdown_n_grid))
    if drawdown_n_grid and any(n < 1 for n in drawdown_n_grid):
        parser.error('--drawdown-n-grid values must be >= 1')
    if drawdown_n_grid and args.decoder != 'cnn':
        parser.error('--drawdown-n-grid requires --decoder cnn')
    skew_n_grid = tuple(int(s) for s in _split_tickers(args.skew_n_grid))
    if skew_n_grid and any(n < 3 for n in skew_n_grid):
        parser.error('--skew-n-grid values must be >= 3')
    if skew_n_grid and args.decoder != 'cnn':
        parser.error('--skew-n-grid requires --decoder cnn')
    kurt_n_grid = tuple(int(s) for s in _split_tickers(args.kurt_n_grid))
    if kurt_n_grid and any(n < 3 for n in kurt_n_grid):
        parser.error('--kurt-n-grid values must be >= 3')
    if kurt_n_grid and args.decoder != 'cnn':
        parser.error('--kurt-n-grid requires --decoder cnn')

    # SSL / freeze-backbone validation.
    if args.decoder == 'masked-ae':
        if (rsi_n_grid or rsi_w_grid or cci_n_grid or cci_w_grid
                or vol_n_grid or macd_fast_grid
                or momentum_n_grid or drawdown_n_grid
                or skew_n_grid or kurt_n_grid):
            parser.error('--rsi-/--cci-/--vol-/--macd-/--momentum-/'
                         '--drawdown-/--skew-/--kurt- conditioning grids '
                         'are not supported with --decoder masked-ae (SSL '
                         'has no per-target heads)')
        if args.freeze_backbone is not None:
            parser.error('--freeze-backbone is for the supervised CNN '
                         'probe, not --decoder masked-ae')
        if effective_window_cols <= args.cnn_kernel * args.cnn_layers:
            parser.error(f'--decoder masked-ae needs effective K > '
                         f'cnn_kernel * cnn_layers ({args.cnn_kernel} * '
                         f'{args.cnn_layers}); got {effective_window_cols} '
                         f'(window_cols={args.window_cols}, '
                         f'compress={args.compress})')
        if not (0.0 < args.mask_ratio < 1.0):
            parser.error(f'--mask-ratio must be in (0, 1); got '
                         f'{args.mask_ratio}')
    if args.freeze_backbone is not None:
        if args.decoder != 'cnn':
            parser.error('--freeze-backbone requires --decoder cnn '
                         '(probe protocol trains heads on a frozen backbone)')
        if not Path(args.freeze_backbone).is_file():
            parser.error(f'--freeze-backbone path does not exist: '
                         f'{args.freeze_backbone}')

    load_kwargs = dict(
        stooq_dir=args.stooq_dir, kaggle_dir=args.kaggle_dir,
        use_yahoo=args.yahoo,
        start=args.start, end=args.end,
        scales=scales, lookback=args.lookback,
        window_cols=args.window_cols,
        rsi_n=args.rsi_n, macd_fast=args.macd_fast,
        macd_slow=args.macd_slow, macd_signal=args.macd_signal,
        vol_window=args.vol_window,
        cci_n=args.cci_n,
        momentum_n=args.momentum_n, drawdown_n=args.drawdown_n,
        skew_n=args.skew_n, kurt_n=args.kurt_n,
        rsi_n_grid=rsi_n_grid, rsi_w_grid=rsi_w_grid,
        cci_n_grid=cci_n_grid, cci_w_grid=cci_w_grid,
        vol_n_grid=vol_n_grid, macd_fast_grid=macd_fast_grid,
        momentum_n_grid=momentum_n_grid,
        drawdown_n_grid=drawdown_n_grid,
        skew_n_grid=skew_n_grid,
        kurt_n_grid=kurt_n_grid,
        compression=compression,
    )

    primary = load_ticker(args.ticker, **load_kwargs)
    extra_train = [load_ticker(t, **load_kwargs)
                   for t in _split_tickers(args.train_tickers)]
    train_data = [primary, *extra_train]

    val_data = []
    if args.val_ticker is not None:
        val_data = [load_ticker(args.val_ticker, **load_kwargs)]

    _log_tinygrad_device(args.decoder)

    cnn_channels_per_lag = primary.features.shape[1] // effective_window_cols

    if args.decoder == 'masked-ae':
        _run_ssl(
            args, train_data, val_data,
            cnn_channels_per_lag=cnn_channels_per_lag,
            scales=scales, extra_scales=extra_scales)
        return

    results, params_per_target = fit_and_evaluate(
        train_data, val_data,
        decoder=args.decoder, cnn_channels_per_lag=cnn_channels_per_lag,
        targets=targets,
        mlp_hidden=args.mlp_hidden, mlp_layers=args.mlp_layers,
        mlp_steps=args.mlp_steps, mlp_batch_size=args.mlp_batch_size,
        cnn_hidden=args.cnn_hidden, cnn_kernel=args.cnn_kernel,
        cnn_layers=args.cnn_layers, cnn_steps=args.cnn_steps,
        cnn_batch_size=args.cnn_batch_size,
        cnn_microbatch_size=args.cnn_microbatch_size,
        cnn_film_hidden=args.cnn_film_hidden,
        rsi_n_grid=rsi_n_grid, rsi_w_grid=rsi_w_grid,
        rsi_anchor_n=args.rsi_n, rsi_anchor_w=args.rsi_anchor_w,
        cci_n_grid=cci_n_grid, cci_w_grid=cci_w_grid,
        cci_anchor_n=args.cci_n, cci_anchor_w=args.cci_anchor_w,
        vol_n_grid=vol_n_grid, vol_anchor_n=args.vol_window,
        macd_fast_grid=macd_fast_grid, macd_anchor_fast=args.macd_fast,
        momentum_n_grid=momentum_n_grid, momentum_anchor_n=args.momentum_n,
        drawdown_n_grid=drawdown_n_grid, drawdown_anchor_n=args.drawdown_n,
        skew_n_grid=skew_n_grid, skew_anchor_n=args.skew_n,
        kurt_n_grid=kurt_n_grid, kurt_anchor_n=args.kurt_n,
        frozen_backbone_path=args.freeze_backbone,
        use_bf16=not args.cnn_no_bf16,
    )

    n_features = primary.features.shape[1]
    train_names = ', '.join(d.name for d in train_data)
    if compression is not None:
        compress_str = (f'compress={args.compress}/L{args.compress_levels}/'
                        f'{args.compress_wavelet}, '
                        f'effective_K={effective_window_cols}, ')
    else:
        compress_str = ''
    print(f'train pool: {train_names}  |  '
          f'{sum(d.valid.sum() for d in train_data)} valid rows  |  '
          f'{len(scales)} scales, lookback={args.lookback}, '
          f'window_cols={args.window_cols}, '
          f'{compress_str}'
          f'decoder={args.decoder}, targets={",".join(targets)}, '
          f'n_features={n_features} '
          f'(channels_per_lag={cnn_channels_per_lag})')

    for d in train_data:
        per_target = {n: results[d.name][n]['stats'] for n in targets}
        print(f'\n{d.name} (train, {d.valid.sum()} valid bars):')
        _print_target_stats('train', per_target)
    for d in val_data:
        per_target = {n: results[d.name][n]['stats'] for n in targets}
        print(f'\n{d.name} (zero-shot val, {d.valid.sum()} valid bars):')
        _print_target_stats('val  ', per_target)

    os.makedirs(args.output_dir, exist_ok=True)

    train_tag = _short_train_tag(
        primary.name, [t.name for t in extra_train])

    primary_targets = {n: primary.targets[n] for n in targets}
    primary_recon = {n: results[primary.name][n]['recon'] for n in targets}
    primary_stats = {n: results[primary.name][n]['stats'] for n in targets}
    fig = plot_reconstruction(
        primary.name, primary.dates, primary_targets, primary_recon,
        primary_stats,
        rsi_n=args.rsi_n, macd_fast=args.macd_fast,
        macd_slow=args.macd_slow, macd_signal=args.macd_signal,
        n_features=n_features, decoder=args.decoder,
        window_cols=args.window_cols,
        vol_window=args.vol_window,
        cci_n=args.cci_n,
        momentum_n=args.momentum_n, drawdown_n=args.drawdown_n,
        skew_n=args.skew_n, kurt_n=args.kurt_n)
    fname = Path(args.output_dir) / f'{primary.name}-replay.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fname}')

    for d in val_data:
        v_targets = {n: d.targets[n] for n in targets}
        v_recon = {n: results[d.name][n]['recon'] for n in targets}
        v_stats = {n: results[d.name][n]['stats'] for n in targets}
        title = f'{d.name} (zero-shot from {train_tag})'
        val_fig = plot_reconstruction(
            title, d.dates, v_targets, v_recon, v_stats,
            rsi_n=args.rsi_n, macd_fast=args.macd_fast,
            macd_slow=args.macd_slow, macd_signal=args.macd_signal,
            n_features=n_features, decoder=args.decoder,
            window_cols=args.window_cols,
            vol_window=args.vol_window,
            cci_n=args.cci_n,
            momentum_n=args.momentum_n, drawdown_n=args.drawdown_n,
            skew_n=args.skew_n, kurt_n=args.kurt_n)
        val_fname = (Path(args.output_dir) /
                     f'{d.name}-replay-zeroshot-from-{train_tag}.png')
        val_fig.savefig(val_fname, dpi=150)
        plt.close(val_fig)
        print(f'Saved {val_fname}')

    sha, dirty = _git_sha_and_dirty()
    sha_tag = sha + ('-dirty' if dirty else '')
    targets_tag = '+'.join(targets)
    weights_arrays: dict[str, np.ndarray] = {}
    for target_name, params in params_per_target.items():
        for key, arr in params.items():
            weights_arrays[f'{target_name}__{key}'] = arr
    metadata = {
        'train_tickers': [d.name for d in train_data],
        'val_tickers': [d.name for d in val_data],
        'targets': list(targets),
        'decoder': args.decoder,
        'window_cols': args.window_cols,
        'effective_window_cols': effective_window_cols,
        'compress': args.compress,
        'compress_levels': args.compress_levels,
        'compress_wavelet': args.compress_wavelet,
        'compress_pad_mode': args.compress_pad_mode,
        'lookback': args.lookback,
        'scales': scales,
        'extra_high_freq_scales': sorted(extra_scales),
        'n_features': n_features,
        'rsi_n': args.rsi_n,
        'rsi_n_grid': list(rsi_n_grid),
        'rsi_w_grid': list(rsi_w_grid),
        'rsi_anchor_w': args.rsi_anchor_w,
        'cci_n': args.cci_n,
        'cci_n_grid': list(cci_n_grid),
        'cci_w_grid': list(cci_w_grid),
        'cci_anchor_w': args.cci_anchor_w,
        'vol_n_grid': list(vol_n_grid),
        'macd_fast_grid': list(macd_fast_grid),
        'momentum_n': args.momentum_n,
        'momentum_n_grid': list(momentum_n_grid),
        'drawdown_n': args.drawdown_n,
        'drawdown_n_grid': list(drawdown_n_grid),
        'skew_n': args.skew_n,
        'skew_n_grid': list(skew_n_grid),
        'kurt_n': args.kurt_n,
        'kurt_n_grid': list(kurt_n_grid),
        'macd_fast': args.macd_fast,
        'macd_slow': args.macd_slow,
        'macd_signal': args.macd_signal,
        'vol_window': args.vol_window,
        'mlp_hidden': args.mlp_hidden,
        'mlp_layers': args.mlp_layers,
        'mlp_steps': args.mlp_steps,
        'mlp_batch_size': args.mlp_batch_size,
        'cnn_hidden': args.cnn_hidden,
        'cnn_kernel': args.cnn_kernel,
        'cnn_layers': args.cnn_layers,
        'cnn_steps': args.cnn_steps,
        'cnn_batch_size': args.cnn_batch_size,
        'cnn_film_hidden': args.cnn_film_hidden,
        'freeze_backbone': args.freeze_backbone,
        'start': args.start,
        'end': args.end,
        'git_sha': sha,
        'git_dirty': dirty,
        'train_stats': {
            d.name: {
                t: results[d.name][t]['stats'] for t in targets
            } for d in train_data
        },
        'val_stats': {
            d.name: {
                t: results[d.name][t]['stats'] for t in targets
            } for d in val_data
        },
    }
    weights_arrays['_meta'] = np.array(json.dumps(metadata, default=float))
    weights_fname = (Path(args.output_dir) /
                     f'{train_tag}-{targets_tag}-{args.decoder}-'
                     f'{sha_tag}.npz')
    np.savez(weights_fname, **weights_arrays)
    print(f'Saved {weights_fname}')


if __name__ == '__main__':
    main()
