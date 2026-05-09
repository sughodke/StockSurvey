"""Pure-CWT featurizer head-to-head against the IndicatorGridConfig
forecast probe.

Question: when we said "the indicators forecast and the CWT doesn't,"
the CWT comparison was actually at 30 tickers / rebal=5d
(apps/docs/docs/notes.md "What we already know about supervision being
the binding constraint"), not at the 297 / 20d setup
where indicators show val IC +0.0120 (returns) and +0.4743 (vol). Those
aren't matched. This driver re-runs the CWT side at the matched setup.

Featurizer is the **pure** CWT bundle: per-bar a stack of (coeffs,
power) per scale, lag-windowed over the trailing `window_cols` bars.
**No price**, **no close**, **no raw / log returns**, **no rolling
z-norm stats** — those would all be direct or near-direct leakage of
forward-return-correlated information into a forward-return-target
loss. The `coeffs` themselves come out of `causal_cwt`, which z-norms
prices over the lookback before the Ricker convolution, so absolute
price level is already stripped at the wavelet stage.

  channels per lag  C = 2 * n_scales
  feature row        = (window_cols, C) flattened to (window_cols * C,)
  identity backbone  = K = window_cols, F = C   (z-norm + flatten only)

Default `window_cols=1` — point-in-time CWT (no extra lag history)
to match `IndicatorGridConfig`'s K=1 framing exactly. Both featurizers
then compress time history into per-bar scalars; the comparison
isolates the basis (Ricker wavelets at multiple scales vs RSI/MACD/CCI/
vol). Larger K is feasible at cost of memory — `(D, N, K, F)` aligned
tensor is ~19 GB at K=96 / F=26 / 297 tickers, which OOMs a 32GB Mac.
K=1 stays under 200 MB.

Two arms, mirroring `forecast_probe_walkforward.py`:
  * `log_return`     — direct comparison vs the +0.0120 indicator baseline.
  * `vol_innovation` — direct comparison vs the +0.4743 indicator vol arm.

Same 297-ticker stooq_us_long universe, rebal=20d, linear head,
n_steps=200, AdamW lr=1e-2 wd=1e-3, 6 walk-forward windows. Only the
featurizer differs.

Run from the repo root:
    uv run python apps/factor/scripts/cwt_bundle_walkforward.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _resolve_ticker_list(min_history_bars: int, max_tickers: int) -> list[str]:
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    if min_history_bars > 0:
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} tickers; {len(entries)} pass '
          f'min_history_bars={min_history_bars}')
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers} for smoke run')
    return names


def _build_one_ticker(args):
    """Pure CWT bundle, no price / return / zscore-stat channels."""
    ticker, stooq_dir, scales, lookback, window_cols, start, end = args
    try:
        import numpy as _np
        from ss_features import (
            TickerData, load_prices, compute_scalogram, build_lagged_features,
        )
        series = load_prices(ticker, stooq_dir=stooq_dir, start=start, end=end)
        prices = series.values.astype(_np.float64)
        dates = _np.asarray(series.index)
        coeffs, power = compute_scalogram(prices, scales, lookback=lookback)
        # `channels_cn` shape (C, n_dates) with C = 2 * n_scales: [coeffs ; power].
        # No mu/std (price level proxy), no raw returns, no return sign.
        channels_cn = _np.vstack([
            coeffs.astype(_np.float32),
            power.astype(_np.float32),
        ])
        feats = build_lagged_features(channels_cn, window_cols)
        # Valid = warm-up complete AND every channel finite.
        valid = _np.zeros(len(prices), dtype=bool)
        valid[max(lookback, window_cols - 1):] = True
        valid &= _np.isfinite(feats).all(axis=1)
        if not valid.any():
            return ticker, None, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ), None
    except Exception as e:
        return ticker, None, f'({type(e).__name__}: {e})'


def _summarize_arm(label: str, wf, save_path: Path) -> dict:
    print(f'\n=== {label} (target={wf.forward_target_kind}, F={wf.feature_width}) ===')
    print(f'{"win":>3}  {"train_ic":>9}  {"val_ic":>9}  {"train_sh":>9}  '
          f'{"val_sh":>9}  {"n_train":>7}  {"n_val":>5}')
    for w in wf.windows:
        print(f'{w.window_idx:>3}  {w.train_ic:+.4f}    {w.val_ic:+.4f}    '
              f'{w.train_sharpe:+.3f}    {w.val_sharpe:+.3f}     '
              f'{w.n_train_bars:>7d}  {w.n_val_bars:>5d}')
    print(f'mean val IC    : {wf.mean_val_ic:+.4f}')
    print(f'median val IC  : {wf.median_val_ic:+.4f}')
    print(f'mean val Sharpe: {wf.mean_val_sharpe:+.3f}')
    print(f'pos val IC frac: {wf.positive_val_ic_fraction:.2f}  '
          f'({sum(1 for w in wf.windows if w.val_ic > 0)}/{wf.n_windows})')

    blob: dict[str, np.ndarray] = {
        'window_idx':  np.array([w.window_idx for w in wf.windows]),
        'train_ic':    np.array([w.train_ic for w in wf.windows]),
        'val_ic':      np.array([w.val_ic for w in wf.windows]),
        'train_sharpe': np.array([w.train_sharpe for w in wf.windows]),
        'val_sharpe':  np.array([w.val_sharpe for w in wf.windows]),
    }
    blob['_meta'] = np.array(json.dumps({
        'forward_target_kind': wf.forward_target_kind,
        'scorer': wf.scorer, 'n_steps': wf.n_steps,
        'feature_width': wf.feature_width,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
    }))
    np.savez(save_path, **blob)
    print(f'-> {save_path.name}')

    return {
        'arm':             label,
        'target':          wf.forward_target_kind,
        'feature_width':   wf.feature_width,
        'mean_val_ic':     wf.mean_val_ic,
        'median_val_ic':   wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',  default='2000-01-01')
    p.add_argument('--end',    default='2026-04-01')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--train-window-blocks', type=int, default=63)
    p.add_argument('--val-window-blocks',   type=int, default=39)
    p.add_argument('--step-window-blocks',  type=int, default=39)
    p.add_argument('--n-steps',       type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=1e-2)
    p.add_argument('--weight-decay',  type=float, default=1e-3)
    p.add_argument('--scorer',        default='linear')
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers',      type=int, default=0)
    p.add_argument('--n-workers',     type=int, default=mp.cpu_count())
    p.add_argument('--window-cols',  type=int, default=1,
                   help='K: lag-window size. Default 1 = point-in-time CWT '
                        '(matches IndicatorGridConfig K=1 framing). Larger '
                        'K adds lag history but scales memory as K*N*D*F*4 — '
                        'K>32 OOMs at 297 tickers / 26 channels.')
    p.add_argument('--lookback',     type=int, default=252)
    p.add_argument('--targets', default='log_return,vol_innovation',
                   help='comma-separated subset of '
                        '{log_return,sign_demeaned,vol_innovation}')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    valid_targets = {'log_return', 'sign_demeaned', 'vol_innovation'}
    if not set(targets).issubset(valid_targets):
        raise SystemExit(
            f'--targets must be subset of {sorted(valid_targets)}; got {targets}')

    from factor import (
        compute_input_stats, identity_backbone, train_scorer_walkforward,
    )
    from ss_wavelets import ALL_SCALES

    scales = list(ALL_SCALES)
    n_scales = len(scales)
    F = 2 * n_scales            # coeffs + power per scale, no extras
    K = args.window_cols
    print(f'pure-CWT bundle: scales={scales} (n={n_scales}), '
          f'lookback={args.lookback}, window_cols={K}, '
          f'channels_per_lag={F}, hidden_flat={K * F}')
    print(f'  channels: [coeffs] x {n_scales} + [power] x {n_scales}  '
          f'(NO price, NO return, NO log return, NO zscore stats)')

    names = _resolve_ticker_list(args.min_history_bars, args.max_tickers)

    print(f'\nbuilding pure-CWT features over {len(names)} tickers '
          f'(workers={args.n_workers}) ...')
    t0 = time.perf_counter()
    pool_args = [
        (n, str(STOOQ_SUBSET), scales, args.lookback, K, args.start, args.end)
        for n in names
    ]
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            results = pool.map(_build_one_ticker, pool_args)
    else:
        results = [_build_one_ticker(a) for a in pool_args]
    ticker_data = []
    failed = []
    for name, td, err in results:
        if td is None:
            failed.append((name, err))
        else:
            ticker_data.append(td)
    print(f'built {len(ticker_data)} / {len(names)} tickers in '
          f'{time.perf_counter() - t0:.1f}s')
    if failed:
        print(f'  failed: {len(failed)}')
        for name, err in failed[:10]:
            print(f'    {name}: {err}')

    # Identity backbone over (K, F) — z-norm + flatten, no learned weights.
    mu, sd = compute_input_stats(ticker_data, K=K, F=F)
    backbone = identity_backbone(K=K, F=F, feat_mu=mu, feat_sd=sd)
    print(f'identity_backbone(K={K}, F={F}, hidden_flat={backbone.hidden_flat})')

    label_for: dict[str, str] = {
        'log_return':     'cwt-return',
        'sign_demeaned':  'cwt-sign',
        'vol_innovation': 'cwt-vol',
    }

    summary_arms: list[dict] = []
    for target_kind in targets:
        label = label_for[target_kind]
        save_path = output / f'cwt-bundle-{label}-windows.npz'
        print(f'\n--- arm: {label} (forward_target_kind={target_kind}) ---')
        t1 = time.perf_counter()
        wf = train_scorer_walkforward(
            ticker_data, backbone,
            rebal_days=args.rebal_days,
            train_window_blocks=args.train_window_blocks,
            val_window_blocks=args.val_window_blocks,
            step_window_blocks=args.step_window_blocks,
            scorer=args.scorer,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            forward_target_kind=target_kind,
        )
        print(f'arm wall: {time.perf_counter() - t1:.1f}s')
        summary_arms.append(_summarize_arm(label, wf, save_path))

    # Side-by-side vs documented IndicatorGridConfig baselines.
    print('\n' + '=' * 96)
    print('Pure-CWT bundle leaderboard — vs IndicatorGridConfig baselines '
          '(297 tickers, rebal=20d, linear)')
    print('=' * 96)
    print(f'{"arm":<22} {"target":<18} {"mean_ic":>10} '
          f'{"median_ic":>10} {"mean_sh":>10} {"posfrac":>10}')
    for a in summary_arms:
        print(f'{a["arm"]:<22} {a["target"]:<18} '
              f'{a["mean_val_ic"]:>+10.4f} {a["median_val_ic"]:>+10.4f} '
              f'{a["mean_val_sharpe"]:>+10.3f} '
              f'{a["positive_val_ic_fraction"]:>10.2f}')
    print()
    print('IndicatorGridConfig reference (NOTES 2026-05-04, factor commits '
          'def3ac9 / ff90762):')
    print(f'{"indicator-control":<22} {"log_return":<18} '
          f'{"+0.0120":>10} {"+0.0168":>10} {"+0.440":>10} {"0.83":>10}')
    print(f'{"indicator-vol":<22} {"vol_innovation":<18} '
          f'{"+0.4743":>10} {"+0.4735":>10} {"+0.515":>10} {"1.00":>10}')

    summary_path = output / 'cwt-bundle-summary.json'
    summary_path.write_text(json.dumps({
        'universe_size': len(ticker_data),
        'feature_width': F, 'window_cols': K, 'hidden_flat': K * F,
        'scales': list(map(int, scales)),
        'rebal_days': args.rebal_days,
        'train_window_blocks': args.train_window_blocks,
        'val_window_blocks': args.val_window_blocks,
        'step_window_blocks': args.step_window_blocks,
        'scorer': args.scorer, 'n_steps': args.n_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'arms': summary_arms,
        'reference_indicator_baseline': {
            'log_return':     {'mean_val_ic': 0.0120, 'mean_val_sharpe': 0.440},
            'vol_innovation': {'mean_val_ic': 0.4743, 'mean_val_sharpe': 0.515},
        },
    }, indent=2))
    print(f'\n-> {summary_path}')


if __name__ == '__main__':
    main()
