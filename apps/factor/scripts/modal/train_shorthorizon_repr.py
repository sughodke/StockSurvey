"""Modal walk-forward: short-horizon × fixed (C,L) representation sweep.

Pre-registered experiment
(`apps/docs/docs/TODO/factor-shorthorizon-representation.md`):
sweep `encoder ∈ {indicator, spectral, minirocket}` ×
`rebal_days ∈ {5, 10, 20}` on the `factor-narrow` universe (297 tickers,
`min_history_bars=6500`, ~2000→2026 — the exact operating condition of
the `+0.0120 ; +0.440` baseline leaderboard row).

`encoder=indicator` at `rebal_days=20` reproduces that baseline row and
binds the whole sweep to the leaderboard. The indicator arm at *every*
horizon is the confound control: it isolates "short horizon moved the
signal" from "the encoder moved it".

Windowing — **year-comparable block scaling.** `rebal_days` is both the
rebal cadence and the forward-return horizon, and windows are counted in
blocks (= `rebal_days` bars). To hold the train/val *calendar* spans
(and window count ≈ 6) fixed across horizons, block counts scale
inversely with `rebal_days` from the 20-day anchor (63/39/39) — the
convention the `6-window factor (q)` row established:

  rebal_days=20 → 63/39/39   (anchor; reproduces baseline)
  rebal_days=10 → 126/78/78
  rebal_days=5  → 252/156/156

Metric guardrail: the decision metric is **mean val IC** (commission-
free). Sharpe is recorded for every cell but is only comparable
*within* a horizon — at rebal_days=5 commission drag is ~4× heavier
than 20d so cross-horizon Sharpe is mechanically confounded (cf. the
`2026-05-04` quarterly `reversed-OOS` row).

Usage
-----
Smoke (~1-2 min wall, <$0.02):
    uvx modal run apps/factor/scripts/modal/train_shorthorizon_repr.py \\
        --encoders spectral --rebal-days-grid 20 \\
        --max-tickers 30 --n-steps 30

Full pre-registered Phase-1 sweep (9 cells, ~1-1.5h T4, <$1):
    uvx modal run apps/factor/scripts/modal/train_shorthorizon_repr.py \\
        --encoders indicator,spectral,minirocket \\
        --rebal-days-grid 20,10,5

Returns one `sh-{encoder}-r{rebal}-windows.npz` per cell plus
`shorthorizon-summary.json` and `shorthorizon-comparison.png`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'
STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_us_long'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'

# Year-comparable block scaling: base 63/39/39 at rebal_days=20.
BASE_BLOCKS = (63, 39, 39)


def _scaled_blocks(rebal_days: int) -> tuple[int, int, int]:
    f = 20.0 / rebal_days
    return tuple(int(round(b * f)) for b in BASE_BLOCKS)  # type: ignore[return-value]


image = (
    modal.Image.from_registry(
        'nvidia/cuda:12.4.0-devel-ubuntu22.04',
        add_python='3.12',
    )
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**', '.venv/**', 'Output/**', 'StooqData/**',
            'Nasdaq3347/**',
            'apps/relational/src/**', 'apps/regime/src/**',
            'apps/v1/src/**', 'apps/replay/src/**',
            '**/__pycache__/**', '**/*.pyc',
        ],
    )
)

app = modal.App('factor-shorthorizon-repr', image=image)


def _build_one_ticker(args):
    """Worker: load prices + build the chosen encoder's (C,L) stack.

    Top-level for `mp.Pool` picklability. Imports inside the worker so
    the module imports cheaply on the local side.
    """
    ticker, stooq_subset, encoder, cfg, start, end = args
    import numpy as np
    from ss_features import TickerData, load_prices
    try:
        series = load_prices(ticker, stooq_dir=stooq_subset,
                             start=start, end=end)
        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        if encoder == 'indicator':
            from factor import build_indicator_features
            feats, valid = build_indicator_features(prices, cfg)
        elif encoder == 'spectral':
            from factor import build_spectral_features
            feats, valid = build_spectral_features(prices, cfg)
        elif encoder == 'minirocket':
            from factor import build_minirocket_features
            feats, valid = build_minirocket_features(prices, cfg)
        else:
            return ticker, f'(unknown encoder {encoder!r})'
        if not valid.any():
            return ticker, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid)
    except Exception as e:
        return ticker, f'({type(e).__name__}: {e})'


def _resolve_ticker_list(tickers, max_tickers, min_history_bars):
    manifest = json.loads((Path(STOOQ_SUBSET) / 'manifest.json').read_text())
    if tickers:
        want = {t.strip().upper() for t in tickers.split(',') if t.strip()}
        entries = [t for t in manifest['tickers']
                   if t['ticker'].upper() in want]
    else:
        entries = list(manifest['tickers'])
    if min_history_bars > 0:
        before = len(entries)
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
        print(f'  min_history_bars={min_history_bars}: dropped '
              f'{before - len(entries)} short-history tickers')
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
    return names


@app.function(gpu='T4', cpu=4, memory=16384, timeout=3 * 60 * 60)
def sweep(
    encoders: str, rebal_days_grid: str, tickers: str, start: str, end: str,
    scorer: str, n_steps: int, learning_rate: float, weight_decay: float,
    max_tickers: int, min_history_bars: int,
) -> dict[str, bytes]:
    import os
    import subprocess
    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')
    os.environ['CUDA'] = '1'

    print('=== uv sync workspace deps ===', flush=True)
    subprocess.run(['uv', 'sync', '--package', 'factor', '--inexact'],
                   cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(
            f'tinygrad Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    import multiprocessing as mp
    from factor import (
        IndicatorGridConfig, MiniRocketGridConfig, SpectralGridConfig,
        train_scorer_indicators_walkforward,
        train_scorer_minirocket_walkforward,
        train_scorer_spectral_walkforward,
    )

    enc_list = [e.strip() for e in encoders.split(',') if e.strip()]
    rebal_list = [int(r) for r in rebal_days_grid.split(',') if r.strip()]
    ticker_list = _resolve_ticker_list(tickers, max_tickers, min_history_bars)
    print(f'  universe: {len(ticker_list)} tickers', flush=True)
    n_workers = max(1, int(os.environ.get('FACTOR_FEATURE_WORKERS',
                                          os.cpu_count() or 4)))

    cfg_for = {
        'indicator': IndicatorGridConfig(),
        'spectral': SpectralGridConfig(),
        'minirocket': MiniRocketGridConfig(),
    }
    wf_for = {
        'indicator': train_scorer_indicators_walkforward,
        'spectral': train_scorer_spectral_walkforward,
        'minirocket': train_scorer_minirocket_walkforward,
    }

    summary: list[dict] = []
    for encoder in enc_list:
        cfg = cfg_for[encoder]
        print(f'\n=== encoder={encoder}  width={cfg.feature_width()} '
              f'=== build features', flush=True)
        t0 = time.perf_counter()
        work = [(t, STOOQ_SUBSET, encoder, cfg, start, end)
                for t in ticker_list]
        ticker_data, skipped = [], []
        with mp.Pool(n_workers) as pool:
            for i, (tk, res) in enumerate(
                    pool.imap_unordered(_build_one_ticker, work)):
                (ticker_data if not isinstance(res, str) else skipped).append(
                    res if not isinstance(res, str) else f'{tk} {res}')
                if (i + 1) % 50 == 0:
                    print(f'  built {i + 1}/{len(work)} '
                          f'({time.perf_counter()-t0:.0f}s)', flush=True)
        ticker_data.sort(key=lambda td: td.name)
        print(f'  {len(ticker_data)} usable / {len(skipped)} skipped '
              f'({time.perf_counter()-t0:.0f}s)', flush=True)
        if len(ticker_data) < 4:
            summary.append({'encoder': encoder, 'failed': True,
                            'error': f'only {len(ticker_data)} tickers'})
            continue

        for rebal in rebal_list:
            tr, va, st = _scaled_blocks(rebal)
            cell = f'sh-{encoder}-r{rebal}'
            print(f'\n  >>> {cell}  blocks {tr}/{va}/{st}', flush=True)
            t1 = time.perf_counter()
            try:
                wf = wf_for[encoder](
                    ticker_data, cfg, rebal_days=rebal,
                    train_window_blocks=tr, val_window_blocks=va,
                    step_window_blocks=st, scorer=scorer, n_steps=n_steps,
                    learning_rate=learning_rate, weight_decay=weight_decay,
                    verbose=True)
            except Exception as e:
                print(f'    FAILED: {type(e).__name__}: {e}', flush=True)
                summary.append({'cell': cell, 'encoder': encoder,
                                'rebal_days': rebal, 'failed': True,
                                'error': f'{type(e).__name__}: {e}'})
                continue
            wall = time.perf_counter() - t1
            _save_npz(output / f'{cell}-windows.npz', wf, cell)
            per_window = [
                {'window_idx': w.window_idx, 'val_ic': w.val_ic,
                 'train_ic': w.train_ic, 'val_sharpe': w.val_sharpe,
                 'train_sharpe': w.train_sharpe} for w in wf.windows]
            summary.append({
                'cell': cell, 'encoder': encoder, 'rebal_days': rebal,
                'blocks': [tr, va, st], 'n_windows': wf.n_windows,
                'mean_val_ic': wf.mean_val_ic,
                'median_val_ic': wf.median_val_ic,
                'mean_val_sharpe': wf.mean_val_sharpe,
                'positive_val_ic_fraction': wf.positive_val_ic_fraction,
                'wall_seconds': round(wall, 1), 'windows': per_window,
                'failed': False})
            print(f'    {wf.n_windows} win  mean val IC='
                  f'{wf.mean_val_ic:+.4f}  mean val Sh='
                  f'{wf.mean_val_sharpe:+.3f}  pos-IC frac='
                  f'{wf.positive_val_ic_fraction:.2f}  wall={wall:.0f}s',
                  flush=True)

    (output / 'shorthorizon-summary.json').write_text(json.dumps({
        'universe_size_requested': len(ticker_list),
        'start': start, 'end': end, 'scorer': scorer, 'n_steps': n_steps,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'min_history_bars': min_history_bars, 'cells': summary}, indent=2))
    _plot(summary, output / 'shorthorizon-comparison.png')

    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name.startswith(('sh-', 'shorthorizon-')):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


def _save_npz(path: Path, wf, cell: str) -> None:
    import numpy as np
    blob = {
        'window_idx': np.array([w.window_idx for w in wf.windows], np.int32),
        'val_ic': np.array([w.val_ic for w in wf.windows], np.float32),
        'train_ic': np.array([w.train_ic for w in wf.windows], np.float32),
        'val_sharpe': np.array([w.val_sharpe for w in wf.windows], np.float32),
        'train_sharpe': np.array(
            [w.train_sharpe for w in wf.windows], np.float32),
        '_summary': np.array(json.dumps({
            'cell': cell, 'scorer': wf.scorer, 'n_steps': wf.n_steps,
            'rebal_days': wf.rebal_days, 'n_windows': wf.n_windows,
            'mean_val_ic': wf.mean_val_ic,
            'mean_val_sharpe': wf.mean_val_sharpe,
            'positive_val_ic_fraction': wf.positive_val_ic_fraction})),
    }
    np.savez(path, **blob)
    print(f'    -> {path.name}', flush=True)


def _plot(summary, out_path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ok = [c for c in summary if not c.get('failed') and 'cell' in c]
    if not ok:
        print('  no cells to plot'); return
    ok.sort(key=lambda c: (c['encoder'], c['rebal_days']))
    labels = [c['cell'] for c in ok]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(8, len(ok) * 0.7), 7),
                                 sharex=True)
    a1.bar(range(len(ok)), [c['mean_val_ic'] for c in ok])
    a1.axhline(0, color='k', lw=0.5)
    a1.axhline(0.0120, color='green', ls='--', lw=0.8,
               label='20d indicator baseline +0.0120')
    a1.set_ylabel('mean val IC'); a1.legend(fontsize=8)
    a1.set_title('Short-horizon × (C,L) representation — IC is the decision metric')
    a2.bar(range(len(ok)), [c['mean_val_sharpe'] for c in ok], color='gray')
    a2.axhline(0, color='k', lw=0.5)
    a2.set_ylabel('mean val Sharpe\n(within-horizon only)')
    a2.set_xticks(range(len(ok)))
    a2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f'  -> {out_path.name}', flush=True)


@app.local_entrypoint()
def main(
    encoders: str = 'indicator,spectral,minirocket',
    rebal_days_grid: str = '20,10,5',
    tickers: str = '',
    start: str = '2000-01-01',
    end: str = '2026-04-01',
    scorer: str = 'linear',
    n_steps: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    max_tickers: int = 0,
    min_history_bars: int = 6500,
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'launching factor-shorthorizon-repr (encoders={encoders}, '
          f'rebal_days={rebal_days_grid}, max_tickers={max_tickers})')
    artifacts = sweep.remote(
        encoders=encoders, rebal_days_grid=rebal_days_grid, tickers=tickers,
        start=start, end=end, scorer=scorer, n_steps=n_steps,
        learning_rate=learning_rate, weight_decay=weight_decay,
        max_tickers=max_tickers, min_history_bars=min_history_bars)
    for name, data in artifacts.items():
        (LOCAL_OUTPUT_DIR / name).write_bytes(data)
        print(f'  wrote {LOCAL_OUTPUT_DIR / name} ({len(data)//1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
