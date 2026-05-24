"""Modal entrypoint: head-to-head factor head training across three
losses on identical 5d walk-forward windows.

Per TODO/factor-studentized-sharpe-diff-loss.md (committed at fdab384
BEFORE this eval runs). The local version at
`apps/factor/scripts/train_studentized_sharpe_diff.py` crashed mid-run
on the Intel-Mac laptop; this Modal port runs the same eval on a T4
in ~5-15 min wall.

The eval is unchanged: factor-narrow universe, rebal_days=5,
forward_skip=1, n_steps=200, three loss arms:
  - reference: rank_ic
  - baseline:  ir_vs_ew
  - candidate: studentized_sharpe_diff_vs_ew
plus identical seed (42), train/val/step blocks (200/100/100).

Per-arc pooled OOS bootstrap-CI of ΔSR-vs-EW is computed locally on
the host after artifacts return — keeps the Modal function clean of
bootstrap noise.

Usage
-----
Smoke (~1-2 min wall, <$0.05):
    uvx modal run apps/factor/scripts/modal/train_studentized_sharpe_diff.py \\
        --max-tickers 30 --n-steps 30

Full pre-registered run (~5-15 min wall, <$0.20):
    uvx modal run apps/factor/scripts/modal/train_studentized_sharpe_diff.py

Returns `factor-studentized-sharpe-diff-{rank_ic,ir_vs_ew,
studentized_sharpe_diff_vs_ew}-windows.npz` per arm plus
`factor-studentized-sharpe-diff-summary.json` (which the local
post-process consumes to compute the bootstrap-CI verdict per
locked pre-reg bar).
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
            '.git/**', '.venv/**', '.claude/**',
            'Output/**', 'StooqData/**', 'Nasdaq3347/**',
            'apps/relational/src/**', 'apps/regime/src/**',
            'apps/v1/src/**', 'apps/replay/src/**',
            '**/__pycache__/**', '**/*.pyc',
        ],
    )
)

app = modal.App('factor-studentized-sharpe-diff', image=image)


def _build_one_ticker(args):
    """Worker — load prices + build the IndicatorGridConfig (C,L) stack."""
    ticker, stooq_subset, start, end = args
    import numpy as np
    from factor import IndicatorGridConfig, build_indicator_features
    from ss_features import TickerData, load_prices
    try:
        series = load_prices(ticker, stooq_dir=stooq_subset,
                             start=start, end=end)
        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        cfg = IndicatorGridConfig()
        feats, valid = build_indicator_features(prices, cfg)
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


@app.function(gpu='T4', cpu=4, memory=16384, timeout=2 * 60 * 60)
def head_to_head(
    tickers: str, start: str, end: str,
    rebal_days: int, forward_skip: int,
    train_window_blocks: int, val_window_blocks: int, step_window_blocks: int,
    scorer: str, n_steps: int,
    learning_rate: float, weight_decay: float, commission_bps: float,
    max_tickers: int, min_history_bars: int, seed: int,
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
    import numpy as np
    from factor import (
        IndicatorGridConfig, train_scorer_indicators_walkforward,
    )

    cfg = IndicatorGridConfig()
    ticker_list = _resolve_ticker_list(tickers, max_tickers, min_history_bars)
    print(f'  universe: {len(ticker_list)} tickers', flush=True)
    n_workers = max(1, int(os.environ.get('FACTOR_FEATURE_WORKERS',
                                          os.cpu_count() or 4)))

    print(f'\n=== building TickerData panel (n_workers={n_workers}) ===',
          flush=True)
    t0 = time.perf_counter()
    work = [(t, STOOQ_SUBSET, start, end) for t in ticker_list]
    ticker_data, skipped = [], []
    with mp.Pool(n_workers) as pool:
        for i, (tk, res) in enumerate(
                pool.imap_unordered(_build_one_ticker, work)):
            if isinstance(res, str):
                skipped.append(f'{tk} {res}')
            else:
                ticker_data.append(res)
            if (i + 1) % 50 == 0:
                print(f'  built {i + 1}/{len(work)} '
                      f'({time.perf_counter()-t0:.0f}s)', flush=True)
    ticker_data.sort(key=lambda td: td.name)
    print(f'  {len(ticker_data)} usable / {len(skipped)} skipped '
          f'({time.perf_counter()-t0:.0f}s)', flush=True)

    arms = [
        ('rank_ic',                       'reference'),
        ('ir_vs_ew',                      'baseline'),
        ('studentized_sharpe_diff_vs_ew', 'candidate'),
    ]
    summary: list[dict] = []
    for loss_kind, role in arms:
        cell = f'factor-stud-sh-diff-{loss_kind}'
        print(f'\n{"=" * 70}\n{role.upper()}: loss_kind={loss_kind}\n{"=" * 70}',
              flush=True)
        t0 = time.perf_counter()
        wf = train_scorer_indicators_walkforward(
            ticker_data, cfg,
            rebal_days=rebal_days, forward_skip=forward_skip,
            train_window_blocks=train_window_blocks,
            val_window_blocks=val_window_blocks,
            step_window_blocks=step_window_blocks,
            scorer=scorer, loss_kind=loss_kind, n_steps=n_steps,
            learning_rate=learning_rate, weight_decay=weight_decay,
            commission_bps=commission_bps, seed=seed, verbose=True,
        )
        wall = time.perf_counter() - t0

        # Persist per-window OOS streams and the EW reference per window
        # so the local bootstrap-CI step can run cleanly. The
        # walk-forward result already has oos_block_returns; we don't
        # have a stored EW stream, so reconstruct it per window from
        # the panel's block_log_ret + mask on the val slice.
        per_window_payload = []
        for w in wf.windows:
            per_window_payload.append({
                'window_idx': int(w.window_idx),
                'val_ic': float(w.val_ic),
                'train_ic': float(w.train_ic),
                'val_sharpe': float(w.val_sharpe),
                'train_sharpe': float(w.train_sharpe),
                'n_val_bars': int(w.n_val_bars),
            })

        blob = {
            'window_idx': np.array([w.window_idx for w in wf.windows], np.int32),
            'val_ic': np.array([w.val_ic for w in wf.windows], np.float32),
            'train_ic': np.array([w.train_ic for w in wf.windows], np.float32),
            'val_sharpe': np.array([w.val_sharpe for w in wf.windows], np.float32),
            'train_sharpe': np.array(
                [w.train_sharpe for w in wf.windows], np.float32),
            'oos_block_returns': np.asarray(wf.oos_block_returns, np.float64),
            'oos_block_returns_long_short':
                np.asarray(wf.oos_block_returns_long_short, np.float64),
            'periods_per_year': np.float64(252.0 / wf.rebal_days),
            '_summary': np.array(json.dumps({
                'cell': cell, 'loss_kind': loss_kind, 'role': role,
                'scorer': wf.scorer, 'n_steps': wf.n_steps,
                'rebal_days': wf.rebal_days, 'forward_skip': forward_skip,
                'n_windows': wf.n_windows,
                'mean_val_ic': wf.mean_val_ic,
                'mean_val_sharpe': wf.mean_val_sharpe,
                'positive_val_ic_fraction': wf.positive_val_ic_fraction,
                'commission_bps': commission_bps,
                'seed': seed,
                'wall_seconds': round(wall, 1),
            })),
        }
        np.savez(output / f'{cell}-windows.npz', **blob)

        summary.append({
            'cell': cell, 'loss_kind': loss_kind, 'role': role,
            'n_windows': wf.n_windows,
            'mean_val_ic': float(wf.mean_val_ic),
            'mean_val_sharpe': float(wf.mean_val_sharpe),
            'positive_val_ic_fraction': float(wf.positive_val_ic_fraction),
            'wall_seconds': round(wall, 1),
            'windows': per_window_payload,
        })
        print(f'  {wf.n_windows} win  mean val IC={wf.mean_val_ic:+.4f}  '
              f'mean val Sh={wf.mean_val_sharpe:+.3f}  '
              f'pos-IC frac={wf.positive_val_ic_fraction:.2f}  '
              f'wall={wall:.0f}s', flush=True)

    (output / 'factor-studentized-sharpe-diff-summary.json').write_text(
        json.dumps({
            'pre_reg': 'apps/docs/docs/TODO/factor-studentized-sharpe-diff-loss.md',
            'commit_pre_reg': 'fdab384',
            'n_tickers': len(ticker_data),
            'rebal_days': rebal_days, 'forward_skip': forward_skip,
            'train_window_blocks': train_window_blocks,
            'val_window_blocks': val_window_blocks,
            'step_window_blocks': step_window_blocks,
            'n_steps': n_steps,
            'commission_bps': commission_bps, 'seed': seed,
            'arms': summary,
        }, indent=2))

    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and (
                p.name.startswith('factor-stud-sh-diff-')
                or p.name == 'factor-studentized-sharpe-diff-summary.json'):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def main(
    tickers: str = '',
    start: str = '2000-01-01',
    end: str = '2026-04-01',
    rebal_days: int = 5,
    forward_skip: int = 1,
    train_window_blocks: int = 200,
    val_window_blocks: int = 100,
    step_window_blocks: int = 100,
    scorer: str = 'linear',
    n_steps: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    commission_bps: float = 10.0,
    max_tickers: int = 0,
    min_history_bars: int = 6500,
    seed: int = 42,
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'launching factor-studentized-sharpe-diff '
          f'(rebal={rebal_days}, skip={forward_skip}, '
          f'blocks={train_window_blocks}/{val_window_blocks}/{step_window_blocks}, '
          f'n_steps={n_steps}, max_tickers={max_tickers})')
    artifacts = head_to_head.remote(
        tickers=tickers, start=start, end=end,
        rebal_days=rebal_days, forward_skip=forward_skip,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        scorer=scorer, n_steps=n_steps,
        learning_rate=learning_rate, weight_decay=weight_decay,
        commission_bps=commission_bps,
        max_tickers=max_tickers, min_history_bars=min_history_bars, seed=seed,
    )
    for name, data in artifacts.items():
        (LOCAL_OUTPUT_DIR / name).write_bytes(data)
        print(f'  wrote {LOCAL_OUTPUT_DIR / name} ({len(data)//1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
    print(f'\nNext step (host-local): post-process the artifacts to')
    print(f'compute pooled OOS bootstrap-CI of ΔSR-vs-EW and apply')
    print(f'the locked pre-reg verdict bar. Driver:')
    print(f'  uv run python apps/factor/scripts/verdict_studentized_sharpe_diff.py')
