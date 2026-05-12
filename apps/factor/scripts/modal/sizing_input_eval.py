"""Modal entrypoint for the sizing-input v0 eval (factor-narrow,
two-arm rank_ic vs mse_alpha walk-forward).

Motivating TODO:
    apps/docs/docs/TODO/factor-sizing-input-reframe.md
Smoke driver (same shape, local):
    apps/factor/scripts/sizing_input_eval.py
Modal harness pattern lifted from `train_indicator.py::train_walkforward`
(T4 GPU + uv sync + CUDA pin + parallel feature build over the baked-in
312-ticker `stooq_us_long` subset).

Two arms, same head + windows + commission as `loss_pivot_eval.py`:

  - `rank_ic`   — `pearson_rank_ic` loss (existing baseline; score
                  magnitude uncalibrated).
  - `mse_alpha` — `masked_mse` on per-bar cross-sectional alpha
                  targets (new; score magnitude calibrated to alpha
                  units).

Per arm, emit per-val-bar signal-quality (top-decile − bottom-decile
predicted alpha), val_start_date, val_mse_alpha. The sizing-input
artifact this run ships is `Output/sizing-input-{arm}-windows.npz`
with the full per-bar dispersion time series, consumed by the v1
macro-meta-gate wiring.

Usage:
    uvx modal token new   # one-time

    # Smoke (~30s wall, ~$0.005):
    uvx modal run apps/factor/scripts/modal/sizing_input_eval.py \\
        --max-tickers 30 --n-steps 50

    # Full factor-narrow (~30 min wall at 297 tickers × 200 steps × 2 arms):
    uvx modal run apps/factor/scripts/modal/sizing_input_eval.py

Returns `Output/sizing-input-{arm}-windows.npz` per arm plus
`Output/sizing-input-eval-summary.json` with the pre-registered
verdict.
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
            '.git/**',
            '.venv/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            'apps/relational/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            'apps/replay/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('factor-sizing-input', image=image)


def _resolve_ticker_list_remote(
    min_history_bars: int, max_tickers: int,
) -> list[str]:
    """Read the baked-in stooq_us_long manifest from the shipped repo,
    filter by manifest `n_bars`, optionally cap for smoke."""
    manifest_path = Path(STOOQ_SUBSET) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    if min_history_bars > 0:
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} tickers; {len(entries)} pass '
          f'min_history_bars={min_history_bars}', flush=True)
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers} for smoke run', flush=True)
    return names


def _build_one_ticker(args):
    """Top-level for mp.Pool pickling. Same shape as
    `train_indicator.py::_build_one_ticker`."""
    ticker, stooq_subset, cfg, start, end = args
    import numpy as np
    from factor import build_indicator_features
    from ss_features import TickerData, load_prices
    try:
        series = load_prices(ticker, stooq_dir=stooq_subset,
                             start=start, end=end)
        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        feats, valid = build_indicator_features(prices, cfg)
        if not valid.any():
            return ticker, None, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ), None
    except Exception as e:
        return ticker, None, f'({type(e).__name__}: {e})'


def _lag1_autocorr(x):
    import numpy as np
    if x.size < 2:
        return float('nan')
    a = x[:-1]; b = x[1:]
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 5:
        return float('nan')
    av = a[valid]; bv = b[valid]
    am = av.mean(); bm = bv.mean()
    num = ((av - am) * (bv - bm)).sum()
    den = (((av - am) ** 2).sum() * ((bv - bm) ** 2).sum()) ** 0.5 + 1e-18
    return float(num / den)


def _spearman(x, y):
    import numpy as np
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float('nan')
    xv = x[mask]; yv = y[mask]
    rx = np.argsort(np.argsort(xv)).astype(np.float64)
    ry = np.argsort(np.argsort(yv)).astype(np.float64)
    rxm = rx.mean(); rym = ry.mean()
    num = ((rx - rxm) * (ry - rym)).sum()
    den = (((rx - rxm) ** 2).sum() * ((ry - rym) ** 2).sum()) ** 0.5 + 1e-18
    return float(num / den)


@app.function(gpu='T4', cpu=4, memory=8192, timeout=60 * 60)
def sizing_input(
    losses: str,
    start: str,
    end: str,
    rebal_days: int,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    n_steps: int,
    learning_rate: float,
    weight_decay: float,
    min_history_bars: int,
    max_tickers: int,
    seed: int,
) -> dict[str, bytes]:
    """Run the two-arm sizing-input walk-forward on T4. Returns the
    npz per arm + summary JSON + verdict as a bundle."""
    import io
    import os
    import subprocess

    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')

    os.environ['CUDA'] = '1'

    print('=== Step 1/4: uv sync workspace deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'factor', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(
            f'tinygrad picked Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    print('\n=== Step 2/4: resolve universe + build features ===', flush=True)
    import multiprocessing as mp
    import numpy as np

    from factor import IndicatorGridConfig, train_scorer_indicators_walkforward
    from ss_features import TickerData

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'  cfg.feature_width() = {F} channels', flush=True)

    names = _resolve_ticker_list_remote(min_history_bars, max_tickers)
    n_workers = max(1, os.cpu_count() or 4)
    print(f'  parallelizing feature build across {n_workers} workers',
          flush=True)
    t0 = time.perf_counter()
    work_args = [(t, STOOQ_SUBSET, cfg, start, end) for t in names]
    ticker_data: list[TickerData] = []
    skipped: list[str] = []
    with mp.Pool(n_workers) as pool:
        for i, (ticker, td, err) in enumerate(
                pool.imap_unordered(_build_one_ticker, work_args)):
            if td is not None:
                ticker_data.append(td)
            else:
                skipped.append(f'{ticker} {err}')
            if (i + 1) % 50 == 0:
                print(f'  built {i + 1}/{len(names)}  '
                      f'({time.perf_counter() - t0:.0f}s)', flush=True)
    ticker_data.sort(key=lambda td: td.name)
    print(f'  feature build done: {len(ticker_data)} usable / '
          f'{len(skipped)} skipped  ({time.perf_counter() - t0:.0f}s)',
          flush=True)
    if len(ticker_data) < 4:
        raise RuntimeError(
            f'only {len(ticker_data)} tickers built — too few for IC training')

    print('\n=== Step 3/4: two-arm walk-forward (rank_ic + mse_alpha) ===',
          flush=True)
    loss_list = [s.strip() for s in losses.split(',') if s.strip()]
    valid = {'rank_ic', 'mse_alpha', 'block_sharpe', 'ir_vs_ew'}
    if not set(loss_list).issubset(valid):
        raise RuntimeError(f'--losses must be subset of {sorted(valid)}')

    arms: dict[str, dict] = {}
    artifacts: dict[str, bytes] = {}

    for loss_kind in loss_list:
        prefix = f'sizing-input-{loss_kind}'
        print(f'\n  >>> {prefix}', flush=True)
        t1 = time.perf_counter()
        wf = train_scorer_indicators_walkforward(
            ticker_data, cfg=cfg,
            rebal_days=rebal_days,
            train_window_blocks=train_window_blocks,
            val_window_blocks=val_window_blocks,
            step_window_blocks=step_window_blocks,
            scorer='linear',
            n_steps=n_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            forward_target_kind='log_return',
            loss_kind=loss_kind,
            verbose=True,
        )
        wall = time.perf_counter() - t1
        print(f'    arm wall: {wall:.1f}s', flush=True)

        # Per-window detail + verdict stats.
        sq_per_bar = np.stack(
            [w.signal_quality_per_val_bar for w in wf.windows], axis=0)
        pooled = sq_per_bar.reshape(-1)
        lag1 = _lag1_autocorr(pooled)
        per_window_sq = np.array(
            [w.signal_quality_mean for w in wf.windows])
        per_window_sh = np.array([w.val_sharpe for w in wf.windows])
        rho = _spearman(per_window_sq, per_window_sh)

        print(f'    per-window val IC: '
              f'{[round(w.val_ic, 4) for w in wf.windows]}', flush=True)
        print(f'    mean val IC: {wf.mean_val_ic:+.4f}', flush=True)
        print(f'    mean val Sharpe: {wf.mean_val_sharpe:+.3f}', flush=True)
        print(f'    mean val MSE-alpha: {wf.mean_val_mse_alpha:.3e}',
              flush=True)
        print(f'    mean signal-quality: '
              f'{wf.mean_signal_quality:+.3e}', flush=True)
        print(f'    pooled lag-1 sq autocorr: {lag1:+.4f}', flush=True)
        print(f'    Spearman(sq_mean, val_sh): {rho:+.4f}', flush=True)

        blob = {
            'window_idx': np.array(
                [w.window_idx for w in wf.windows]),
            'val_start_date': np.array(
                [w.val_start_date for w in wf.windows], dtype='S10'),
            'train_ic': np.array([w.train_ic for w in wf.windows]),
            'val_ic': np.array([w.val_ic for w in wf.windows]),
            'val_sharpe': np.array(
                [w.val_sharpe for w in wf.windows]),
            'val_mse_alpha': np.array(
                [w.val_mse_alpha for w in wf.windows]),
            'signal_quality_per_val_bar': sq_per_bar,
            'signal_quality_mean': per_window_sq,
            'signal_quality_std': np.array(
                [w.signal_quality_std for w in wf.windows]),
        }
        blob['_meta'] = np.array(json.dumps({
            'loss_kind': loss_kind,
            'forward_target_kind': wf.forward_target_kind,
            'rebal_days': wf.rebal_days,
            'feature_width': wf.feature_width,
            'mean_val_ic': wf.mean_val_ic,
            'mean_val_sharpe': wf.mean_val_sharpe,
            'mean_val_mse_alpha': wf.mean_val_mse_alpha,
            'mean_signal_quality': wf.mean_signal_quality,
            'lag1_autocorr_signal_quality_pooled': lag1,
            'spearman_sq_vs_val_sharpe': rho,
            'n_universe': len(ticker_data),
            'wall_seconds': round(wall, 1),
        }))
        buf = io.BytesIO()
        np.savez(buf, **blob)
        artifacts[f'{prefix}-windows.npz'] = buf.getvalue()

        arms[loss_kind] = {
            'arm': loss_kind,
            'n_windows': wf.n_windows,
            'mean_val_ic': wf.mean_val_ic,
            'mean_val_sharpe': wf.mean_val_sharpe,
            'mean_val_mse_alpha': wf.mean_val_mse_alpha,
            'mean_signal_quality': wf.mean_signal_quality,
            'lag1_autocorr_signal_quality_pooled': lag1,
            'spearman_sq_vs_val_sharpe': rho,
            'wall_seconds': round(wall, 1),
            'per_window': [{
                'window_idx': w.window_idx,
                'val_start_date': w.val_start_date,
                'train_ic': w.train_ic, 'val_ic': w.val_ic,
                'val_sharpe': w.val_sharpe,
                'val_mse_alpha': w.val_mse_alpha,
                'signal_quality_mean': w.signal_quality_mean,
                'signal_quality_std': w.signal_quality_std,
            } for w in wf.windows],
        }

    print('\n=== Step 4/4: verdict ===', flush=True)
    if 'mse_alpha' in arms and 'rank_ic' in arms:
        new = arms['mse_alpha']; base = arms['rank_ic']
        a_pass = (
            new['lag1_autocorr_signal_quality_pooled'] >= 0.20
            and (new['lag1_autocorr_signal_quality_pooled']
                 - base['lag1_autocorr_signal_quality_pooled']) >= 0.10
        )
        b_pass = (
            new['spearman_sq_vs_val_sharpe'] >= 0.40
            and (new['spearman_sq_vs_val_sharpe']
                 - base['spearman_sq_vs_val_sharpe']) >= 0.10
        )
        if a_pass and b_pass:
            verdict = 'PASS  (both criteria clear)'
        elif not a_pass and not b_pass:
            verdict = 'FAIL  (neither criterion clears)'
        else:
            which = 'lag1-autocorr' if a_pass else 'Spearman'
            verdict = f'INCONCLUSIVE  (only {which} clears)'
    else:
        verdict = 'N/A (missing arm)'
    print(f'verdict: {verdict}', flush=True)

    summary = {
        'universe_size': len(ticker_data),
        'feature_width': F,
        'rebal_days': rebal_days,
        'train_window_blocks': train_window_blocks,
        'val_window_blocks': val_window_blocks,
        'step_window_blocks': step_window_blocks,
        'n_steps': n_steps,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'arms': arms,
        'verdict': verdict,
    }
    artifacts['sizing-input-eval-summary.json'] = json.dumps(
        summary, indent=2).encode()

    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def main(
    losses: str = 'rank_ic,mse_alpha',
    start: str = '2000-01-01',
    end: str = '2025-12-11',
    rebal_days: int = 20,
    train_window_blocks: int = 63,
    val_window_blocks: int = 39,
    step_window_blocks: int = 39,
    n_steps: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    min_history_bars: int = 6500,
    max_tickers: int = 0,
    seed: int = 0,
) -> None:
    """Launch the two-arm sizing-input eval on Modal T4 and write
    artifacts back to local `Output/`.
    """
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[local] launching sizing-input Modal eval '
          f'(losses={losses}, max_tickers={max_tickers}, n_steps={n_steps})',
          flush=True)
    t0 = time.perf_counter()
    artifacts = sizing_input.remote(
        losses=losses,
        start=start, end=end,
        rebal_days=rebal_days,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_history_bars=min_history_bars,
        max_tickers=max_tickers,
        seed=seed,
    )
    print(f'[local] remote done in {time.perf_counter() - t0:.0f}s',
          flush=True)
    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.0f} KB)',
              flush=True)
