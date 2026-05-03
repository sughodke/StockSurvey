"""Modal entrypoint for apps/factor's deterministic-indicator scorer.

Sweeps a grid over (scorer, n_steps, weight_decay) per call against the
baked-in ~312-ticker stooq_us_long subset; trains one head per cell and
returns the artifacts to the caller's local Output/.

Why CPU and not GPU: the deterministic-indicator path has no encoder —
just a (n, 79) -> 1 linear (or tiny MLP) Adam loop. The bottleneck is
the strided RSI/CCI Python recurrences during feature building, not
matmuls; tinygrad's CPU backend matches GPU within fractions of a
second on this workload, at ~10x lower cost.

Usage
-----
One-time setup (local):
    uvx modal token new          # or, if modal is already installed, `modal token new`

Smoke test (~3-5 min wall, ~$0.05):
    uvx modal run apps/factor/scripts/modal/train_indicator.py \\
        --scorers linear --n-steps-grid 100 --weight-decay-grid 1e-3 \\
        --max-tickers 30

Full grid sweep (~10-30 min wall, ~$0.10-0.30):
    uvx modal run apps/factor/scripts/modal/train_indicator.py \\
        --scorers linear,mlp \\
        --n-steps-grid 200,500,1000 \\
        --weight-decay-grid 0,1e-3,1e-2

Modal is declared as a dep on `apps/factor` (in `pyproject.toml`), so
either `uvx modal …` (ephemeral) or `uv run modal …` (after sync) works.

Returns one `{prefix}-indicator-head.npz` per cell plus a summary
`indicator-grid-summary.json` with val IC / val Sharpe / training time
across all cells, and a `indicator-grid-comparison.png` heatmap.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal


# REPO_ROOT only matters on the local side (image build + artifact write).
# Inside the Modal container this script is dropped at /root/<basename> with
# only 2 parents, so parents[4] would IndexError at import time.
try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')   # remote fallback (unused there)
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'

# Baked-in 312-ticker curated subset (~140 MB). Built locally via
# `apps/notebook/data/build_stooq_us_long.py` from the user's StooqData/
# archive; all tickers have >=22y of history starting from 2000-01-01.
STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_us_long'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'

# debian_slim is fine here — no GPU, no nvcc. uv handles the workspace.
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'curl', 'build-essential')
    .pip_install('uv')
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('factor-indicator-grid', image=image)


@app.function(cpu=4, memory=8192, timeout=60 * 90)
def train_grid(
    scorers: str,
    n_steps_grid: str,
    weight_decay_grid: str,
    tickers: str,
    start: str,
    end: str,
    rebal_days: int,
    learning_rate: float,
    train_frac: float,
    max_tickers: int,
    mlp_hidden: int,
    mlp_layers: int,
) -> dict[str, bytes]:
    """Build features once across the universe, then train one IC head per
    grid cell. Returns every file under Output/ as {filename: bytes}.
    """
    import os
    import subprocess
    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')

    print('=== Step 1/4: uv sync workspace deps (one-time per cold start) ===',
          flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'factor', '--inexact'],
        cwd=REMOTE_REPO, check=True)

    # Activate the editable workspace install so this in-process function
    # can `import factor` etc. (mirrors the train_cnn_multihead pattern).
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    # ---------- Step 2: load tickers + build features (shared across cells) ----------
    print('\n=== Step 2/4: load tickers + build deterministic indicator features ===',
          flush=True)
    ticker_list = _resolve_ticker_list(tickers, max_tickers)
    print(f'  universe: {len(ticker_list)} tickers '
          f'(first 5: {ticker_list[:5]} ...)')

    from factor import IndicatorGridConfig, train_scorer_indicators
    from ss_features import TickerData

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'  cfg.feature_width() = {F} channels')

    import multiprocessing as mp
    import os
    n_workers = max(1, int(os.environ.get('FACTOR_FEATURE_WORKERS',
                                          os.cpu_count() or 4)))
    print(f'  parallelizing feature build across {n_workers} workers')

    t0 = time.perf_counter()
    ticker_data: list[TickerData] = []
    skipped: list[str] = []
    work_args = [(t, STOOQ_SUBSET, cfg, start, end) for t in ticker_list]
    # imap_unordered returns results as workers complete — gives sensible
    # progress output and overlaps load-prices I/O with indicator compute.
    with mp.Pool(n_workers) as pool:
        for i, (ticker, result) in enumerate(
                pool.imap_unordered(_build_one_ticker, work_args)):
            if isinstance(result, TickerData):
                ticker_data.append(result)
            else:
                skipped.append(f'{ticker} {result}')
            if (i + 1) % 50 == 0:
                print(f'  built {i + 1}/{len(ticker_list)}  '
                      f'({time.perf_counter()-t0:.0f}s)', flush=True)
    # Stable order across runs even though workers complete out of order.
    ticker_data.sort(key=lambda td: td.name)
    print(f'  feature build done: {len(ticker_data)} usable / '
          f'{len(skipped)} skipped  ({time.perf_counter()-t0:.0f}s)')
    if skipped[:5]:
        print(f'  first 5 skipped: {skipped[:5]}')
    if len(ticker_data) < 4:
        raise RuntimeError(
            f'only {len(ticker_data)} tickers built — too few for IC training')

    # ---------- Step 3: grid sweep ----------
    print('\n=== Step 3/4: grid sweep over (scorer, n_steps, weight_decay) ===',
          flush=True)
    cells = list(_cartesian(scorers, n_steps_grid, weight_decay_grid))
    print(f'  {len(cells)} cells: {cells[:3]} ...' if len(cells) > 3
          else f'  {len(cells)} cells: {cells}')

    summary: list[dict] = []
    for ci, (scorer, n_steps, weight_decay) in enumerate(cells):
        prefix = f'ind-{scorer}-s{n_steps}-wd{weight_decay:g}'
        print(f'\n  [{ci+1}/{len(cells)}] {prefix} ...', flush=True)
        t1 = time.perf_counter()
        try:
            res = train_scorer_indicators(
                ticker_data, cfg,
                rebal_days=rebal_days, train_frac=train_frac,
                scorer=scorer,
                mlp_hidden=mlp_hidden, mlp_layers=mlp_layers,
                n_steps=n_steps, learning_rate=learning_rate,
                weight_decay=weight_decay,
                finetune_steps=0, verbose=False,
            )
        except Exception as e:
            print(f'    FAILED: {type(e).__name__}: {e}', flush=True)
            summary.append({'cell': prefix, 'failed': True,
                            'error': f'{type(e).__name__}: {e}'})
            continue
        wall = time.perf_counter() - t1
        npz_path = output / f'{prefix}-head.npz'
        _save_head_npz(npz_path, res, cfg)
        summary.append({
            'cell': prefix, 'scorer': scorer,
            'n_steps': n_steps, 'weight_decay': weight_decay,
            'train_ic': res.train_ic, 'val_ic': res.val_ic,
            'train_sharpe': res.train_sharpe, 'val_sharpe': res.val_sharpe,
            'n_train_bars': res.n_train_bars, 'n_val_bars': res.n_val_bars,
            'wall_seconds': round(wall, 1),
            'failed': False,
        })
        print(f'    train IC={res.train_ic:+.4f}  val IC={res.val_ic:+.4f}  '
              f'val Sharpe={res.val_sharpe:+.3f}  wall={wall:.1f}s')

    (output / 'indicator-grid-summary.json').write_text(
        json.dumps({
            'universe_size': len(ticker_data),
            'feature_width': F,
            'rebal_days': rebal_days, 'learning_rate': learning_rate,
            'train_frac': train_frac, 'start': start, 'end': end,
            'cells': summary,
        }, indent=2))

    # ---------- Step 4: comparison plot ----------
    print('\n=== Step 4/4: comparison plot ===', flush=True)
    _plot_grid_comparison(summary, output / 'indicator-grid-comparison.png')

    # Bundle every file under Output/ for return.
    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name.startswith(('ind-', 'indicator-')):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts')
    return artifacts


def _build_one_ticker(args):
    """Worker: load prices + build deterministic indicator features for one ticker.

    Returns `(ticker, TickerData)` on success or `(ticker, error_str)` on
    failure. Top-level function so `multiprocessing.Pool` can pickle it.
    Imports happen inside the worker so this module can be imported on the
    local side (where `factor` may not yet be importable) without dragging
    in the workspace deps at module-load time.
    """
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
            return ticker, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        )
    except Exception as e:
        return ticker, f'({type(e).__name__}: {e})'


def _resolve_ticker_list(tickers: str, max_tickers: int) -> list[str]:
    """Either parse the user's comma-separated list, or fall back to the
    full manifest. `max_tickers > 0` caps either result for smoke runs."""
    if tickers:
        names = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    else:
        manifest_path = Path(STOOQ_SUBSET) / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        names = [t['ticker'] for t in manifest['tickers']]
    if max_tickers > 0:
        names = names[:max_tickers]
    return names


def _cartesian(scorers: str, n_steps_grid: str, weight_decay_grid: str):
    """Yield (scorer, n_steps, weight_decay) for every combination."""
    s_list = [s.strip() for s in scorers.split(',') if s.strip()]
    n_list = [int(n) for n in n_steps_grid.split(',') if n.strip()]
    w_list = [float(w) for w in weight_decay_grid.split(',') if w.strip()]
    for s in s_list:
        for n in n_list:
            for w in w_list:
                yield s, n, w


def _save_head_npz(path: Path, res, cfg) -> None:
    """Pack the trained head + per-channel labels + history into one npz so
    a downstream notebook can read it without depending on `factor`."""
    import numpy as np
    blob: dict[str, np.ndarray] = {}
    for k, v in res.params.items():
        blob[f'head_{k}'] = np.asarray(v, dtype=np.float32)
    blob['channel_names'] = np.array(cfg.channel_names())
    blob['train_history'] = np.array(res.train_history, dtype=np.float32)
    if res.val_history:
        vh = np.array(res.val_history)   # (steps, ic, sharpe)
        blob['val_history_step'] = vh[:, 0].astype(np.int32)
        blob['val_history_ic'] = vh[:, 1].astype(np.float32)
        blob['val_history_sharpe'] = vh[:, 2].astype(np.float32)
    blob['_summary'] = np.array(json.dumps({
        'scorer': res.scorer,
        'train_ic': res.train_ic, 'val_ic': res.val_ic,
        'train_sharpe': res.train_sharpe, 'val_sharpe': res.val_sharpe,
        'n_train_bars': res.n_train_bars, 'n_val_bars': res.n_val_bars,
    }))
    np.savez(path, **blob)
    print(f'    -> {path.name} ({path.stat().st_size // 1024}KB)')


def _plot_grid_comparison(summary: list[dict], out_path: Path) -> None:
    """Two-panel plot: val IC and val Sharpe across the grid, one bar
    per cell, color-coded by scorer."""
    import matplotlib.pyplot as plt

    ok = [c for c in summary if not c.get('failed')]
    if not ok:
        print(f'  no successful cells — skipping {out_path.name}')
        return
    labels = [c['cell'] for c in ok]
    val_ic = [c['val_ic'] for c in ok]
    val_sh = [c['val_sharpe'] for c in ok]
    colors = ['steelblue' if c['scorer'] == 'linear' else 'darkorange'
              for c in ok]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(ok) * 0.6), 7),
                                    sharex=True)
    ax1.bar(range(len(ok)), val_ic, color=colors)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('val IC')
    ax1.set_title('Deterministic-indicator grid sweep')
    ax2.bar(range(len(ok)), val_sh, color=colors)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('val Sharpe')
    ax2.set_xticks(range(len(ok)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  -> {out_path.name}')


@app.local_entrypoint()
def main(
    scorers: str = 'linear,mlp',
    n_steps_grid: str = '200,500',
    weight_decay_grid: str = '0,1e-3',
    tickers: str = '',
    start: str = '2000-01-01',
    end: str = '2026-04-01',
    rebal_days: int = 20,
    learning_rate: float = 1e-2,
    train_frac: float = 0.7,
    max_tickers: int = 0,         # 0 = use all from manifest
    mlp_hidden: int = 64,
    mlp_layers: int = 1,
) -> None:
    """Local entrypoint that calls the remote `train_grid` and writes
    artifacts back to repo-root Output/."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'launching factor-indicator-grid on Modal '
          f'(scorers={scorers}, n_steps={n_steps_grid}, '
          f'weight_decay={weight_decay_grid}, max_tickers={max_tickers})')
    artifacts = train_grid.remote(
        scorers=scorers,
        n_steps_grid=n_steps_grid,
        weight_decay_grid=weight_decay_grid,
        tickers=tickers,
        start=start,
        end=end,
        rebal_days=rebal_days,
        learning_rate=learning_rate,
        train_frac=train_frac,
        max_tickers=max_tickers,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
    )
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
