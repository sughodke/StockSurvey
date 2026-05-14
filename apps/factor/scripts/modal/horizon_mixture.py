"""Modal entrypoint for the endogenous-horizon mixture walk-forward.

Mirrors `train_indicator.py::walkforward` but invokes
`train_scorer_horizon_walkforward` with the multi-horizon mixture loss
and the dual-head scorer. Returns per-window endog Sharpe, fixed-h
baselines, random-π baseline, π entropy + argmax-bin histogram, all
packed into a single npz + a summary json.

Pre-registered hypothesis (see commit history): state-conditional
horizon selection beats best-fixed-h by ≥ 0.10 Sharpe AND beats
random-π. Verdict labels follow the leaderboard vocabulary
(confirmed-OOS / partial-OOS / confirmed-null).

Usage
-----
Smoke (~3 min wall, ~$0.04 at T4 prices):
    uvx modal run apps/factor/scripts/modal/horizon_mixture.py \\
        --max-tickers 30 --n-steps 50

Full 297-ticker walkforward (~15-25 min wall, ~$0.30-0.50):
    uvx modal run apps/factor/scripts/modal/horizon_mixture.py
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
            # `.claude/` carries the scheduled-tasks lock + agent transcripts;
            # Modal aborts the build with "modified during build process" if
            # the harness writes a wakeup mid-upload.
            '.claude/**',
            '.modal_metadata/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            '.edgar-cache/**',
            '.macro-cache/**',
            'apps/relational/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            'apps/replay/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('factor-horizon-mixture', image=image)


@app.function(gpu='T4', cpu=4, memory=8192, timeout=60 * 60)
def train_horizon_walkforward_remote(
    horizons_csv: str,
    n_steps: int,
    learning_rate: float,
    weight_decay: float,
    entropy_weight: float,
    mlp_hidden: int,
    mlp_layers: int,
    commission_bps: float,
    temperature: float,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    seed: int,
    tickers: str,
    start: str,
    end: str,
    max_tickers: int,
    min_history_bars: int,
) -> dict[str, bytes]:
    """Remote: build features once on the universe, run horizon-mixture
    walk-forward, bundle per-window metrics + summary into artifacts."""
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

    print('\n=== Step 2/4: load tickers + build deterministic indicator features ===',
          flush=True)
    ticker_list = _resolve_ticker_list(tickers, max_tickers, min_history_bars)
    print(f'  universe: {len(ticker_list)} tickers '
          f'(first 5: {ticker_list[:5]} ...)')

    from factor import (
        IndicatorGridConfig, make_indicator_backbone,
        train_scorer_horizon_walkforward,
    )
    from ss_features import TickerData

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'  cfg.feature_width() = {F} channels')

    horizons = tuple(int(h) for h in horizons_csv.split(','))
    print(f'  horizons = {horizons} (h_min={min(horizons)}, K={len(horizons)})')

    import multiprocessing as mp
    n_workers = max(1, int(os.environ.get('FACTOR_FEATURE_WORKERS',
                                          os.cpu_count() or 4)))
    print(f'  parallelizing feature build across {n_workers} workers')

    t0 = time.perf_counter()
    ticker_data: list[TickerData] = []
    skipped: list[str] = []
    work_args = [(t, STOOQ_SUBSET, cfg, start, end) for t in ticker_list]
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
    ticker_data.sort(key=lambda td: td.name)
    print(f'  feature build done: {len(ticker_data)} usable / '
          f'{len(skipped)} skipped  ({time.perf_counter()-t0:.0f}s)')
    if len(ticker_data) < 4:
        raise RuntimeError(
            f'only {len(ticker_data)} tickers built — too few for IC training')

    print('\n=== Step 3/4: horizon-mixture walk-forward ===', flush=True)
    backbone = make_indicator_backbone(ticker_data, cfg)
    t1 = time.perf_counter()
    res = train_scorer_horizon_walkforward(
        ticker_data, backbone,
        horizons=horizons,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        entropy_weight=entropy_weight,
        commission_bps=commission_bps,
        temperature=temperature,
        seed=seed,
        verbose=True,
    )
    wall = time.perf_counter() - t1
    print(f'  walk-forward wall: {wall:.1f}s', flush=True)

    # Aggregates + null-rejection.
    endog_mean = res.mean_val_endog_sharpe
    random_mean = res.mean_val_random_sharpe
    h_best, best_fixed_mean = res.best_fixed_horizon
    delta_best = endog_mean - best_fixed_mean
    delta_rand = endog_mean - random_mean

    print('\n=== summary ===', flush=True)
    print(f'  mean endog Sharpe       = {endog_mean:+.3f}', flush=True)
    for h in horizons:
        marker = '  <-- best fixed' if h == h_best else ''
        print(f'  mean fixed-h={h:<3} Sharpe = {res.mean_fixed_sharpe(h):+.3f}{marker}',
              flush=True)
    print(f'  mean random-π Sharpe    = {random_mean:+.3f}', flush=True)
    print(f'  delta vs best fixed     = {delta_best:+.3f}   '
          f'(success threshold >= +0.10)', flush=True)
    print(f'  delta vs random-π       = {delta_rand:+.3f}   '
          f'(success threshold >= 0)', flush=True)

    # Argmax bin shares across all windows.
    import numpy as np
    all_counts = np.zeros(len(horizons), dtype=np.int64)
    total_bars = 0
    for w in res.windows:
        for k, h in enumerate(horizons):
            all_counts[k] += w.val_pi_argmax_counts[h]
        total_bars += sum(w.val_pi_argmax_counts.values())
    pi_global = all_counts / max(total_bars, 1)
    print(f'  argmax bin shares       = '
          f'{ {h: f"{pi_global[k]:.2f}" for k, h in enumerate(horizons)} }',
          flush=True)
    collapse_max = float(np.max(pi_global))
    verdict_n1 = collapse_max <= 0.90
    verdict_n2 = endog_mean > res.mean_fixed_sharpe(max(horizons))
    verdict_n3 = delta_best >= 0.10
    verdict_n4 = delta_rand > 0.0
    verdict_pass = verdict_n1 and verdict_n2 and verdict_n3 and verdict_n4
    print('\n=== null-rejection checks ===', flush=True)
    print(f'  N1 (no π collapse, worst share={collapse_max:.2f}): '
          f'{"PASS" if verdict_n1 else "FAIL"}', flush=True)
    print(f'  N2 (beats fixed h_max={max(horizons)}): '
          f'{"PASS" if verdict_n2 else "FAIL"}', flush=True)
    print(f'  N3 (beats best fixed by >=0.10): '
          f'{"PASS" if verdict_n3 else "FAIL"} (delta {delta_best:+.3f})',
          flush=True)
    print(f'  N4 (beats random-π): '
          f'{"PASS" if verdict_n4 else "FAIL"} (delta {delta_rand:+.3f})',
          flush=True)
    print(f'  Overall verdict: '
          f'{"confirmed-OOS" if verdict_pass else "partial-OOS/null"}',
          flush=True)

    # ---------- Step 4: pack artifacts ----------
    print('\n=== Step 4/4: pack artifacts ===', flush=True)
    blob: dict[str, np.ndarray] = {
        'window_idx':           np.array([w.window_idx for w in res.windows], dtype=np.int32),
        'train_block_start':    np.array([w.train_block_start for w in res.windows], dtype=np.int32),
        'train_block_end':      np.array([w.train_block_end for w in res.windows], dtype=np.int32),
        'val_block_start':      np.array([w.val_block_start for w in res.windows], dtype=np.int32),
        'val_block_end':        np.array([w.val_block_end for w in res.windows], dtype=np.int32),
        'val_daily_start':      np.array([w.val_daily_start for w in res.windows], dtype=np.int32),
        'val_daily_end':        np.array([w.val_daily_end for w in res.windows], dtype=np.int32),
        'train_loss':           np.array([w.train_loss for w in res.windows], dtype=np.float32),
        'val_endog_sharpe':     np.array([w.val_endog_sharpe for w in res.windows], dtype=np.float32),
        'val_random_sharpe':    np.array([w.val_random_sharpe for w in res.windows], dtype=np.float32),
        'val_endog_mean_holding': np.array([w.val_endog_mean_holding for w in res.windows], dtype=np.float32),
        'val_endog_n_rebals':   np.array([w.val_endog_n_rebals for w in res.windows], dtype=np.int32),
        'val_endog_avg_turnover': np.array([w.val_endog_avg_turnover for w in res.windows], dtype=np.float32),
        'val_pi_entropy_mean':  np.array([w.val_pi_entropy_mean for w in res.windows], dtype=np.float32),
        'val_start_date':       np.array([w.val_start_date for w in res.windows]),
    }
    for h in horizons:
        blob[f'val_fixed_sharpe_h{h}'] = np.array(
            [w.val_fixed_sharpes[h] for w in res.windows], dtype=np.float32)
        blob[f'val_argmax_count_h{h}'] = np.array(
            [w.val_pi_argmax_counts[h] for w in res.windows], dtype=np.int32)
    blob['_summary'] = np.array(json.dumps({
        'horizons':             list(horizons),
        'mean_endog_sharpe':    endog_mean,
        'mean_random_sharpe':   random_mean,
        'best_fixed_horizon':   h_best,
        'best_fixed_sharpe':    best_fixed_mean,
        'delta_vs_best_fixed':  delta_best,
        'delta_vs_random':      delta_rand,
        'pi_argmax_global_shares': {
            int(h): float(pi_global[k]) for k, h in enumerate(horizons)},
        'verdict_n1':           verdict_n1,
        'verdict_n2':           verdict_n2,
        'verdict_n3':           verdict_n3,
        'verdict_n4':           verdict_n4,
        'verdict_pass':         verdict_pass,
        'verdict_label':        'confirmed-OOS' if verdict_pass else (
            'partial-OOS' if (verdict_n1 and verdict_n2 and verdict_n4) else
            'confirmed-null'),
        'n_windows':            res.n_windows,
        'universe_size':        len(ticker_data),
        'feature_width':        F,
        'mlp_hidden':           mlp_hidden,
        'mlp_layers':           mlp_layers,
        'n_steps':              n_steps,
        'learning_rate':        learning_rate,
        'weight_decay':         weight_decay,
        'entropy_weight':       entropy_weight,
        'commission_bps':       commission_bps,
        'temperature':          temperature,
        'train_window_blocks':  train_window_blocks,
        'val_window_blocks':    val_window_blocks,
        'step_window_blocks':   step_window_blocks,
        'wall_seconds':         round(wall, 1),
    }, indent=2))
    npz_path = output / 'horizon-mixture-windows.npz'
    np.savez(npz_path, **blob)
    print(f'  -> {npz_path.name} ({npz_path.stat().st_size // 1024}KB)')

    # Per-window comparison plot.
    plot_path = output / 'horizon-mixture-comparison.png'
    _plot_horizon_mixture(res, horizons, plot_path)

    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name.startswith('horizon-mixture'):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts')
    return artifacts


def _build_one_ticker(args):
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


def _resolve_ticker_list(
    tickers: str, max_tickers: int, min_history_bars: int = 0,
) -> list[str]:
    manifest_path = Path(STOOQ_SUBSET) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if tickers:
        requested = {t.strip().upper() for t in tickers.split(',') if t.strip()}
        entries = [t for t in manifest['tickers'] if t['ticker'].upper() in requested]
    else:
        entries = list(manifest['tickers'])
    if min_history_bars > 0:
        before = len(entries)
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
        dropped = before - len(entries)
        if dropped:
            print(f'  min_history_bars={min_history_bars}: '
                  f'dropped {dropped} short-history tickers')
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
    return names


def _plot_horizon_mixture(res, horizons, out_path: Path) -> None:
    """Three-panel plot:
      1. Per-window Sharpes (endog vs each fixed-h vs random-π).
      2. Per-window argmax-bin histogram (which horizons the model picked).
      3. Mean π entropy per window (proxy for confidence vs uncertainty).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not res.windows:
        return
    n = res.n_windows
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(max(10, n * 0.4), 11),
                             sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1]})

    # Panel 1: Sharpes.
    ax1 = axes[0]
    endog = [w.val_endog_sharpe for w in res.windows]
    ax1.bar(x, endog, color='crimson', label='endog (argmax π)', alpha=0.85)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(horizons)))
    for k, (h, c) in enumerate(zip(horizons, colors)):
        ys = [w.val_fixed_sharpes[h] for w in res.windows]
        ax1.plot(x, ys, marker='o', markersize=4, color=c, label=f'fix h={h}',
                 linewidth=1.2)
    rand = [w.val_random_sharpe for w in res.windows]
    ax1.plot(x, rand, marker='x', markersize=5, color='gray',
             label='random-π', linestyle='--', linewidth=1.2)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Sharpe (daily PnL, net of costs)')
    ax1.set_title(
        f'Horizon-mixture walk-forward — mean endog={res.mean_val_endog_sharpe:+.3f}  '
        f'best-fix(h={res.best_fixed_horizon[0]})={res.best_fixed_horizon[1]:+.3f}  '
        f'random-π={res.mean_val_random_sharpe:+.3f}')
    ax1.legend(fontsize=8, ncol=3)

    # Panel 2: argmax-bin histogram per window.
    ax2 = axes[1]
    bottom = np.zeros(n)
    for k, (h, c) in enumerate(zip(horizons, colors)):
        heights = np.array([w.val_pi_argmax_counts[h] for w in res.windows],
                           dtype=float)
        ax2.bar(x, heights, bottom=bottom, color=c,
                label=f'h={h}', width=0.85)
        bottom += heights
    ax2.set_ylabel('argmax(π_t) bar counts')
    ax2.legend(fontsize=8, ncol=len(horizons))

    # Panel 3: mean entropy per window.
    ax3 = axes[2]
    ent = [w.val_pi_entropy_mean for w in res.windows]
    ax3.plot(x, ent, marker='o', color='steelblue', linewidth=1.2)
    ax3.axhline(np.log(len(horizons)), color='gray', linestyle=':',
                label=f'log(K)={np.log(len(horizons)):.2f}')
    ax3.set_ylabel('mean H(π_t)')
    ax3.set_xlabel('walk-forward window')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'w{i}' for i in range(n)], fontsize=7,
                       rotation=45 if n > 10 else 0)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  -> {out_path.name}')


@app.local_entrypoint()
def main(
    horizons: str = '5,10,20,40,60',
    n_steps: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    entropy_weight: float = 0.0,
    mlp_hidden: int = 32,
    mlp_layers: int = 1,
    commission_bps: float = 10.0,
    temperature: float = 1.0,
    train_window_blocks: int = 252,
    val_window_blocks: int = 156,
    step_window_blocks: int = 156,
    seed: int = 0,
    tickers: str = '',
    start: str = '2000-01-01',
    end: str = '2026-04-01',
    max_tickers: int = 0,
    min_history_bars: int = 6500,
) -> None:
    """Local entrypoint: kicks off remote walk-forward and downloads artifacts."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'launching factor horizon-mixture on Modal '
          f'(horizons={horizons}, n_steps={n_steps}, lr={learning_rate}, '
          f'wd={weight_decay}, ent_w={entropy_weight}, '
          f'train/val/step blocks={train_window_blocks}/{val_window_blocks}/'
          f'{step_window_blocks}, max_tickers={max_tickers}, '
          f'min_history_bars={min_history_bars})')
    artifacts = train_horizon_walkforward_remote.remote(
        horizons_csv=horizons,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        entropy_weight=entropy_weight,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
        commission_bps=commission_bps,
        temperature=temperature,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        seed=seed,
        tickers=tickers,
        start=start, end=end,
        max_tickers=max_tickers,
        min_history_bars=min_history_bars,
    )
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
