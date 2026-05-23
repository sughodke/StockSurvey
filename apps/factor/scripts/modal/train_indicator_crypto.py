"""Modal walk-forward of the 74-channel deterministic indicator grid on
CryptoCompare daily OHLCV — top-50 crypto by 2026-vintage market cap.

Pre-registered hypothesis (from `.research-venue-fit.md`):
    The 74-channel indicator grid that delivered val IC +0.0212 (6/6
    positive windows) on US equities at rebal_days=5 also clears
    mean val IC > +0.025 on top-50 crypto at the same horizon,
    with 4/5 positive windows and DSR t > +1.5.

Falsification bar (record honestly per CLAUDE.md):
    Mean val IC > +0.025 ; 4/5 windows positive ; DSR t > +1.5.
If any of the three miss, the result is `confirmed-null` for this
venue port — do NOT post-hoc adjust the bar.

Mechanism (Liu–Tsyvinski 2022, J. Finance): crypto cross-section
exhibits a 3-factor structure (market / size / momentum) ~3–4× the
strength of the equity momentum factor. If the indicator grid extracts
any cross-sectional alpha on US equities, crypto's larger inefficiency
should lift the val-IC ceiling above the +0.025 deflation-friendly
threshold.

Periods-per-year discipline: crypto trades 7d/wk. With rebal_days=5,
`periods_per_year = 365/5 = 73.0`. NOT `252/5`. The npz records this
explicitly under `periods_per_year` so the cross-arc DSR harness
(`ss_portfolio.standardize_oos`) annualizes correctly.

Usage
-----
One-time prep (local — fetches CryptoCompare, ~1 min):
    uv run python apps/factor/scripts/prep_crypto_universe.py

Smoke test (local, no Modal — see `if __name__ == '__main__'` in
the bottom of this file): runs the remote function body locally for
fast iteration. Use `--max-tickers 8 --n-steps 30 --n-windows 2`.

Modal launch (full walk-forward, T4 GPU):
    uvx modal run apps/factor/scripts/modal/train_indicator_crypto.py \\
        --scorers linear,mlp --n-steps 200 --weight-decay 1e-3

Returns one `walkforward-crypto-{scorer}-...-windows.npz` per scorer
plus `walkforward-crypto-summary.json` + a comparison plot.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# `modal` is only needed for the Modal app/image/decorators. Smoke-test
# path (`__main__`) imports this file directly from `uv run python` and
# must not require modal to be installed in the project venv. Import is
# guarded; only the Modal-specific top-level objects below are gated.
try:
    import modal  # type: ignore
    _HAVE_MODAL = True
except ImportError:
    modal = None  # type: ignore[assignment]
    _HAVE_MODAL = False


# REPO_ROOT only matters on the local side (image build + artifact write).
try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
LOCAL_PICKLE = LOCAL_OUTPUT_DIR / 'crypto_universe_panel.pkl'
REMOTE_REPO = '/root/StockSurvey'

# Pre-registered falsification bar — recorded into the result NPZ.
PRE_REGISTERED_BAR = (
    'mean val IC > +0.025 ; 4/5 positive windows ; DSR t > +1.5'
)

# Image: same as train_indicator.py — NVIDIA CUDA devel for tinygrad's
# CUDA backend. The prep step (fetching CryptoCompare) does NOT run on
# Modal — we pickle locally and ship via RPC.
if _HAVE_MODAL:
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
    app = modal.App('factor-indicator-crypto', image=image)
else:
    # Stub so module load doesn't error in the smoke-test path. The
    # decorators below short-circuit via _HAVE_MODAL.
    image = None  # type: ignore[assignment]
    app = None    # type: ignore[assignment]


# =============================================================================
# Core driver — usable both as a Modal remote function and (for the smoke
# test) as a plain function on the local Mac. Kept side-effect-free apart
# from writing files into `output_dir`, which the caller pre-creates.
# =============================================================================

def _run_walkforward(
    panel_bytes: bytes,
    *,
    scorers: str,
    n_steps: int,
    weight_decay: float,
    rebal_days: int,
    learning_rate: float,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    max_tickers: int,
    n_windows_cap: int,
    mlp_hidden: int,
    mlp_layers: int,
    commission_bps: float,
    output_dir: Path,
    require_cuda: bool,
    dump_returns: bool,
) -> dict[str, bytes]:
    """Build TickerData from the pickled panel, run walk-forward eval per
    scorer, write artifacts under `output_dir`, return them as bytes."""
    import pickle as _pickle
    import numpy as np

    # Backend pin (Modal remote: CUDA; local smoke: whatever tinygrad picks).
    from tinygrad import Device
    if require_cuda and Device.DEFAULT != 'CUDA':
        raise RuntimeError(
            f"tinygrad picked Device.DEFAULT={Device.DEFAULT!r} but expected "
            "'CUDA' — check the image (nvidia/cuda:*-devel-* required) and "
            "the gpu='T4' decorator.")
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    payload = _pickle.loads(panel_bytes)
    panel: dict = payload['panel']
    print(f'  loaded panel: {len(panel)} tickers, '
          f'universe_label={payload.get("universe_label", "?")}')

    from factor import (
        IndicatorGridConfig, build_indicator_features,
        train_scorer_indicators_walkforward,
    )
    from ss_features import TickerData

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'  cfg.feature_width() = {F} channels')

    # Build TickerData inline (no I/O — the pickle already has all closes).
    tickers_in = list(panel.keys())
    if max_tickers > 0:
        tickers_in = tickers_in[:max_tickers]

    ticker_data: list[TickerData] = []
    skipped: list[str] = []
    t0 = time.perf_counter()
    for name in tickers_in:
        df = panel[name]
        prices = df['close'].to_numpy(dtype=np.float64)
        dates = np.asarray(df.index)
        try:
            feats, valid = build_indicator_features(prices, cfg)
        except Exception as e:
            skipped.append(f'{name} ({type(e).__name__}: {e})')
            continue
        if not valid.any():
            skipped.append(f'{name} (no valid bars)')
            continue
        ticker_data.append(TickerData(
            name=name, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ))
    ticker_data.sort(key=lambda td: td.name)
    print(f'  built {len(ticker_data)} TickerData / '
          f'{len(skipped)} skipped  ({time.perf_counter()-t0:.1f}s)')
    if skipped[:5]:
        print(f'  first 5 skipped: {skipped[:5]}')
    if len(ticker_data) < 4:
        raise RuntimeError(
            f'only {len(ticker_data)} tickers built — too few for IC training')

    s_list = [s.strip() for s in scorers.split(',') if s.strip()]
    print(f'\n=== walk-forward per scorer: {s_list} ===', flush=True)
    print(f'  train={train_window_blocks} val={val_window_blocks} '
          f'step={step_window_blocks} (rebal_days={rebal_days})')

    periods_per_year = 365.0 / float(rebal_days)
    print(f'  periods_per_year = {periods_per_year:.3f}  '
          f'(crypto trades 7d/wk, NOT 252/yr)')

    summary: list[dict] = []
    for scorer in s_list:
        prefix = f'walkforward-crypto-{scorer}-s{n_steps}-wd{weight_decay:g}'
        print(f'\n  >>> {prefix}', flush=True)
        t1 = time.perf_counter()
        try:
            wf = train_scorer_indicators_walkforward(
                ticker_data, cfg,
                rebal_days=rebal_days,
                train_window_blocks=train_window_blocks,
                val_window_blocks=val_window_blocks,
                step_window_blocks=step_window_blocks,
                scorer=scorer,
                mlp_hidden=mlp_hidden, mlp_layers=mlp_layers,
                n_steps=n_steps, learning_rate=learning_rate,
                weight_decay=weight_decay,
                commission_bps=commission_bps,
                verbose=True,
            )
        except Exception as e:
            print(f'    FAILED: {type(e).__name__}: {e}', flush=True)
            summary.append({'scorer': scorer, 'failed': True,
                            'error': f'{type(e).__name__}: {e}'})
            continue
        wall = time.perf_counter() - t1
        # Optional cap on number of windows (smoke runs).
        if n_windows_cap > 0 and len(wf.windows) > n_windows_cap:
            wf.windows = wf.windows[:n_windows_cap]
            print(f'    capped to first {n_windows_cap} windows for smoke')
        npz_path = output_dir / f'{prefix}-windows.npz'
        _save_walkforward_npz(
            npz_path, wf, cfg,
            periods_per_year=periods_per_year,
            commission_bps=commission_bps,
            universe_label=payload.get('universe_label', '?'),
            pre_registered_bar=PRE_REGISTERED_BAR,
            dump_returns=dump_returns,
        )
        per_window = [
            {
                'window_idx': w.window_idx,
                'train_block_start': w.train_block_start,
                'train_block_end': w.train_block_end,
                'val_block_start': w.val_block_start,
                'val_block_end': w.val_block_end,
                'train_ic': w.train_ic, 'val_ic': w.val_ic,
                'train_sharpe': w.train_sharpe, 'val_sharpe': w.val_sharpe,
                'val_sharpe_long_short': w.val_sharpe_long_short,
                'n_train_bars': w.n_train_bars, 'n_val_bars': w.n_val_bars,
                'val_start_date': w.val_start_date,
            } for w in wf.windows
        ]
        summary.append({
            'scorer': scorer, 'n_steps': n_steps, 'weight_decay': weight_decay,
            'n_windows': wf.n_windows,
            'mean_val_ic': wf.mean_val_ic,
            'median_val_ic': wf.median_val_ic,
            'mean_val_sharpe': wf.mean_val_sharpe,
            'mean_val_sharpe_long_short': wf.mean_val_sharpe_long_short,
            'positive_val_ic_fraction': wf.positive_val_ic_fraction,
            'wall_seconds': round(wall, 1),
            'windows': per_window,
            'failed': False,
        })
        print(f'    {wf.n_windows} windows  mean val IC={wf.mean_val_ic:+.4f}  '
              f'median val IC={wf.median_val_ic:+.4f}  '
              f'pos-val-IC frac={wf.positive_val_ic_fraction:.2f}  '
              f'mean val Sharpe L/O={wf.mean_val_sharpe:+.3f}  '
              f'mean val Sharpe L/S={wf.mean_val_sharpe_long_short:+.3f}  '
              f'wall={wall:.1f}s')

    (output_dir / 'walkforward-crypto-summary.json').write_text(
        json.dumps({
            'universe_label': payload.get('universe_label', '?'),
            'universe_size': len(ticker_data),
            'feature_width': F,
            'rebal_days': rebal_days, 'learning_rate': learning_rate,
            'train_window_blocks': train_window_blocks,
            'val_window_blocks': val_window_blocks,
            'step_window_blocks': step_window_blocks,
            'commission_bps': commission_bps,
            'periods_per_year': periods_per_year,
            'pre_registered_bar': PRE_REGISTERED_BAR,
            'scorers': summary,
        }, indent=2))

    print('\n=== comparison plot ===', flush=True)
    _plot_walkforward(
        summary,
        output_dir / 'walkforward-crypto-comparison.png')

    artifacts: dict[str, bytes] = {}
    for p in sorted(output_dir.iterdir()):
        if p.is_file() and p.name.startswith('walkforward-crypto'):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts')
    return artifacts


# =============================================================================
# Modal remote function — thin wrapper around _run_walkforward.
# =============================================================================

def _modal_decorator_fn():
    return app.function(gpu='T4', cpu=8, memory=8192, timeout=2 * 60 * 60) \
        if _HAVE_MODAL else (lambda f: f)


@_modal_decorator_fn()
def train_walkforward_crypto(
    panel_bytes: bytes,
    scorers: str,
    n_steps: int,
    weight_decay: float,
    rebal_days: int,
    learning_rate: float,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    max_tickers: int,
    n_windows_cap: int,
    mlp_hidden: int,
    mlp_layers: int,
    commission_bps: float,
    dump_returns: bool,
) -> dict[str, bytes]:
    import os
    import subprocess
    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')

    # Pin tinygrad to CUDA *before* any tinygrad import in any worker.
    os.environ['CUDA'] = '1'

    print('=== uv sync workspace deps (one-time per cold start) ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'factor', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    return _run_walkforward(
        panel_bytes,
        scorers=scorers, n_steps=n_steps, weight_decay=weight_decay,
        rebal_days=rebal_days, learning_rate=learning_rate,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        max_tickers=max_tickers, n_windows_cap=n_windows_cap,
        mlp_hidden=mlp_hidden, mlp_layers=mlp_layers,
        commission_bps=commission_bps,
        output_dir=output, require_cuda=True,
        dump_returns=dump_returns,
    )


# =============================================================================
# Artifact writers (npz + plot).
# =============================================================================

def _save_walkforward_npz(
    path: Path, wf, cfg, *,
    periods_per_year: float,
    commission_bps: float,
    universe_label: str,
    pre_registered_bar: str,
    dump_returns: bool,
) -> None:
    """Pack per-window head params + per-window metrics into one npz.

    When `dump_returns=True`, also includes `oos_block_returns` and
    `oos_block_returns_long_short` for the cross-arc DSR harness.
    `periods_per_year` is the crypto-7d-week-corrected annualization
    factor (365/rebal_days).
    """
    import numpy as np
    blob: dict[str, np.ndarray] = {}
    blob['channel_names'] = np.array(cfg.channel_names())
    blob['window_idx'] = np.array([w.window_idx for w in wf.windows], dtype=np.int32)
    blob['train_block_start'] = np.array(
        [w.train_block_start for w in wf.windows], dtype=np.int32)
    blob['train_block_end'] = np.array(
        [w.train_block_end for w in wf.windows], dtype=np.int32)
    blob['val_block_start'] = np.array(
        [w.val_block_start for w in wf.windows], dtype=np.int32)
    blob['val_block_end'] = np.array(
        [w.val_block_end for w in wf.windows], dtype=np.int32)
    blob['train_ic'] = np.array([w.train_ic for w in wf.windows], dtype=np.float32)
    blob['val_ic']   = np.array([w.val_ic   for w in wf.windows], dtype=np.float32)
    blob['train_sharpe'] = np.array(
        [w.train_sharpe for w in wf.windows], dtype=np.float32)
    blob['val_sharpe'] = np.array(
        [w.val_sharpe for w in wf.windows], dtype=np.float32)
    blob['val_sharpe_long_short'] = np.array(
        [w.val_sharpe_long_short for w in wf.windows], dtype=np.float32)
    if dump_returns:
        blob['oos_block_returns'] = np.asarray(
            wf.oos_block_returns, dtype=np.float64)
        blob['oos_block_returns_long_short'] = np.asarray(
            wf.oos_block_returns_long_short, dtype=np.float64)
    # Crypto-corrected annualization. NOT 252/rebal_days.
    blob['periods_per_year'] = np.float64(periods_per_year)
    blob['commission_bps'] = np.float64(commission_bps)
    # Crypto spot has no borrow leg; long-short here is conceptual
    # (we are not actually shorting on spot data). Record 0 explicitly
    # so the downstream DSR harness reads the borrow convention.
    blob['borrow_bps_yr'] = np.float64(0.0)
    if wf.windows:
        for k in wf.windows[0].head_params:
            blob[f'head_{k}'] = np.stack(
                [np.asarray(w.head_params[k], dtype=np.float32)
                 for w in wf.windows])
    blob['_summary'] = np.array(json.dumps({
        'scorer': wf.scorer, 'n_steps': wf.n_steps,
        'learning_rate': wf.learning_rate, 'weight_decay': wf.weight_decay,
        'rebal_days': wf.rebal_days,
        'train_window_blocks': wf.train_window_blocks,
        'val_window_blocks': wf.val_window_blocks,
        'step_window_blocks': wf.step_window_blocks,
        'feature_width': wf.feature_width,
        'n_windows': wf.n_windows,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'mean_val_sharpe_long_short': wf.mean_val_sharpe_long_short,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
        'periods_per_year': periods_per_year,
        'commission_bps': commission_bps,
        'universe_label': universe_label,
    }))
    blob['pre_registered_bar'] = np.array(pre_registered_bar)
    blob['universe_label'] = np.array(universe_label)
    np.savez(path, **blob)
    print(f'    -> {path.name} ({path.stat().st_size // 1024}KB)')


def _plot_walkforward(summary: list[dict], out_path: Path) -> None:
    """Per-window train + val IC bars per scorer."""
    import matplotlib.pyplot as plt
    import numpy as np

    ok = [s for s in summary if not s.get('failed')]
    if not ok:
        print(f'  no successful scorers — skipping {out_path.name}')
        return
    n_scorers = len(ok)
    fig, axes = plt.subplots(n_scorers, 1, figsize=(10, 3.2 * n_scorers),
                             sharex=False)
    if n_scorers == 1:
        axes = [axes]
    for ax, s in zip(axes, ok):
        wins = s['windows']
        idx = np.array([w['window_idx'] for w in wins])
        tr = np.array([w['train_ic'] for w in wins])
        va = np.array([w['val_ic'] for w in wins])
        x = np.arange(len(idx))
        ax.bar(x - 0.2, tr, width=0.4, label='train IC', color='steelblue')
        ax.bar(x + 0.2, va, width=0.4, label='val IC', color='darkorange')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(0.025, color='green', linewidth=0.5, linestyle='--',
                   label='pre-reg bar (+0.025)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'w{i}' for i in idx], fontsize=8)
        ax.set_title(
            f"crypto top-50  {s['scorer']} "
            f"(n_steps={s['n_steps']}, wd={s['weight_decay']:g})  "
            f"— mean val IC={s['mean_val_ic']:+.4f}  "
            f"pos-val frac={s['positive_val_ic_fraction']:.2f}")
        ax.set_ylabel('IC')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  -> {out_path.name}')


# =============================================================================
# Local entrypoint (Modal driver). Reads the pickle, ships to remote.
# Stays free of project-venv-only deps (no `pandas`, `ss_loaders`).
# =============================================================================

def _modal_local_entrypoint_fn():
    return app.local_entrypoint() if _HAVE_MODAL else (lambda f: f)


@_modal_local_entrypoint_fn()
def walkforward(
    scorers: str = 'linear,mlp',
    n_steps: int = 200,
    weight_decay: float = 1e-3,
    rebal_days: int = 5,                 # matches equity +0.0212 winner
    learning_rate: float = 1e-2,
    # 110 / 55 / 55 blocks at rebal_days=5: train ~1.51y, val ~0.75y,
    # step = val (non-overlapping). At the default prep (min_bars=2000,
    # 42 tickers, common axis ~405 rebal blocks) this fits exactly 5
    # walk-forward windows — matching the pre-registered "5 windows"
    # target in `.research-venue-fit.md`.
    train_window_blocks: int = 110,
    val_window_blocks: int = 55,
    step_window_blocks: int = 55,
    max_tickers: int = 0,                # 0 = use all from pickle
    n_windows_cap: int = 0,              # 0 = no cap
    mlp_hidden: int = 64,
    mlp_layers: int = 1,
    commission_bps: float = 10.0,        # equity baseline; crypto retail may be higher
    dump_returns: bool = True,
) -> None:
    """Modal walk-forward driver. Reads the locally-built pickle and ships
    it to the remote function as raw bytes."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LOCAL_PICKLE.exists():
        print(f'ERROR: pickle not found at {LOCAL_PICKLE}', file=sys.stderr)
        print('  Run `uv run python apps/factor/scripts/prep_crypto_universe.py` first.',
              file=sys.stderr)
        sys.exit(1)
    panel_bytes = LOCAL_PICKLE.read_bytes()
    print(f'loaded panel pickle: {len(panel_bytes) / 1024 / 1024:.2f} MB')
    print(f'launching factor-indicator-crypto on Modal '
          f'(scorers={scorers}, n_steps={n_steps}, weight_decay={weight_decay}, '
          f'rebal_days={rebal_days}, '
          f'train_blocks={train_window_blocks}, val_blocks={val_window_blocks}, '
          f'commission_bps={commission_bps})')
    print(f'pre-registered bar: {PRE_REGISTERED_BAR}')
    artifacts = train_walkforward_crypto.remote(
        panel_bytes=panel_bytes,
        scorers=scorers,
        n_steps=n_steps,
        weight_decay=weight_decay,
        rebal_days=rebal_days,
        learning_rate=learning_rate,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        max_tickers=max_tickers,
        n_windows_cap=n_windows_cap,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
        commission_bps=commission_bps,
        dump_returns=dump_returns,
    )
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')


# =============================================================================
# Smoke-test entrypoint — runs the full driver locally (no Modal), against
# the same pickle. Lets us validate the scaffold on the local Intel Mac per
# CLAUDE.md's "smoke test locally first" rule before kicking off a Modal run.
#
# Invocation:
#   uv run python apps/factor/scripts/modal/train_indicator_crypto.py \
#       --max-tickers 8 --n-steps 30 --n-windows 2
# =============================================================================

def _smoke_main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description='Local smoke test for train_indicator_crypto walk-forward')
    ap.add_argument('--scorers', default='linear')
    ap.add_argument('--n-steps', type=int, default=30)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--rebal-days', type=int, default=5)
    ap.add_argument('--learning-rate', type=float, default=1e-2)
    ap.add_argument('--train-window-blocks', type=int, default=110)
    ap.add_argument('--val-window-blocks', type=int, default=55)
    ap.add_argument('--step-window-blocks', type=int, default=55)
    ap.add_argument('--max-tickers', type=int, default=8)
    ap.add_argument('--n-windows', type=int, default=2)
    ap.add_argument('--mlp-hidden', type=int, default=64)
    ap.add_argument('--mlp-layers', type=int, default=1)
    ap.add_argument('--commission-bps', type=float, default=10.0)
    ap.add_argument('--no-dump-returns', action='store_true')
    args = ap.parse_args()

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LOCAL_PICKLE.exists():
        print(f'ERROR: pickle not found at {LOCAL_PICKLE}', file=sys.stderr)
        print('  Run `uv run python apps/factor/scripts/prep_crypto_universe.py` first.',
              file=sys.stderr)
        sys.exit(1)
    panel_bytes = LOCAL_PICKLE.read_bytes()
    print(f'[smoke] loaded panel pickle: {len(panel_bytes) / 1024 / 1024:.2f} MB')
    print(f'[smoke] pre-registered bar: {PRE_REGISTERED_BAR}')

    _run_walkforward(
        panel_bytes,
        scorers=args.scorers,
        n_steps=args.n_steps,
        weight_decay=args.weight_decay,
        rebal_days=args.rebal_days,
        learning_rate=args.learning_rate,
        train_window_blocks=args.train_window_blocks,
        val_window_blocks=args.val_window_blocks,
        step_window_blocks=args.step_window_blocks,
        max_tickers=args.max_tickers,
        n_windows_cap=args.n_windows,
        mlp_hidden=args.mlp_hidden,
        mlp_layers=args.mlp_layers,
        commission_bps=args.commission_bps,
        output_dir=LOCAL_OUTPUT_DIR,
        require_cuda=False,
        dump_returns=not args.no_dump_returns,
    )
    print('[smoke] done — artifacts in', LOCAL_OUTPUT_DIR)


if __name__ == '__main__':
    _smoke_main()
