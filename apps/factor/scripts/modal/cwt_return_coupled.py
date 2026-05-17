"""Modal entrypoint — return-coupled recurrent CWT embedding k-sweep.

The pre-registered close-out of the
`apps/docs/docs/findings/cwt-recursive-compression.md` arc: a GRU state
over the 13-scale causal CWT, trained END-TO-END against the factor
cross-sectional rank-IC (encoder in the autograd graph, NOT a frozen
reservoir), swept over hidden dim `k`. See
`apps/docs/docs/TODO/factor-cwt-return-coupled.md` for the hypothesis +
pre-registered kill criterion.

Pre-registered (do not move post-hoc):
  * Positive (confirmed-OOS candidate): mean val-IC at k≤4 ≥ +0.0140
    AND ≥5/6 windows positive AND within −0.002 of the k=13 value.
  * Null: val-IC ≤ +0.0120 indicator baseline, or monotone-in-k with
    no plateau. Hard stop.

Usage
-----
Smoke (~3-5 min wall, ~$0.05 at T4 prices):
    uvx modal run apps/factor/scripts/modal/cwt_return_coupled.py \\
        --max-tickers 30 --n-steps 50 --ks 2,8

Full 297-ticker k-sweep (~1-2h wall):
    uvx modal run apps/factor/scripts/modal/cwt_return_coupled.py
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
            '.claude/**',
            # `apps/docs/site/` is the mkdocs build output; a live
            # ss-docs-serve livereload watcher rewrites it mid-upload
            # and Modal aborts with "modified during build process".
            'apps/docs/site/**',
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

app = modal.App('factor-cwt-return-coupled', image=image)

# Indicator-baseline factor-narrow bar (2026-04-30 confirmed-OOS).
INDICATOR_BASELINE_IC = 0.0120
POSITIVE_IC_CUT = 0.0140        # +0.002 over baseline
SATURATION_TOL = 0.002          # k≤4 within this of k=13


# 3h (vs the 2h MLP-head convention): the L=32 GRU BPTT × 200 steps ×
# 6 windows × 6 k is materially heavier than a feed-forward head, and
# artifacts only stream back at function end — a mid-sweep timeout
# loses the whole run, so budget generously rather than tightly.
@app.function(gpu='T4', cpu=4, memory=16384, timeout=3 * 60 * 60)
def cwt_gru_sweep_remote(
    ks_csv: str,
    rebal_days: int,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    seq_len: int,
    lookback: int,
    n_steps: int,
    learning_rate: float,
    weight_decay: float,
    commission_bps: float,
    seed: int,
    tickers: str,
    start: str,
    end: str,
    max_tickers: int,
    min_history_bars: int,
) -> dict[str, bytes]:
    """Build the causal-CWT panel once on the universe, then run a
    fresh leak-free walk-forward per `k`. Bundle one npz + a sweep
    summary json + a val-IC-vs-k plot."""
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
            f'tinygrad Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    print('\n=== Step 2/4: load tickers + build causal-CWT panels ===',
          flush=True)
    ticker_list = _resolve_ticker_list(tickers, max_tickers, min_history_bars)
    print(f'  universe: {len(ticker_list)} tickers '
          f'(first 5: {ticker_list[:5]} ...)')

    import multiprocessing as mp
    import numpy as np
    from factor import train_cwt_gru_walkforward
    from ss_features import TickerData

    n_workers = max(1, int(os.environ.get('FACTOR_FEATURE_WORKERS',
                                          os.cpu_count() or 4)))
    print(f'  parallelizing panel build across {n_workers} workers')
    t0 = time.perf_counter()
    work = [(t, STOOQ_SUBSET, start, end, lookback) for t in ticker_list]
    ticker_data: list[TickerData] = []
    skipped: list[str] = []
    with mp.Pool(n_workers) as pool:
        for i, (tk, res) in enumerate(
                pool.imap_unordered(_build_one_ticker, work)):
            if isinstance(res, TickerData):
                ticker_data.append(res)
            else:
                skipped.append(f'{tk} {res}')
            if (i + 1) % 50 == 0:
                print(f'  built {i + 1}/{len(ticker_list)}  '
                      f'({time.perf_counter()-t0:.0f}s)', flush=True)
    ticker_data.sort(key=lambda td: td.name)
    print(f'  panel build done: {len(ticker_data)} usable / '
          f'{len(skipped)} skipped  ({time.perf_counter()-t0:.0f}s)')
    if len(ticker_data) < 4:
        raise RuntimeError(
            f'only {len(ticker_data)} tickers built — too few for IC')

    ks = [int(x) for x in ks_csv.split(',') if x.strip()]
    print(f'\n=== Step 3/4: rank-IC-trained GRU-over-CWT k-sweep '
          f'(ks={ks}) ===', flush=True)

    sweep: list[dict] = []
    blob: dict[str, np.ndarray] = {}
    for k in ks:
        print(f'\n  --- k={k} ---', flush=True)
        t1 = time.perf_counter()
        res = train_cwt_gru_walkforward(
            ticker_data, k=k,
            rebal_days=rebal_days,
            train_window_blocks=train_window_blocks,
            val_window_blocks=val_window_blocks,
            step_window_blocks=step_window_blocks,
            seq_len=seq_len, lookback=lookback,
            n_steps=n_steps, learning_rate=learning_rate,
            weight_decay=weight_decay, commission_bps=commission_bps,
            seed=seed, verbose=True)
        wall = time.perf_counter() - t1
        per_win_ic = [w.val_ic for w in res.windows]
        print(f'    k={k}: mean val IC={res.mean_val_ic:+.4f}  '
              f'pos-frac={res.positive_val_ic_fraction:.2f}  '
              f'val Sharpe={res.mean_val_sharpe:+.3f}  '
              f'val IR-vs-EW={res.mean_val_ir_vs_ew:+.3f}  '
              f'({wall:.0f}s)', flush=True)
        blob[f'val_ic_k{k}'] = np.array(per_win_ic, dtype=np.float32)
        blob[f'train_ic_k{k}'] = np.array(
            [w.train_ic for w in res.windows], dtype=np.float32)
        blob[f'val_sharpe_k{k}'] = np.array(
            [w.val_sharpe for w in res.windows], dtype=np.float32)
        blob[f'val_ir_vs_ew_k{k}'] = np.array(
            [w.val_ir_vs_ew for w in res.windows], dtype=np.float32)
        sweep.append({
            'k': k,
            'mean_val_ic': res.mean_val_ic,
            'positive_val_ic_fraction': res.positive_val_ic_fraction,
            'mean_val_sharpe': res.mean_val_sharpe,
            'mean_val_ir_vs_ew': res.mean_val_ir_vs_ew,
            'per_window_val_ic': per_win_ic,
            'per_window_train_ic': [w.train_ic for w in res.windows],
            'per_window_val_start': [w.val_start_date for w in res.windows],
            'wall_seconds': round(wall, 1),
        })

    print('\n=== Step 4/4: pre-registered verdict ===', flush=True)
    by_k = {s['k']: s for s in sweep}
    ref = by_k.get(13)
    low_ks = [s for s in sweep if s['k'] <= 4]
    best_low = max(low_ks, key=lambda s: s['mean_val_ic']) if low_ks else None

    verdict = 'confirmed-null'
    reason = ''
    if best_low is not None:
        ic_lo = best_low['mean_val_ic']
        pos_lo = best_low['positive_val_ic_fraction']
        ic_pass = ic_lo >= POSITIVE_IC_CUT
        pos_pass = pos_lo >= 5.0 / 6.0
        sat_pass = (ref is not None
                    and ic_lo >= ref['mean_val_ic'] - SATURATION_TOL)
        if ic_pass and pos_pass and sat_pass:
            verdict = 'confirmed-OOS'
            reason = (f'k={best_low["k"]} val-IC {ic_lo:+.4f} ≥ '
                      f'{POSITIVE_IC_CUT}, pos {pos_lo:.2f}, saturates')
        elif ic_pass and pos_pass:
            verdict = 'partial-OOS'
            reason = (f'k={best_low["k"]} clears IC bar but needs ~k=13 '
                      f'(no k≤4 plateau within {SATURATION_TOL})')
        else:
            verdict = 'confirmed-null'
            reason = (f'best k≤4 val-IC {ic_lo:+.4f} below '
                      f'{POSITIVE_IC_CUT} cut / pos {pos_lo:.2f}')
    print(f'  verdict: {verdict} — {reason}', flush=True)
    print(f'  {"k":>4}  {"mean_val_ic":>12}  {"pos":>4}  '
          f'{"val_sh":>7}  {"val_ir":>7}', flush=True)
    for s in sweep:
        print(f'  {s["k"]:>4}  {s["mean_val_ic"]:>+12.4f}  '
              f'{s["positive_val_ic_fraction"]:>4.2f}  '
              f'{s["mean_val_sharpe"]:>+7.3f}  '
              f'{s["mean_val_ir_vs_ew"]:>+7.3f}', flush=True)

    summary = {
        'experiment': 'cwt-return-coupled-gru',
        'universe_size': len(ticker_data),
        'ks': ks,
        'rebal_days': rebal_days,
        'train_window_blocks': train_window_blocks,
        'val_window_blocks': val_window_blocks,
        'step_window_blocks': step_window_blocks,
        'seq_len': seq_len,
        'lookback': lookback,
        'n_steps': n_steps,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'commission_bps': commission_bps,
        'seed': seed,
        'indicator_baseline_ic': INDICATOR_BASELINE_IC,
        'positive_ic_cut': POSITIVE_IC_CUT,
        'saturation_tol': SATURATION_TOL,
        'verdict': verdict,
        'verdict_reason': reason,
        'arms': sweep,
    }
    blob['_summary'] = np.array(json.dumps(summary, indent=2))
    npz_path = output / 'cwt-return-coupled-windows.npz'
    np.savez(npz_path, **blob)
    summary_path = output / 'cwt-return-coupled-summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    plot_path = output / 'cwt-return-coupled-ic-vs-k.png'
    _plot_ic_vs_k(sweep, plot_path)
    print(f'  -> {npz_path.name}, {summary_path.name}, {plot_path.name}',
          flush=True)

    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name.startswith('cwt-return-coupled'):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts')
    return artifacts


def _build_one_ticker(args):
    ticker, stooq_subset, start, end, lookback = args
    from factor import load_ticker_cwt
    from ss_features import TickerData
    try:
        td = load_ticker_cwt(
            ticker, stooq_dir=stooq_subset, start=start, end=end,
            lookback=lookback)
        if not td.valid.any():
            return ticker, '(no valid bars)'
        return ticker, td
    except Exception as e:  # noqa: BLE001 — worker reports, never crashes pool
        return ticker, f'({type(e).__name__}: {e})'


def _resolve_ticker_list(
    tickers: str, max_tickers: int, min_history_bars: int = 0,
) -> list[str]:
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
        dropped = before - len(entries)
        if dropped:
            print(f'  min_history_bars={min_history_bars}: '
                  f'dropped {dropped} short-history tickers')
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
    return names


def _plot_ic_vs_k(sweep: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not sweep:
        return
    ks = [s['k'] for s in sweep]
    mean_ic = [s['mean_val_ic'] for s in sweep]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, mean_ic, marker='o', color='crimson',
            label='mean val rank-IC (return-coupled GRU)')
    ax.axhline(INDICATOR_BASELINE_IC, color='gray', linestyle='--',
               label=f'indicator baseline +{INDICATOR_BASELINE_IC:.4f}')
    ax.axhline(POSITIVE_IC_CUT, color='green', linestyle=':',
               label=f'positive cut +{POSITIVE_IC_CUT:.4f}')
    ax.axhline(0.0, color='black', linewidth=0.5)
    for s in sweep:
        ax.annotate(f'{s["mean_val_ic"]:+.4f}', (s['k'], s['mean_val_ic']),
                    textcoords='offset points', xytext=(0, 6), fontsize=7,
                    ha='center')
    ax.set_xscale('log', base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('GRU hidden dim k (log2)')
    ax.set_ylabel('mean val rank-IC (6-window walk-forward)')
    ax.set_title('Return-coupled recurrent CWT embedding — does predictive '
                 'structure saturate at low k?')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@app.local_entrypoint()
def main(
    ks: str = '2,4,8,13,16,32',
    rebal_days: int = 20,
    train_window_blocks: int = 63,
    val_window_blocks: int = 39,
    step_window_blocks: int = 39,
    seq_len: int = 32,
    lookback: int = 90,
    n_steps: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    commission_bps: float = 10.0,
    seed: int = 0,
    tickers: str = '',
    start: str = '2000-01-01',
    end: str = '2026-04-01',
    max_tickers: int = 0,
    min_history_bars: int = 6500,
) -> None:
    """Kick off the remote k-sweep and download artifacts.

    Defaults are the pre-registered factor-narrow config (297
    stooq_us_long, min_history_bars=6500, the deterministic-indicator-
    baseline 6-window windowing). Smoke with `--max-tickers 30
    --n-steps 50 --ks 2,8`.
    """
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'launching cwt-return-coupled GRU k-sweep on Modal '
          f'(ks={ks}, rebal_days={rebal_days}, '
          f'train/val/step={train_window_blocks}/{val_window_blocks}/'
          f'{step_window_blocks}, seq_len={seq_len}, lookback={lookback}, '
          f'n_steps={n_steps}, lr={learning_rate}, wd={weight_decay}, '
          f'max_tickers={max_tickers}, min_history_bars={min_history_bars})')
    artifacts = cwt_gru_sweep_remote.remote(
        ks_csv=ks,
        rebal_days=rebal_days,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        seq_len=seq_len,
        lookback=lookback,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        commission_bps=commission_bps,
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
