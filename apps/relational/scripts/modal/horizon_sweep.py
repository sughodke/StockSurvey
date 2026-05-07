"""Modal entrypoint: rebal-horizon sweep on the two wider-universe winners.

!!! UNVALIDATED — never produced a successful end-to-end run !!!
This script was authored by background agent a7a468ab53674115f, which
died via stream watchdog (600s no progress) before the Modal launch
completed. The code passes static review and matches the
prep-then-ship pattern of `apps/factor/scripts/modal/{prep_universe_
pivot_data, universe_pivot_vol_arm}.py`, but no run output has been
checked. Treat the first invocation as a smoke test. The arc itself
(Phase-13 horizon sweep) was labeled non-load-bearing for the live-
trade deploy decision — preserved here for future revisitation, not
because the result is needed.

Phase 13 of the `apps/relational/` arc. Tests whether `rebal_days=20`
is actually optimal for the two shippable wide-universe constructions
on the 312-ticker `apps/notebook/data/stooq_us_long/` subset:

  * **transition-triggered** — Phase 9 (`diagnostic_transition_triggered`),
    Sharpe 0.63 reported at `rebal_days=20` (transition events fire
    rebals; the 20d cadence is only the union/fallback grid — for the
    `transition-only` arm there is no scheduled cadence to vary, so
    this construction's `rebal_days` cell is effectively a control).
    To make the sweep *meaningful* for this construction, we vary the
    union grid: `transition-or-{5,10,20,63}d` keeps the transition
    triggers and adjusts the fallback cadence.
  * **velocity-magnitude** — Phase 11 (`diagnostic_velocity`),
    Sharpe 0.60 reported at `rebal_days=20`. Scheduled rebal at the
    horizon under test; W=21 fixed across cells (a separate sweep
    target if needed).

Cells:  4 horizons × 2 constructions = 8 backtests.

Compute model:
  - One CWT precompute (the dominant cost, ~30-60s on T4 CPU).
  - One cluster-ID refit + Hungarian-stabilization pass (only needed
    for the transition arm).
  - One fingerprint + velocity-magnitude score pass (only needed for
    the velocity arm).
  - Per-cell: weight matrix → bt backtest. bt is single-threaded but
    each backtest on the 312-ticker x ~5500-day panel runs in tens of
    seconds, so the 8-cell sweep is ~few-min wall.

Image base: GPU CUDA dev image to reuse the cached layer the rest of
the repo's Modal entrypoints use. We don't actually need the GPU here
(numpy + bt + sklearn), but reusing the cached image avoids a fresh
3-GB pull. `cpu=8 timeout=2*60*60` matches the repo convention.

Usage:
    # 1. One-time local prep (uses project venv, pickles close DF):
    uv run python apps/relational/scripts/modal/prep_horizon_sweep_data.py

    # 2. Ship to Modal:
    uvx modal run apps/relational/scripts/modal/horizon_sweep.py

Returns
-------
  Output/relational-horizon-sweep-stats.txt
  Output/relational-horizon-sweep-leaderboard.csv
  Output/relational-horizon-sweep-stats.json
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


# CUDA dev image — reused across the repo's Modal entrypoints so the
# layer is cached on Modal's side. We don't actually need the GPU; this
# workload is numpy + bt + sklearn. The `cpu=8` request asks Modal for
# 8 cores (T4 instances expose ~24).
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
        # NOTE: `apps/relational/src/**` is intentionally INCLUDED here
        # (the factor-side Modal scripts ignore it for build-speed; we
        # need it). Other apps stay excluded so concurrent edits in
        # those trees don't perturb the image hash.
        ignore=[
            '.git/**',
            '.venv/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            '.iv-cache/**',
            '.scalogram-cache/**',
            'apps/relational/.scalogram-cache/**',
            'apps/factor/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            'apps/replay/src/**',
            'apps/notebook/src/**',
            # Ship the 312-ticker data subset since it's only ~140MB —
            # avoids a Modal Volume round-trip just for an 8-cell sweep.
            # The rest of `apps/notebook/data` is excluded.
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('relational-horizon-sweep', image=image)


@app.function(cpu=8, memory=32768, timeout=2 * 60 * 60)
def horizon_sweep(
    close_pickle: bytes,
    *,
    horizons: list[int],
    top_n: int,
    lookback: int,
    n_tail: int,
    fp_window: int,
    w_delta: int,
    k_clusters: int,
    refit_days: int,
    persistence: int,
    commission_bps: float,
    seed: int,
) -> dict[str, bytes]:
    """Run the 4-horizon × 2-construction sweep on a single container.

    Compute is shared across cells:
      - CWT precompute happens once (depends only on prices/scales/lookback).
      - Cluster IDs + transition mask: once (only used by transition arm).
      - Velocity-magnitude scores: once (only used by velocity arm).

    Per cell: build weight matrix and bt.run a single backtest.
    """
    import os
    import pickle
    import subprocess
    os.environ['CUDA'] = '1'

    print('=== Step 1/5: uv sync workspace deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'relational',
         '--extra', 'research', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    import io
    import numpy as np
    import pandas as pd
    import bt

    from ss_portfolio import apply_nan_mask, select_top_n_matrix
    from ss_portfolio.bt_helpers import (
        bt_safe_prices, build_strategy, make_commission_fn,
    )

    from relational.cluster_tracking import stabilize_cluster_ids
    from relational.empirical_sectors import (
        empirical_excess_divergence_scores,
    )
    from relational.fingerprints import extract_fingerprints
    from relational.regime_velocity import velocity_magnitude_scores
    from relational.scalogram_cache import load_or_compute_cwt
    from relational.transitions import (
        detect_transition_dates, trigger_dates_from_transitions,
    )

    print('\n=== Step 2/5: deserialize close DataFrame from RPC ===',
          flush=True)
    close: pd.DataFrame = pickle.loads(close_pickle)
    print(f'  shape: {close.shape}  '
          f'date range: {close.index[0].date()} .. {close.index[-1].date()}',
          flush=True)

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, k_clusters={k_clusters}, '
          f'fp_window={fp_window}, refit_days={refit_days}, '
          f'persistence={persistence}, w_delta={w_delta}', flush=True)

    # CWT cache lives in the container's tmp; freshly written this run.
    cache_dir = Path('/tmp/scalogram-cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    print('\n=== Step 3/5: empirical scores + cluster IDs (transition arm) ===',
          flush=True)
    t0 = time.perf_counter()
    emp_scores, raw_cluster_ids_eval = empirical_excess_divergence_scores(
        close, lookback=lookback, n_tail=n_tail, scales=scales,
        fp_window=fp_window, k_clusters=k_clusters,
        refit_days=refit_days, return_clusters=True,
        cache_dir=cache_dir.as_posix())
    n_dates_full = len(close)
    raw_cluster_ids_full = np.full(
        (n_dates_full, close.shape[1]), -1, dtype=np.int64)
    raw_cluster_ids_full[lookback:] = raw_cluster_ids_eval

    coeffs = load_or_compute_cwt(
        close, scales, lookback, cache_dir=cache_dir.as_posix())
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)

    stable_full = stabilize_cluster_ids(
        raw_cluster_ids_full, fps,
        refit_days=refit_days, lookback=lookback)
    cluster_ids_eval = stable_full[lookback:]

    transitions = detect_transition_dates(
        cluster_ids_eval, persistence=persistence)
    n_events = int(transitions.sum())
    print(f'  empirical+clusters+transitions in '
          f'{time.perf_counter() - t0:.0f}s; '
          f'{n_events} transition events total', flush=True)

    # idea-A top-N (NaN-protected)
    masked_emp = apply_nan_mask(emp_scores, close.values, lookback)
    emp_top_arr = select_top_n_matrix(masked_emp, top_n, ascending=False)
    emp_top = pd.DataFrame(
        emp_top_arr, index=close.index[lookback:], columns=close.columns)
    # zero out NaN/inactive cells, renormalize per row
    valid = np.isfinite(close.reindex(
        index=emp_top.index, columns=emp_top.columns).values)
    w_arr = emp_top.fillna(0).values * valid
    sums = w_arr.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    emp_top = pd.DataFrame(
        w_arr / sums, index=emp_top.index, columns=emp_top.columns)

    pick_mask = emp_top.values > 0
    transition_dates = trigger_dates_from_transitions(
        transitions, close.index, lookback, selected_columns=pick_mask)
    print(f'  filtered transition trigger dates: {len(transition_dates)}',
          flush=True)

    print('\n=== Step 4/5: velocity-magnitude scores (velocity arm) ===',
          flush=True)
    t0 = time.perf_counter()
    vel_scores = velocity_magnitude_scores(
        close, lookback=lookback, scales=scales,
        fp_window=fp_window, w_delta=w_delta,
        cache_dir=cache_dir.as_posix())
    masked_vel = apply_nan_mask(vel_scores, close.values, lookback)
    vel_top_arr = select_top_n_matrix(masked_vel, top_n, ascending=False)
    vel_top = pd.DataFrame(
        vel_top_arr, index=close.index[lookback:], columns=close.columns)
    valid = np.isfinite(close.reindex(
        index=vel_top.index, columns=vel_top.columns).values)
    w_arr = vel_top.fillna(0).values * valid
    sums = w_arr.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    vel_top = pd.DataFrame(
        w_arr / sums, index=vel_top.index, columns=vel_top.columns)
    print(f'  velocity scores in {time.perf_counter() - t0:.0f}s',
          flush=True)

    print('\n=== Step 5/5: build + run 8 bt backtests ===', flush=True)
    bt_safe = bt_safe_prices(close)
    weights_index = emp_top.index

    # ----- transition arm -----
    # Phase 9's "transition-only" arm has no scheduled cadence to vary
    # — it fires solely on the 25 transition events. We *do* sweep the
    # union grid (transition-or-{5,10,20,63}d) which exposes how the
    # fallback cadence interacts with transition triggers. We also keep
    # a `transition-only` cell as a horizon-invariant control (so the
    # leaderboard has the Phase-9 reference Sharpe to compare against).
    transition_cells: list[tuple[str, list[pd.Timestamp]]] = [
        ('transition|only', transition_dates),
    ]
    for h in horizons:
        scheduled = list(weights_index[::h])
        union = sorted(set(scheduled) | set(transition_dates))
        transition_cells.append((f'transition|or-{h}d', union))

    # ----- velocity arm -----
    # Pure scheduled rebal at horizon `h`; W=21 fixed across cells.
    velocity_cells: list[tuple[str, int]] = [
        (f'velocity|{h}d', h) for h in horizons
    ]

    print(f'  total cells: {len(transition_cells) + len(velocity_cells)} '
          f'({len(transition_cells)} transition + '
          f'{len(velocity_cells)} velocity)', flush=True)

    # Build all backtests, then run them in one bt.run for shared price
    # frame caching.
    backtests: list[bt.Backtest] = []
    rebal_counts: dict[str, int] = {}

    def _build_with_dates(label, weights_df, dates):
        aligned = sorted(set(d for d in dates if d in weights_df.index))
        if not aligned:
            aligned = [weights_df.index[0]]
        rw = weights_df.loc[aligned]
        nonzero = rw.abs().sum(axis=1) > 0.1
        if nonzero.any():
            rw = rw.loc[nonzero]
        rebal_counts[label] = len(rw)
        strategy = bt.Strategy(label, [
            bt.algos.RunOnDate(*rw.index),
            bt.algos.WeighTarget(rw),
            bt.algos.Rebalance(),
        ])
        return bt.Backtest(
            strategy, bt_safe,
            commissions=make_commission_fn(commission_bps),
            integer_positions=False)

    for label, dates in transition_cells:
        backtests.append(_build_with_dates(label, emp_top, dates))

    for label, h in velocity_cells:
        bt_obj = build_strategy(
            label, close, vel_top,
            rebal_days=h, commission_bps=commission_bps,
            drop_empty=True, safe_prices=True)
        backtests.append(bt_obj)
        # mirror the bt_helpers slicing logic for reporting
        rebal_counts[label] = int(
            (vel_top.iloc[::h].abs().sum(axis=1) > 0.1).sum())

    print(f'  bt.run({len(backtests)}) ...', flush=True)
    t0 = time.perf_counter()
    result = bt.run(*backtests)
    print(f'  bt.run wall: {time.perf_counter() - t0:.0f}s', flush=True)

    # ----- format leaderboard -----
    stats = result.stats
    headline = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                'daily_vol', 'total_return']
    sharpe_row = stats.loc['daily_sharpe'].astype(float)
    order = sharpe_row.sort_values(ascending=False).index.tolist()
    leaderboard = stats.loc[headline, order].T.copy()
    leaderboard['n_rebals'] = leaderboard.index.map(rebal_counts)
    leaderboard = leaderboard[
        ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
         'n_rebals', 'daily_vol']
    ]

    print('\n' + '=' * 100, flush=True)
    print('Phase 13 horizon-sweep leaderboard — sorted by daily Sharpe',
          flush=True)
    print(f'  universe: {close.shape[1]} tickers, '
          f'{close.index[0].date()} → {close.index[-1].date()}', flush=True)
    print('=' * 100, flush=True)
    with pd.option_context(
        'display.float_format', lambda x: f'{x:.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print(leaderboard.to_string(), flush=True)

    # ----- pack artifacts -----
    leaderboard_csv = leaderboard.to_csv()

    stats_txt_lines = [
        'Phase 13 — relational rebal-horizon sweep',
        f'  universe: {close.shape[1]} tickers '
        f'({close.index[0].date()} → {close.index[-1].date()})',
        f'  top_n={top_n}  lookback={lookback}  n_tail={n_tail}  '
        f'fp_window={fp_window}',
        f'  k_clusters={k_clusters}  refit_days={refit_days}  '
        f'persistence={persistence}',
        f'  w_delta={w_delta}  commission_bps={commission_bps}',
        f'  horizons={horizons}',
        f'  total transition events: {n_events}',
        f'  transition trigger dates (filtered to top-N picks): '
        f'{len(transition_dates)}',
        '=' * 100,
        leaderboard.to_string(),
        '',
        'Per-cell rebal counts:',
    ]
    for label in order:
        stats_txt_lines.append(
            f'  {label:>32s}  n_rebals={rebal_counts.get(label, 0)}')
    stats_txt_lines.append('')
    stats_txt_lines.append('Full bt stats:')
    stats_txt_lines.append(str(stats))
    stats_txt = '\n'.join(stats_txt_lines)

    stats_json = json.dumps({
        'universe': {
            'n_tickers': int(close.shape[1]),
            'first_date': str(close.index[0].date()),
            'last_date': str(close.index[-1].date()),
        },
        'config': {
            'horizons': horizons,
            'top_n': top_n,
            'lookback': lookback,
            'n_tail': n_tail,
            'fp_window': fp_window,
            'w_delta': w_delta,
            'k_clusters': k_clusters,
            'refit_days': refit_days,
            'persistence': persistence,
            'commission_bps': commission_bps,
            'seed': seed,
        },
        'transition_events_total': int(n_events),
        'transition_trigger_dates_filtered': len(transition_dates),
        'rebal_counts': rebal_counts,
        'leaderboard': leaderboard.astype(float).to_dict(orient='index'),
        'order_desc': order,
    }, indent=2, default=float)

    return {
        'relational-horizon-sweep-leaderboard.csv': leaderboard_csv.encode(),
        'relational-horizon-sweep-stats.txt':       stats_txt.encode(),
        'relational-horizon-sweep-stats.json':      stats_json.encode(),
    }


@app.local_entrypoint()
def main(
    horizons: str = '5,10,20,63',
    top_n: int = 20,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    w_delta: int = 21,
    k_clusters: int = 11,
    refit_days: int = 252,
    persistence: int = 5,
    commission_bps: float = 10.0,
    seed: int = 0,
) -> None:
    """Read pre-prepped pickle as raw bytes and ship to Modal.

    Run `prep_horizon_sweep_data.py` first to populate the pickle.
    """
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pickle_path = LOCAL_OUTPUT_DIR / 'relational_horizon_sweep_close.pkl'
    if not pickle_path.exists():
        raise SystemExit(
            f'pickle not found at {pickle_path}. Run prep first:\n'
            f'  uv run python apps/relational/scripts/modal/'
            f'prep_horizon_sweep_data.py')

    horizons_list = [int(h) for h in horizons.split(',') if h.strip()]
    print(f'[local] reading {pickle_path} '
          f'({pickle_path.stat().st_size / 1024 / 1024:.1f} MB)', flush=True)
    close_pickle = pickle_path.read_bytes()

    print(f'[local] launching Modal horizon_sweep.remote '
          f'(horizons={horizons_list}, top_n={top_n}) ...', flush=True)
    t0 = time.perf_counter()
    artifacts = horizon_sweep.remote(
        close_pickle,
        horizons=horizons_list,
        top_n=top_n,
        lookback=lookback,
        n_tail=n_tail,
        fp_window=fp_window,
        w_delta=w_delta,
        k_clusters=k_clusters,
        refit_days=refit_days,
        persistence=persistence,
        commission_bps=commission_bps,
        seed=seed,
    )
    print(f'[local] remote done in {time.perf_counter() - t0:.0f}s',
          flush=True)

    for name, blob in artifacts.items():
        out_path = LOCAL_OUTPUT_DIR / name
        out_path.write_bytes(blob)
        print(f'[local] wrote {out_path} ({len(blob) / 1024:.1f} KB)',
              flush=True)
