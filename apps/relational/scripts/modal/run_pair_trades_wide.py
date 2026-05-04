"""Modal entrypoint: wide-universe pair-trade diagnostic on stooq_us_long.

Mirrors `relational/research/diagnostic_pair_trades_wide.py` but targets
Modal CPU instances. The relational pair-trade workload is **CPU-bound
numpy + bt** — no GPU, unlike `apps/replay` and `apps/factor`. This file
matches the Modal-app conventions established in those scripts (Image
+ App + remote function decorators + add_local_dir + uv-based env
sync) but with a CPU-only base image and a Modal Volume for persisted
data.

This file is a **scaffold**. It has not been deployed. Sections marked
``# DEPLOY ME`` are deliberately minimal stubs until someone runs it
end-to-end.

Design notes (see also: this docstring + the per-section comments)
==================================================================

Modal volume layout
-------------------
One volume, ``relational-data``, mounted at ``/vol`` inside the
container. First-run upload step (``upload_data``) populates three
trees from the caller's local repo, all read-only thereafter:

    /vol/stooq_us_long/                ~140 MB   (Stooq daily CSV archive,
                                                   the same 312-ticker
                                                   curated subset apps/factor
                                                   uses)
    /vol/iv-cache/                     ~22 MB    (DoltHub volatility_history.parquet
                                                   only — the full 7.9 GB dolt
                                                   clone stays local)
    /vol/scalogram-cache/              grows     (causal_cwt npz outputs;
                                                   keyed by content hash on
                                                   prices, so identical input
                                                   panels short-circuit)

The scalogram cache **must** live on the volume rather than in the
ephemeral container filesystem. ``relational.scalogram_cache``
content-hashes the input prices and writes ``cwt-{hash}.npz``; the same
panel of 312 tickers hashes to the same key, so cache writes from
prior Modal runs are reused on subsequent runs. Without volume
persistence each cold container would recompute the CWT from scratch
(~2-3 min wall on 312 × 6618 × 8 scales).

The DoltHub IV parquet is included for forward compatibility with
options / vol-arb cross-tests on this scaffold; the equity-only
pair-trade diagnostic does not read it. Tiny enough (22 MB) that it
costs nothing to keep co-located.

The full `pkgs.dolt` clone (~7.5 GB) is intentionally **not** uploaded
— the upstream source-of-truth is DoltHub, the parquet is the
extracted slice we actually need, and re-cloning into the volume on
every refresh would burn upload bandwidth + storage for no gain.

Compute breakdown — what's worth Modal-izing
--------------------------------------------
*Modal-eligible (CPU-parallel)*:
  - **Causal CWT precomputation** is parallelizable per ticker (pure
    numpy + scipy convolve, embarrassingly parallel across the
    n_tickers axis). On a 16-core CPU instance this turns the ~2-3 min
    serial CWT into ~15-20s. Output goes into the persistent volume's
    `scalogram-cache/` and is reused across runs.
  - **bt backtests are independent across the 10 strategies** —
    `_run_one_backtest` already accepts a single `(label, prices,
    weights, ...)` tuple and reconstructs the bt graph in the worker.
    Modal's `Function.map` parallelizes the 10 jobs across as many CPU
    instances as we want; bt is single-threaded per backtest so this
    gives a 10× wall-time speedup if we provision 10 instances.

*Stays local / inside one container*:
  - **Score computation** (per-stock CWT-power KL, empirical k-means
    clustering, centroid distance) is sequential per scorer and runs
    in <30s each on the wider universe; not worth distributing.
  - **Cluster-id refit** (k-means every refit_days bars) is fast and
    must happen before `cluster_pair_weights` can run; sequential by
    construction.
  - **Final stitching + plot** is a few seconds.

The natural Modal shape is therefore: one orchestrator function that
loads prices, computes scores (using the cached CWT from the volume),
and `Function.map`s the 10 bt jobs out to a fan-out of CPU workers.
The orchestrator can run on a smaller box (n_cpu=4) since it's mostly
glue.

Cost / time estimates
---------------------
Local (M-class macOS, 8 perf cores, parallel bt):
  - Load + CWT (cold cache):       ~3 min
  - Score (3 scorers):              ~2-3 min
  - 10 bt backtests, 8-way pool:   ~10-15 min
  - **total ≈ 15-25 min wall**

Local (cold, no parallelism):
  - **total ≈ 30-50 min wall**, mostly bt rebalance solver

Modal CPU (warm volume, 10 parallel bt workers, n_cpu=4 each):
  - Cold start of orchestrator:    ~30s (uv sync, image cached)
  - Load + CWT (warm cache):       ~10s (npz load from volume)
  - Score (3 scorers):              ~2-3 min
  - 10 bt backtests fanned out:    ~3-4 min (each on its own worker)
  - Stitching + artifact return:   ~5s
  - **total ≈ 6-8 min wall**

  Cost (Modal CPU pricing, generic billable rate ≈ $0.000050/cpu-sec):
  - Orchestrator: ~5 cpu-min            ≈ $0.015
  - 10 × bt workers: ~5 cpu-min each    ≈ $0.150
  - **total ≈ $0.15-0.20 per run**

Modal CPU first-run (cold volume, full CWT recompute):
  - Add ~3 min wall + ~$0.04 to upload + CWT-compute. Subsequent runs
    against the same Stooq subset hit the cached scalogram (the
    content-hash key is on the price bytes), so re-runs are ~6-8 min.

When Modal helps
----------------
For the 10-backtest equity-only pair-trade diagnostic specifically,
local parallelism on a half-decent laptop is already within 2-3× of
Modal wall time. The Modal payoff is real only when:

  1. You're running this on a thin laptop (4 cores or fewer) — Modal's
     16-core orchestrator + fan-out is a 4-5× speedup.
  2. You expand the variant grid (e.g., scorer × {KL, JS, cosine, L2} ×
     {top-N=10, 20, 30} × {rebal=10, 20, 40}) — the local linear cost
     scales but Modal can fan out to dozens of workers in parallel.
  3. You're running in a CI environment without ~32 GB RAM available
     for the wide-universe panel (Modal's CPU instances default to
     16 GB, sufficient here, and easy to bump).

For the "did wider-universe move the needle" one-shot test, the local
runtime is fine.

Concrete next steps to actually deploy
--------------------------------------
1. ``pip install --user modal`` (or ``uv tool install modal``); then
   ``modal token new`` for the browser auth flow.
2. Run the bootstrap once to build & seed the volume:

       uvx modal run apps/relational/scripts/modal/run_pair_trades_wide.py::bootstrap

   This uploads ``apps/notebook/data/stooq_us_long`` and
   ``.iv-cache/volatility_history.parquet`` into the persistent volume.
   Subsequent runs see them already mounted at ``/vol/``.
3. Smoke-test the diagnostic on a small slice:

       uvx modal run apps/relational/scripts/modal/run_pair_trades_wide.py \\
           --max-tickers 30

   This runs the full pipeline against 30 tickers; verify it produces
   ``Output/relational-pair-trades-wide-{equity.png,stats.txt}`` locally.
4. Full run:

       uvx modal run apps/relational/scripts/modal/run_pair_trades_wide.py

   Expect ~6-8 min wall, ~$0.20.
5. To add a variant grid (#2 in "When Modal helps"), wrap the inner
   `_run_one_backtest`-equivalent in a `@app.function` decorated
   helper and dispatch via `Function.map` — see
   `apps/factor/scripts/modal/train_indicator.py::train_grid` for the
   ``cells`` cartesian-product pattern; the same shape applies here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Repo layout / paths
# ---------------------------------------------------------------------------

# REPO_ROOT only matters on the local side (image build + artifact write).
# Inside the Modal container this script is dropped at /root/<basename>
# with only 2 parents, so parents[4] would IndexError at import time.
try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')   # remote fallback (unused there)
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'

STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_us_long'
LOCAL_STOOQ_SUBSET = REPO_ROOT / STOOQ_SUBSET_REL
LOCAL_IV_PARQUET = REPO_ROOT / '.iv-cache' / 'volatility_history.parquet'

# Volume mount point inside the container. All persisted data lives
# under /vol/. The scalogram cache lives there too so a content-hash
# hit from a prior run skips the CWT recompute on a new container.
VOL_MOUNT = '/vol'
VOL_STOOQ_SUBSET = f'{VOL_MOUNT}/stooq_us_long'
VOL_IV_PARQUET   = f'{VOL_MOUNT}/iv-cache/volatility_history.parquet'
VOL_SCALOGRAM_CACHE = f'{VOL_MOUNT}/scalogram-cache'


# ---------------------------------------------------------------------------
# Modal image — CPU-only. No GPU base image; this workload is numpy + bt.
# ---------------------------------------------------------------------------

# Mirrors the conventions in apps/replay and apps/factor: thin python
# image + uv + add_local_dir of the repo source. The CUDA dev base
# image used by the GPU scripts is unnecessary here and ~3 GB heavier.
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'curl', 'build-essential')
    .pip_install('uv')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            # Heavy / regenerable / per-run state — never bake into image.
            '.iv-cache/**',
            '.scalogram-cache/**',
            'apps/relational/.scalogram-cache/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            # `uv sync --package relational` walks every workspace
            # member's pyproject.toml so we keep them, but skip the
            # `src/` trees of apps that aren't deps of relational
            # (factor / regime / replay / v1) — concurrent edits there
            # have raced Modal's directory hash before.
            'apps/factor/src/**',
            'apps/regime/src/**',
            'apps/replay/src/**',
            'apps/v1/src/**',
            'apps/notebook/src/**',
            # `apps/notebook/data/stooq_us_long` is the data we want on
            # the *volume*, not in the image — uploaded once via
            # `bootstrap` and read from /vol thereafter. Excluding it
            # from the image avoids re-baking 140 MB on every code edit.
            'apps/notebook/data/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('relational-pair-trades-wide', image=image)

# The persistent volume. `create_if_missing=True` makes the first-run
# bootstrap idempotent — subsequent runs reuse the same volume by name.
vol = modal.Volume.from_name('relational-data', create_if_missing=True)


# ---------------------------------------------------------------------------
# Bootstrap: seed the volume from the caller's local disk.
# ---------------------------------------------------------------------------
# Modal 0.65+ supports `volume.batch_upload` from a local path. We use
# it once to populate /vol/stooq_us_long and /vol/iv-cache. The local
# entrypoint reads the local files and streams them up; the function
# below runs *inside* a container with the volume mounted and only
# verifies the layout.

@app.function(volumes={VOL_MOUNT: vol}, cpu=2, memory=4096, timeout=60 * 30)
def verify_volume_layout() -> dict:
    """Quick sanity check that the volume has the expected trees.

    Returns a small dict the local entrypoint prints. Does **not**
    upload anything — that happens locally via `vol.batch_upload`.
    """
    import os
    out = {}
    out['stooq_subset_present'] = os.path.isdir(VOL_STOOQ_SUBSET)
    out['stooq_manifest_present'] = os.path.isfile(
        f'{VOL_STOOQ_SUBSET}/manifest.json')
    out['iv_parquet_present'] = os.path.isfile(VOL_IV_PARQUET)
    out['scalogram_cache_dir_exists'] = os.path.isdir(VOL_SCALOGRAM_CACHE)
    if out['stooq_subset_present']:
        out['stooq_n_txt_files'] = sum(
            1 for _ in Path(VOL_STOOQ_SUBSET).rglob('*.txt'))
    return out


@app.local_entrypoint()
def bootstrap() -> None:
    """One-time seeding of the persistent volume from local disk.

    Idempotent — re-running re-uploads the same files (Modal's
    `batch_upload` is overwrite-aware). Skip the parquet upload if the
    `.iv-cache/volatility_history.parquet` file is missing locally;
    the equity-only pair-trade diagnostic doesn't need it.
    """
    # Modal's volume.batch_upload accepts (local_path, remote_path)
    # tuples. `force=True` overwrites; without it, existing files on
    # the volume are kept. We use force=True for the Stooq archive
    # (it's tiny and the source-of-truth lives in-repo) and force=False
    # for the IV parquet (it's regenerated from a slow dolt clone, so
    # we don't want to clobber a possibly-newer remote copy).
    if not LOCAL_STOOQ_SUBSET.is_dir():
        raise SystemExit(
            f'ERROR: expected {LOCAL_STOOQ_SUBSET} on local disk; '
            f'build it via `apps/notebook/data/build_stooq_us_long.py` first.')
    print(f'>>> seeding volume {vol!r}')
    print(f'  uploading {LOCAL_STOOQ_SUBSET} → {VOL_STOOQ_SUBSET}')
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(LOCAL_STOOQ_SUBSET.as_posix(), VOL_STOOQ_SUBSET)
        if LOCAL_IV_PARQUET.exists():
            print(f'  uploading {LOCAL_IV_PARQUET} → {VOL_IV_PARQUET}')
            batch.put_file(LOCAL_IV_PARQUET.as_posix(), VOL_IV_PARQUET)
        else:
            print(f'  skipping IV parquet (not present at {LOCAL_IV_PARQUET})')

    print('\n>>> verifying volume layout')
    info = verify_volume_layout.remote()
    print(json.dumps(info, indent=2))


# ---------------------------------------------------------------------------
# Main pipeline: load prices → compute scores → fan out 10 bt backtests.
# ---------------------------------------------------------------------------

# `Function.map` over this signature distributes the 10 backtests
# across as many container replicas as Modal will allocate. Each
# worker reconstructs the bt graph from the inputs (same pattern as
# the local `_run_one_backtest`), runs it, and returns
# (label, stats_series, equity_series_bytes).
#
# We can't return a pandas Series directly from a Modal function in
# the general case (pickle works for most pandas types, but the
# stats Series carries a numpy dtype that some Modal versions
# round-trip oddly). Returning bytes via pickle.dumps + the caller
# pickle.loads is the safest hand-shake.
@app.function(volumes={VOL_MOUNT: vol}, cpu=4, memory=16384, timeout=60 * 60)
def run_one_backtest(
    label: str,
    prices_pkl: bytes,
    weights_pkl: bytes,
    rebal_days: int,
    commission_bps: float,
) -> dict[str, bytes]:
    """Run a single bt backtest and return (label, stats, equity) bytes.

    Volume mount is unused here (no scalogram cache reads) but kept
    consistent with the orchestrator so future variants can reach
    `/vol/scalogram-cache/` without re-plumbing.
    """
    import pickle
    import subprocess
    subprocess.run(
        ['uv', 'sync', '--package', 'relational', '--extra', 'research', '--inexact'],
        cwd=REMOTE_REPO, check=True)

    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    import bt
    prices = pickle.loads(prices_pkl)
    weights = pickle.loads(weights_pkl)

    def _commission(q, p):
        return abs(q) * p * (commission_bps / 10000.0)

    # Forward/back-fill prices so a delisted/gapped name's last known
    # price persists for bt's solver (it raises if a held position's
    # price becomes NaN mid-holding). Mirrors `_bt_safe_prices` in
    # `relational/research/diagnostic_pair_trades_wide.py`.
    bt_prices = prices.ffill().bfill()

    rebal_weights = weights.iloc[::rebal_days]
    nonzero = rebal_weights.abs().sum(axis=1) > 0.1
    if nonzero.any():
        rebal_weights = rebal_weights.loc[nonzero]
    strategy = bt.Strategy(label, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])
    backtest = bt.Backtest(strategy, bt_prices,
                            commissions=_commission,
                            integer_positions=False)
    result = bt.run(backtest)
    stats = result.stats[label].copy()
    equity = result.prices[label].copy()
    return {
        'label':  label.encode(),
        'stats':  pickle.dumps(stats),
        'equity': pickle.dumps(equity),
    }


@app.function(volumes={VOL_MOUNT: vol}, cpu=8, memory=32768, timeout=60 * 60)
def orchestrate(
    top_n: int,
    lookback: int,
    n_tail: int,
    fp_window: int,
    k_clusters: int,
    start: str | None,
    end: str | None,
    rebal_days: int,
    commission_bps: float,
    max_tickers: int,
) -> dict[str, bytes]:
    """Load prices, compute scores against the cached CWT (on /vol),
    build the 10 weight matrices, and `Function.map` the bt jobs out.

    Returns artifacts (`equity.png`, `stats.txt`, `stats.json`) as
    a `{filename: bytes}` dict so the local entrypoint can mirror them
    to the caller's Output/.
    """
    import pickle
    import subprocess
    subprocess.run(
        ['uv', 'sync', '--package', 'relational', '--extra', 'research', '--inexact'],
        cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from ss_loaders import load_stooq_matrix
    from ss_portfolio import (
        apply_nan_mask, select_top_n_matrix,
        weights_regime as _baseline_long,
    )
    from relational.empirical_sectors import (
        empirical_excess_divergence_scores,
        weights_excess_regime_empirical,
    )
    from relational.farthest import (
        centroid_distance_scores, weights_regime_farthest,
    )
    from relational.pairs import (
        cluster_pair_weights, market_neutral_weights, rank_spread_weights,
    )

    # Load prices from the volume-mounted Stooq archive.
    print(f'loading Stooq panel from {VOL_STOOQ_SUBSET}', flush=True)
    min_history = lookback + n_tail + 10
    tickers = None
    if max_tickers and max_tickers > 0:
        # Smoke-test path: take the first `max_tickers` from the manifest
        # in alphabetic order so the choice is deterministic across runs.
        manifest = json.loads(
            Path(f'{VOL_STOOQ_SUBSET}/manifest.json').read_text())
        names = sorted(t['ticker'] for t in manifest['tickers'])
        tickers = names[:max_tickers]
        print(f'  smoke mode: {len(tickers)} tickers', flush=True)
    prices, _, _, _ = load_stooq_matrix(
        VOL_STOOQ_SUBSET, min_history=min_history,
        start_date=start, end_date=end, tickers=tickers)
    print(f'  loaded {prices.shape[0]} dates × {prices.shape[1]} tickers',
          flush=True)

    scales = [5, 7, 10, 12, 21, 26, 50, 90]

    # Plumb the volume-resident scalogram cache so the CWT precompute
    # is content-hash deduplicated across runs. The cache_dir kwarg is
    # respected by load_or_compute_cwt → flows into all three scorers
    # below.
    Path(VOL_SCALOGRAM_CACHE).mkdir(parents=True, exist_ok=True)
    cache_dir = VOL_SCALOGRAM_CACHE

    # ---- scoring (sequential — three scorers, ~30s each) ----
    print('scoring: baseline (CWT-power KL)', flush=True)
    from ss_indicators import get_divergence
    from ss_wavelets import precompute_windows
    from relational.scalogram_cache import load_or_compute_cwt
    coeffs = load_or_compute_cwt(prices, scales, lookback, cache_dir=cache_dir)
    power = (coeffs ** 2).astype(np.float32)
    recent, hist = precompute_windows(power, lookback, n_tail)
    div_fn = get_divergence('kl')
    base_scores = np.array(
        div_fn(recent, hist, np.zeros(len(scales), dtype=np.float32)),
        copy=True)
    base_top = _baseline_long(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n, scales=scales)
    base_bot = pd.DataFrame(
        select_top_n_matrix(
            apply_nan_mask(base_scores, prices.values, lookback),
            top_n, ascending=True),
        index=prices.index[lookback:], columns=prices.columns)

    print('scoring: empirical (k-means on scalogram fingerprints)', flush=True)
    emp_scores, emp_cluster_ids = empirical_excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        fp_window=fp_window, k_clusters=k_clusters,
        return_clusters=True, cache_dir=cache_dir)
    emp_top = weights_excess_regime_empirical(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, k_clusters=k_clusters, fp_window=fp_window,
        cache_dir=cache_dir)
    emp_bot = pd.DataFrame(
        select_top_n_matrix(
            apply_nan_mask(emp_scores, prices.values, lookback),
            top_n, ascending=True),
        index=prices.index[lookback:], columns=prices.columns)
    emp_cluster_pair = cluster_pair_weights(
        emp_scores, emp_cluster_ids, prices, lookback=lookback)

    print('scoring: farthest (cross-sectional centroid distance)', flush=True)
    far_scores = centroid_distance_scores(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, cache_dir=cache_dir)
    far_top = weights_regime_farthest(
        prices, lookback=lookback, top_n=top_n, scales=scales,
        fp_window=fp_window, cache_dir=cache_dir)
    far_bot = pd.DataFrame(
        select_top_n_matrix(
            apply_nan_mask(far_scores, prices.values, lookback),
            top_n, ascending=True),
        index=prices.index[lookback:], columns=prices.columns)

    # Mirror the local NaN-protection helper so weights for tickers
    # that aren't yet (or no longer) trading get zeroed out before
    # they reach bt's rebalance solver.
    def _mask_weights_to_active(w: pd.DataFrame) -> pd.DataFrame:
        valid = np.isfinite(prices.reindex(
            index=w.index, columns=w.columns).values)
        ww = w.fillna(0).values * valid
        is_long_only = (ww >= 0).all(axis=1) & (ww.sum(axis=1) > 0)
        if is_long_only.any():
            sums = ww[is_long_only].sum(axis=1, keepdims=True)
            sums = np.where(sums > 0, sums, 1.0)
            ww[is_long_only] = ww[is_long_only] / sums
        return pd.DataFrame(ww, index=w.index, columns=w.columns)

    # Build all 10 weight matrices and pickle prices once.
    prices_pkl = pickle.dumps(prices)
    base_strategies = {
        'baseline':  (base_top, base_bot),
        'empirical': (emp_top, emp_bot),
        'farthest':  (far_top, far_bot),
    }
    bt_jobs: list[tuple] = []
    for sname, (top, bot) in base_strategies.items():
        for vname, w in [
            ('long-only',   _mask_weights_to_active(top)),
            ('mkt-neutral', _mask_weights_to_active(
                market_neutral_weights(top, prices=prices))),
            ('rank-spread', _mask_weights_to_active(
                rank_spread_weights(top, bot))),
        ]:
            bt_jobs.append(
                (f'{sname}|{vname}', prices_pkl, pickle.dumps(w),
                 rebal_days, commission_bps))
    bt_jobs.append(
        ('empirical|cluster-pair', prices_pkl,
         pickle.dumps(_mask_weights_to_active(emp_cluster_pair)),
         rebal_days, commission_bps))

    # ---- fan out 10 backtests via Function.map ----
    print(f'fanning out {len(bt_jobs)} bt backtests', flush=True)
    stats_by_label: dict[str, pd.Series] = {}
    equity_by_label: dict[str, pd.Series] = {}
    for res in run_one_backtest.starmap(bt_jobs):
        label = res['label'].decode()
        stats_by_label[label] = pickle.loads(res['stats'])
        equity_by_label[label] = pickle.loads(res['equity'])
        print(f'  done: {label}', flush=True)

    # ---- stitch + plot ----
    job_order = [j[0] for j in bt_jobs]
    stats = pd.concat(
        [stats_by_label[label].rename(label) for label in job_order],
        axis=1)
    sharpe = stats.loc['daily_sharpe'].astype(float)
    order = sharpe.sort_values(ascending=False).index.tolist()
    headline = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                'daily_vol', 'total_return', 'worst_year']
    leaderboard = stats.loc[headline, order].T

    fig, ax = plt.subplots(figsize=(14, 8))
    for label in order:
        eq = equity_by_label[label]
        ax.plot(eq.index, eq.values, label=label, linewidth=1.0)
    ax.set_title(
        f'Wide-universe pair-trade overlays (Modal) — '
        f'{prices.index[0].date()} → {prices.index[-1].date()}, '
        f'{prices.shape[1]} tickers, top-{top_n}, rebal={rebal_days}d, '
        f'commission={commission_bps}bps')
    ax.set_ylabel('equity (start = 100)')
    ax.set_xlabel('date')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_bytes_path = Path(f'{REMOTE_REPO}/Output')
    fig_bytes_path.mkdir(parents=True, exist_ok=True)
    fig_path = fig_bytes_path / 'relational-pair-trades-wide-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    stats_path = fig_bytes_path / 'relational-pair-trades-wide-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Wide-universe pair-trade leaderboard (Modal) — '
                'sorted by daily Sharpe\n')
        f.write(f'  universe: {prices.shape[1]} tickers\n')
        f.write(f'  date range: {prices.index[0].date()} → '
                f'{prices.index[-1].date()}\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        f.write('Full bt stats:\n')
        f.write(stats.to_string() + '\n')

    json_path = fig_bytes_path / 'relational-pair-trades-wide-stats.json'
    json_path.write_text(json.dumps({
        'leaderboard': leaderboard.to_dict(),
        'order': order,
    }, indent=2, default=float))

    # Make sure the volume's scalogram cache flushes before the
    # container is destroyed — Modal volumes are write-back.
    vol.commit()

    artifacts = {p.name: p.read_bytes()
                 for p in fig_bytes_path.iterdir()
                 if p.is_file() and p.name.startswith(
                     'relational-pair-trades-wide-')}
    print(f'returning {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def main(
    top_n: int = 20,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    start: str = '',
    end: str = '',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    max_tickers: int = 0,
) -> None:
    """Local entrypoint. Calls `orchestrate.remote(...)` and writes the
    returned artifacts back to repo-root Output/.

    `--max-tickers > 0` is the smoke-test path (deterministic prefix
    of the manifest by alphabetic ticker). Default 0 = full 312-ticker
    subset.

    Run `bootstrap` first to seed the volume:
        uvx modal run apps/relational/scripts/modal/run_pair_trades_wide.py::bootstrap
    """
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'>>> launching wide-universe pair-trade diagnostic on Modal '
          f'(top_n={top_n}, max_tickers={max_tickers or "all"})')
    artifacts = orchestrate.remote(
        top_n=top_n, lookback=lookback, n_tail=n_tail,
        fp_window=fp_window, k_clusters=k_clusters,
        start=(start or None), end=(end or None),
        rebal_days=rebal_days, commission_bps=commission_bps,
        max_tickers=max_tickers,
    )
    for name, blob in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(blob)
        print(f'  ← {out.name}  ({len(blob):,} bytes)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')


if __name__ == '__main__':
    # Allow `python run_pair_trades_wide.py` to print the docstring
    # (the actual run is invoked through `modal run`).
    print(__doc__, file=sys.stderr)
