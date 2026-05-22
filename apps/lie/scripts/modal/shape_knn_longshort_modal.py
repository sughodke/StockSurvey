"""Modal: shape-kNN 1-month-reversal long/short on the WIDE liquid universe.

Breadth-expansion test. Local Phase-2 (21 names) gave IC t=+3.75 but a
long/short deflated-t of -1.55 — the fundamental law (IR~IC*sqrt(breadth)*TC)
said 21-name breadth was the binding constraint. This re-runs the identical
signal on factor-narrow (297 stooq_us_long names, n_bars>=6500), sqrt(breadth)
~3.8x higher, to see whether the L/S deflated-t clears 0.

CPU-bound numpy kNN (no GPU). The brute-force predictor is O(queries x train),
so we predict ONLY on the non-overlapping rebal-date (date,ticker) samples
(~15k), not all test pairs (~600k). Data: the baked-in stooq_us_long subset
(same one factor's Modal entrypoint uses); loaded remotely via
`ss_loaders.load_stooq_matrix`, no pickle RPC needed.

    uvx modal run apps/lie/scripts/modal/shape_knn_longshort_modal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[4]
REMOTE_REPO = '/root/StockSurvey'
STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_us_long'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'

# Reuse factor's proven workspace image (it uv-syncs the monorepo cleanly).
# No GPU needed here — the kNN is numpy — so the function requests cpu only.
image = (
    modal.Image.from_registry(
        'nvidia/cuda:12.4.0-devel-ubuntu22.04', add_python='3.12',
    )
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**', '.venv/**', 'Output/**', 'StooqData/**', 'Nasdaq3347/**',
            # lie depends only on ss_* packages; skip every other app's src
            # tree (keep their pyprojects so uv's workspace walk succeeds).
            'apps/factor/src/**', 'apps/relational/src/**', 'apps/regime/src/**',
            'apps/v1/src/**', 'apps/replay/src/**', 'apps/cfr/src/**',
            'apps/gate/src/**', 'apps/pairs/src/**', 'apps/vol/src/**',
            'apps/dca/src/**', 'apps/notebook/src/**',
            '**/__pycache__/**', '**/*.pyc',
        ],
    )
)

app = modal.App('lie-shape-knn-ls', image=image)


@app.function(cpu=8.0, memory=32768, timeout=2 * 60 * 60)
def run_longshort(
    start: str, end: str, horizon: int, lookback: int, k: int,
    temporal_gap: int, train_frac: float, min_history_bars: int,
    max_tickers: int, commission_bps: float, borrow_bps_yr: float,
    query_batch: int,
) -> dict[str, bytes]:
    import os
    import subprocess
    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)

    print('=== uv sync lie deps ===', flush=True)
    subprocess.run(['uv', 'sync', '--package', 'lie', '--inexact'],
                   cwd=REMOTE_REPO, check=True)
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    import numpy as np
    from ss_loaders import load_stooq_matrix
    from ss_portfolio import standardize_oos
    from lie.cross_sectional import cross_sectional_ic_summary
    from lie.longshort import long_short_net_returns
    from lie.predictor import TimelessPredictor
    from lie.ticker_features import TickerFeatureConfig, build_ticker_features

    # factor-narrow universe from the baked manifest.
    man = json.loads(Path(f'{STOOQ_SUBSET}/manifest.json').read_text())
    rows = man['tickers'] if isinstance(man, dict) else man
    names = sorted(t['ticker'].upper() for t in rows
                   if t.get('n_bars', 0) >= min_history_bars)
    if max_tickers > 0:
        names = names[:max_tickers]
    print(f'universe: {len(names)} names (n_bars>={min_history_bars})', flush=True)

    H = horizon
    closes, _, _, _ = load_stooq_matrix(
        STOOQ_SUBSET, tickers=names, start_date=start, end_date=end,
        min_history=lookback + H + 10)
    panel = closes.to_numpy()
    dates = closes.index
    T, N = panel.shape
    print(f'panel: {T} dates x {N} tickers', flush=True)

    ticker_feats, valid_tick = build_ticker_features(panel, TickerFeatureConfig())
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(np.isfinite(panel) & (panel > 0), np.log(panel), np.nan)
    fwd = np.full((T, N), np.nan)
    if T > H:
        fwd[:T - H] = log_p[H:] - log_p[:T - H]
    with np.errstate(invalid='ignore'):
        excess = fwd - np.nanmean(fwd, axis=1, keepdims=True)

    valid_full = valid_tick & np.isfinite(excess)
    t_arr, i_arr = np.where(valid_full)
    print(f'long-format samples: M = {len(t_arr)}', flush=True)

    split_date = int(np.percentile(np.unique(t_arr), train_frac * 100))
    is_train = t_arr <= split_date
    tf = ticker_feats[t_arr, i_arr]
    mu = tf[is_train].mean(axis=0)
    sd = tf[is_train].std(axis=0, ddof=1)
    sd = np.where(sd <= 0, 1.0, sd)
    X = (tf - mu) / sd
    y = excess[t_arr, i_arr]
    t_int = t_arr.astype(np.int64)

    pred = TimelessPredictor(k=k, temporal_gap=temporal_gap,
                             weighting='inverse_distance')
    pred.fit(X[is_train], t_int[is_train], y[is_train])
    print(f'fit on {int(is_train.sum())} train samples', flush=True)

    # Non-overlapping rebal dates over the TEST span; predict only on those
    # (date,ticker) samples to keep brute-force kNN tractable.
    test_dates = np.unique(t_arr[~is_train])
    rebal_dates = test_dates[::H]
    rebal_dates = rebal_dates[rebal_dates + H < T]
    rebal_set = set(int(d) for d in rebal_dates)
    q = np.array([j for j in range(len(t_arr))
                  if (not is_train[j]) and int(t_arr[j]) in rebal_set])
    print(f'rebal blocks: {len(rebal_dates)}; query samples: {len(q)}', flush=True)

    yhat_q = np.full(len(q), np.nan)
    for s in range(0, len(q), query_batch):
        e = min(s + query_batch, len(q))
        p, _ = pred.predict(X[q[s:e]], t_int[q[s:e]])
        yhat_q[s:e] = p

    ic = cross_sectional_ic_summary(yhat_q, y[q], t_int[q], method='spearman')
    print(f'signal check (rebal dates): mean IC {ic["mean_ic"]:+.4f} '
          f't={ic["t_stat"]:+.2f} over {ic["n_dates"]} dates', flush=True)

    score_grid = np.full((T, N), np.nan)
    score_grid[t_arr[q], i_arr[q]] = yhat_q
    scores = np.nan_to_num(score_grid[rebal_dates], nan=0.0)
    blr = np.where(np.isfinite(fwd[rebal_dates]), fwd[rebal_dates], 0.0)
    mask = (np.isfinite(score_grid[rebal_dates])
            & np.isfinite(fwd[rebal_dates])).astype(np.float64)

    net = long_short_net_returns(scores, blr, mask, commission_bps / 1e4)
    net = net - (borrow_bps_yr / 1e4) * (H / 252.0) * 0.5
    ppy = 252.0 / H
    mb = standardize_oos(net, periods_per_year=ppy, n_trials=9)
    print(f'\n--- shape-kNN L/S (factor-narrow {N}, H={H}, {len(net)} blocks) ---',
          flush=True)
    print(f'  ann Sharpe {mb.ann_sharpe:+.3f}  DSR {mb.dsr:.3f}  '
          f'deflated t {mb.deflated_tstat:+.3f}', flush=True)

    import io
    buf = io.BytesIO()
    np.savez(buf, ls_block_returns=net.astype(np.float64),
             periods_per_year=np.float64(ppy),
             commission_bps=np.float64(commission_bps),
             borrow_bps_yr=np.float64(borrow_bps_yr),
             n_names=np.int64(N))
    summary = {
        'universe_size': int(N), 'horizon': H, 'lookback': lookback, 'k': k,
        'n_blocks': int(len(net)), 'ic_mean': ic['mean_ic'],
        'ic_tstat': ic['t_stat'], 'ic_n_dates': ic['n_dates'],
        'ann_sharpe': mb.ann_sharpe, 'dsr': mb.dsr,
        'deflated_tstat': mb.deflated_tstat, 'n_trials': 9,
    }
    return {
        'lie-shape-knn-wide-returns.npz': buf.getvalue(),
        'lie-shape-knn-wide-summary.json': json.dumps(summary, indent=2).encode(),
    }


@app.local_entrypoint()
def main(
    start: str = '2000-01-01', end: str = '2026-04-01', horizon: int = 21,
    lookback: int = 60, k: int = 50, temporal_gap: int = 60,
    train_frac: float = 0.7, min_history_bars: int = 6500, max_tickers: int = 0,
    commission_bps: float = 10.0, borrow_bps_yr: float = 50.0,
    query_batch: int = 128,
) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'launching lie shape-kNN L/S on Modal (min_history_bars='
          f'{min_history_bars}, max_tickers={max_tickers})')
    artifacts = run_longshort.remote(
        start=start, end=end, horizon=horizon, lookback=lookback, k=k,
        temporal_gap=temporal_gap, train_frac=train_frac,
        min_history_bars=min_history_bars, max_tickers=max_tickers,
        commission_bps=commission_bps, borrow_bps_yr=borrow_bps_yr,
        query_batch=query_batch)
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out} ({len(data) // 1024}KB)')


if __name__ == '__main__':
    main()
