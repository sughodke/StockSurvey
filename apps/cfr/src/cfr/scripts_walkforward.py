"""Phase 1 walk-forward — full eval against passive-EW + baselines.

Mirrors `apps/gate/scripts/run_walkforward.py`'s shape so the
leaderboard ingestion is uniform. Loads the `stooq_us_long`
universe by default (same canonical universe `apps/gate`,
`apps/pairs`, `apps/factor sizing-input` used), runs 6
walk-forward windows with the canonical 5y/3y/3y split, reports
per-window CFR Sharpe vs passive-EW vs trailing-best-greedy vs
naive-uniform, and applies the pre-registered cuts.

Run from repo root:
    uv run python -m cfr.scripts_walkforward
    uv run ss-cfr walkforward --output Output/cfr-phase1.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cfr.menu import default_phase1_menu
from cfr.state import default_infoset_builder
from cfr.walkforward import CFRWalkForward


REPO_ROOT = Path(__file__).resolve().parents[4]


def run_walkforward(
    *,
    data_dir: str = './StooqData',
    manifest: str = 'apps/notebook/data/stooq_us_long/manifest.json',
    start: str = '2000-01-01',
    end:   str = '2025-12-11',
    rebal_days: int = 20,
    top_k: int = 20,
    train_window_days: int = 1260,
    val_window_days:   int = 780,
    step_window_days:  int = 780,
    n_training_passes: int = 1,
    output: Path = Path('Output/cfr-walkforward-summary.json'),
    seed: int = 0,
) -> int:
    print(f'Loading universe from manifest: {manifest}...')
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    universe_raw = json.loads(manifest_path.read_text())
    tickers = sorted(t['ticker'].upper() for t in universe_raw['tickers'])
    print(f'  manifest has {len(tickers)} tickers')

    t0 = time.time()
    from ss_loaders import load_stooq_matrix
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=150,
        start_date=start, end_date=end,
        tickers=tickers,
    )
    print(f'  loaded {prices.shape[1]} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()} '
          f'in {time.time() - t0:.1f}s')

    driver = CFRWalkForward(
        menu_builder=lambda: default_phase1_menu(top_k=top_k),
        infoset_builder_factory=default_infoset_builder,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        step_window_days=step_window_days,
        rebal_days=rebal_days,
        commission_bps=10.0,
        n_training_passes=n_training_passes,
        rng_seed=seed,
    )

    menu = driver.menu_builder()
    print(f'\nMenu ({menu.n_actions} actions): {menu.action_keys}')

    t0 = time.time()
    per_window, summary = driver.run(prices)
    print(f'\nWalk-forward complete in {time.time() - t0:.1f}s '
          f'({len(per_window)} windows)')

    print(f'\n{"win":>3s} {"val_dates":>23s} {"cfr_sh":>7s} {"pas_sh":>7s} '
          f'{"trl_sh":>7s} {"nai_sh":>7s} {"alpha":>7s} {"vs_trl":>7s} '
          f'{"gross":>6s}')
    print('-' * 105)
    for w in per_window:
        print(f'{w.window_idx:>3d} {w.val_start}→{w.val_end} '
              f'{w.cfr_sharpe:>+7.3f} {w.passive_ew_sharpe:>+7.3f} '
              f'{w.trailing_best_sharpe:>+7.3f} '
              f'{w.naive_uniform_sharpe:>+7.3f} '
              f'{w.cfr_alpha:>+7.3f} '
              f'{w.cfr_sharpe - w.trailing_best_sharpe:>+7.3f} '
              f'{w.cfr_avg_gross:>6.2f}')

    print('\n' + '=' * 100)
    print(f'mean CFR Sharpe        = {summary["mean_cfr_sharpe"]:+.3f}')
    print(f'mean passive EW Sharpe = {summary["mean_passive_sharpe"]:+.3f}')
    print(f'mean trailing-best     = {summary["mean_trailing_sharpe"]:+.3f}')
    print(f'mean CFR alpha (vs EW) = {summary["mean_cfr_alpha"]:+.3f}')
    print(f'mean CFR vs trailing   = {summary["mean_cfr_minus_trailing_best"]:+.3f}')
    print(f'positive-alpha frac    = {summary["positive_alpha_fraction"]:.2f}')
    print(f'\nverdict: {summary["verdict"]}')

    out_path = output
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        'config': {
            'data_dir': str(data_dir),
            'manifest': str(manifest),
            'start': start, 'end': end,
            'rebal_days': rebal_days,
            'top_k': top_k,
            'train_window_days': train_window_days,
            'val_window_days': val_window_days,
            'step_window_days': step_window_days,
            'n_training_passes': n_training_passes,
            'seed': seed,
        },
        'summary': summary,
        'per_window': [w.__dict__ for w in per_window],
    }, indent=2, default=str))
    print(f'\n-> {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(run_walkforward())
