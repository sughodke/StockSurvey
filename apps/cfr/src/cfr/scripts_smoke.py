"""Phase 0 smoke test — verify the pipeline runs end-to-end.

Loads a small subset of the Stooq universe (default 30 names, 8 years
of bars), trains the tabular CFR on the first 5 years, evaluates on
the last 3, and prints the result. Should complete in well under a
minute on the local Intel Mac.

No leaderboard row produced. The purpose is to confirm:
  - All modules import and wire together.
  - Action menu precomputes without errors.
  - Infoset bucketing produces non-degenerate labels.
  - CFR training visits a non-trivial number of infosets.
  - Eval produces a finite Sharpe number.
  - CFR Sharpe is roughly in the ballpark of the trailing-best
    baseline (no signal expected on 30 tickers / 1 window — just
    "the code runs").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cfr.menu import default_phase1_menu
from cfr.state import default_infoset_builder
from cfr.walkforward import CFRWalkForward


def _load_small_stooq(data_dir: str, n_tickers: int, max_bars: int) -> pd.DataFrame:
    """Load a small subset for smoke testing.

    `data_dir` defaults to the curated `stooq_us_long` subset
    (`apps/notebook/data/stooq_us_long`) which loads in ~10s vs
    ~minutes for the full StooqData/ tree.

    Picks the first `n_tickers` US tickers that have at least
    `max_bars + 600` of history, slices the last `max_bars` bars.
    """
    from ss_loaders import load_stooq_matrix
    closes, _, _, _ = load_stooq_matrix(
        data_dir, min_history=max_bars + 600,
        include_etfs=False,
    )
    if closes.shape[1] < n_tickers:
        raise SystemExit(
            f'only {closes.shape[1]} tickers passed min_history filter, '
            f'need {n_tickers}')
    tickers = sorted(closes.columns)[:n_tickers]
    prices = closes[tickers].iloc[-max_bars:]
    return prices.dropna(axis=1, how='all')


def run_smoke(
    *, data_dir: str = './StooqData',
    n_tickers: int = 30,
    max_bars: int = 2000,
) -> int:
    print(f'Loading small Stooq subset: {n_tickers} tickers, {max_bars} bars...')
    prices = _load_small_stooq(data_dir, n_tickers, max_bars)
    print(f'  loaded shape {prices.shape}, '
          f'{prices.index[0].date()} → {prices.index[-1].date()}')

    print('\nBuilding menu + infoset builder + walkforward...')
    driver = CFRWalkForward(
        menu_builder=lambda: default_phase1_menu(top_k=min(10, n_tickers // 3 or 1)),
        infoset_builder_factory=default_infoset_builder,
        train_window_days=max(500, max_bars // 2),
        val_window_days=max(300, max_bars // 3),
        step_window_days=max(300, max_bars // 3),
        rebal_days=20,
    )
    menu = driver.menu_builder()
    print(f'  menu actions ({menu.n_actions}): {menu.action_keys}')

    print('\nRunning walk-forward...')
    per_window, summary = driver.run(prices)
    if not per_window:
        print('FAIL — no walk-forward windows fit in the smoke universe.')
        return 1

    print(f'  built {len(per_window)} window(s)')
    print(f'\n{"win":>3s} {"cfr_sh":>7s} {"pas_sh":>7s} {"trl_sh":>7s} '
          f'{"alpha":>7s} {"vs_trl":>7s} {"visited":>7s} {"gross":>7s}')
    print('-' * 75)
    for w in per_window:
        print(f'{w.window_idx:>3d} {w.cfr_sharpe:>+7.3f} '
              f'{w.passive_ew_sharpe:>+7.3f} {w.trailing_best_sharpe:>+7.3f} '
              f'{w.cfr_alpha:>+7.3f} '
              f'{w.cfr_sharpe - w.trailing_best_sharpe:>+7.3f} '
              f'{w.visited_infosets:>7d} {w.cfr_avg_gross:>7.3f}')
    print()
    for k, v in summary.items():
        if isinstance(v, list):
            continue
        if isinstance(v, float):
            print(f'  {k} = {v:+.4f}')
        else:
            print(f'  {k} = {v}')

    out_dir = Path('Output')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'cfr-smoke.json'
    out_path.write_text(json.dumps({
        'summary': summary,
        'per_window': [w.__dict__ for w in per_window],
    }, indent=2, default=str))
    print(f'\n-> {out_path}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(run_smoke())
