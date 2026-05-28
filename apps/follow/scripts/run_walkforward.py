"""Pre-registered walk-forward eval for the leadership-disclosure follower.

Test design (locked to NPZ `pre_registered_bar` BEFORE the eval runs):

* Universe       : US equities disclosed in the 'Congressional Trades.xlsx'
                   feed (Quiver Quant aggregate), ∩ Stooq archive.
* Folds          : Fold-1 train 2014-2018 / val 2019-2021;
                   Fold-2 train 2016-2020 / val 2022-2024.
* Friction       : 10 bps round-trip per |Δw|.
* Entry rule     : `filed + 1` trading day (disclosure-lag-honest).
* Hold horizons  : {30, 60, 90} trading days
* Top-K basket   : {10, 25, 50}
* Filter mode    : {recency, frequency}
  → 18 grid cells; we record per-cell + best-cell-per-fold + mean.

Bar:
  confirmed-OOS  : ann_sharpe ≥ +1.0 net AND defl-t > +2.0
                   AND alpha vs SPY ≥ +5pp/yr AND ≥60% pos-quarter
  partial-OOS    : ann_sharpe ≥ +0.5 AND defl-t > +1.0
                   AND alpha vs SPY ≥ +2pp/yr
  confirmed-null : ann_sharpe < +0.3 OR alpha vs SPY < +1pp/yr
  reversed-OOS   : alpha vs SPY < 0
  diagnostic     : anything else

The driver runs BOTH the leadership-only arm AND the all-members
baseline so the load-bearing 'leadership-filter helps' question
gets answered from the same artifact.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from follow.data import build_eligible_disclosures
from follow.backtest import run_backtest, BacktestResult

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'

HOLD_GRID = [30, 60, 90]
TOPK_GRID = [10, 25, 50]
FILTER_GRID = ['recency', 'frequency']

# Folds — (label, train_start, train_end, val_start, val_end).
# Train block is currently informational only — the strategy has no
# trainable params (leadership cohort is point-in-time and the grid
# is pre-registered). Carried for documentation / future arms that
# might learn over the train slice.
FOLDS = [
    ('fold1', '2014-01-01', '2018-12-31', '2019-01-01', '2021-12-31'),
    ('fold2', '2016-01-01', '2020-12-31', '2022-01-01', '2024-12-31'),
]


def _ann_sharpe(r: np.ndarray, ppy: int = 252) -> float:
    if len(r) < 5 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(ppy))


def _max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def _pos_quarter_fraction(r: pd.Series) -> float:
    """Fraction of calendar quarters with positive cumulative return."""
    if len(r) == 0:
        return 0.0
    eq = (1 + r).cumprod()
    qreturns = eq.resample('QE').last().pct_change().dropna()
    if len(qreturns) == 0:
        return 0.0
    return float((qreturns > 0).mean())


def _slice_returns(res: BacktestResult, lo: str, hi: str) -> pd.Series:
    r = res.daily_returns
    mask = (r.index >= pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))
    return r.loc[mask].fillna(0.0)


def _alpha_vs_spy(strat: pd.Series, spy: pd.Series) -> tuple[float, float]:
    """Returns (annualized excess-return alpha, beta-adjusted alpha) in %/yr."""
    idx = strat.index.intersection(spy.index)
    s = strat.loc[idx].values
    b = spy.loc[idx].values
    if len(s) < 30:
        return 0.0, 0.0
    excess_ann = (s.mean() - b.mean()) * 252 * 100
    # Beta-adjusted: regress strat on bench.
    var_b = b.var()
    beta = float(np.cov(s, b, ddof=0)[0, 1] / max(var_b, 1e-12))
    alpha_daily = s.mean() - beta * b.mean()
    return excess_ann, float(alpha_daily * 252 * 100)


def run_grid(
    *,
    stooq_dir: str,
    leadership_only: bool,
    folds=FOLDS,
    hold_grid=HOLD_GRID,
    topk_grid=TOPK_GRID,
    filter_grid=FILTER_GRID,
    cache_dir: str | None = None,
) -> dict:
    """Run the grid for one arm. Returns dict ready for JSON dump.

    Builds the panel ONCE over the union span (cheap), then re-uses
    it across all grid cells × folds.
    """
    union_start = min(f[1] for f in folds)
    union_end = max(f[4] for f in folds)
    panel = build_eligible_disclosures(
        stooq_dir=stooq_dir,
        leadership_only=leadership_only,
        start=union_start,
        end=union_end,
        cache_dir=cache_dir,
    )
    print(f'[panel] drop_stats = {panel.drop_stats}')
    # SPY benchmark from the panel itself — if SPY isn't in the
    # disclosed universe we re-fetch.
    if 'SPY' not in panel.closes.columns:
        from ss_loaders import load_stooq_matrix
        spy_closes, _, _, _ = load_stooq_matrix(
            stooq_dir, min_history=100,
            start_date=union_start, end_date=union_end,
            tickers=['SPY'], include_etfs=True,
        )
        spy = spy_closes['SPY']
    else:
        spy = panel.closes['SPY']
    spy_ret = spy.pct_change().fillna(0.0)

    rows = []
    per_fold_results = {f[0]: [] for f in folds}
    daily_returns_per_cell: dict[str, pd.Series] = {}

    for hold_days, top_k, filter_mode in itertools.product(
            hold_grid, topk_grid, filter_grid):
        res = run_backtest(panel, hold_days=hold_days, top_k=top_k,
                           filter_mode=filter_mode, commission_bps=10.0)
        cell_key = f'h{hold_days}_k{top_k}_{filter_mode}'
        daily_returns_per_cell[cell_key] = res.daily_returns

        for fold_label, _tr_s, _tr_e, va_s, va_e in folds:
            r_val = _slice_returns(res, va_s, va_e)
            spy_val = spy_ret.loc[
                (spy_ret.index >= va_s) & (spy_ret.index <= va_e)]
            ann_sh = _ann_sharpe(r_val.values)
            mdd = _max_dd(r_val.values)
            pq = _pos_quarter_fraction(r_val)
            alpha_excess, alpha_beta = _alpha_vs_spy(r_val, spy_val)
            row = {
                'arm': 'leadership' if leadership_only else 'all_members',
                'fold': fold_label,
                'hold_days': hold_days, 'top_k': top_k,
                'filter_mode': filter_mode,
                'val_start': va_s, 'val_end': va_e,
                'n_days': int(len(r_val)),
                'ann_sharpe_net': ann_sh,
                'max_dd': mdd,
                'pos_quarter_fraction': pq,
                'alpha_excess_pct_yr': alpha_excess,
                'alpha_beta_adj_pct_yr': alpha_beta,
                'mean_n_holdings': float(res.n_holdings.mean()),
            }
            rows.append(row)
            per_fold_results[fold_label].append(row)

    # Aggregate: mean over folds per (cell), and best cell per fold.
    cells = pd.DataFrame(rows)
    agg = (cells.groupby(['hold_days', 'top_k', 'filter_mode'])
                  [['ann_sharpe_net', 'alpha_excess_pct_yr',
                    'pos_quarter_fraction', 'max_dd']]
                  .mean().reset_index())
    agg = agg.sort_values('ann_sharpe_net', ascending=False)
    best_cell = agg.iloc[0].to_dict() if len(agg) else None

    return {
        'arm': 'leadership' if leadership_only else 'all_members',
        'drop_stats': panel.drop_stats,
        'grid_cells': rows,
        'grid_mean_over_folds': agg.to_dict(orient='records'),
        'best_cell_mean': best_cell,
        'daily_returns_per_cell': {
            k: v.to_dict() for k, v in daily_returns_per_cell.items()
        },
        'spy_ret': spy_ret.to_dict(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stooq-dir', default='./StooqData')
    p.add_argument('--output-stem', default='follow-walkforward')
    args = p.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print('--- Arm 1: leadership-only ---')
    lead = run_grid(stooq_dir=args.stooq_dir, leadership_only=True)
    print('--- Arm 2: all-members baseline ---')
    allm = run_grid(stooq_dir=args.stooq_dir, leadership_only=False)

    # Pre-registered bar.
    pre_reg = {
        'confirmed_OOS': {
            'ann_sharpe_net_geq': 1.0,
            'deflated_t_gt': 2.0,
            'alpha_vs_spy_pct_yr_geq': 5.0,
            'pos_quarter_fraction_geq': 0.60,
        },
        'partial_OOS': {
            'ann_sharpe_net_geq': 0.5,
            'deflated_t_gt': 1.0,
            'alpha_vs_spy_pct_yr_geq': 2.0,
        },
        'confirmed_null': {
            'ann_sharpe_net_lt': 0.3,
            'OR_alpha_vs_spy_pct_yr_lt': 1.0,
        },
        'reversed_OOS': {'alpha_vs_spy_pct_yr_lt': 0.0},
        'notes': (
            'Bar locked BEFORE eval. The load-bearing arm is the '
            'leadership-only arm; all-members is the apples-to-apples '
            'baseline that tests whether the leadership filter actually '
            'helps. Bowne 2024 predicts the disclosure-lag treatment '
            'collapses the apparent alpha — `filed + 1` entry is the '
            'falsification test.'),
    }

    # Build the deployable daily-return stream for compute_dsr.py:
    # leadership-only arm at the median grid cell (h=60, k=25,
    # filter=recency) across BOTH val slices concatenated.
    cell_key = 'h60_k25_recency'
    series_lead = pd.Series(lead['daily_returns_per_cell'][cell_key])
    series_lead.index = pd.to_datetime(series_lead.index)
    series_lead = series_lead.sort_index()
    val_mask = (
        ((series_lead.index >= '2019-01-01') & (series_lead.index <= '2021-12-31'))
        | ((series_lead.index >= '2022-01-01') & (series_lead.index <= '2024-12-31'))
    )
    oos = series_lead.loc[val_mask].values.astype(np.float64)

    spy_series = pd.Series(lead['spy_ret'])
    spy_series.index = pd.to_datetime(spy_series.index)
    spy_series = spy_series.sort_index().reindex(series_lead.index).fillna(0.0)
    spy_oos = spy_series.loc[val_mask].values.astype(np.float64)

    # JSON dump (human-readable summary).
    summary = {
        'pre_registered_bar': pre_reg,
        'leadership': {k: v for k, v in lead.items()
                       if k not in ('daily_returns_per_cell', 'spy_ret')},
        'all_members': {k: v for k, v in allm.items()
                        if k not in ('daily_returns_per_cell', 'spy_ret')},
        'leadership_best_cell_mean': lead['best_cell_mean'],
        'all_members_best_cell_mean': allm['best_cell_mean'],
        'deployable_cell': cell_key,
    }
    json_path = OUTPUT / f'{args.output_stem}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'-> {json_path}')

    # NPZ for compute_dsr.py.
    npz_path = OUTPUT / f'{args.output_stem}.npz'
    np.savez(
        npz_path,
        oos_block_returns=oos,
        spy_ret=spy_oos,
        periods_per_year=np.float64(252.0),
        n_grid_cells=np.int64(len(HOLD_GRID) * len(TOPK_GRID) * len(FILTER_GRID)),
        deployable_cell=cell_key,
        universe_label=np.array('congressional-leadership'),
        pre_registered_bar=np.array(json.dumps(pre_reg)),
    )
    print(f'-> {npz_path}  (oos n={len(oos)}, spy n={len(spy_oos)})')

    # Print headline numbers.
    print('\n=== leadership-only arm — per-fold best-cell ===')
    df_lead = pd.DataFrame(lead['grid_cells'])
    for fl in df_lead['fold'].unique():
        sub = df_lead[df_lead['fold'] == fl].sort_values(
            'ann_sharpe_net', ascending=False).head(3)
        print(f'  -- {fl}: top 3 cells')
        print(sub[['hold_days', 'top_k', 'filter_mode', 'ann_sharpe_net',
                   'alpha_excess_pct_yr', 'pos_quarter_fraction',
                   'max_dd']].to_string(index=False))

    print('\n=== all-members arm — per-fold best-cell ===')
    df_all = pd.DataFrame(allm['grid_cells'])
    for fl in df_all['fold'].unique():
        sub = df_all[df_all['fold'] == fl].sort_values(
            'ann_sharpe_net', ascending=False).head(3)
        print(f'  -- {fl}: top 3 cells')
        print(sub[['hold_days', 'top_k', 'filter_mode', 'ann_sharpe_net',
                   'alpha_excess_pct_yr', 'pos_quarter_fraction',
                   'max_dd']].to_string(index=False))

    # Headline cross-arm comparison.
    print('\n=== Cross-arm best-mean-over-folds ===')
    print('leadership_best_cell_mean:', lead['best_cell_mean'])
    print('all_members_best_cell_mean:', allm['best_cell_mean'])


if __name__ == '__main__':
    main()
