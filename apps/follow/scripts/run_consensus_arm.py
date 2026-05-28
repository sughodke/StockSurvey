"""Pre-registered fold-3 OOS verdict for the all-members consensus arm.

Locked cell: leadership_only=False, filter_mode='frequency',
hold_days=30, top_k=10, commission_bps=10, entry='filed+1' (Bowne 2024).

Folds:
  fold-1  : val 2019-01-01 → 2021-12-31  (v0 fold-1 reproduction)
  fold-2  : val 2022-01-01 → 2024-12-31  (v0 fold-2 reproduction)
  fold-3  : val 2025-01-01 → 2025-10-16  (UNSEEN OOS verdict — bounded
            by xlsx end 2025-10-16)

Pre-registered bar (per TODO/follow-consensus-arm.md, the locked
source of truth):

  confirmed-OOS  : ann_Sharpe ≥ +0.85 (pooled folds 1+2+3) AND
                   alpha vs SPY ≥ +3pp/yr (pooled) AND
                   fold-3 alpha ≥ +1pp/yr
  partial-OOS    : fold-1+2 reproduce v0 numbers (alpha ≥ +5pp/yr)
                   AND fold-3 alpha ≥ 0 AND pooled defl-t > +1.0
  confirmed-null : fold-3 alpha < +1pp/yr OR pooled defl-t < 0
  reversed-OOS   : fold-3 alpha < 0

Also computes (per the v1 brief) a stationary-bootstrap CI on
fold-3 alpha vs SPY at 95% with seed=42, n_boot=2000.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from follow.data import build_eligible_disclosures
from follow.backtest import run_backtest

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'

# Locked cell.
HOLD_DAYS = 30
TOP_K = 10
FILTER_MODE = 'frequency'
COMMISSION_BPS = 10.0

FOLDS = [
    ('fold1', '2019-01-01', '2021-12-31'),
    ('fold2', '2022-01-01', '2024-12-31'),
    ('fold3', '2025-01-01', '2025-10-16'),
]


def _ann_sharpe(r: np.ndarray, ppy: int = 252) -> float:
    if len(r) < 5 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(ppy))


def _max_dd(r: np.ndarray) -> float:
    if len(r) == 0:
        return 0.0
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def _pos_quarter_fraction(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    eq = (1 + r).cumprod()
    qreturns = eq.resample('QE').last().pct_change().dropna()
    if len(qreturns) == 0:
        return 0.0
    return float((qreturns > 0).mean())


def _alpha_vs_spy(strat: pd.Series, spy: pd.Series) -> tuple[float, float]:
    idx = strat.index.intersection(spy.index)
    s = strat.loc[idx].values
    b = spy.loc[idx].values
    if len(s) < 30:
        return 0.0, 0.0
    excess_ann = (s.mean() - b.mean()) * 252 * 100
    var_b = b.var()
    beta = float(np.cov(s, b, ddof=0)[0, 1] / max(var_b, 1e-12))
    alpha_daily = s.mean() - beta * b.mean()
    return excess_ann, float(alpha_daily * 252 * 100)


def _stationary_bootstrap_ci(
    edge: np.ndarray, *, n_boot: int = 2000, seed: int = 42,
    avg_block: float = 20.0, ppy: int = 252, alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    """Studentized stationary-bootstrap CI on annualized mean of `edge`.

    Returns (lo, hi, samples) in pp/yr units.
    """
    rng = np.random.default_rng(seed)
    n = len(edge)
    if n < 30:
        return 0.0, 0.0, np.array([])
    p = 1.0 / avg_block
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        # Stationary bootstrap (Politis-Romano): random starts + geom block lengths.
        idxs = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            for j in range(length):
                if i >= n:
                    break
                idxs[i] = (start + j) % n
                i += 1
        s = edge[idxs]
        samples[b] = s.mean() * ppy * 100
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi, samples


def _deflated_t(r: np.ndarray, *, n_trials: int = 1, ppy: int = 252) -> float:
    """Deflated Sharpe t-stat (Bailey-López de Prado).

    n_trials=1 → no multiple-testing deflation; reduces to studentized
    Sharpe with skew/kurt correction (the JKP form).
    """
    n = len(r)
    if n < 30 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()  # per-period
    sr_ann = sr * math.sqrt(ppy)
    # Skew/kurt adjustment.
    rs = (r - r.mean()) / r.std()
    skew = float((rs ** 3).mean())
    kurt = float((rs ** 4).mean())  # raw 4th moment, not excess
    # Sharpe std under non-normal returns (Mertens 2002 / Bailey-LdP eq 7):
    var_sr = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)
    if var_sr <= 0:
        return 0.0
    se_sr = math.sqrt(var_sr)  # per-period SE
    # E[max SR] across n_trials at sharpe_std=0 (single config → 0).
    if n_trials <= 1:
        emax = 0.0
    else:
        from math import gamma
        z = 1 - 1 / n_trials
        # Approximate maximum order statistic; here we use 0 since
        # n_trials=1 by pre-reg.
        emax = 0.0
    return float((sr - emax) / se_sr)  # per-period t-stat; matches DSR convention


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stooq-dir', default='./StooqData')
    p.add_argument('--output-stem', default='follow-consensus-fold3-walkforward')
    p.add_argument('--cache-dir', default='.congress-cache')
    args = p.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    union_start = min(f[1] for f in FOLDS)
    union_end = max(f[2] for f in FOLDS)

    print(f'[panel] building all-members panel {union_start} → {union_end}')
    panel = build_eligible_disclosures(
        stooq_dir=args.stooq_dir,
        leadership_only=False,
        start=union_start,
        end=union_end,
        cache_dir=args.cache_dir,
    )
    print(f'[panel] drop_stats = {panel.drop_stats}')

    # SPY benchmark.
    if 'SPY' not in panel.closes.columns:
        from ss_loaders import load_stooq_matrix
        spy_closes, _, _, _ = load_stooq_matrix(
            args.stooq_dir, min_history=100,
            start_date=union_start, end_date=union_end,
            tickers=['SPY'], include_etfs=True,
        )
        spy = spy_closes['SPY']
    else:
        spy = panel.closes['SPY']
    spy_ret = spy.pct_change().fillna(0.0)

    # Run the locked cell once across the union span.
    print(f'[bt] running locked cell h={HOLD_DAYS} k={TOP_K} '
          f'filter={FILTER_MODE} commission={COMMISSION_BPS}bps')
    res = run_backtest(
        panel, hold_days=HOLD_DAYS, top_k=TOP_K,
        filter_mode=FILTER_MODE, commission_bps=COMMISSION_BPS)
    daily = res.daily_returns.sort_index()
    print(f'[bt] daily_returns: {len(daily)} days, '
          f'mean n_holdings={res.n_holdings.mean():.2f}')

    spy_aligned = spy_ret.reindex(daily.index).fillna(0.0)

    # Per-fold metrics.
    fold_rows = []
    fold_edges: dict[str, np.ndarray] = {}
    for label, va_s, va_e in FOLDS:
        mask = (daily.index >= pd.Timestamp(va_s)) & (daily.index <= pd.Timestamp(va_e))
        r = daily.loc[mask].fillna(0.0)
        b = spy_aligned.loc[mask].fillna(0.0)
        ann_sh = _ann_sharpe(r.values)
        mdd = _max_dd(r.values)
        pq = _pos_quarter_fraction(r)
        alpha_ex, alpha_beta = _alpha_vs_spy(r, b)
        edge = (r.values - b.values)
        fold_edges[label] = edge
        spy_sh = _ann_sharpe(b.values)
        row = {
            'fold': label,
            'val_start': va_s, 'val_end': va_e,
            'n_days': int(len(r)),
            'ann_sharpe_net': ann_sh,
            'spy_ann_sharpe': spy_sh,
            'max_dd': mdd,
            'pos_quarter_fraction': pq,
            'alpha_excess_pct_yr': alpha_ex,
            'alpha_beta_adj_pct_yr': alpha_beta,
            'mean_n_holdings': float(res.n_holdings.loc[mask].mean()),
        }
        fold_rows.append(row)
        print(f'  {label}: n={len(r):4d}d  Sh={ann_sh:+.3f}  '
              f'SPY-Sh={spy_sh:+.3f}  α_excess={alpha_ex:+.2f}pp/yr  '
              f'α_β-adj={alpha_beta:+.2f}pp/yr  MDD={mdd:.2%}  pos-Q={pq:.2f}')

    # Pooled across all three folds.
    pool_mask = np.zeros(len(daily), dtype=bool)
    for _, va_s, va_e in FOLDS:
        pool_mask |= ((daily.index >= pd.Timestamp(va_s))
                      & (daily.index <= pd.Timestamp(va_e)))
    pooled = daily.loc[pool_mask].fillna(0.0)
    pooled_spy = spy_aligned.loc[pool_mask].fillna(0.0)
    pooled_edge = pooled.values - pooled_spy.values
    pooled_ann_sh = _ann_sharpe(pooled.values)
    pooled_alpha_ex, pooled_alpha_beta = _alpha_vs_spy(pooled, pooled_spy)
    pooled_defl_t = _deflated_t(pooled_edge, n_trials=1)
    pooled_strat_defl_t = _deflated_t(pooled.values, n_trials=1)
    print(f'\n[pooled 1+2+3] n={len(pooled)}  Sh={pooled_ann_sh:+.3f}  '
          f'α_excess={pooled_alpha_ex:+.2f}pp/yr  '
          f'α_β-adj={pooled_alpha_beta:+.2f}pp/yr  '
          f'pooled-edge defl-t={pooled_defl_t:+.3f}  '
          f'pooled-strat defl-t={pooled_strat_defl_t:+.3f}')

    # Fold-3 bootstrap CI on the *edge* (strategy − SPY).
    f3_edge = fold_edges['fold3']
    f3_lo, f3_hi, f3_samples = _stationary_bootstrap_ci(
        f3_edge, n_boot=2000, seed=42, avg_block=20.0, ppy=252)
    print(f'[fold3 boot] edge-α 95% CI = [{f3_lo:+.2f}, {f3_hi:+.2f}] pp/yr  '
          f'(point {fold_rows[2]["alpha_excess_pct_yr"]:+.2f})')

    # Fold-1+2 confirmation against v0 +5.25 pp/yr alpha.
    f12_mask = np.zeros(len(daily), dtype=bool)
    for label, va_s, va_e in FOLDS[:2]:
        f12_mask |= ((daily.index >= pd.Timestamp(va_s))
                     & (daily.index <= pd.Timestamp(va_e)))
    f12_strat = daily.loc[f12_mask].fillna(0.0)
    f12_spy = spy_aligned.loc[f12_mask].fillna(0.0)
    f12_ann_sh = _ann_sharpe(f12_strat.values)
    f12_alpha_ex, f12_alpha_beta = _alpha_vs_spy(f12_strat, f12_spy)
    print(f'[pooled 1+2] Sh={f12_ann_sh:+.3f}  '
          f'α_excess={f12_alpha_ex:+.2f}pp/yr  '
          f'α_β-adj={f12_alpha_beta:+.2f}pp/yr  '
          f'(v0 reported +1.006 Sh / +5.25 pp/yr)')

    # Pre-reg verdict per TODO.
    fold3 = fold_rows[2]
    f3_alpha = fold3['alpha_excess_pct_yr']
    ci_excludes_zero = (f3_lo > 0) or (f3_hi < 0)

    if f3_alpha < 0:
        verdict = 'reversed-OOS'
    elif f3_alpha < 1.0:
        verdict = 'confirmed-null'
    elif (pooled_ann_sh >= 0.85 and pooled_alpha_ex >= 3.0
          and f3_alpha >= 1.0):
        verdict = 'confirmed-OOS'
    elif (f12_alpha_ex >= 5.0 and f3_alpha >= 0
          and pooled_defl_t > 1.0):
        verdict = 'partial-OOS'
    else:
        verdict = 'diagnostic'

    # Also evaluate user-brief bars (alternative reading).
    user_brief = {
        'confirmed_OOS': (f3_alpha >= 1.0 and ci_excludes_zero
                          and pooled_defl_t > 2.0),
        'partial_OOS': (f3_alpha >= 1.0 and (not ci_excludes_zero
                                              or pooled_defl_t < 2.0)),
        'confirmed_null': (f3_alpha < 1.0 or f3_alpha < 0),
        'reversed_OOS': f3_alpha < -1.0,
    }

    print(f'\n=== VERDICT (TODO pre-reg) === {verdict}')
    print(f'  pooled-Sh={pooled_ann_sh:.3f}  pooled-α={pooled_alpha_ex:.2f}pp/yr')
    print(f'  fold-3 α={f3_alpha:.2f}pp/yr  CI=[{f3_lo:.2f},{f3_hi:.2f}]  '
          f'CI excludes 0: {ci_excludes_zero}')
    print(f'  pooled defl-t (edge)={pooled_defl_t:+.3f}')
    print(f'  user-brief bar: {user_brief}')

    # NPZ for the ladder.
    pre_reg = {
        'confirmed_OOS': {
            'pooled_ann_sharpe_geq': 0.85,
            'pooled_alpha_vs_spy_pct_yr_geq': 3.0,
            'fold3_alpha_vs_spy_pct_yr_geq': 1.0,
        },
        'partial_OOS': {
            'fold12_alpha_vs_spy_pct_yr_geq': 5.0,
            'fold3_alpha_vs_spy_pct_yr_geq': 0.0,
            'pooled_deflated_t_gt': 1.0,
        },
        'confirmed_null': {'fold3_alpha_vs_spy_pct_yr_lt': 1.0},
        'reversed_OOS': {'fold3_alpha_vs_spy_pct_yr_lt': 0.0},
        'cell': {
            'leadership_only': False, 'hold_days': HOLD_DAYS,
            'top_k': TOP_K, 'filter_mode': FILTER_MODE,
            'commission_bps': COMMISSION_BPS,
        },
        'entry_rule': 'filed+1 trading day (Bowne 2024 disclosure-lag-honest)',
        'note': 'Locked BEFORE eval per TODO/follow-consensus-arm.md.',
    }

    npz_path = OUTPUT / f'{args.output_stem}.npz'
    np.savez(
        npz_path,
        oos_block_returns=pooled.values.astype(np.float64),
        spy_ret=pooled_spy.values.astype(np.float64),
        fold3_strat_returns=daily.loc[
            (daily.index >= pd.Timestamp(FOLDS[2][1]))
            & (daily.index <= pd.Timestamp(FOLDS[2][2]))
        ].fillna(0.0).values.astype(np.float64),
        fold3_spy_returns=spy_aligned.loc[
            (spy_aligned.index >= pd.Timestamp(FOLDS[2][1]))
            & (spy_aligned.index <= pd.Timestamp(FOLDS[2][2]))
        ].fillna(0.0).values.astype(np.float64),
        fold3_edge_returns=f3_edge.astype(np.float64),
        fold3_boot_samples=f3_samples.astype(np.float64),
        val_sharpe=np.float64(pooled_ann_sh),
        pooled_alpha_excess_pct_yr=np.float64(pooled_alpha_ex),
        pooled_alpha_beta_adj_pct_yr=np.float64(pooled_alpha_beta),
        pooled_deflated_t_edge=np.float64(pooled_defl_t),
        pooled_deflated_t_strat=np.float64(pooled_strat_defl_t),
        fold3_alpha_excess_pct_yr=np.float64(f3_alpha),
        fold3_alpha_beta_adj_pct_yr=np.float64(fold3['alpha_beta_adj_pct_yr']),
        fold3_ci_lo=np.float64(f3_lo),
        fold3_ci_hi=np.float64(f3_hi),
        fold3_ci_excludes_zero=np.bool_(ci_excludes_zero),
        verdict=np.array(verdict),
        periods_per_year=np.float64(252.0),
        n_grid_cells=np.int64(1),
        universe_label=np.array('congressional-all-members'),
        pre_registered_bar=np.array(json.dumps(pre_reg)),
    )
    print(f'-> {npz_path}')

    # JSON dump.
    summary = {
        'pre_registered_bar': pre_reg,
        'verdict': verdict,
        'folds': fold_rows,
        'pooled_1_2_3': {
            'n_days': int(len(pooled)),
            'ann_sharpe_net': pooled_ann_sh,
            'alpha_excess_pct_yr': pooled_alpha_ex,
            'alpha_beta_adj_pct_yr': pooled_alpha_beta,
            'pooled_deflated_t_edge': pooled_defl_t,
            'pooled_deflated_t_strat': pooled_strat_defl_t,
        },
        'pooled_1_2': {
            'n_days': int(len(f12_strat)),
            'ann_sharpe_net': f12_ann_sh,
            'alpha_excess_pct_yr': f12_alpha_ex,
            'alpha_beta_adj_pct_yr': f12_alpha_beta,
            'v0_reported_ann_sharpe': 1.006,
            'v0_reported_alpha_pct_yr': 5.25,
        },
        'fold3_bootstrap': {
            'n_boot': 2000, 'seed': 42, 'avg_block': 20,
            'ci_lo_pp_yr': f3_lo, 'ci_hi_pp_yr': f3_hi,
            'ci_excludes_zero': ci_excludes_zero,
            'point_alpha_pp_yr': f3_alpha,
        },
        'user_brief_bar_check': user_brief,
        'drop_stats': panel.drop_stats,
    }
    json_path = OUTPUT / f'{args.output_stem}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'-> {json_path}')


if __name__ == '__main__':
    main()
