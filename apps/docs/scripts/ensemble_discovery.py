"""Cross-arc ensemble discovery — does ANY combination of the leaderboard's
positive-deflated-t arcs honestly beat DCA-alone on the same date overlap?

Reads every available arc OOS net-return stream under ``Output/``, date-aligns
each pair / triple / quad on the **intersection** of their dates, computes the
ensemble deflated-t under the workspace's ``sharpe_std_ann=0.25`` calibration,
and compares it to **DCA-alone evaluated on the same intersection window** —
NOT to DCA's full-sample +1.93, which would be sample-window selection bias.

Honest scoring rules (encoded here, see brief at
``.research-leaderboard-ensembles.md``):

1.  Date-align by intersection. Daily streams are aligned bar-for-bar by
    inferred date axis (each stream's tail-anchored business-day grid).
    Block streams (vol, monthly factor, etc.) are aligned by compounding
    the higher-frequency stream's daily returns over each lower-frequency
    block window.
2.  ``n_trials`` for a k-arc ensemble = sum of component n_trials + (k-1)
    for the (k-1)-dim mixing-weight search step. ``sharpe_std_ann=0.25``.
3.  Weight rule = one of: equal-weight, inverse-variance, or tangency on
    the same overlap window. Per spec we report all three but rank on
    inverse-variance (single fixed rule, avoids weight grid-search).
4.  Capital semantics: vol-v3 streams are *capital-free overlays*
    (additive). Everything else (DCA, relational, factor-LO, momentum LS,
    low-vol LS) is treated as capital-competing — equal weighted sum.
    Sum of weights normalized to 1 for capital-competing component;
    overlay vol added on top.
5.  Apples-to-apples reference: DCA-alone evaluated on the SAME date
    intersection as the ensemble. Without this, sample-window selection
    fakes a lift.

The most credible deliverable from this exercise might be "no ensemble
honestly beats DCA-alone once the n_trials penalty is applied" — and
that's a valuable negative answer.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ss_portfolio import standardize_oos

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / 'Output'

# Match the leaderboard's calibration.
SHARPE_STD_ANN = 0.25


@dataclass(frozen=True)
class ArcStream:
    key: str
    returns: np.ndarray            # 1-D net return series at native frequency
    dates: pd.DatetimeIndex        # one date per return entry
    ppy: float                     # periods per year at native frequency
    n_trials: int                  # arc-level deflation count
    capital_class: str             # 'cash' | 'overlay'
    note: str = ''


# ---------------------------------------------------------------------------
# Stream loaders. Each function returns an ArcStream with native frequency
# dates. For arcs without explicit date metadata, we infer a tail-anchored
# business-day grid — flagged as approximate in the brief.
# ---------------------------------------------------------------------------

def _load_dca() -> ArcStream:
    d = np.load(OUTPUT / 'dca-returns.npz')
    r = np.asarray(d['daily_ret'], dtype=np.float64)
    # DCA close panel runs 2005-02-25 -> 2025-12-11 over 5232 daily bars.
    # The daily_returns dumper keeps all 5232 finite entries; tail-anchor
    # to 2025-12-11 over 5232 business days for exactness.
    end = pd.Timestamp('2025-12-11')
    dates = pd.bdate_range(end=end, periods=r.size)
    return ArcStream('dca', r, dates, ppy=252.0, n_trials=4,
                     capital_class='cash',
                     note='daily, 2005-02-25→2025-12-11')


def _load_relational() -> ArcStream:
    d = np.load(OUTPUT / 'relational-returns.npz', allow_pickle=True)
    r = np.asarray(d['val_daily_ret'], dtype=np.float64)
    # Val window: 2021-01-01 → 2025-12-11 per
    # apps/relational/.../idea_b_analog_knn_dwt_walkforward.py.
    end = pd.Timestamp('2025-12-11')
    dates = pd.bdate_range(end=end, periods=r.size)
    return ArcStream('relational-analog', r, dates, ppy=252.0, n_trials=16,
                     capital_class='cash',
                     note='daily, ~2021-01→2025-12-11')


def _load_vol_v3_dolthub() -> ArcStream:
    d = np.load(OUTPUT / 'vol-v3-dolthub-oos-returns.npz', allow_pickle=True)
    r = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    dates = pd.to_datetime(d['rebal_dates'].astype(str))
    return ArcStream('vol-v3-dolthub', r, pd.DatetimeIndex(dates), ppy=12.6,
                     n_trials=12, capital_class='overlay',
                     note='monthly-ish rebal, explicit dates 2023-08→2026-03')


def _load_vol_gauss314() -> ArcStream:
    d = np.load(OUTPUT / 'vol-returns.npz')
    r = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    # gauss314 IV file covered 2019-01→2023-07 (30 monthly rebals).
    # Tail-anchor monthly with ppy=12.6 (~28-day rebal). Approximate.
    end = pd.Timestamp('2023-07-31')
    dates = pd.date_range(end=end, periods=r.size, freq=pd.tseries.offsets.BDay(20))
    return ArcStream('vol-v3-gauss314', r, pd.DatetimeIndex(dates), ppy=12.6,
                     n_trials=12, capital_class='overlay',
                     note='approx tail-anchored 2019→2023')


def _load_momentum() -> ArcStream:
    d = np.load(OUTPUT / 'momentum-12-1-returns.npz')
    r = np.asarray(d['ls_block_returns'], dtype=np.float64)
    # 302 monthly blocks; data 2000-01→2026-04 with 273-bar formation+skip
    # ⇒ first rebal ~2001-02; monthly hold=21bdays.
    start = pd.Timestamp('2001-02-01')
    dates = pd.date_range(start=start, periods=r.size, freq=pd.tseries.offsets.BDay(21))
    return ArcStream('momentum-12-1-LS', r, pd.DatetimeIndex(dates), ppy=12.0,
                     n_trials=1, capital_class='cash',  # L/S but capital-using
                     note='monthly, ~2001-02→2026-04 (inferred)')


def _load_low_vol() -> ArcStream:
    d = np.load(OUTPUT / 'low-vol-bab-returns.npz')
    r = np.asarray(d['ls_block_returns'], dtype=np.float64)
    start = pd.Timestamp('2001-02-01')
    dates = pd.date_range(start=start, periods=r.size, freq=pd.tseries.offsets.BDay(21))
    return ArcStream('low-vol-bab-LS', r, pd.DatetimeIndex(dates), ppy=12.0,
                     n_trials=1, capital_class='cash',
                     note='monthly, ~2001-02→2026-04 (inferred)')


def _load_factor_5d_LO() -> ArcStream:
    d = np.load(OUTPUT / 'sh-indicator-r5-s1-windows.npz', allow_pickle=True)
    r = np.asarray(d['oos_block_returns'], dtype=np.float64)
    # 6 walkforward windows × 156 blocks of 5 trading days = 936 entries.
    # Span ~2004-2026 (factor walkforward 2000→2026 with train warmup).
    # Tail-anchor: end at 2026-04-01, step 5 BDays.
    end = pd.Timestamp('2026-04-01')
    dates = pd.date_range(end=end, periods=r.size, freq=pd.tseries.offsets.BDay(5))
    return ArcStream('factor-5d-LO-focused', r, pd.DatetimeIndex(dates), ppy=50.4,
                     n_trials=8, capital_class='cash',
                     note='5-day blocks, tail-anchored ~2004→2026 (approx)')


LOADERS = [
    _load_dca, _load_relational, _load_vol_v3_dolthub, _load_vol_gauss314,
    _load_momentum, _load_low_vol, _load_factor_5d_LO,
]


# ---------------------------------------------------------------------------
# Frequency reconciliation: compound a daily stream onto a block stream's
# date grid by summing intra-block daily returns (log-additive approximation
# good for small returns; matches workspace block-Sharpe convention).
# ---------------------------------------------------------------------------

def _to_series(arc: ArcStream) -> pd.Series:
    return pd.Series(arc.returns, index=arc.dates).sort_index()


def _compound_to_grid(daily: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Compound a daily return series onto a coarser block grid.

    Each entry at grid[i] = sum of daily returns over (grid[i-1], grid[i]].
    For the first grid block we use the daily series tail of length =
    median block span. Log-additive approximation (small-return regime).
    """
    g = grid.sort_values()
    # Use a 1+r → log → cumsum → diff approach for exactness:
    log_d = np.log1p(daily).cumsum()
    # Align at the grid; ffill so missing trading days take last cumlog.
    aligned = log_d.reindex(daily.index.union(g)).sort_index().ffill().reindex(g)
    blocks = aligned.diff()
    # First block: from start-of-history to g[0]
    if len(g) > 0 and not pd.isna(aligned.iloc[0]):
        blocks.iloc[0] = aligned.iloc[0]
    return np.expm1(blocks)


def align_pair(a: ArcStream, b: ArcStream) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex, float]:
    """Return aligned (series_a, series_b, date_index, periods_per_year).

    The coarser arc dictates the grid. The finer-frequency arc gets
    compounded onto the coarse grid.
    """
    sa, sb = _to_series(a), _to_series(b)
    if a.ppy >= b.ppy:
        fine, coarse = a, b
        s_fine, s_coarse = sa, sb
    else:
        fine, coarse = b, a
        s_fine, s_coarse = sb, sa
    # Coarse grid restricted to fine's date span.
    grid = s_coarse.index[(s_coarse.index >= s_fine.index.min())
                           & (s_coarse.index <= s_fine.index.max())]
    if len(grid) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DatetimeIndex([]), coarse.ppy
    fine_on_grid = _compound_to_grid(s_fine, grid)
    coarse_on_grid = s_coarse.reindex(grid)
    df = pd.concat([fine_on_grid, coarse_on_grid], axis=1).dropna()
    if a.ppy >= b.ppy:
        return df.iloc[:, 0], df.iloc[:, 1], df.index, coarse.ppy
    else:
        return df.iloc[:, 1], df.iloc[:, 0], df.index, coarse.ppy


def align_many(arcs: list[ArcStream]) -> tuple[list[pd.Series], pd.DatetimeIndex, float]:
    """Align an arbitrary list onto the coarsest arc's grid."""
    coarsest = min(arcs, key=lambda a: a.ppy)
    grid = _to_series(coarsest).index
    # Restrict grid to intersection of all arcs' date spans.
    lo = max(a.dates.min() for a in arcs)
    hi = min(a.dates.max() for a in arcs)
    grid = grid[(grid >= lo) & (grid <= hi)]
    aligned = []
    for a in arcs:
        s = _to_series(a)
        if a.ppy > coarsest.ppy:
            aligned.append(_compound_to_grid(s, grid))
        else:
            aligned.append(s.reindex(grid))
    df = pd.concat(aligned, axis=1).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])], df.index, coarsest.ppy


# ---------------------------------------------------------------------------
# Ensemble construction. Capital-competing arcs share weight space; overlay
# arcs add additively.
# ---------------------------------------------------------------------------

def build_ensemble(arcs: list[ArcStream], series: list[pd.Series], weight_rule: str) -> np.ndarray:
    """Combine arc return series into one ensemble net-return stream.

    weight_rule ∈ {'equal', 'invvar', 'tangency'} controls the cash-book
    weights. Overlay arcs are always added 1× on top (vega-budget scaled).
    """
    cash_idx = [i for i, a in enumerate(arcs) if a.capital_class == 'cash']
    overlay_idx = [i for i, a in enumerate(arcs) if a.capital_class == 'overlay']
    mat = np.column_stack([s.values for s in series])
    n = mat.shape[0]
    if not cash_idx:
        # All overlay — just sum them (no capital base).
        return mat.sum(axis=1)
    cash_mat = mat[:, cash_idx]
    if weight_rule == 'equal':
        w = np.ones(cash_mat.shape[1]) / cash_mat.shape[1]
    elif weight_rule == 'invvar':
        var = cash_mat.var(axis=0, ddof=0)
        var = np.where(var > 0, var, 1.0)
        w = (1.0 / var)
        w = w / w.sum()
    elif weight_rule == 'tangency':
        # Single-rule tangency on the overlap window. Pin to nonnegative.
        mu = cash_mat.mean(axis=0)
        cov = np.cov(cash_mat, rowvar=False, ddof=0)
        if cash_mat.shape[1] == 1:
            w = np.array([1.0])
        else:
            try:
                w_raw = np.linalg.solve(cov + 1e-8 * np.eye(cov.shape[0]), mu)
                w_raw = np.clip(w_raw, 0.0, None)
                if w_raw.sum() <= 0:
                    w_raw = np.ones_like(w_raw)
                w = w_raw / w_raw.sum()
            except np.linalg.LinAlgError:
                w = np.ones(cash_mat.shape[1]) / cash_mat.shape[1]
    else:
        raise ValueError(weight_rule)
    cash_book = cash_mat @ w
    if overlay_idx:
        overlay_sum = mat[:, overlay_idx].sum(axis=1)
        return cash_book + overlay_sum
    return cash_book


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_stream(r: np.ndarray, ppy: float, n_trials: int) -> dict:
    sharpe_std_pp = SHARPE_STD_ANN / math.sqrt(ppy)
    mb = standardize_oos(r, periods_per_year=ppy, n_trials=n_trials,
                         sharpe_std=sharpe_std_pp)
    return {'n_obs': mb.n_obs, 'ann_sharpe': mb.ann_sharpe,
            'deflated_t': mb.deflated_tstat, 'dsr': mb.dsr,
            'skew': mb.skew, 'kurt': mb.kurtosis,
            'expected_max_sharpe': mb.expected_max_sharpe}


def evaluate_combo(arcs: list[ArcStream], weight_rule: str = 'invvar') -> dict:
    series, idx, ppy = align_many(arcs)
    if len(idx) < 5:
        return {'n_overlap': len(idx), 'skipped': True}
    ens = build_ensemble(arcs, series, weight_rule)
    k = len(arcs)
    n_trials = sum(a.n_trials for a in arcs) + max(k - 1, 0)
    res = {
        'components': [a.key for a in arcs],
        'weight_rule': weight_rule,
        'n_overlap': len(idx),
        'ppy': ppy,
        'date_start': str(idx.min().date()),
        'date_end': str(idx.max().date()),
        'n_trials': n_trials,
        'ensemble': eval_stream(ens, ppy, n_trials),
    }
    # Component stats on same overlap.
    comp_stats = {}
    for a, s in zip(arcs, series):
        comp_stats[a.key] = eval_stream(s.values, ppy, a.n_trials)
    res['components_on_overlap'] = comp_stats
    # Correlation matrix on overlap.
    mat = np.column_stack([s.values for s in series])
    if mat.shape[1] > 1:
        corr = np.corrcoef(mat, rowvar=False)
        res['corr_matrix'] = corr.tolist()
    # DCA-on-same-overlap reference (always recomputed for any combo that
    # doesn't already include DCA, OR taken from ensemble's DCA component).
    dca_keys = [a.key for a in arcs if a.key == 'dca']
    if dca_keys:
        res['dca_on_overlap'] = comp_stats['dca']
    else:
        # Compound DCA onto same grid for honest comparison.
        dca_arc = _load_dca()
        dca_s = _compound_to_grid(_to_series(dca_arc), idx)
        dca_s = dca_s.reindex(idx).dropna()
        if len(dca_s) >= 5:
            res['dca_on_overlap'] = eval_stream(dca_s.values, ppy, dca_arc.n_trials)
        else:
            res['dca_on_overlap'] = None
    # Lift vs DCA-on-overlap.
    if res['dca_on_overlap']:
        res['lift_vs_dca_on_overlap'] = (
            res['ensemble']['deflated_t'] - res['dca_on_overlap']['deflated_t'])
    return res


def main() -> None:
    arcs = [ld() for ld in LOADERS]
    print('Loaded arcs:')
    for a in arcs:
        print(f'  {a.key:25s} n={a.returns.size:5d}  ppy={a.ppy:6.2f}  '
              f'class={a.capital_class:7s} trials={a.n_trials:3d}  '
              f'{a.dates.min().date()} → {a.dates.max().date()}  ({a.note})')

    results = {'singles': {}, 'pairs': [], 'triples': [], 'quads': [], 'all_in': None}

    # Singles (sanity check: native-frequency standalone deflated-t).
    for a in arcs:
        results['singles'][a.key] = eval_stream(a.returns, a.ppy, a.n_trials)

    # Pairs.
    for combo in itertools.combinations(arcs, 2):
        for rule in ('equal', 'invvar', 'tangency'):
            r = evaluate_combo(list(combo), weight_rule=rule)
            r['size'] = 2
            results['pairs'].append(r)

    # Triples (invvar only, to keep output manageable).
    for combo in itertools.combinations(arcs, 3):
        r = evaluate_combo(list(combo), weight_rule='invvar')
        r['size'] = 3
        results['triples'].append(r)

    # Quads.
    for combo in itertools.combinations(arcs, 4):
        r = evaluate_combo(list(combo), weight_rule='invvar')
        r['size'] = 4
        results['quads'].append(r)

    # The all-in inv-var ensemble.
    results['all_in'] = evaluate_combo(arcs, weight_rule='invvar')

    out = OUTPUT / 'ensemble-discovery.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'\n-> {out}')

    # Print top-10 each tier by deflated_t.
    def _key(r):
        return r.get('ensemble', {}).get('deflated_t', -1e9) if not r.get('skipped') else -1e9

    print('\n=== TOP 10 PAIRS by ensemble deflated-t (any weight rule) ===')
    pairs_sorted = sorted(results['pairs'], key=_key, reverse=True)
    for r in pairs_sorted[:10]:
        if r.get('skipped'):
            continue
        print(f"  {'+'.join(r['components']):55s} [{r['weight_rule']:8s}] "
              f"n={r['n_overlap']:4d} ens_t={r['ensemble']['deflated_t']:+5.2f} "
              f"DCA_overlap_t={r['dca_on_overlap']['deflated_t'] if r['dca_on_overlap'] else float('nan'):+5.2f} "
              f"lift={r.get('lift_vs_dca_on_overlap', float('nan')):+5.2f}")

    print('\n=== TOP 5 TRIPLES (invvar) ===')
    for r in sorted(results['triples'], key=_key, reverse=True)[:5]:
        if r.get('skipped'):
            continue
        print(f"  {'+'.join(r['components']):55s} n={r['n_overlap']:4d} "
              f"ens_t={r['ensemble']['deflated_t']:+5.2f} "
              f"DCA_overlap_t={r['dca_on_overlap']['deflated_t'] if r['dca_on_overlap'] else float('nan'):+5.2f} "
              f"lift={r.get('lift_vs_dca_on_overlap', float('nan')):+5.2f}")

    print('\n=== TOP 5 QUADS (invvar) ===')
    for r in sorted(results['quads'], key=_key, reverse=True)[:5]:
        if r.get('skipped'):
            continue
        print(f"  {'+'.join(r['components']):55s} n={r['n_overlap']:4d} "
              f"ens_t={r['ensemble']['deflated_t']:+5.2f} "
              f"DCA_overlap_t={r['dca_on_overlap']['deflated_t'] if r['dca_on_overlap'] else float('nan'):+5.2f} "
              f"lift={r.get('lift_vs_dca_on_overlap', float('nan')):+5.2f}")

    print('\n=== ALL-IN (every arc, invvar) ===')
    r = results['all_in']
    if r and not r.get('skipped'):
        print(f"  {'+'.join(r['components'])} n={r['n_overlap']} "
              f"ens_t={r['ensemble']['deflated_t']:+.2f} "
              f"DCA_overlap_t={r['dca_on_overlap']['deflated_t']:+.2f} "
              f"lift={r['lift_vs_dca_on_overlap']:+.2f}")
    else:
        print('  (no overlap)')

    # DCA-alone full-sample reference for context.
    dca = arcs[0]
    print(f'\nDCA full-sample deflated-t (n={dca.returns.size}): '
          f"{results['singles']['dca']['deflated_t']:+.3f}  "
          f"(this is the +1.93-class number; ensembles are scored on their "
          'OWN overlap window vs DCA-on-same-overlap.)')


if __name__ == '__main__':
    main()
