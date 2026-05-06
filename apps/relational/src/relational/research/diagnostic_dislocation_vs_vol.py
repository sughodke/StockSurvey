"""Diagnostic — does any of the relational scorers forecast forward
realized vol expansion?

We don't have a historical IV feed, but IV market-makers anchor to
trailing realized vol. The diagnostic question is therefore: **for each
scorer, do its top-N picks show forward 20d realized vol that
systematically exceeds their trailing 20d realized vol?**

  * If yes → IV anchored to trailing realized would be too low → long
    straddles on top-N picks have an edge worth chasing with a real
    IV feed.
  * If no  → trailing realized already prices what's coming → no edge
    from this angle.

Scorers compared (`--scorer all`):
  - baseline  — `ss_portfolio.weights_regime` (temporal divergence)
  - farthest  — idea C, centroid distance
  - empirical — idea A, k-means peer-divergence
  - analog    — idea B, k-NN forward-return forecast
  - gics      — idea #1, static-GICS sector excess

Idea D (selector) is omitted; it operates on top of another ranking
rather than producing a per-ticker score matrix.

Output: prints a row per scorer and writes
`Output/relational-vol-expansion-diagnostic.txt`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ss_features import realized_vol_matrix
from ss_indicators import get_divergence
from ss_loaders import load_stooq_matrix
from ss_wavelets import causal_cwt, precompute_windows

from relational.analog_knn import analog_knn_scores
from relational.bocpd import changepoint_scores
from relational.empirical_sectors import empirical_excess_divergence_scores
from relational.farthest import centroid_distance_scores
from relational.iv_data import (
    load_atm_iv, load_dolthub_iv, load_dolthub_iv_parquet)
from relational.ot_stress import ot_stress_scores
from relational.scale_energy import (
    scale_energy_ratio_scores, scale_entropy_scores)
from relational.scoring import baseline_divergence_scores, excess_divergence_scores
from relational.sectors import PHASE2_TICKERS

warnings.filterwarnings('ignore')


SCORER_CHOICES = (
    'baseline', 'farthest', 'empirical', 'analog', 'gics',
    'n1_ratio', 'n1_entropy', 'n2_bocpd', 'r1_ot',
    'all', 'brainstorm',
)
ALL_SCORERS = ('baseline', 'farthest', 'empirical', 'analog', 'gics')
BRAINSTORM_SCORERS = ('n1_ratio', 'n1_entropy', 'n2_bocpd', 'r1_ot')


def _compute_scores(name: str, prices: pd.DataFrame, *,
                    lookback: int, n_tail: int, scales: list[int],
                    fp_window: int, k_clusters: int) -> np.ndarray:
    if name == 'baseline':
        return baseline_divergence_scores(
            prices, lookback=lookback, n_tail=n_tail, scales=scales)
    if name == 'farthest':
        return centroid_distance_scores(
            prices, lookback=lookback, scales=scales, fp_window=fp_window)
    if name == 'empirical':
        return empirical_excess_divergence_scores(
            prices, lookback=lookback, n_tail=n_tail, scales=scales,
            fp_window=fp_window, k_clusters=k_clusters)
    if name == 'analog':
        return analog_knn_scores(
            prices, lookback=lookback, scales=scales, fp_window=fp_window)
    if name == 'gics':
        return excess_divergence_scores(
            prices, lookback=lookback, n_tail=n_tail, scales=scales)
    if name == 'n1_ratio':
        return scale_energy_ratio_scores(
            prices, lookback=lookback, scales=scales)
    if name == 'n1_entropy':
        return scale_entropy_scores(
            prices, lookback=lookback, scales=scales)
    if name == 'n2_bocpd':
        return changepoint_scores(
            prices, lookback=lookback, scales=scales)
    if name == 'r1_ot':
        return ot_stress_scores(
            prices, lookback=lookback, scales=scales, fp_window=fp_window)
    raise ValueError(f'unknown scorer {name!r}')


def _evaluate(
    scores: np.ndarray,
    forward_ann: np.ndarray,
    anchor_ann: np.ndarray,
    prices: pd.DataFrame,
    *,
    lookback: int, top_n: int, rebal_days: int, vol_window: int,
) -> dict:
    """Per-scorer evaluation. `anchor_ann` is the IV-or-trailing-realized
    anchor at signal time (annualized, fractional units). `forward_ann`
    is the realized vol over the forward window (annualized, fractional).
    Both are `(n_dates, n_tickers)`.

    Reports forward/anchor expansion ratio and paired top-vs-rest gap.
    """
    n_dates = prices.shape[0]
    n_eval = scores.shape[0]
    rebal_eval_idx = np.arange(0, n_eval, rebal_days)

    rows: list[dict] = []
    for e in rebal_eval_idx:
        t = e + lookback
        if t + vol_window >= n_dates:
            break
        score_row = scores[e]
        anchor = anchor_ann[t]
        forward = forward_ann[t + vol_window]
        ok = (np.isfinite(score_row) & np.isfinite(anchor)
              & np.isfinite(forward) & (anchor > 0))
        if ok.sum() < top_n + 1:
            continue
        active = np.where(ok)[0]
        order = np.argsort(-score_row[active])
        ranked = active[order]
        top_idx = set(ranked[:top_n].tolist())
        date = prices.index[t]
        for j in active:
            rows.append({
                'date': date,
                'group': 'top' if j in top_idx else 'rest',
                'anchor': float(anchor[j]),
                'forward': float(forward[j]),
                'expansion': float(forward[j] / anchor[j]),
                'log_expansion': float(np.log(forward[j] / anchor[j])),
            })

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return {'n_rebals': 0, 'n_top_rows': 0,
                'top_anchor_ann': float('nan'),
                'top_forward_ann': float('nan'),
                'rest_anchor_ann': float('nan'),
                'rest_forward_ann': float('nan'),
                'top_median_expansion': float('nan'),
                'top_hit_rate': float('nan'),
                'paired_log_gap_mean': float('nan'),
                'paired_t_stat': float('nan'),
                'frac_rebals_top_gt_rest': float('nan')}
    top = df[df['group'] == 'top']
    rest = df[df['group'] == 'rest']

    grouped = df.groupby(['date', 'group'])['log_expansion'].mean().unstack('group')
    grouped = grouped.dropna(subset=['top', 'rest'])
    paired = grouped['top'] - grouped['rest']
    t_stat = (paired.mean() / (paired.std() / np.sqrt(len(paired)))
              if len(paired) > 1 else float('nan'))

    return {
        'n_rebals': len(grouped),
        'n_top_rows': len(top),
        'top_anchor_ann': top['anchor'].mean(),
        'top_forward_ann': top['forward'].mean(),
        'rest_anchor_ann': rest['anchor'].mean(),
        'rest_forward_ann': rest['forward'].mean(),
        'top_median_expansion': top['expansion'].median(),
        'top_hit_rate': float((top['forward'] > top['anchor']).mean()),
        'paired_log_gap_mean': float(paired.mean()),
        'paired_t_stat': float(t_stat),
        'frac_rebals_top_gt_rest': float((paired > 0).mean()),
    }


def run(
    *, data_dir: str,
    scorer: str = 'all',
    top_n: int = 10,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    rebal_days: int = 20,
    vol_window: int = 20,
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    output_dir: str = 'Output',
    all_tickers: bool = False,
    iv_anchor: bool = False,
    iv_source: str = 'gauss314',
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=lookback + 50,
        start_date=start, end_date=end,
        tickers=None if all_tickers else list(PHASE2_TICKERS))
    print(f'  {prices.shape[0]} dates x {prices.shape[1]} tickers')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'fp_window={fp_window}, k_clusters={k_clusters}')
    print(f'  top_n={top_n}, rebal_days={rebal_days}, vol_window={vol_window}')
    print(f'  anchor: {"ATM IV (HF gauss314)" if iv_anchor else "trailing realized vol"}')

    print('\nComputing per-ticker rolling realized vol (forward target)...')
    rv = realized_vol_matrix(prices, vol_window)
    forward_ann = rv * np.sqrt(252)  # daily std → annualized fractional

    if iv_anchor:
        print(f'Loading ATM IV (source={iv_source})...')
        if iv_source == 'gauss314':
            iv = load_atm_iv()
            iv_aligned = iv.reindex(index=prices.index, columns=prices.columns)
        elif iv_source == 'dolthub':
            # Weekly snapshots — forward-fill to daily before reindexing.
            iv = load_dolthub_iv_parquet(tickers=list(prices.columns))
            iv = iv.reindex(index=prices.index).ffill(limit=7)
            iv_aligned = iv.reindex(columns=prices.columns)
        else:
            raise ValueError(f'unknown iv_source: {iv_source!r}')
        anchor_ann = iv_aligned.values.astype(np.float64)
        n_iv = int(np.isfinite(anchor_ann).sum())
        n_total = anchor_ann.size
        print(f'  IV coverage on this panel: {n_iv}/{n_total} '
              f'({100 * n_iv / n_total:.1f}%) cells finite')
    else:
        anchor_ann = forward_ann.copy()
        # Anchor is trailing realized; align by shifting nothing — at date t,
        # rv[t] is the trailing-window vol ending at t.
        # forward_ann[t + vol_window] is the forward-window vol.

    if scorer == 'all':
        scorer_list = list(ALL_SCORERS)
    elif scorer == 'brainstorm':
        scorer_list = list(BRAINSTORM_SCORERS)
    else:
        scorer_list = [scorer]
    rows: list[dict] = []
    for name in scorer_list:
        print(f'\n[{name}] computing scores...')
        scores = _compute_scores(
            name, prices, lookback=lookback, n_tail=n_tail,
            scales=scales, fp_window=fp_window, k_clusters=k_clusters)
        print(f'  scores shape: {scores.shape}')
        result = _evaluate(
            scores, forward_ann, anchor_ann, prices,
            lookback=lookback, top_n=top_n,
            rebal_days=rebal_days, vol_window=vol_window)
        result = {'scorer': name, **result}
        rows.append(result)

    summary = pd.DataFrame.from_records(rows)
    print('\n' + '=' * 110)
    print('Forward vs trailing realized vol — per scorer')
    print('=' * 110)
    cols_levels = ['scorer', 'top_anchor_ann', 'top_forward_ann',
                   'rest_anchor_ann', 'rest_forward_ann']
    cols_test = ['scorer', 'top_median_expansion', 'top_hit_rate',
                 'paired_log_gap_mean', 'paired_t_stat',
                 'frac_rebals_top_gt_rest', 'n_rebals']
    with pd.option_context(
        'display.float_format', lambda x: f'{x:.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print('\nVol levels (annualized):')
        print(summary[cols_levels].to_string(index=False))
        print('\nExpansion test:')
        print(summary[cols_test].to_string(index=False))

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    suffix = '-iv' if iv_anchor else ''
    txt_path = out / f'relational-vol-expansion-diagnostic{suffix}.txt'
    with open(txt_path, 'w') as f:
        f.write('Forward vs trailing realized vol — per scorer\n')
        f.write('=' * 110 + '\n')
        f.write('Vol levels (annualized):\n')
        f.write(summary[cols_levels].to_string(index=False) + '\n\n')
        f.write('Expansion test:\n')
        f.write(summary[cols_test].to_string(index=False) + '\n')
    print(f'\nSaved {txt_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--scorer', default='all', choices=SCORER_CHOICES)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--vol-window', type=int, default=20)
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--output-dir', default='Output')
    p.add_argument('--all-tickers', action='store_true',
                   help='Skip the Phase-2 filter; use every ticker in --data-dir')
    p.add_argument('--iv-anchor', action='store_true',
                   help='Use ATM IV as the vol anchor instead of trailing realized')
    p.add_argument('--iv-source', default='gauss314',
                   choices=('gauss314', 'dolthub'),
                   help='IV provider: gauss314 (HF, daily, 2019-10→2023-07) '
                        'or dolthub (weekly, 2019-02→2026-04)')
    args = p.parse_args()
    run(**vars(args))
