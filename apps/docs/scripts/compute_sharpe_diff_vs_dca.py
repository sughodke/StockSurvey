"""Step 4 of TODO/ladder-methodology-rewrite.md: add the Ledoit-Wolf
studentized stationary-bootstrap Sharpe-difference CI column to the
DSR ladder.

For each leaderboard arc with a return-stream NPZ on disk, compute
the Sharpe difference vs DCA-canonical over the date-aligned common
window with frequency collapsed to the lower (block) cadence. Bootstrap
CI per `ss_portfolio.sharpe_difference_ci` (Politis-Romano stationary
block bootstrap; Ledoit-Wolf studentized inversion).

Output `Output/sharpe-diff-vs-dca.json` for the leaderboard's new
column. Arcs lacking explicit dates in their NPZ are aligned via
tail-matching with a clearly-flagged caveat in the output.

Run from repo root:
    uv run python apps/docs/scripts/compute_sharpe_diff_vs_dca.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ss_portfolio import sharpe_difference_ci

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'

# DCA canonical = the baseline every other arc is measured against
DCA_NPZ = 'dca-returns.npz'
DCA_STREAM_KEY = 'daily_ret'
DCA_PPY = 252.0
DCA_START = pd.Timestamp('2005-02-25')   # known from cfr_phase4d pickle


def _load_dca_daily() -> pd.Series:
    """Load DCA daily-return stream as a date-indexed Series."""
    import pickle
    with open(REPO_ROOT / 'Output/cfr_phase4d_multiasset_close.pkl', 'rb') as f:
        close = pickle.load(f)
    from cfr.baselines import PassiveEW
    daily = np.asarray(
        PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(close),
        dtype=np.float64)
    return pd.Series(daily, index=close.index, dtype=np.float64).dropna()


def _block_aggregate_daily(daily: pd.Series, block_dates: pd.DatetimeIndex,
                           block_size: int = 20) -> np.ndarray:
    """For each date in `block_dates`, compound the next `block_size`
    daily returns. Used to align a daily stream to a block-cadence arc.
    """
    out = []
    for d in block_dates:
        pos = daily.index.searchsorted(d, side='left')
        if pos + block_size >= len(daily):
            out.append(np.nan); continue
        win = daily.iloc[pos+1 : pos+1+block_size]
        out.append((1.0 + win).prod() - 1.0 if win.size >= block_size * 0.7
                   else np.nan)
    return np.asarray(out, dtype=np.float64)


def _read_arc_stream(npz: str, stream_key: str) -> tuple[np.ndarray, pd.DatetimeIndex | None]:
    """Return (returns_array, dates_or_None). Dates non-None when the
    NPZ explicitly carries rebal_dates / dates / etc."""
    p = OUTPUT / npz
    if not p.exists():
        return np.array([]), None
    d = np.load(p, allow_pickle=True)
    if stream_key not in d.files:
        return np.array([]), None
    r = np.asarray(d[stream_key], dtype=np.float64)
    dates = None
    for k in ('rebal_dates', 'dates'):
        if k in d.files:
            dates = pd.to_datetime(np.asarray(d[k], dtype=str))
            break
    return r, dates


# Arc class -> (npz, stream_key, freq_per_year, alignment_strategy)
# alignment_strategy: 'daily' (align to DCA daily) or 'block20' (align to
# 20-trading-day blocks, with DCA block-aggregated to match).
ARCS = [
    ('dca-canonical', 'dca-returns.npz', 'daily_ret', 252.0, 'daily'),
    ('dca-basket-search-winner-4etf', 'dca-winner-4etf-returns.npz',
     'daily_ret', 252.0, 'daily'),
    ('relational-analog', 'relational-returns.npz', 'val_daily_ret',
     252.0, 'daily-tail'),
    ('gate-v0', 'gate-returns.npz', 'gated_ret', 252.0, 'daily-tail'),
    ('pairs-v0', 'pairs-returns.npz', 'agg_ret', 252.0, 'daily-tail'),
    ('vol-v3-dolthub-c0', 'vol-v3-dolthub-oos-returns.npz',
     'full_panel_alpha', 12.6, 'block20-dates'),
    ('vol-v3-dolthub-c200', 'vol-v3-dolthub-oos-c200-returns.npz',
     'full_panel_alpha', 12.6, 'block20-dates'),
    ('momentum-12-1-LS', 'momentum-12-1-returns.npz', 'ls_block_returns',
     12.0, 'block-tail'),
    ('low-vol-bab-LS', 'low-vol-bab-returns.npz', 'ls_block_returns',
     12.0, 'block-tail'),
    ('lie-shape-knn-LS-phase2', 'lie-shape-knn-returns.npz',
     'ls_block_returns', 12.0, 'block-tail'),
    ('lie-shape-knn-LS-wide', 'lie-shape-knn-wide-returns.npz',
     'ls_block_returns', 12.0, 'block-tail'),
    ('factor-LO-baseline', 'walkforward-linear-s200-wd0.001-windows.npz',
     'oos_block_returns', 12.6, 'block-tail'),
    ('factor-LS-baseline', 'walkforward-linear-s200-wd0.001-windows.npz',
     'oos_block_returns_long_short', 12.6, 'block-tail'),
    ('factor-5d-LO-skip1', 'sh-indicator-r5-s1-windows.npz',
     'oos_block_returns', 50.4, 'block-tail'),
    ('factor-5d-LS-skip1', 'sh-indicator-r5-s1-windows.npz',
     'oos_block_returns_long_short', 50.4, 'block-tail'),
]


def compare_to_dca(arc_key: str, npz: str, stream_key: str,
                   freq_per_year: float, alignment: str,
                   dca_daily: pd.Series) -> dict:
    r, dates = _read_arc_stream(npz, stream_key)
    if r.size == 0:
        return {'arc': arc_key, 'status': 'skip',
                'reason': f'{npz} missing or lacks {stream_key}'}

    aligned_a, aligned_b, n_overlap, caveat = None, None, 0, None

    if alignment == 'daily':
        # Daily-frequency arc; truncate to DCA index intersection.
        # We don't have arc dates here, so assume arc's daily series
        # ends at the same date as DCA. Tail-align (matches dca-canonical
        # which is just DCA itself — perfect overlap).
        n_overlap = min(r.size, dca_daily.size)
        aligned_a = r[-n_overlap:]
        aligned_b = dca_daily.values[-n_overlap:]
        caveat = 'tail-aligned (arc NPZ lacks explicit dates)' if arc_key != 'dca-canonical' else None

    elif alignment == 'daily-tail':
        # Daily arc; tail-align to DCA over the arc's length.
        n_overlap = min(r.size, dca_daily.size)
        aligned_a = r[-n_overlap:]
        aligned_b = dca_daily.values[-n_overlap:]
        caveat = 'tail-aligned (arc NPZ lacks explicit dates)'

    elif alignment == 'block20-dates':
        # Block arc with explicit dates — block-aggregate DCA on those
        # dates exactly. This is the gold-standard alignment.
        if dates is None or len(dates) != r.size:
            return {'arc': arc_key, 'status': 'skip',
                    'reason': 'block20-dates needs explicit rebal_dates'}
        dca_blocks = _block_aggregate_daily(dca_daily, dates, block_size=20)
        valid = ~np.isnan(dca_blocks)
        n_overlap = int(valid.sum())
        aligned_a = r[valid]
        aligned_b = dca_blocks[valid]
        caveat = None

    elif alignment == 'block-tail':
        # Block arc without dates. Tail-align DCA-block to the arc's
        # n_obs, assuming the arc ends at roughly the same date as DCA.
        # Block size inferred from freq_per_year (252/freq).
        block_size = max(1, int(round(252.0 / freq_per_year)))
        # Build DCA block-returns at this cadence from the tail of DCA.
        # Take the last (block_size * r.size) daily bars, group into
        # blocks, compound.
        need_bars = block_size * r.size
        if dca_daily.size < need_bars:
            need_bars = dca_daily.size - (dca_daily.size % block_size)
        tail = dca_daily.iloc[-need_bars:]
        # Reshape into (n_blocks, block_size) and compound
        n_full_blocks = tail.size // block_size
        tail = tail.iloc[-n_full_blocks * block_size:]
        blocks = tail.values.reshape(n_full_blocks, block_size)
        dca_blocks = (1.0 + blocks).prod(axis=1) - 1.0
        n_overlap = min(r.size, dca_blocks.size)
        aligned_a = r[-n_overlap:]
        aligned_b = dca_blocks[-n_overlap:]
        caveat = (f'tail-aligned (arc NPZ lacks dates; DCA aggregated to '
                  f'{block_size}-day blocks)')
    else:
        return {'arc': arc_key, 'status': 'skip',
                'reason': f'unknown alignment {alignment}'}

    if n_overlap < 10:
        return {'arc': arc_key, 'status': 'skip',
                'reason': f'overlap n={n_overlap} too small (<10)'}

    res = sharpe_difference_ci(aligned_a, aligned_b,
                               n_bootstraps=2000, confidence=0.95,
                               seed=42)
    # Annualize SR diff by sqrt(freq_per_year) for cross-arc readability
    ann_factor = math.sqrt(freq_per_year)
    return {
        'arc': arc_key,
        'status': 'ok',
        'npz': npz, 'stream_key': stream_key,
        'freq_per_year': freq_per_year,
        'alignment': alignment,
        'n_overlap': n_overlap,
        'sr_arc_pp': res.sr_a,
        'sr_dca_pp': res.sr_b,
        'delta_sr_pp': res.delta_sr,
        'delta_sr_ann': res.delta_sr * ann_factor,
        'se_delta_sr_pp': res.se_delta_sr,
        'ci_lo_ann': res.ci_lo * ann_factor,
        'ci_hi_ann': res.ci_hi * ann_factor,
        'block_length': res.block_length,
        'includes_zero': bool(res.includes_zero),
        'caveat': caveat,
    }


def main() -> None:
    print('Loading DCA daily reference stream...')
    dca_daily = _load_dca_daily()
    print(f'  DCA: {dca_daily.size} daily bars '
          f'{dca_daily.index[0].date()} → {dca_daily.index[-1].date()}')
    print()
    print(f'{"arc":34s} {"n":>6} {"ΔSR ann":>9} {"CI lo":>8} {"CI hi":>8} '
          f'{"≠0?":>5} note')
    print('-' * 95)

    results = []
    for arc_key, npz, stream_key, freq, align in ARCS:
        r = compare_to_dca(arc_key, npz, stream_key, freq, align, dca_daily)
        results.append(r)
        if r['status'] == 'skip':
            print(f'{arc_key:34s} skip: {r["reason"]}')
            continue
        flag = 'YES' if not r['includes_zero'] else 'no'
        note = r.get('caveat') or ''
        print(f'{arc_key:34s} {r["n_overlap"]:>6d} '
              f'{r["delta_sr_ann"]:>+9.3f} '
              f'{r["ci_lo_ann"]:>+8.3f} {r["ci_hi_ann"]:>+8.3f} '
              f'{flag:>5} {note[:32]}')

    out = OUTPUT / 'sharpe-diff-vs-dca.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
