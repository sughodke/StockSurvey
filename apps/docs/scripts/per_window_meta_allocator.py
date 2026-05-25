"""Per-window meta-allocator hypothesis test.

Tests whether picking a *single best arc per window* (oracle or causal
selection rule) clears the `confirmed-OOS` bar (Ledoit-Wolf 95% CI on
ΔSR vs DCA excludes 0) that no individual arc on the ladder achieves.

Hypotheses:
  H1 — Oracle: pick best per-block realized return (upper bound, peeks at future).
  H2 — Lagged-Sharpe: pick arc with highest trailing K-block Sharpe.
  H3 — VIX regime gate: pick between vol-v3 (fired-VIX) and DCA (calm).
  H4 — Markowitz: trailing-covariance one-period MV optimal weights.

Per-window grid: 20-trading-day blocks aligned to DCA's daily date index.

NOT for commit. Research probe only.
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

BLOCK = 20  # trading days per window-block


def load_dca_daily() -> pd.Series:
    import pickle
    with open(REPO_ROOT / 'Output/cfr_phase4d_multiasset_close.pkl', 'rb') as f:
        close = pickle.load(f)
    from cfr.baselines import PassiveEW
    daily = np.asarray(
        PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(close),
        dtype=np.float64)
    return pd.Series(daily, index=close.index, dtype=np.float64).dropna()


def block_aggregate_daily(daily: pd.Series, block_starts: pd.DatetimeIndex,
                          block_size: int = BLOCK) -> np.ndarray:
    """Compound block_size daily returns starting at each block_start."""
    out = []
    for d in block_starts:
        pos = daily.index.searchsorted(d, side='left')
        if pos + block_size > len(daily):
            out.append(np.nan); continue
        win = daily.iloc[pos: pos + block_size]
        if win.size < block_size * 0.7:
            out.append(np.nan)
        else:
            out.append((1.0 + win).prod() - 1.0)
    return np.asarray(out, dtype=np.float64)


def make_block_grid(daily: pd.Series, block_size: int = BLOCK) -> pd.DatetimeIndex:
    """Build a contiguous non-overlapping block-start grid over DCA's daily
    index. Each block_starts[i] is the date of the first daily bar in block i."""
    n_blocks = len(daily) // block_size
    starts = [daily.index[i * block_size] for i in range(n_blocks)]
    return pd.DatetimeIndex(starts)


def block_aggregate_arc_daily(arc_daily: pd.Series,
                              block_starts: pd.DatetimeIndex,
                              block_size: int = BLOCK) -> np.ndarray:
    """Aggregate a daily arc stream onto the block grid (NaN where the arc
    isn't yet active or has insufficient coverage in the block)."""
    out = []
    for d in block_starts:
        pos = arc_daily.index.searchsorted(d, side='left')
        if pos >= len(arc_daily) or pos + block_size > len(arc_daily):
            out.append(np.nan); continue
        win = arc_daily.iloc[pos: pos + block_size]
        if win.size < block_size * 0.7:
            out.append(np.nan)
        else:
            out.append((1.0 + win).prod() - 1.0)
    return np.asarray(out, dtype=np.float64)


# -----------------------------------------------------------------------------
# Arc loaders → block-grid-aligned return arrays (NaN where arc is inactive)
# -----------------------------------------------------------------------------

def load_arcs_on_block_grid(dca_daily: pd.Series,
                            block_starts: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    arcs: dict[str, np.ndarray] = {}

    # DCA baseline (block-aggregated from daily)
    arcs['dca'] = block_aggregate_daily(dca_daily, block_starts, BLOCK)

    # gate-v0 (has dates, daily). Tail-aligned via its own dates.
    d = np.load(OUTPUT / 'gate-returns.npz', allow_pickle=True)
    gate_dates = pd.to_datetime(np.asarray(d['dates'], dtype=str))
    gate_daily = pd.Series(np.asarray(d['gated_ret'], dtype=np.float64),
                           index=gate_dates).dropna()
    arcs['gate'] = block_aggregate_arc_daily(gate_daily, block_starts, BLOCK)

    # gate benchmark (unc_ret) — same dates
    gate_unc = pd.Series(np.asarray(d['unc_ret'], dtype=np.float64),
                         index=gate_dates).dropna()
    arcs['gate_unc'] = block_aggregate_arc_daily(gate_unc, block_starts, BLOCK)

    # relational-analog: 1241 daily obs, no dates → tail-align to DCA end
    d = np.load(OUTPUT / 'relational-returns.npz', allow_pickle=True)
    rel_arr = np.asarray(d['val_daily_ret'], dtype=np.float64)
    # tail-align to the last len(rel_arr) bars of DCA
    rel_idx = dca_daily.index[-rel_arr.size:]
    rel_daily = pd.Series(rel_arr, index=rel_idx)
    arcs['relational'] = block_aggregate_arc_daily(rel_daily, block_starts, BLOCK)

    # vol-v3-dolthub c200: 33 obs, ~monthly rebals (28 trading days), HAS dates.
    # Map each rebal_date to the nearest block_start (forward).
    d = np.load(OUTPUT / 'vol-v3-dolthub-oos-c200-returns.npz', allow_pickle=True)
    vol_dates = pd.to_datetime(np.asarray(d['rebal_dates'], dtype=str))
    vol_alpha = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    # Build a sparse series, then for each block_start in the vol date range,
    # take the rebal whose date falls inside the block.
    vol_stream = np.full(len(block_starts), np.nan)
    for vd, va in zip(vol_dates, vol_alpha):
        # find the block_start that this rebal falls into (i.e., block_starts[i] <= vd < block_starts[i+1])
        pos = int(block_starts.searchsorted(vd, side='right')) - 1
        if 0 <= pos < len(block_starts):
            # NOTE: vol is an *overlay alpha* on DCA-like exposure; here we treat
            # it as a standalone block return for the meta-allocator selection
            # (the meta-allocator picks WHICH overlay to deploy that block,
            # versus DCA which is full exposure).
            # To make it commensurate with DCA block-return, we treat vol's
            # alpha as the overlay's net return — additive. The meta-allocator
            # framing: "this block, deploy DCA alone vs deploy DCA+vol overlay"
            # collapses to the same selection: when vol > 0, prefer DCA+vol.
            # For oracle competition purposes, we use vol's panel alpha as
            # the standalone block-return PROXY (small absolute scale).
            vol_stream[pos] = va
    arcs['vol_v3'] = vol_stream

    # factor-5d-LO-skip1: 936 obs at 5d cadence = 187 weeks ≈ 3.6 years
    # Aggregate 4 of those 5d blocks → 1 of our 20d blocks. Tail-align.
    d = np.load(OUTPUT / 'sh-indicator-r5-s1-windows.npz', allow_pickle=True)
    fac_r5 = np.asarray(d['oos_block_returns'], dtype=np.float64)
    # 4 consecutive 5d blocks → 1 of 20d block, compound
    n_groups = fac_r5.size // 4
    fac_r5 = fac_r5[-n_groups * 4:]
    fac_blocks = (1.0 + fac_r5.reshape(n_groups, 4)).prod(axis=1) - 1.0
    fac_stream = np.full(len(block_starts), np.nan)
    # Tail-align: last n_groups block_starts
    if n_groups <= len(block_starts):
        fac_stream[-n_groups:] = fac_blocks
    arcs['factor_5d_LO'] = fac_stream

    return arcs


# -----------------------------------------------------------------------------
# Meta-allocator strategies
# -----------------------------------------------------------------------------

def h1_oracle(arc_matrix: np.ndarray) -> np.ndarray:
    """Per-block, pick the arc with the highest realized return that block.
    arc_matrix: (n_blocks, n_arcs). Returns: (n_blocks,) — block returns of winner.
    NaN entries are ignored in the argmax.
    """
    n_blocks = arc_matrix.shape[0]
    out = np.full(n_blocks, np.nan)
    for t in range(n_blocks):
        row = arc_matrix[t]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        i = np.flatnonzero(valid)[np.argmax(row[valid])]
        out[t] = row[i]
    return out


def trailing_sharpe(stream: np.ndarray, k: int) -> np.ndarray:
    """Trailing K-block Sharpe at each block (uses [t-k, t-1])."""
    n = len(stream)
    out = np.full(n, np.nan)
    for t in range(k, n):
        win = stream[t - k: t]
        m = np.nanmean(win); s = np.nanstd(win, ddof=1)
        if s > 1e-12 and not np.isnan(m):
            out[t] = m / s
    return out


def h2_lagged_sharpe(arc_matrix: np.ndarray, k: int) -> np.ndarray:
    """Pick arc with highest trailing-K Sharpe (causal — uses blocks [t-k, t-1])."""
    n_blocks, n_arcs = arc_matrix.shape
    tr = np.column_stack([trailing_sharpe(arc_matrix[:, i], k) for i in range(n_arcs)])
    out = np.full(n_blocks, np.nan)
    for t in range(n_blocks):
        row_tr = tr[t]
        row_ret = arc_matrix[t]
        valid = ~np.isnan(row_tr) & ~np.isnan(row_ret)
        if not valid.any():
            continue
        i = np.flatnonzero(valid)[np.argmax(row_tr[valid])]
        out[t] = row_ret[i]
    return out


def h3_vix_regime(dca_block: np.ndarray, vol_block: np.ndarray,
                  block_starts: pd.DatetimeIndex,
                  vix_lookback_days: int = 126) -> np.ndarray:
    """Pick vol_v3 in fired-VIX blocks, DCA in calm blocks.
    Fired = VIX at block_start > 126-day rolling median.
    """
    from ss_macro.loaders import load_fred_series
    try:
        vix = load_fred_series('VIXCLS')  # returns pd.Series with date index
    except Exception as e:
        print(f'  H3: failed to load VIX ({e}); using fallback (always DCA)')
        return dca_block
    # `vix` might be DataFrame or Series; coerce
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix = vix.dropna().astype(float)
    vix_med = vix.rolling(vix_lookback_days).median()
    n = len(dca_block)
    out = np.full(n, np.nan)
    for t, d in enumerate(block_starts):
        # VIX level at block start
        pos = vix.index.searchsorted(d, side='right') - 1
        if pos < vix_lookback_days:
            # not enough history → default DCA
            out[t] = dca_block[t]
            continue
        fired = vix.iloc[pos] > vix_med.iloc[pos]
        if fired and not np.isnan(vol_block[t]):
            out[t] = vol_block[t]
        elif not np.isnan(dca_block[t]):
            out[t] = dca_block[t]
    return out


def h4_markowitz(arc_matrix: np.ndarray, k: int = 12, shrink: float = 0.1) -> np.ndarray:
    """Trailing-K-block covariance + mean → MV optimal weights, applied this block.
    Long-only with simplex constraint via softmax of unconstrained MV weights.
    Skip blocks where any active arc has NaN trailing history.
    """
    n_blocks, n_arcs = arc_matrix.shape
    out = np.full(n_blocks, np.nan)
    for t in range(k, n_blocks):
        win = arc_matrix[t - k: t]  # (k, n_arcs)
        # require all arcs valid over the full window AND this block
        col_ok = (~np.isnan(win)).all(axis=0) & (~np.isnan(arc_matrix[t]))
        if col_ok.sum() < 2:
            # fallback: use whatever single arc is valid
            valid_now = ~np.isnan(arc_matrix[t])
            if valid_now.any():
                # pick highest trailing mean of valid
                tr = np.nanmean(win, axis=0)
                i_candidates = np.flatnonzero(valid_now & (~np.isnan(tr)))
                if i_candidates.size > 0:
                    i = i_candidates[np.argmax(tr[i_candidates])]
                    out[t] = arc_matrix[t, i]
            continue
        sub = win[:, col_ok]
        mu = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False)
        # shrinkage to diagonal
        cov_diag = np.diag(np.diag(cov))
        cov = (1 - shrink) * cov + shrink * cov_diag
        # ridge
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        try:
            w = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            continue
        # Long-only via clip + renorm (simpler than QP)
        w = np.maximum(w, 0)
        s = w.sum()
        if s < 1e-12:
            # fall back to equal weight on valid arcs
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        out[t] = float(np.dot(w, arc_matrix[t, col_ok]))
    return out


# -----------------------------------------------------------------------------
# CI computation
# -----------------------------------------------------------------------------

def ci_vs_dca(strat: np.ndarray, dca: np.ndarray, label: str,
              ppy: float = 252.0 / BLOCK) -> dict:
    valid = ~np.isnan(strat) & ~np.isnan(dca)
    a = strat[valid]; b = dca[valid]
    if a.size < 10:
        return {'label': label, 'n': int(a.size), 'status': 'skip'}
    res = sharpe_difference_ci(a, b, n_bootstraps=2000, confidence=0.95, seed=42)
    ann = math.sqrt(ppy)
    return {
        'label': label,
        'n': int(a.size),
        'sr_strat_pp': res.sr_a,
        'sr_dca_pp': res.sr_b,
        'delta_sr_pp': res.delta_sr,
        'delta_sr_ann': res.delta_sr * ann,
        'ci_lo_ann': res.ci_lo * ann,
        'ci_hi_ann': res.ci_hi * ann,
        'includes_zero': bool(res.includes_zero),
        'block_length': res.block_length,
    }


def main() -> None:
    print('Loading DCA daily reference...')
    dca_daily = load_dca_daily()
    print(f'  DCA: n={len(dca_daily)} {dca_daily.index[0].date()} → '
          f'{dca_daily.index[-1].date()}')

    block_starts = make_block_grid(dca_daily, BLOCK)
    print(f'  block grid: {len(block_starts)} blocks of {BLOCK} trading days '
          f'each ({block_starts[0].date()} → {block_starts[-1].date()})')

    arcs = load_arcs_on_block_grid(dca_daily, block_starts)

    # Overlap audit
    print('\n--- per-arc block coverage ---')
    for name, arr in arcs.items():
        valid = ~np.isnan(arr)
        print(f'  {name:18s}  n_blocks_valid={valid.sum():>4d} / {len(arr)}')

    # Stack into matrix (n_blocks, n_arcs).
    # IMPORTANT: only arcs WITH EXPLICIT DATES can be honestly compared
    # per-block. Tail-aligning a no-dates arc (relational, factor_5d_LO)
    # to DCA breaks per-block alignment — the arc's block-t return refers
    # to a different calendar block than DCA's block-t. Oracle / H1 picks
    # the per-block winner ONLY IF the returns refer to the same block.
    # So we use DCA + GATE + VOL_V3 as the date-aligned competitive set.
    # We separately report (with the no-dates arcs) a SECOND run that
    # shows what the inflation looks like — to make the issue visible.
    arc_names = ['dca', 'gate', 'vol_v3']
    arc_matrix = np.column_stack([arcs[n] for n in arc_names])
    print(f'\n=== DATE-ALIGNED COMPETITIVE SET: {arc_names} ===')

    # Per-arc individual baselines vs DCA
    print('\n--- individual-arc baselines vs DCA (block-aligned) ---')
    results = []
    for i, name in enumerate(arc_names):
        if name == 'dca': continue
        r = ci_vs_dca(arc_matrix[:, i], arc_matrix[:, 0], f'individual:{name}')
        results.append(r); print(f'  {r}')

    print('\n--- H1 ORACLE (max per-block return; future-peeking upper bound) ---')
    h1 = h1_oracle(arc_matrix)
    r_h1 = ci_vs_dca(h1, arc_matrix[:, 0], 'H1_oracle')
    results.append(r_h1); print(f'  {r_h1}')

    print('\n--- H2 lagged-Sharpe selection (causal, K=3 blocks) ---')
    h2_3 = h2_lagged_sharpe(arc_matrix, k=3)
    r_h2_3 = ci_vs_dca(h2_3, arc_matrix[:, 0], 'H2_lagsharpe_k3')
    results.append(r_h2_3); print(f'  {r_h2_3}')

    print('\n--- H2 lagged-Sharpe selection (causal, K=6 blocks) ---')
    h2_6 = h2_lagged_sharpe(arc_matrix, k=6)
    r_h2_6 = ci_vs_dca(h2_6, arc_matrix[:, 0], 'H2_lagsharpe_k6')
    results.append(r_h2_6); print(f'  {r_h2_6}')

    print('\n--- H3 VIX-regime gate (vol_v3 if fired, else DCA) ---')
    h3 = h3_vix_regime(arc_matrix[:, 0], arc_matrix[:, 2], block_starts)
    r_h3 = ci_vs_dca(h3, arc_matrix[:, 0], 'H3_vix_regime')
    results.append(r_h3); print(f'  {r_h3}')

    print('\n--- H4 Markowitz (trailing-12-block cov + mean, long-only) ---')
    h4 = h4_markowitz(arc_matrix, k=12, shrink=0.2)
    r_h4 = ci_vs_dca(h4, arc_matrix[:, 0], 'H4_markowitz_k12')
    results.append(r_h4); print(f'  {r_h4}')

    # ------------------------------------------------------------
    # Second pass: include the tail-aligned arcs (relational, factor)
    # to make the date-misalignment inflation visible.
    # ------------------------------------------------------------
    print('\n\n=== INCL. TAIL-ALIGNED ARCS (relational, factor_5d_LO) — '
          'INFLATED, see brief ===')
    arc_names_v2 = ['dca', 'gate', 'vol_v3', 'relational', 'factor_5d_LO']
    arc_matrix_v2 = np.column_stack([arcs[n] for n in arc_names_v2])
    h1_v2 = h1_oracle(arc_matrix_v2)
    r_h1_v2 = ci_vs_dca(h1_v2, arc_matrix_v2[:, 0], 'H1_oracle_with_tail_aligned')
    results.append(r_h1_v2); print(f'  {r_h1_v2}')
    h2_v2 = h2_lagged_sharpe(arc_matrix_v2, k=6)
    r_h2_v2 = ci_vs_dca(h2_v2, arc_matrix_v2[:, 0], 'H2_k6_with_tail_aligned')
    results.append(r_h2_v2); print(f'  {r_h2_v2}')

    out = OUTPUT / 'per-window-meta-allocator.json'
    out.write_text(json.dumps({
        'block_size_trading_days': BLOCK,
        'block_grid_start': str(block_starts[0].date()),
        'block_grid_end': str(block_starts[-1].date()),
        'arc_coverage': {n: int((~np.isnan(arcs[n])).sum()) for n in arcs},
        'arc_names_in_matrix': arc_names,
        'results': results,
    }, indent=2, default=str))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
