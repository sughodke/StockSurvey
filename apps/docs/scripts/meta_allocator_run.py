"""Meta-allocator regime-forecasting walk-forward.

Pre-registered in `apps/docs/docs/TODO/meta-allocator-regime-forecasting.md`.
Tests whether any causal allocator across the 6 strategy arcs (dca, gate,
pairs, relational, dca_winner_4etf, vol_v3) beats the triple benchmark
(persistence / 1-over-N / inverse-arc-vol) on a locked walk-forward.

Output: `Output/meta-allocator-results.json` + per-candidate per-fold
return streams.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'apps' / 'docs' / 'scripts'))
OUTPUT = REPO_ROOT / 'Output'

from ss_portfolio import sharpe_difference_ci
from ss_portfolio.deflated import standardize_oos

REBAL_TD = 20
COMMISSION_BPS = 10.0
TRADING_DAYS = 252.0

# Locked OOS folds
FOLDS = [
    ('fold1', pd.Timestamp('2015-01-01'), pd.Timestamp('2018-12-31')),
    ('fold2', pd.Timestamp('2019-01-01'), pd.Timestamp('2022-12-31')),
    ('fold3', pd.Timestamp('2023-01-01'), pd.Timestamp('2025-12-11')),
]

ARC_COLS = ['dca', 'gate', 'pairs', 'relational', 'dca_winner_4etf', 'vol_v3']


# ---------------------------------------------------------------------------
# Data loading — reuse build_master
# ---------------------------------------------------------------------------

def load_arc_panel() -> pd.DataFrame:
    from count_regimes_since_2005 import build_master, load_dca_daily
    dca = load_dca_daily()
    df = build_master(dca)
    # Standardize column order
    return df[ARC_COLS]


def load_macro_panel(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    from ss_macro.loaders import load_macro_panel
    cache_dir = REPO_ROOT / '.macro-cache'
    panel = load_macro_panel(target_index=target_index, cache_dir=cache_dir)
    # Just the canonical 6-stack
    keep = ['fed_funds', 'slope_10y_3m', 'credit_baa', 'm2_yoy',
            'real_yield_10y', 'vix']
    keep = [k for k in keep if k in panel.columns]
    return panel[keep]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rebal_dates(arc_df: pd.DataFrame, cadence: int = REBAL_TD) -> pd.DatetimeIndex:
    """Trading-day rebal grid (every cadence trading days)."""
    return arc_df.index[::cadence]


def arc_available_mask(arc_df: pd.DataFrame, dt: pd.Timestamp,
                       min_history: int = 252) -> np.ndarray:
    """For each arc, is it available at rebal date dt? Available if it has
    >= min_history obs strictly before dt."""
    avail = np.zeros(arc_df.shape[1], dtype=bool)
    hist = arc_df.loc[arc_df.index < dt]
    for j, c in enumerate(arc_df.columns):
        avail[j] = hist[c].dropna().size >= min_history
    return avail


def trailing_sharpe(returns: np.ndarray, L: int) -> float:
    """Annualized Sharpe over last L obs (NaN-aware)."""
    a = returns[~np.isnan(returns)]
    if a.size < int(0.7 * L):
        return np.nan
    a = a[-L:]
    if a.std(ddof=1) < 1e-12:
        return np.nan
    return float(a.mean() / a.std(ddof=1) * math.sqrt(TRADING_DAYS))


def trailing_vol(returns: np.ndarray, L: int) -> float:
    a = returns[~np.isnan(returns)]
    if a.size < int(0.7 * L):
        return np.nan
    return float(a[-L:].std(ddof=1))


def realize_block(arc_df: pd.DataFrame, dt: pd.Timestamp, weights: np.ndarray,
                  cadence: int = REBAL_TD,
                  prev_weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Realize a block of `cadence` trading days starting at the bar AFTER dt.
    Returns (block_return_net_of_costs, daily_return_stream)."""
    pos = arc_df.index.searchsorted(dt, side='right')
    if pos + cadence > len(arc_df):
        return np.nan, np.array([])
    block = arc_df.iloc[pos: pos + cadence].to_numpy()  # (cadence, n_arcs)
    # Treat NaN as zero contribution; renormalize weights over available arcs
    # at this rebal date
    avail = ~np.isnan(block).all(axis=0)  # arc available across the full block
    if not avail.any():
        return np.nan, np.array([])
    # Renormalize over available arcs
    w = np.where(avail, weights, 0.0)
    s = w.sum()
    if s < 1e-12:
        w = np.zeros_like(weights)
    else:
        w = w / s
    # Fill NaN daily returns with 0 for unavailable arcs (allocation
    # already zeroed those out)
    block_filled = np.where(np.isnan(block), 0.0, block)
    daily = block_filled @ w  # (cadence,)
    # Switching cost: paid at start of block on |Δw|/2 × bps
    if prev_weights is None:
        cost = w.sum() * COMMISSION_BPS / 1e4  # initial entry cost
    else:
        cost = 0.5 * np.abs(w - prev_weights).sum() * COMMISSION_BPS / 1e4
    # Subtract cost from first day's return
    if daily.size > 0:
        daily = daily.copy()
        daily[0] -= cost
    return float((1.0 + daily).prod() - 1.0), daily, w


# ---------------------------------------------------------------------------
# Candidate allocators
# ---------------------------------------------------------------------------

def b1_persistence(arc_hist: pd.DataFrame, avail: np.ndarray, L: int) -> np.ndarray:
    """Pick the available arc with the highest trailing-L Sharpe. 100% allocation."""
    n = arc_hist.shape[1]
    w = np.zeros(n)
    best_i = -1; best_s = -np.inf
    for j in range(n):
        if not avail[j]:
            continue
        s = trailing_sharpe(arc_hist.iloc[:, j].to_numpy(), L)
        if not np.isnan(s) and s > best_s:
            best_s = s; best_i = j
    if best_i >= 0:
        w[best_i] = 1.0
    else:
        # Fallback: equal weight on available
        w[avail] = 1.0 / max(avail.sum(), 1)
    return w


def b2_equal_weight(avail: np.ndarray) -> np.ndarray:
    w = np.zeros_like(avail, dtype=float)
    if avail.any():
        w[avail] = 1.0 / avail.sum()
    return w


def b3_inverse_vol(arc_hist: pd.DataFrame, avail: np.ndarray, L: int) -> np.ndarray:
    n = arc_hist.shape[1]
    inv_vol = np.zeros(n)
    for j in range(n):
        if not avail[j]:
            continue
        v = trailing_vol(arc_hist.iloc[:, j].to_numpy(), L)
        if not np.isnan(v) and v > 1e-12:
            inv_vol[j] = 1.0 / v
    if inv_vol.sum() < 1e-12:
        return b2_equal_weight(avail)
    return inv_vol / inv_vol.sum()


def c1_markov(regime_sequence: list[str], arc_names: list[str],
              current_winner: str, avail: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Laplace-smoothed transition matrix. regime_sequence is list of
    arc-name winners observed strictly before the rebal date. Predict the
    next-winner distribution conditional on `current_winner`, then mask to
    available arcs and renormalize."""
    n = len(arc_names)
    idx = {a: i for i, a in enumerate(arc_names)}
    counts = np.full((n, n), alpha, dtype=float)
    for prev, nxt in zip(regime_sequence[:-1], regime_sequence[1:]):
        if prev in idx and nxt in idx:
            counts[idx[prev], idx[nxt]] += 1.0
    row_sum = counts.sum(axis=1, keepdims=True)
    P = counts / row_sum
    if current_winner not in idx:
        # Use stationary distribution
        try:
            eigvals, eigvecs = np.linalg.eig(P.T)
            stat = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
            stat = np.abs(stat); stat /= stat.sum()
            row = stat
        except Exception:
            row = np.ones(n) / n
    else:
        row = P[idx[current_winner]]
    # Mask & renormalize on availability
    row = np.where(avail, row, 0.0)
    if row.sum() < 1e-12:
        return b2_equal_weight(avail)
    return row / row.sum()


def turbulence_score(macro_hist: pd.DataFrame, dt: pd.Timestamp,
                     fit_window: int = 504) -> float:
    """K=2 turbulence proxy: Mahalanobis distance of current macro vector
    from the in-sample mean, using shrunk covariance. Mapped to [0,1] via
    rank in the trailing fit_window history (proportion of past obs with
    lower distance). High = turbulent.

    Implemented without hmmlearn — equivalent K=2 Gaussian-mixture
    semantics with a hard turbulence-quantile cutoff. Per literature brief,
    the published gain is in the "switch to cash" tail, which a Mahalanobis
    proxy captures."""
    hist = macro_hist.loc[macro_hist.index < dt].dropna()
    if hist.shape[0] < 60:
        return 0.0
    hist = hist.iloc[-fit_window:] if hist.shape[0] > fit_window else hist
    cur = hist.iloc[-1].to_numpy()
    X = hist.to_numpy()
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # Diagonal shrinkage
    cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov))
    cov += 1e-6 * np.eye(cov.shape[0])
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    d_cur = float(np.sqrt((cur - mu) @ inv @ (cur - mu)))
    # Rank in past
    diffs = X - mu
    d_all = np.sqrt(np.einsum('ij,jk,ik->i', diffs, inv, diffs))
    return float((d_all < d_cur).mean())


def c2_turbulence_overlay(b2_weights: np.ndarray, p_turb: float) -> np.ndarray:
    """Scale B2 weights by (1 - P_turbulent). Excess goes to cash (return 0)."""
    return b2_weights * (1.0 - p_turb)


def c3_meta_label(arc_df: pd.DataFrame, macro_df: pd.DataFrame,
                  dt: pd.Timestamp, avail: np.ndarray,
                  H: int = 63, arc_full_sharpes: dict[str, float] | None = None
                  ) -> np.ndarray:
    """6 logistic regressions. Label y_i,t = 1 if arc i's realized forward-H
    Sharpe (from t) exceeds arc's full-sample-up-to-fit Sharpe. Features:
    macro 6-stack + arc's trailing 63d Sharpe + trailing 252d Sharpe +
    trailing 63d vol. Fit at each rebal date; predict P(y=1) for the
    current feature vector."""
    from sklearn.linear_model import LogisticRegression

    macro_hist = macro_df.loc[macro_df.index < dt].dropna()
    arc_hist = arc_df.loc[arc_df.index < dt]
    if macro_hist.shape[0] < 200 or arc_hist.shape[0] < 252:
        return b2_equal_weight(avail)

    n = arc_df.shape[1]
    probs = np.zeros(n)
    # Build training set: for each historical rebal-grid date, compute
    # label + features. Use 60-day spacing to reduce overlap.
    spacing = 30
    train_dates = arc_hist.index[252:-H:spacing]
    if len(train_dates) < 30:
        return b2_equal_weight(avail)

    macro_cur_row = macro_df.loc[macro_df.index < dt].iloc[-1].to_numpy()
    if np.isnan(macro_cur_row).any():
        return b2_equal_weight(avail)

    for j, arc_name in enumerate(arc_df.columns):
        if not avail[j]:
            continue
        arc_full = arc_df.iloc[:, j].loc[arc_df.index < dt].dropna()
        if arc_full.size < 252:
            continue
        full_sh = float(arc_full.mean() / arc_full.std(ddof=1) * math.sqrt(TRADING_DAYS))
        X = []; y = []
        for td in train_dates:
            arc_up = arc_df.iloc[:, j].loc[arc_df.index < td].dropna()
            if arc_up.size < 252:
                continue
            macro_row = macro_df.loc[macro_df.index < td]
            if macro_row.empty:
                continue
            mrow = macro_row.iloc[-1].to_numpy()
            if np.isnan(mrow).any():
                continue
            tr_63 = arc_up.iloc[-63:]
            tr_252 = arc_up.iloc[-252:]
            sh_63 = float(tr_63.mean() / tr_63.std(ddof=1) * math.sqrt(TRADING_DAYS)) if tr_63.std(ddof=1) > 1e-12 else 0.0
            sh_252 = float(tr_252.mean() / tr_252.std(ddof=1) * math.sqrt(TRADING_DAYS)) if tr_252.std(ddof=1) > 1e-12 else 0.0
            vol_63 = float(tr_63.std(ddof=1))
            feats = np.concatenate([mrow, [sh_63, sh_252, vol_63]])
            # Forward H label
            fwd = arc_df.iloc[:, j].loc[(arc_df.index >= td) & (arc_df.index < td + pd.Timedelta(days=int(H*1.6)))].dropna().iloc[:H]
            if fwd.size < int(H * 0.7) or fwd.std(ddof=1) < 1e-12:
                continue
            fwd_sh = float(fwd.mean() / fwd.std(ddof=1) * math.sqrt(TRADING_DAYS))
            X.append(feats); y.append(int(fwd_sh > full_sh))
        if len(y) < 30 or len(set(y)) < 2:
            continue
        X = np.asarray(X); y = np.asarray(y)
        # Standardize features
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-12] = 1.0
        Xs = (X - mu) / sd
        try:
            clf = LogisticRegression(C=1.0, max_iter=500, solver='liblinear')
            clf.fit(Xs, y)
        except Exception:
            continue
        # Current features
        tr_63 = arc_full.iloc[-63:]
        tr_252 = arc_full.iloc[-252:]
        sh_63 = float(tr_63.mean() / tr_63.std(ddof=1) * math.sqrt(TRADING_DAYS)) if tr_63.std(ddof=1) > 1e-12 else 0.0
        sh_252 = float(tr_252.mean() / tr_252.std(ddof=1) * math.sqrt(TRADING_DAYS)) if tr_252.std(ddof=1) > 1e-12 else 0.0
        vol_63 = float(tr_63.std(ddof=1))
        cur_feats = np.concatenate([macro_cur_row, [sh_63, sh_252, vol_63]])
        cur_feats_s = (cur_feats - mu) / sd
        try:
            probs[j] = float(clf.predict_proba(cur_feats_s.reshape(1, -1))[0, 1])
        except Exception:
            continue
    if probs.sum() < 1e-12:
        return b2_equal_weight(avail)
    return probs / probs.sum()


def c4_cusum(arc_hist: pd.DataFrame, avail: np.ndarray, L: int,
             current_alloc: int | None, h_thresh: float = 4.0) -> tuple[np.ndarray, int]:
    """CUSUM on trailing-21d Sharpe of currently-held arc. If a downward CP
    fires, re-evaluate persistence on the full board. Returns (weights, new_alloc_idx)."""
    n = arc_hist.shape[1]
    re_eval = current_alloc is None or not avail[current_alloc]
    if not re_eval and current_alloc is not None:
        # CUSUM on the trailing rolling-21d Sharpe of the held arc
        held = arc_hist.iloc[:, current_alloc].dropna()
        if held.size > 84:
            roll = held.rolling(21).mean() / held.rolling(21).std(ddof=1)
            r = roll.dropna().to_numpy()
            if r.size > 30:
                mu = r[:-30].mean() if r.size > 60 else r.mean()
                sd = r[:-30].std(ddof=1) if r.size > 60 else r.std(ddof=1)
                if sd < 1e-12:
                    sd = 1.0
                # One-sided downward CUSUM on the last 30 standardized obs
                z = -(r[-30:] - mu) / sd
                cs = 0.0; max_cs = 0.0
                for zi in z:
                    cs = max(0.0, cs + zi - 0.5)
                    if cs > max_cs:
                        max_cs = cs
                if max_cs > h_thresh:
                    re_eval = True
    if re_eval:
        w = b1_persistence(arc_hist, avail, L)
        new_alloc = int(np.argmax(w))
        return w, new_alloc
    w = np.zeros(n)
    w[current_alloc] = 1.0
    return w, current_alloc


# ---------------------------------------------------------------------------
# Regime sequence builder (causal) — used by C1 Markov
# ---------------------------------------------------------------------------

def build_regime_sequence(arc_df: pd.DataFrame, end_date: pd.Timestamp,
                          L: int = 252) -> list[str]:
    """Compute the daily rolling-Sharpe winner sequence for dates < end_date,
    then collapse to maximal contiguous runs."""
    hist = arc_df.loc[arc_df.index < end_date]
    if hist.shape[0] < L + 10:
        return []
    mn = hist.rolling(L, min_periods=int(L * 0.7)).mean()
    sd = hist.rolling(L, min_periods=int(L * 0.7)).std(ddof=1)
    sh = mn / sd
    sh = sh.where(sd > 1e-12)
    arr = sh.to_numpy()
    cols = hist.columns.tolist()
    winners: list[str] = []
    for t in range(arr.shape[0]):
        row = arr[t]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        idxs = np.flatnonzero(valid)
        i = idxs[np.argmax(row[valid])]
        winners.append(cols[i])
    if not winners:
        return []
    # Collapse to maximal runs
    seq = [winners[0]]
    for w in winners[1:]:
        if w != seq[-1]:
            seq.append(w)
    return seq


def current_winner(arc_df: pd.DataFrame, end_date: pd.Timestamp, L: int = 252) -> str | None:
    seq = build_regime_sequence(arc_df, end_date, L)
    return seq[-1] if seq else None


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------

def run_walkforward(arc_df: pd.DataFrame, macro_df: pd.DataFrame,
                    L: int = 252) -> dict:
    """Returns per-candidate daily return arrays + per-fold metadata."""
    rebals = rebal_dates(arc_df)

    # Filter to those that begin BEFORE end of fold3 and have room for a full block
    rebals = [d for d in rebals if d <= FOLDS[-1][2]]

    cands = ['B1_persist_L126', 'B1_persist_L252', 'B1_persist_L504',
             'B2_equal_weight', 'B3_inv_vol',
             'C1_markov', 'C2_turb_overlay', 'C3_meta_label',
             'C4_cusum', 'C5_combo']

    out_daily: dict[str, list[float]] = {c: [] for c in cands}
    out_dates: list[pd.Timestamp] = []

    prev_w: dict[str, np.ndarray] = {c: None for c in cands}
    c4_alloc: int | None = None

    n_arcs = arc_df.shape[1]

    # First fold-start
    fold1_start = FOLDS[0][1]

    for i, dt in enumerate(rebals):
        if dt < fold1_start:
            continue
        if i + 1 >= len(rebals):
            break
        # data up-to-but-not-including dt
        arc_hist = arc_df.loc[arc_df.index < dt]
        if arc_hist.shape[0] < 252:
            continue
        avail = arc_available_mask(arc_df, dt, min_history=252)
        if not avail.any():
            continue

        # B1 family (3 lookback choices)
        w_b1_126 = b1_persistence(arc_hist, avail, 126)
        w_b1_252 = b1_persistence(arc_hist, avail, 252)
        w_b1_504 = b1_persistence(arc_hist, avail, 504)
        w_b2 = b2_equal_weight(avail)
        w_b3 = b3_inverse_vol(arc_hist, avail, L)

        # C1 Markov
        seq = build_regime_sequence(arc_df, dt, L)
        cur_w = seq[-1] if seq else None
        if cur_w is not None:
            w_c1 = c1_markov(seq, list(arc_df.columns), cur_w, avail)
        else:
            w_c1 = w_b2.copy()

        # C2 turbulence overlay on B2
        p_turb = turbulence_score(macro_df, dt)
        w_c2 = c2_turbulence_overlay(w_b2, p_turb)

        # C3 meta-label
        try:
            w_c3 = c3_meta_label(arc_df, macro_df, dt, avail, H=63)
        except Exception as e:
            print(f'  C3 fallback at {dt.date()}: {e}')
            w_c3 = w_b2.copy()

        # C4 CUSUM
        w_c4, c4_alloc = c4_cusum(arc_hist, avail, L, c4_alloc)

        # C5 combo: mean of C1, C2, C3
        w_c5 = (w_c1 + w_c2 + w_c3) / 3.0
        # Renormalize over availability
        w_c5 = np.where(avail, w_c5, 0.0)
        if w_c5.sum() > 1e-12:
            w_c5 = w_c5 / w_c5.sum()
        else:
            w_c5 = w_b2.copy()

        candidates = {
            'B1_persist_L126': w_b1_126,
            'B1_persist_L252': w_b1_252,
            'B1_persist_L504': w_b1_504,
            'B2_equal_weight': w_b2,
            'B3_inv_vol': w_b3,
            'C1_markov': w_c1,
            'C2_turb_overlay': w_c2,
            'C3_meta_label': w_c3,
            'C4_cusum': w_c4,
            'C5_combo': w_c5,
        }

        # Realize the block for each candidate
        for name, w in candidates.items():
            res = realize_block(arc_df, dt, w, cadence=REBAL_TD,
                                prev_weights=prev_w[name])
            if res[0] != res[0]:  # NaN
                continue
            block_ret, daily, w_used = res
            out_daily[name].extend(daily.tolist())
            prev_w[name] = w_used
        # Track block dates for all candidates simultaneously (they share calendar)
        pos = arc_df.index.searchsorted(dt, side='right')
        block_dates = arc_df.index[pos: pos + REBAL_TD]
        out_dates.extend(block_dates.tolist())

    # Truncate each to common min length (in case any candidate failed
    # at edge cases). Actually, since we extend all simultaneously and skip
    # in-tandem, they should all have len == n_blocks_processed * REBAL_TD.
    # We just collect a master dates list of equal length.
    n_min = min(len(v) for v in out_daily.values()) if out_daily else 0
    n_min = min(n_min, len(out_dates))
    for k in out_daily:
        out_daily[k] = out_daily[k][:n_min]
    out_dates = out_dates[:n_min]

    return {
        'daily': {k: np.asarray(v, dtype=np.float64) for k, v in out_daily.items()},
        'dates': pd.DatetimeIndex(out_dates),
        'lookback_L': L,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_candidate(daily: np.ndarray, dates: pd.DatetimeIndex, name: str,
                   benchmark_daily: dict[str, np.ndarray], n_trials: int) -> dict:
    """Pooled OOS Sharpe + CIs vs each benchmark + DSR."""
    rec: dict = {'candidate': name, 'n_obs': int(daily.size)}
    if daily.size < 30:
        rec['status'] = 'too-short'
        return rec
    ann = math.sqrt(TRADING_DAYS)
    sr_pp = float(daily.mean() / daily.std(ddof=1)) if daily.std(ddof=1) > 1e-12 else 0.0
    rec['sharpe_ann'] = sr_pp * ann
    # DSR via standardize_oos with n_trials
    try:
        mb = standardize_oos(daily, periods_per_year=TRADING_DAYS, n_trials=n_trials)
        rec['dsr'] = float(mb.dsr)
        rec['dsr_tstat'] = float(mb.deflated_tstat)
        rec['psr'] = float(mb.psr)
    except Exception as e:
        rec['dsr_error'] = str(e)
    # Per-benchmark deltas
    for bname, bdaily in benchmark_daily.items():
        if bname == name:
            continue
        # Align length
        n = min(daily.size, bdaily.size)
        a = daily[-n:]; b = bdaily[-n:]
        try:
            ci = sharpe_difference_ci(a, b, n_bootstraps=2000,
                                      confidence=0.95, seed=42)
            rec[f'vs_{bname}'] = {
                'n': int(n),
                'delta_sr_ann': ci.delta_sr * ann,
                'ci_lo_ann': ci.ci_lo * ann,
                'ci_hi_ann': ci.ci_hi * ann,
                'includes_zero': bool(ci.includes_zero),
            }
        except Exception as e:
            rec[f'vs_{bname}'] = {'error': str(e)}
    return rec


def classify_verdict(rec: dict, bname: str) -> str:
    v = rec.get(f'vs_{bname}')
    if not v or 'error' in v:
        return 'skip'
    d = v['delta_sr_ann']
    inc0 = v['includes_zero']
    dsr_t = rec.get('dsr_tstat', np.nan)
    if (not inc0) and d > 0 and dsr_t > 3.0 and d >= 0.3:
        return 'confirmed-OOS'
    if d >= 0.15 and dsr_t > 1.5:
        return 'partial-OOS'
    if inc0 and abs(d) < 0.05:
        return 'confirmed-null'
    if d < 0:
        return 'reversed-OOS' if (not inc0) else 'confirmed-null'
    return 'partial-OOS'


def main() -> None:
    print('=== Meta-allocator regime-forecasting walk-forward ===')
    print('Loading arc panel + macro panel...')
    arc_df = load_arc_panel()
    print(f'  arc panel: {arc_df.shape[0]} rows × {arc_df.shape[1]} arcs')
    print(f'  span: {arc_df.index[0].date()} → {arc_df.index[-1].date()}')
    for c in arc_df.columns:
        s = arc_df[c].dropna()
        print(f'    {c:20s} n={s.size:>5d} '
              f'{s.index[0].date() if s.size else "—"} → '
              f'{s.index[-1].date() if s.size else "—"}')

    macro_df = load_macro_panel(target_index=arc_df.index)
    print(f'  macro: {macro_df.shape[1]} features, {macro_df.dropna().shape[0]} fully-populated rows')

    print('\n--- Walk-forward with canonical L=252 ---')
    res = run_walkforward(arc_df, macro_df, L=252)
    n_obs = len(res['dates'])
    print(f'  pooled OOS obs: {n_obs} days '
          f'({res["dates"][0].date()} → {res["dates"][-1].date()})')

    # Locked deflation N = 8 (5 modeling + 3 L choices)
    N_TRIALS = 8

    # Build benchmark dict from B1_L252, B2, B3
    benchmarks = {
        'B1_persist_L252': res['daily']['B1_persist_L252'],
        'B2_equal_weight': res['daily']['B2_equal_weight'],
        'B3_inv_vol': res['daily']['B3_inv_vol'],
    }

    print(f'\n--- per-candidate eval (n_trials={N_TRIALS} for DSR) ---')
    rows = []
    for name, daily in res['daily'].items():
        rec = eval_candidate(daily, res['dates'], name, benchmarks, n_trials=N_TRIALS)
        rec['mean_daily_ret'] = float(np.mean(daily)) if daily.size else float('nan')
        rec['std_daily_ret'] = float(np.std(daily, ddof=1)) if daily.size > 1 else float('nan')
        # Verdicts vs each benchmark (skip benchmarks vs themselves)
        for bn in ['B1_persist_L252', 'B2_equal_weight', 'B3_inv_vol']:
            if bn == name:
                continue
            rec[f'verdict_vs_{bn}'] = classify_verdict(rec, bn)
        rows.append(rec)

        # Pretty print
        print(f'\n  {name}:')
        print(f'    n={rec["n_obs"]}  Sharpe_ann={rec.get("sharpe_ann", float("nan")):+.3f}  '
              f'DSR={rec.get("dsr", float("nan")):.3f}  DSR-t={rec.get("dsr_tstat", float("nan")):+.2f}')
        for bn in ['B1_persist_L252', 'B2_equal_weight', 'B3_inv_vol']:
            if bn == name:
                continue
            v = rec.get(f'vs_{bn}', {})
            if 'error' in v:
                print(f'    vs {bn}: ERROR {v["error"]}')
                continue
            print(f'    vs {bn}: ΔSR_ann {v["delta_sr_ann"]:+.3f} '
                  f'[{v["ci_lo_ann"]:+.3f}, {v["ci_hi_ann"]:+.3f}] '
                  f'incl_0={v["includes_zero"]}  '
                  f'verdict={rec.get(f"verdict_vs_{bn}")}')

    out = OUTPUT / 'meta-allocator-results.json'
    json_safe = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                rr[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                rr[k] = v
        json_safe.append(rr)
    out.write_text(json.dumps({
        'lookback_L': 252,
        'rebal_td': REBAL_TD,
        'commission_bps': COMMISSION_BPS,
        'n_trials_dsr': N_TRIALS,
        'pooled_oos_start': str(res['dates'][0].date()) if n_obs else None,
        'pooled_oos_end': str(res['dates'][-1].date()) if n_obs else None,
        'pooled_oos_n': n_obs,
        'arc_cols': list(arc_df.columns),
        'rows': json_safe,
    }, indent=2, default=str))
    print(f'\n→ {out}')

    # Also save the daily return streams as npz for follow-up analysis
    np.savez(OUTPUT / 'meta-allocator-daily-streams.npz',
             dates=np.array([str(d.date()) for d in res['dates']]),
             **res['daily'])
    print(f'→ {OUTPUT / "meta-allocator-daily-streams.npz"}')


if __name__ == '__main__':
    main()
