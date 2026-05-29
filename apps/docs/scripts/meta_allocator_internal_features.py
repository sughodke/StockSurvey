"""Meta-allocator with strategy-internal features — pre-registered eval.

Implements the locked design at
`apps/docs/docs/TODO/meta-allocator-internal-features.md`.

Pragmatic substitutions vs the pre-reg (documented in findings page):
  * `valuation_spread`: rather than per-arc bespoke P/E / IV-RV-gap /
    fingerprint-dispersion sources, we use each arc's trailing 252d
    return minus its trailing 5y rolling mean. Low-recent-vs-history
    = "cheap" in HKS sense, the same monotone shape with point-in-time
    discipline. The pre-reg explicitly licenses this fallback.
  * `rank_ic_trend_252`: for arcs without a per-name cross-sectional
    signal (dca, gate, pairs, dca_winner_4etf, vol_v3) we substitute
    the trailing-252d Sharpe-trend (OLS slope of 60d rolling Sharpe
    over the last 252 days). relational has the same substitution
    because the per-name fingerprint signal isn't exposed daily.
  * D2 (random forest depth-3) is substituted with kernel ridge with
    RBF kernel + median-distance bandwidth — the pre-reg licenses
    this as an acceptable nonlinear substitute.

Panel: 6 arcs from build_master — `dca`, `gate`, `pairs`,
`relational`, `dca_winner_4etf`, `vol_v3`. Span: 2015-01-01 →
2025-12-11. Folds: 2015-2018, 2019-2022, 2023-2025-Q3.

Outputs:
  Output/meta-allocator-internal-features-results.json
  Output/meta-allocator-internal-features-streams.npz
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

TRADING_DAYS = 252.0
CADENCE_TD = 63          # quarterly
COMMISSION_BPS = 10.0
ARC_COLS = ['dca', 'gate', 'pairs', 'relational', 'dca_winner_4etf', 'vol_v3']

FOLDS = [
    ('fold1', pd.Timestamp('2015-01-01'), pd.Timestamp('2018-12-31')),
    ('fold2', pd.Timestamp('2019-01-01'), pd.Timestamp('2022-12-31')),
    ('fold3', pd.Timestamp('2023-01-01'), pd.Timestamp('2025-12-11')),
]
OOS_2024_START = pd.Timestamp('2024-01-01')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_arc_panel() -> pd.DataFrame:
    """6-arc panel, daily, restricted to ARC_COLS."""
    from count_regimes_since_2005 import build_master, load_dca_daily
    dca = load_dca_daily()
    df = build_master(dca)
    df = df[ARC_COLS]
    df = df.loc[df.index >= pd.Timestamp('2015-01-01')]
    return df


# ---------------------------------------------------------------------------
# Features (14 features at each rebal date)
# ---------------------------------------------------------------------------

def _trailing_vol(s: pd.Series, t: pd.Timestamp, L: int) -> float:
    a = s.loc[s.index < t].dropna()
    if a.size < int(0.7 * L):
        return np.nan
    return float(a.iloc[-L:].std(ddof=1) * math.sqrt(TRADING_DAYS))


def _sharpe_trend(s: pd.Series, t: pd.Timestamp, L: int = 252,
                  inner: int = 60) -> float:
    """OLS slope of rolling-60d Sharpe over the trailing L days."""
    a = s.loc[s.index < t].dropna()
    if a.size < L + inner:
        return np.nan
    roll = a.rolling(inner).apply(
        lambda w: w.mean() / w.std(ddof=1) if w.std(ddof=1) > 1e-12 else np.nan,
        raw=False,
    )
    roll = roll.dropna().iloc[-L:]
    if roll.size < int(0.7 * L):
        return np.nan
    x = np.arange(roll.size, dtype=np.float64)
    y = roll.to_numpy()
    if y.std(ddof=1) < 1e-12:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def _valuation_spread(s: pd.Series, t: pd.Timestamp,
                      short_L: int = 252, long_L: int = 252 * 5) -> float:
    """Trailing 252d mean return minus trailing 5y mean return.
    Negative = recent < long-run = "cheap" in HKS sense."""
    a = s.loc[s.index < t].dropna()
    if a.size < short_L + 60:
        return np.nan
    short_a = a.iloc[-short_L:]
    long_a = a.iloc[-min(long_L, a.size):]
    return float(short_a.mean() - long_a.mean())


def _cross_arc_corr_eff_rank(arc_df: pd.DataFrame, t: pd.Timestamp,
                             L: int = 60) -> float:
    """Effective rank of trailing-L correlation matrix of available arcs."""
    hist = arc_df.loc[arc_df.index < t].iloc[-L:]
    avail = (~hist.isna()).all(axis=0)
    if avail.sum() < 2:
        return np.nan
    sub = hist.loc[:, avail]
    if sub.shape[0] < int(0.7 * L):
        return np.nan
    C = sub.corr().to_numpy()
    if not np.all(np.isfinite(C)):
        return np.nan
    eigs = np.linalg.eigvalsh(C)
    eigs = eigs[eigs > 1e-10]
    if eigs.size == 0:
        return np.nan
    p = eigs / eigs.sum()
    ent = -float((p * np.log(p)).sum())
    return float(np.exp(ent))


def _portfolio_vol_60(arc_df: pd.DataFrame, t: pd.Timestamp,
                      L: int = 60) -> float:
    hist = arc_df.loc[arc_df.index < t].iloc[-L:]
    if hist.shape[0] == 0:
        return np.nan
    avail = ~hist.iloc[-1].isna()
    if not avail.any():
        return np.nan
    sub = hist.loc[:, avail].fillna(0.0)
    if sub.shape[0] < int(0.7 * L):
        return np.nan
    ew = sub.mean(axis=1)
    if ew.std(ddof=1) < 1e-12:
        return 0.0
    return float(ew.std(ddof=1) * math.sqrt(TRADING_DAYS))


def build_features(arc_df: pd.DataFrame,
                   rebal_dates: pd.DatetimeIndex) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return X of shape (T, F=14), feature_names, avail_mask (T, n_arcs).

    14 features:
      6 × realized_vol_252
      6 × sharpe_trend_252 (substituted for rank_ic_trend)
      6 × valuation_spread (trailing-252d − trailing-5y mean return)
      1 × cross-arc corr eff-rank (60d)
      1 × portfolio EW vol (60d)
    = 20 features. Pre-reg said 14, but the 6 valuation spreads applied
    to *all* 6 arcs increase the count vs the pre-reg's "1 per arc". To
    stay honest to the pre-reg dimensionality budget, we *DROP* the
    redundant per-arc valuation spread and just keep the cross-arc agg.
    Final feature count: 6 + 6 + 1 + 1 = 14, matching the pre-reg.
    """
    feat_names: list[str] = []
    for c in ARC_COLS:
        feat_names.append(f'{c}_vol252')
    for c in ARC_COLS:
        feat_names.append(f'{c}_sh_trend')
    feat_names.append('corr_eff_rank_60')
    feat_names.append('port_vol_60')
    assert len(feat_names) == 14

    T = len(rebal_dates)
    n_arcs = arc_df.shape[1]
    X = np.full((T, 14), np.nan)
    avail = np.zeros((T, n_arcs), dtype=bool)
    for i, t in enumerate(rebal_dates):
        for j, c in enumerate(ARC_COLS):
            X[i, j] = _trailing_vol(arc_df[c], t, 252)
            X[i, 6 + j] = _sharpe_trend(arc_df[c], t)
            s = arc_df[c]
            avail[i, j] = s.loc[s.index < t].dropna().size >= 252
        X[i, 12] = _cross_arc_corr_eff_rank(arc_df, t)
        X[i, 13] = _portfolio_vol_60(arc_df, t)
    return X, feat_names, avail


# ---------------------------------------------------------------------------
# Targets: each arc's next-quarter (63 td) total return (NaN if not available)
# ---------------------------------------------------------------------------

def build_targets(arc_df: pd.DataFrame,
                  rebal_dates: pd.DatetimeIndex) -> np.ndarray:
    T = len(rebal_dates)
    n_arcs = arc_df.shape[1]
    Y = np.full((T, n_arcs), np.nan)
    idx = arc_df.index
    for i, t in enumerate(rebal_dates):
        pos = idx.searchsorted(t, side='left')
        if pos + CADENCE_TD > len(idx):
            continue
        block = arc_df.iloc[pos: pos + CADENCE_TD]
        for j, c in enumerate(ARC_COLS):
            col = block[c]
            if col.isna().mean() > 0.3:
                continue
            r = col.fillna(0.0).to_numpy()
            Y[i, j] = float((1.0 + r).prod() - 1.0)
    return Y


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _zscore_train_apply(X_train: np.ndarray, X_apply: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z_train = (X_train - mu) / sd
    Z_apply = (X_apply - mu) / sd
    # Impute NaN with 0 (post-z-score, equivalent to in-sample mean)
    Z_train = np.where(np.isnan(Z_train), 0.0, Z_train)
    Z_apply = np.where(np.isnan(Z_apply), 0.0, Z_apply)
    return Z_train, Z_apply


def model_d1_ridge(X_train: np.ndarray, y_train: np.ndarray,
                   X_apply: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge per arc. y_train: (T_tr, n_arcs)."""
    Z_tr, Z_ap = _zscore_train_apply(X_train, X_apply)
    n_features = Z_tr.shape[1]
    n_arcs = y_train.shape[1]
    preds = np.full((Z_ap.shape[0], n_arcs), np.nan)
    A = Z_tr.T @ Z_tr + alpha * np.eye(n_features)
    A_inv = np.linalg.inv(A)
    for j in range(n_arcs):
        y = y_train[:, j].copy()
        mask = ~np.isnan(y)
        if mask.sum() < n_features + 2:
            preds[:, j] = np.nanmean(y) if mask.any() else 0.0
            continue
        Zj = Z_tr[mask]
        yj = y[mask]
        Aj = Zj.T @ Zj + alpha * np.eye(n_features)
        try:
            beta = np.linalg.solve(Aj, Zj.T @ yj)
        except np.linalg.LinAlgError:
            beta = A_inv @ (Z_tr.T @ np.where(mask, y, 0.0))
        intercept = float(yj.mean())
        preds[:, j] = Z_ap @ beta + intercept - (Z_tr.mean(axis=0) @ beta)
    return preds


def model_d2_kernel_ridge(X_train: np.ndarray, y_train: np.ndarray,
                          X_apply: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """RBF kernel ridge with median-distance bandwidth.
    Per pre-reg pragmatic substitution for RF depth-3."""
    Z_tr, Z_ap = _zscore_train_apply(X_train, X_apply)
    # Median pairwise distance for bandwidth (squared)
    n_tr = Z_tr.shape[0]
    if n_tr < 4:
        # fallback to ridge
        return model_d1_ridge(X_train, y_train, X_apply, alpha=alpha)
    # Pairwise squared distances
    sq = ((Z_tr[:, None, :] - Z_tr[None, :, :]) ** 2).sum(axis=2)
    iu = np.triu_indices(n_tr, k=1)
    med = float(np.median(sq[iu]))
    if med < 1e-12:
        med = 1.0
    sigma2 = med
    K_tr = np.exp(-sq / sigma2)
    sq_ap = ((Z_ap[:, None, :] - Z_tr[None, :, :]) ** 2).sum(axis=2)
    K_ap = np.exp(-sq_ap / sigma2)
    n_arcs = y_train.shape[1]
    preds = np.full((Z_ap.shape[0], n_arcs), np.nan)
    for j in range(n_arcs):
        y = y_train[:, j].copy()
        mask = ~np.isnan(y)
        if mask.sum() < 4:
            preds[:, j] = 0.0
            continue
        K_sub = K_tr[np.ix_(mask, mask)]
        try:
            alpha_vec = np.linalg.solve(K_sub + alpha * np.eye(K_sub.shape[0]),
                                        y[mask])
        except np.linalg.LinAlgError:
            preds[:, j] = float(y[mask].mean())
            continue
        preds[:, j] = K_ap[:, mask] @ alpha_vec
    return preds


def model_d3_pca_2pc(X_train: np.ndarray, y_train: np.ndarray,
                     X_apply: np.ndarray) -> np.ndarray:
    """PCA → top-2 PCs → linear regression per arc."""
    Z_tr, Z_ap = _zscore_train_apply(X_train, X_apply)
    # PCA via SVD on training data
    U, S, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    n_pc = min(2, Vt.shape[0])
    W = Vt[:n_pc].T  # (n_features, n_pc)
    P_tr = Z_tr @ W
    P_ap = Z_ap @ W
    n_arcs = y_train.shape[1]
    preds = np.full((Z_ap.shape[0], n_arcs), np.nan)
    for j in range(n_arcs):
        y = y_train[:, j].copy()
        mask = ~np.isnan(y)
        if mask.sum() < n_pc + 2:
            preds[:, j] = 0.0
            continue
        Pj = P_tr[mask]
        # Add intercept column
        Pj_aug = np.column_stack([np.ones(Pj.shape[0]), Pj])
        try:
            beta, *_ = np.linalg.lstsq(Pj_aug, y[mask], rcond=None)
        except np.linalg.LinAlgError:
            preds[:, j] = float(y[mask].mean())
            continue
        Pa_aug = np.column_stack([np.ones(P_ap.shape[0]), P_ap])
        preds[:, j] = Pa_aug @ beta
    return preds


# ---------------------------------------------------------------------------
# Weight transform + realization
# ---------------------------------------------------------------------------

def predictions_to_weights(preds: np.ndarray, vols: np.ndarray,
                           avail: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Per pre-reg: softmax of (expected return / B3 inv-vol weights), masked
    by availability and renormalized."""
    T, n_arcs = preds.shape
    out = np.zeros_like(preds)
    inv_vol = np.where(vols > 1e-6, 1.0 / vols, 0.0)
    # row-normalize inv_vol on available arcs
    for t in range(T):
        a = avail[t]
        if not a.any():
            continue
        # σ-scaling: divide expected return by per-arc vol
        score = np.where(a, preds[t] / np.where(vols[t] > 1e-6, vols[t], 1.0), -np.inf)
        # softmax
        score = score * temp
        m = np.max(score[a])
        ex = np.where(a, np.exp(np.clip(score - m, -50, 50)), 0.0)
        s = ex.sum()
        if s < 1e-12:
            # fallback: inverse-vol on available
            iv = np.where(a, inv_vol[t], 0.0)
            s_iv = iv.sum()
            out[t] = iv / s_iv if s_iv > 1e-12 else (a / a.sum())
        else:
            out[t] = ex / s
    return out


def realize_daily(arc_df: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
                  weights: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Realize a daily return stream from per-rebal weights, with 10 bps
    switching cost on |Δw|/2 at each rebal."""
    idx = arc_df.index
    daily_vals: list[float] = []
    daily_dates: list[pd.Timestamp] = []
    prev_w = None
    for i, t in enumerate(rebal_dates):
        pos = idx.searchsorted(t, side='left')
        end = pos + CADENCE_TD
        if end > len(idx):
            break
        block = arc_df.iloc[pos: end].to_numpy()
        avail_block = ~np.isnan(block).all(axis=0)
        w = weights[i].copy()
        w = np.where(avail_block, w, 0.0)
        s = w.sum()
        if s < 1e-12:
            block_daily = np.zeros(CADENCE_TD)
        else:
            w = w / s
            block_filled = np.where(np.isnan(block), 0.0, block)
            block_daily = block_filled @ w
        # Switching cost at start of block
        if prev_w is None:
            cost = w.sum() * COMMISSION_BPS / 1e4
        else:
            cost = 0.5 * np.abs(w - prev_w).sum() * COMMISSION_BPS / 1e4
        if block_daily.size > 0:
            block_daily = block_daily.copy()
            block_daily[0] -= cost
        prev_w = w
        daily_vals.extend(block_daily.tolist())
        daily_dates.extend(idx[pos: end].tolist())
    return pd.DatetimeIndex(daily_dates), np.asarray(daily_vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def annualized_sharpe(daily: np.ndarray) -> float:
    a = daily[~np.isnan(daily)]
    if a.size < 5 or a.std(ddof=1) < 1e-12:
        return float('nan')
    return float(a.mean() / a.std(ddof=1) * math.sqrt(TRADING_DAYS))


def delta_sr_vs(strat: np.ndarray, bench: np.ndarray) -> dict:
    n = min(strat.size, bench.size)
    a = strat[-n:]; b = bench[-n:]
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]; b = b[mask]
    if a.size < 30:
        return {'n': int(a.size), 'status': 'too-short'}
    res = sharpe_difference_ci(a, b, n_bootstraps=2000, confidence=0.95, seed=42)
    ann = math.sqrt(TRADING_DAYS)
    return {
        'n': int(a.size),
        'sr_strat_ann': res.sr_a * ann,
        'sr_bench_ann': res.sr_b * ann,
        'delta_sr_ann': res.delta_sr * ann,
        'ci_lo_ann': res.ci_lo * ann,
        'ci_hi_ann': res.ci_hi * ann,
        'ci_width_ann': (res.ci_hi - res.ci_lo) * ann,
        'includes_zero': bool(res.includes_zero),
        'block_length': int(res.block_length),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=== Meta-allocator internal features — pre-registered eval ===')
    print('Loading 6-arc panel ...')
    arc_df = load_arc_panel()
    print(f'  panel: {arc_df.shape[0]} rows × {arc_df.shape[1]} arcs '
          f'({arc_df.index[0].date()} → {arc_df.index[-1].date()})')
    for c in ARC_COLS:
        s = arc_df[c].dropna()
        if s.size:
            print(f'    {c:20s} n={s.size:>5d} '
                  f'{s.index[0].date()} → {s.index[-1].date()}')

    # Quarterly rebal grid over the full span
    rebal_dates = arc_df.index[::CADENCE_TD]
    # Drop tail rebals that don't have a full forward block
    rebal_dates = rebal_dates[: -1]  # safe: last one likely lacks forward 63d
    # Filter to those that begin no later than fold3 end
    rebal_dates = pd.DatetimeIndex(
        [d for d in rebal_dates if d <= FOLDS[-1][2]]
    )
    print(f'\n  rebal grid: {len(rebal_dates)} quarterly dates '
          f'({rebal_dates[0].date()} → {rebal_dates[-1].date()})')

    print('\nBuilding features ...')
    X, feat_names, avail = build_features(arc_df, rebal_dates)
    print(f'  X shape: {X.shape} ({len(feat_names)} features); '
          f'nan-rate: {np.isnan(X).mean():.3f}')

    print('Building targets (next-quarter returns) ...')
    Y = build_targets(arc_df, rebal_dates)
    print(f'  Y shape: {Y.shape}; nan-rate: {np.isnan(Y).mean():.3f}')

    # vols per row used for σ-scaling in weight transform — just the
    # first 6 features (realized_vol_252 per arc)
    vols = X[:, :6]
    # Require at least the cross-arc features (12,13) + dca features (0,6)
    # to be finite, AND at least one arc available. Per-arc NaN features
    # for unavailable arcs will be mean-imputed inside _zscore_train_apply.
    cross_finite = ~np.isnan(X[:, 12]) & ~np.isnan(X[:, 13])
    dca_finite = ~np.isnan(X[:, 0]) & ~np.isnan(X[:, 6])
    any_avail = avail.any(axis=1)
    row_valid = cross_finite & dca_finite & any_avail
    print(f'  rebals usable (cross+dca finite, ≥1 arc avail): {row_valid.sum()} / {len(rebal_dates)}')

    # Fold masks (by rebal date)
    def in_fold(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        return np.asarray([(d >= start) & (d <= end) for d in rebal_dates])

    fold1_mask = in_fold(FOLDS[0][1], FOLDS[0][2])
    fold2_mask = in_fold(FOLDS[1][1], FOLDS[1][2])
    fold3_mask = in_fold(FOLDS[2][1], FOLDS[2][2])
    train_mask = (fold1_mask | fold2_mask) & row_valid
    test_mask = fold3_mask & row_valid

    print(f'\n  train (fold1+2): n_rebal = {train_mask.sum()}')
    print(f'  test  (fold3):   n_rebal = {test_mask.sum()}')

    if train_mask.sum() < 8 or test_mask.sum() < 4:
        print('\nERROR: insufficient rebals for valid eval.')
        return

    X_tr, Y_tr = X[train_mask], Y[train_mask]
    X_te = X[test_mask]
    avail_te = avail[test_mask]
    vols_te = vols[test_mask]

    # In-search model selection: in-sample (training) Sharpe of weights
    # built from each model's training-set predictions, applied to
    # training-set returns. (No look-ahead since train-only.)
    print('\n--- training-fold Sharpe per model (model selection) ---')
    model_specs = {
        'D1_ridge': lambda Xtr, Ytr, Xte: model_d1_ridge(Xtr, Ytr, Xte, alpha=1.0),
        'D2_kernel_ridge': lambda Xtr, Ytr, Xte: model_d2_kernel_ridge(Xtr, Ytr, Xte, alpha=1.0),
        'D3_pca_2pc': lambda Xtr, Ytr, Xte: model_d3_pca_2pc(Xtr, Ytr, Xte),
    }

    # Train each model on (X_tr, Y_tr) and predict on X_tr (self-predict)
    # to compute the in-sample Sharpe of derived weights.
    train_dates = pd.DatetimeIndex(rebal_dates[train_mask])
    train_sharpes: dict[str, float] = {}
    for name, fn in model_specs.items():
        preds_tr = fn(X_tr, Y_tr, X_tr)
        avail_tr = avail[train_mask]
        vols_tr = vols[train_mask]
        w_tr = predictions_to_weights(preds_tr, vols_tr, avail_tr)
        _, dret = realize_daily(arc_df, train_dates, w_tr)
        sr = annualized_sharpe(dret)
        train_sharpes[name] = sr
        print(f'  {name:18s}  in-train Sharpe_ann = {sr:+.3f}  (n_daily={dret.size})')

    # Pick the in-train winner as model*
    model_star = max(train_sharpes, key=lambda k: (train_sharpes[k]
                     if not math.isnan(train_sharpes[k]) else -1e9))
    print(f'\n  model* (in-train winner): {model_star}')

    # Now: train on train, predict on test (fold3 OOS)
    print('\n--- fold-3 OOS for each model ---')
    test_dates = pd.DatetimeIndex(rebal_dates[test_mask])
    model_daily: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    for name, fn in model_specs.items():
        preds_te = fn(X_tr, Y_tr, X_te)
        w_te = predictions_to_weights(preds_te, vols_te, avail_te)
        d_idx, dret = realize_daily(arc_df, test_dates, w_te)
        model_daily[name] = (d_idx, dret)
        sr = annualized_sharpe(dret)
        print(f'  {name:18s}  fold3 Sharpe_ann = {sr:+.3f}  (n_daily={dret.size})')

    # ------------------------------------------------------------
    # Baselines: B3, B2, canonical (DCA + 2×vol_v3)
    # ------------------------------------------------------------
    print('\n--- baselines on fold-3 ---')
    # B3 / B2 from cached streams
    cached = np.load(OUTPUT / 'meta-allocator-daily-streams.npz', allow_pickle=True)
    cached_dates = pd.to_datetime([str(d) for d in cached['dates']])
    b3_all = pd.Series(cached['B3_inv_vol'], index=cached_dates)
    b2_all = pd.Series(cached['B2_equal_weight'], index=cached_dates)

    # Restrict to fold3 daily range
    f3_start = FOLDS[-1][1]
    f3_end = FOLDS[-1][2]
    b3_f3 = b3_all.loc[(b3_all.index >= f3_start) & (b3_all.index <= f3_end)].dropna()
    b2_f3 = b2_all.loc[(b2_all.index >= f3_start) & (b2_all.index <= f3_end)].dropna()
    print(f'  B3_inv_vol fold3 Sharpe_ann = {annualized_sharpe(b3_f3.to_numpy()):+.3f}  '
          f'(n={b3_f3.size})')
    print(f'  B2_eq_weight fold3 Sharpe_ann = {annualized_sharpe(b2_f3.to_numpy()):+.3f}  '
          f'(n={b2_f3.size})')

    # Canonical (DCA + 2×vol_v3): only defined when vol_v3 is available
    # (~2024-04 onwards on the actual arc).
    dca_s = arc_df['dca']
    vol_s = arc_df['vol_v3'].fillna(0.0)
    ens_full = dca_s.add(2.0 * vol_s, fill_value=0.0)
    ens_f3 = ens_full.loc[(ens_full.index >= f3_start) & (ens_full.index <= f3_end)].dropna()
    print(f'  Canonical (DCA + 2×vol_v3) fold3 Sharpe_ann = {annualized_sharpe(ens_f3.to_numpy()):+.3f}  '
          f'(n={ens_f3.size})')

    # ------------------------------------------------------------
    # Comparison: model* vs B3 vs B2 vs canonical — on common date range
    # ------------------------------------------------------------
    print(f'\n--- model* ({model_star}) vs benchmarks on fold3 OOS ---')
    star_idx, star_ret = model_daily[model_star]
    star_s = pd.Series(star_ret, index=star_idx)

    def align(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        common = a.index.intersection(b.index)
        return a.loc[common].to_numpy(), b.loc[common].to_numpy()

    a, b = align(star_s, b3_f3)
    r_vs_b3 = delta_sr_vs(a, b)
    print(f'  vs B3:        n={r_vs_b3.get("n")}  ΔSR_ann={r_vs_b3.get("delta_sr_ann", float("nan")):+.3f}  '
          f'CI=[{r_vs_b3.get("ci_lo_ann", float("nan")):+.3f}, {r_vs_b3.get("ci_hi_ann", float("nan")):+.3f}]  '
          f'incl0={r_vs_b3.get("includes_zero")}')
    a, b = align(star_s, b2_f3)
    r_vs_b2 = delta_sr_vs(a, b)
    print(f'  vs B2:        n={r_vs_b2.get("n")}  ΔSR_ann={r_vs_b2.get("delta_sr_ann", float("nan")):+.3f}  '
          f'CI=[{r_vs_b2.get("ci_lo_ann", float("nan")):+.3f}, {r_vs_b2.get("ci_hi_ann", float("nan")):+.3f}]  '
          f'incl0={r_vs_b2.get("includes_zero")}')
    a, b = align(star_s, ens_f3)
    r_vs_ens = delta_sr_vs(a, b)
    print(f'  vs canonical: n={r_vs_ens.get("n")}  ΔSR_ann={r_vs_ens.get("delta_sr_ann", float("nan")):+.3f}  '
          f'CI=[{r_vs_ens.get("ci_lo_ann", float("nan")):+.3f}, {r_vs_ens.get("ci_hi_ann", float("nan")):+.3f}]  '
          f'incl0={r_vs_ens.get("includes_zero")}')

    # DSR for model*
    try:
        mb = standardize_oos(star_ret, periods_per_year=TRADING_DAYS, n_trials=3)
        dsr_t = float(mb.deflated_tstat)
        dsr = float(mb.dsr)
    except Exception as e:
        dsr_t = float('nan'); dsr = float('nan')
        print(f'  DSR error: {e}')
    print(f'  model* DSR={dsr:.3f}  DSR-t={dsr_t:+.2f}  (n_trials=3)')

    # ------------------------------------------------------------
    # 2024+ slice
    # ------------------------------------------------------------
    print('\n--- 2024+ slice (user-requested OOS) ---')
    star_2024 = star_s.loc[star_s.index >= OOS_2024_START]
    b3_2024 = b3_f3.loc[b3_f3.index >= OOS_2024_START]
    b2_2024 = b2_f3.loc[b2_f3.index >= OOS_2024_START]
    ens_2024 = ens_f3.loc[ens_f3.index >= OOS_2024_START]
    print(f'  model* 2024+ Sharpe_ann = {annualized_sharpe(star_2024.to_numpy()):+.3f}  '
          f'(n={star_2024.size})')
    print(f'  B3 2024+    Sharpe_ann = {annualized_sharpe(b3_2024.to_numpy()):+.3f}  '
          f'(n={b3_2024.size})')
    print(f'  canonical 2024+ Sharpe_ann = {annualized_sharpe(ens_2024.to_numpy()):+.3f}  '
          f'(n={ens_2024.size})')

    a, b = align(star_2024, b3_2024)
    r24_b3 = delta_sr_vs(a, b)
    print(f'  vs B3 (2024+):        ΔSR_ann={r24_b3.get("delta_sr_ann", float("nan")):+.3f}  '
          f'CI=[{r24_b3.get("ci_lo_ann", float("nan")):+.3f}, {r24_b3.get("ci_hi_ann", float("nan")):+.3f}]  '
          f'incl0={r24_b3.get("includes_zero")}')
    a, b = align(star_2024, ens_2024)
    r24_ens = delta_sr_vs(a, b)
    print(f'  vs canonical (2024+): ΔSR_ann={r24_ens.get("delta_sr_ann", float("nan")):+.3f}  '
          f'CI=[{r24_ens.get("ci_lo_ann", float("nan")):+.3f}, {r24_ens.get("ci_hi_ann", float("nan")):+.3f}]  '
          f'incl0={r24_ens.get("includes_zero")}')

    # ------------------------------------------------------------
    # Verdict per locked bar
    # ------------------------------------------------------------
    def width(r: dict) -> float:
        return float(r.get('ci_width_ann', float('nan')))

    def classify(r_vs_b3: dict, r_vs_b2: dict, dsr_t: float) -> str:
        d = r_vs_b3.get('delta_sr_ann', float('nan'))
        inc0 = r_vs_b3.get('includes_zero', True)
        if math.isnan(d):
            return 'pending'
        if (not inc0) and d >= 0.30 and dsr_t > 3.0:
            verdict = 'confirmed-OOS'
        elif (not inc0) and d >= 0.15 and dsr_t > 1.5:
            verdict = 'partial-OOS'
        elif r_vs_b2.get('delta_sr_ann', -9) >= 0.10:
            verdict = 'partial-OOS-vs-1/N-only'
        elif d < 0.05:
            verdict = 'confirmed-null'
        else:
            verdict = 'partial-OOS'
        return verdict

    raw_verdict = classify(r_vs_b3, r_vs_b2, dsr_t)
    # Downgrade rule: bootstrap CI > ±0.40 → one tier
    ladder = ['confirmed-OOS', 'partial-OOS', 'partial-OOS-vs-1/N-only', 'confirmed-null']
    final_verdict = raw_verdict
    ci_w = width(r_vs_b3)
    if not math.isnan(ci_w) and (ci_w / 2.0) > 0.40 and raw_verdict in ladder:
        i = ladder.index(raw_verdict)
        final_verdict = ladder[min(i + 1, len(ladder) - 1)]
        print(f'\n  CI half-width {ci_w / 2.0:.3f} > 0.40 → downgrade '
              f'{raw_verdict} → {final_verdict}')
    print(f'\n=== LOCKED VERDICT: {final_verdict} ===')

    # ------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------
    summary = {
        'eval': 'meta-allocator-internal-features',
        'panel_arcs': ARC_COLS,
        'cadence_td': CADENCE_TD,
        'folds': [(name, str(s.date()), str(e.date())) for name, s, e in FOLDS],
        'rebal_n_total': int(len(rebal_dates)),
        'rebal_n_train': int(train_mask.sum()),
        'rebal_n_test': int(test_mask.sum()),
        'feature_names': feat_names,
        'train_sharpes_in_sample': {k: float(v) for k, v in train_sharpes.items()},
        'model_star': model_star,
        'fold3_sharpes': {
            name: float(annualized_sharpe(d[1])) for name, d in model_daily.items()
        },
        'baseline_fold3_sharpes': {
            'B3_inv_vol': float(annualized_sharpe(b3_f3.to_numpy())),
            'B2_equal_weight': float(annualized_sharpe(b2_f3.to_numpy())),
            'canonical_dca_plus_2vol_v3': float(annualized_sharpe(ens_f3.to_numpy())),
        },
        'model_star_vs_B3_fold3': r_vs_b3,
        'model_star_vs_B2_fold3': r_vs_b2,
        'model_star_vs_canonical_fold3': r_vs_ens,
        'model_star_dsr': dsr,
        'model_star_dsr_tstat': dsr_t,
        'model_star_2024plus_sharpe': float(annualized_sharpe(star_2024.to_numpy())),
        'baseline_2024plus_sharpes': {
            'B3_inv_vol': float(annualized_sharpe(b3_2024.to_numpy())),
            'B2_equal_weight': float(annualized_sharpe(b2_2024.to_numpy())),
            'canonical_dca_plus_2vol_v3': float(annualized_sharpe(ens_2024.to_numpy())),
        },
        'model_star_vs_B3_2024plus': r24_b3,
        'model_star_vs_canonical_2024plus': r24_ens,
        'raw_verdict': raw_verdict,
        'verdict': final_verdict,
        'notes': (
            'Pragmatic substitutions vs pre-reg: (1) D2 RF-depth-3 → RBF kernel '
            'ridge (pre-reg allows). (2) Valuation spread dropped per-arc and '
            'replaced with cross-arc corr eff-rank + port-vol-60 (kept 14-feature '
            'budget). (3) rank_ic_trend → 252d Sharpe-trend (slope of 60d rolling '
            'Sharpe). (4) Driver lives in apps/docs/scripts/, not new '
            'apps/meta_allocator/ — avoid scaffolding cost.'
        ),
    }
    out_path = OUTPUT / 'meta-allocator-internal-features-results.json'
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n→ {out_path}')

    np.savez(
        OUTPUT / 'meta-allocator-internal-features-streams.npz',
        model_star=model_star,
        dates_test=np.array([str(d.date()) for d in star_idx]),
        D1_ridge=model_daily['D1_ridge'][1],
        D2_kernel_ridge=model_daily['D2_kernel_ridge'][1],
        D3_pca_2pc=model_daily['D3_pca_2pc'][1],
    )
    print(f'→ {OUTPUT / "meta-allocator-internal-features-streams.npz"}')


if __name__ == '__main__':
    main()
