"""Pair-level Φ evaluation for pairs (v0.1 rescue).

Different prediction problem than v0:
  - State features = 7 per-pair training-window features + 5 macro context features
  - "Action" is constant (include this pair) — Φ is now a per-pair Sharpe predictor
  - LOO-by-window, hold out ~50 pairs at a time
  - Baselines: per-window-mean (no pair info), linear regression (v1-LR analog)

The point: the pairs v1 LR experiment captured 5.4% of the +1.79 oracle
headroom. This tests whether a non-linear NN + macro context lifts that
capture rate meaningfully.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from critic.dataset import REPO_ROOT, OUTPUT_DIR
from critic.features import _load_vix_series, _load_spy_proxy, _as_of, _trailing_change, _trailing_log_return
from critic.model import train_phi, predict_phi


@dataclass
class PairTriple:
    window_idx: int
    val_start: str
    val_end: str
    pair_a: str
    pair_b: str
    realized_sharpe: float
    # 7 pair features (from pairs v1 predictor):
    log_train_half_life: float
    abs_train_corr: float
    log_eg_pvalue: float
    abs_hedge_beta: float
    train_sharpe: float
    train_pct_in_trade: float
    log_train_n_trades: float


def load_pair_records_npz(output_dir: Path = OUTPUT_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load `pairs-predictor-per-pair-records.npz` produced by the pairs
    predictor walk-forward script.

    Returns (window_idx, features, realized_sharpe, feature_names).
    """
    path = output_dir / "pairs-predictor-per-pair-records.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing per-pair records at {path}. Re-run "
            f"`uv run python apps/pairs/scripts/run_pair_predictor_walkforward.py`."
        )
    d = np.load(path, allow_pickle=False)
    return (
        d["window_idx"].astype(np.int32),
        d["features"].astype(np.float64),
        d["realized_sharpe"].astype(np.float64),
        d["feature_names"].tolist() if "feature_names" in d.files else [],
    )


def _train_features_from_v1_summary(
    output_dir: Path = OUTPUT_DIR,
) -> list[PairTriple]:
    """The pairs v0 walk-forward summary has each pair's training-window
    diagnostics inline. The seven features the v1 LR uses are:

      log_train_half_life
      abs(train_corr)
      log(eg_pvalue)
      abs(hedge_beta)
      train_sharpe
      train_pct_in_trade
      log(train_n_trades)

    These were computed at train time; we read them from the summary
    rather than recomputing.
    """
    src = output_dir / "pairs-walkforward-summary.json"
    if not src.exists():
        return []

    data = json.loads(src.read_text())
    triples: list[PairTriple] = []
    for win in data["per_window"]:
        w_idx = win["window_idx"]
        val_start = win["val_start"]
        val_end = win["val_end"]
        for pair in win["pairs"]:
            # Some fields may be missing depending on how the v0 summary was
            # written. Skip pairs that are missing required features rather
            # than fabricate them.
            try:
                half_life = float(pair["train_half_life"])
                n_trades = int(pair["n_trades"])
                pct_in_trade = float(pair["pct_in_trade"])
                sharpe = float(pair["sharpe"])
            except KeyError:
                continue

            # Other features aren't in the v0 summary per-pair entry.
            # Recompute or use neutral fallbacks; the LR rescue test uses
            # only what's available — if a feature is missing, set to median.
            # This is an artifact of the v0 summary schema, not a real
            # information loss.
            triples.append(
                PairTriple(
                    window_idx=w_idx,
                    val_start=val_start,
                    val_end=val_end,
                    pair_a=str(pair.get("a", "")),
                    pair_b=str(pair.get("b", "")),
                    realized_sharpe=sharpe,
                    log_train_half_life=float(np.log(max(half_life, 1e-6))),
                    abs_train_corr=float(pair.get("abs_train_corr", np.nan)),
                    log_eg_pvalue=float(pair.get("log_eg_pvalue", np.nan)),
                    abs_hedge_beta=float(pair.get("abs_hedge_beta", np.nan)),
                    train_sharpe=float(pair.get("train_sharpe", np.nan)),
                    train_pct_in_trade=pct_in_trade,
                    log_train_n_trades=float(np.log(max(n_trades, 1))),
                )
            )
    return triples


def _macro_state(date_str: str) -> tuple[float, float, float, float, float]:
    """Five macro state features at the val_start date (point-in-time).

    Returns NaN for any feature that can't be resolved; the caller imputes.
    """
    if not date_str:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    vix = _load_vix_series()
    spy = _load_spy_proxy()
    import pandas as pd
    d = pd.Timestamp(date_str)
    return (
        _as_of(vix, d) or np.nan,
        _trailing_change(vix, d, 180) or np.nan,
        _trailing_change(vix, d, 30) or np.nan,
        _trailing_log_return(spy, d, 252) or np.nan,
        _trailing_log_return(spy, d, 63) or np.nan,
    )


def build_pair_dataset(
    triples: list[PairTriple],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix `(n_pairs, 12)` and label `(n_pairs,)` for Φ.

    Features (in order):
      0: log_train_half_life
      1: abs_train_corr
      2: log_eg_pvalue
      3: abs_hedge_beta
      4: train_sharpe
      5: train_pct_in_trade
      6: log_train_n_trades
      7-11: macro at val_start (VIX-level, VIX-6m-chg, VIX-1m-chg,
            SPY-252d-logret, SPY-63d-logret)
    """
    if not triples:
        return np.zeros((0, 12)), np.zeros((0,)), []
    feature_names = [
        "log_train_half_life",
        "abs_train_corr",
        "log_eg_pvalue",
        "abs_hedge_beta",
        "train_sharpe",
        "train_pct_in_trade",
        "log_train_n_trades",
        "vix_level",
        "vix_6m_change_pts",
        "vix_1m_change_pts",
        "spy_trailing_252d_log_ret",
        "spy_trailing_63d_log_ret",
    ]

    # Cache macro features by val_start (much faster than per-row lookup).
    macro_cache: dict[str, tuple[float, ...]] = {}

    rows = []
    labels = []
    for t in triples:
        if t.val_start not in macro_cache:
            macro_cache[t.val_start] = _macro_state(t.val_start)
        m = macro_cache[t.val_start]
        row = [
            t.log_train_half_life,
            t.abs_train_corr,
            t.log_eg_pvalue,
            t.abs_hedge_beta,
            t.train_sharpe,
            t.train_pct_in_trade,
            t.log_train_n_trades,
            m[0], m[1], m[2], m[3], m[4],
        ]
        rows.append(row)
        labels.append(t.realized_sharpe)

    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    # Median-impute NaNs per column.
    for col in range(X.shape[1]):
        nan_mask = ~np.isfinite(X[:, col])
        if nan_mask.any():
            med = np.nanmedian(X[~nan_mask, col]) if (~nan_mask).any() else 0.0
            X[nan_mask, col] = med

    return X, y, feature_names


def _linreg_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """L2-regularized linear regression baseline (the v1-LR analog).

    Centers y in training, fits ridge on (already-standardized) features
    using `np.linalg.lstsq` with explicit augmented design matrix:
        argmin_w ||y - Xw||² + λ||w||²
    via the stacked formulation `[[X], [√λ·I]] w ≈ [y, 0]`. Robust under
    multicollinearity (1200-sample / 12-feature regime). λ = 1.0.
    """
    n, d = X_train.shape
    y_mean = float(y_train.mean())
    y_c = y_train - y_mean
    lam = 1.0
    # Stacked design matrix: top half = X, bottom half = sqrt(λ) * I_d
    A = np.vstack([X_train, np.sqrt(lam) * np.eye(d)])
    b = np.concatenate([y_c, np.zeros(d)])
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return (X_test @ w) + y_mean


@dataclass
class PairFoldResult:
    window_idx: int
    val_start: str
    n_held: int
    phi_preds: list[float]
    lr_preds: list[float]
    window_mean_pred: float
    realized: list[float]
    spearman_r_phi: float
    spearman_r_lr: float
    rmse_phi: float
    rmse_lr: float
    rmse_window_mean: float


@dataclass
class PairQualityResult:
    n_triples: int
    n_features: int
    folds: list[PairFoldResult]
    overall_rmse_phi: float
    overall_rmse_lr: float
    overall_rmse_window_mean: float
    overall_spearman_phi: float
    overall_spearman_lr: float
    relative_rmse_vs_lr: float
    pass_strong: bool
    pass_app: bool
    marginal: bool
    verdict: str
    verdict_label: str
    # Deployment-Sharpe via top-N portfolio (proxy: per-window argmax fraction)
    top_n: int
    mean_topn_sharpe_phi: float
    mean_topn_sharpe_lr: float
    mean_topn_sharpe_oracle: float
    mean_topn_sharpe_all_pairs: float


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def _topn_window_sharpe(
    preds: np.ndarray, realized: np.ndarray, n: int = 10
) -> float:
    """Mean realized Sharpe of the top-N predicted pairs within this window."""
    if len(preds) == 0:
        return 0.0
    n = min(n, len(preds))
    idx = np.argsort(preds)[-n:]
    return float(np.mean(realized[idx]))


def _val_starts_for_windows(triples_or_pred_summary: Path | None = None) -> dict[int, str]:
    """Map window_idx → val_start by reading pairs-predictor-walkforward-summary."""
    if triples_or_pred_summary is None:
        triples_or_pred_summary = OUTPUT_DIR / "pairs-walkforward-summary.json"
    out: dict[int, str] = {}
    if not Path(triples_or_pred_summary).exists():
        return out
    data = json.loads(Path(triples_or_pred_summary).read_text())
    for win in data.get("per_window", []):
        out[int(win["window_idx"])] = str(win["val_start"])
    return out


def pair_phi_quality_loo_npz(
    *,
    hidden: int = 16,
    n_layers: int = 2,
    n_steps: int = 400,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    top_n: int = 50,
    include_macro: bool = True,
    verbose: bool = False,
) -> PairQualityResult:
    """v0.1 rescue runner — uses the per-pair npz dump for the rich 7-feature view."""
    window_idxs_arr, pair_feats, y, feat_names = load_pair_records_npz()
    val_starts_map = _val_starts_for_windows()
    val_starts = [val_starts_map.get(int(w), "") for w in window_idxs_arr]

    if include_macro:
        # Build macro state per row
        macro_cache: dict[str, tuple[float, ...]] = {}
        macro_rows = []
        for vs in val_starts:
            if vs not in macro_cache:
                macro_cache[vs] = _macro_state(vs)
            macro_rows.append(macro_cache[vs])
        macro_arr = np.asarray(macro_rows, dtype=np.float64)
        X = np.hstack([pair_feats, macro_arr])
        feat_names = list(feat_names) + [
            "vix_level", "vix_6m_change_pts", "vix_1m_change_pts",
            "spy_trailing_252d_log_ret", "spy_trailing_63d_log_ret",
        ]
    else:
        X = pair_feats

    # Median-impute any NaNs
    for col in range(X.shape[1]):
        nan_mask = ~np.isfinite(X[:, col])
        if nan_mask.any():
            med = np.nanmedian(X[~nan_mask, col]) if (~nan_mask).any() else 0.0
            X[nan_mask, col] = med

    window_idxs = window_idxs_arr
    n_features = X.shape[1]

    unique_windows = sorted(set(window_idxs.tolist()))
    folds: list[PairFoldResult] = []

    for fold_idx, w in enumerate(unique_windows):
        held_mask = window_idxs == w
        train_mask = ~held_mask
        if not train_mask.any() or not held_mask.any():
            continue

        # Standardize features on train stats
        mu = X[train_mask].mean(axis=0)
        sd = X[train_mask].std(axis=0) + 1e-8
        X_norm = (X - mu) / sd
        X_train = X_norm[train_mask]
        y_train = y[train_mask]
        X_held = X_norm[held_mask]
        y_held = y[held_mask]

        if verbose:
            print(f"[fold {fold_idx+1}/{len(unique_windows)}] w={w} val_start={val_starts[np.where(held_mask)[0][0]]} n_train={len(y_train)} n_held={len(y_held)}")

        # Φ
        res = train_phi(
            X_train,
            y_train,
            hidden=hidden,
            n_layers=n_layers,
            n_steps=n_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + fold_idx,
            verbose=False,
        )
        phi_preds = predict_phi(res.params, X_held)

        # LR baseline
        lr_preds = _linreg_fit_predict(X_train, y_train, X_held)

        # Window-mean baseline (train mean)
        window_mean = float(np.mean(y_train))

        # Spearman per-fold (within-window rank quality)
        if len(y_held) >= 3 and np.std(phi_preds) > 1e-8 and np.std(y_held) > 1e-8:
            r_phi_obj = spearmanr(phi_preds, y_held)
            r_phi = float(r_phi_obj.statistic) if np.isfinite(r_phi_obj.statistic) else 0.0
        else:
            r_phi = 0.0
        if len(y_held) >= 3 and np.std(lr_preds) > 1e-8 and np.std(y_held) > 1e-8:
            r_lr_obj = spearmanr(lr_preds, y_held)
            r_lr = float(r_lr_obj.statistic) if np.isfinite(r_lr_obj.statistic) else 0.0
        else:
            r_lr = 0.0

        folds.append(
            PairFoldResult(
                window_idx=w,
                val_start=val_starts[np.where(held_mask)[0][0]],
                n_held=int(held_mask.sum()),
                phi_preds=[float(x) for x in phi_preds],
                lr_preds=[float(x) for x in lr_preds],
                window_mean_pred=window_mean,
                realized=[float(x) for x in y_held],
                spearman_r_phi=r_phi,
                spearman_r_lr=r_lr,
                rmse_phi=_rmse(phi_preds, y_held),
                rmse_lr=_rmse(lr_preds, y_held),
                rmse_window_mean=_rmse(np.full(len(y_held), window_mean), y_held),
            )
        )

    # Overall pooled metrics
    phi_all = np.concatenate([np.array(f.phi_preds) for f in folds])
    lr_all = np.concatenate([np.array(f.lr_preds) for f in folds])
    real_all = np.concatenate([np.array(f.realized) for f in folds])
    wm_all = np.concatenate([np.full(f.n_held, f.window_mean_pred) for f in folds])
    overall_rmse_phi = _rmse(phi_all, real_all)
    overall_rmse_lr = _rmse(lr_all, real_all)
    overall_rmse_wm = _rmse(wm_all, real_all)
    overall_spearman_phi = float(spearmanr(phi_all, real_all).statistic)
    overall_spearman_lr = float(spearmanr(lr_all, real_all).statistic)
    rel_vs_lr = overall_rmse_phi / max(overall_rmse_lr, 1e-8)

    # Top-N within-window selection Sharpe (deployment proxy)
    phi_topn = [_topn_window_sharpe(np.array(f.phi_preds), np.array(f.realized), n=top_n) for f in folds]
    lr_topn = [_topn_window_sharpe(np.array(f.lr_preds), np.array(f.realized), n=top_n) for f in folds]
    oracle_topn = [_topn_window_sharpe(np.array(f.realized), np.array(f.realized), n=top_n) for f in folds]
    all_pairs = [float(np.mean(f.realized)) for f in folds]

    # Pre-registered v0.1 cuts (laid out in pre-reg before running):
    #   PASS: rel_vs_lr ≤ 0.85 AND overall_spearman_phi ≥ +0.20
    #   MARGINAL: rel_vs_lr ≤ 0.95 AND overall_spearman_phi > 0
    #   STRONG-PASS: rel_vs_lr ≤ 0.70 AND overall_spearman_phi ≥ +0.30 AND
    #                (mean phi_topn > mean lr_topn + 0.5)
    strong = rel_vs_lr <= 0.70 and overall_spearman_phi >= 0.30 and (
        float(np.mean(phi_topn)) > float(np.mean(lr_topn)) + 0.5
    )
    app_pass = rel_vs_lr <= 0.85 and overall_spearman_phi >= 0.20
    marginal = (not app_pass) and (rel_vs_lr <= 0.95 and overall_spearman_phi > 0)

    if strong:
        verdict = "STRONG-PASS"
        label = "partial-OOS"
    elif app_pass:
        verdict = "PASS"
        label = "partial-OOS"
    elif marginal:
        verdict = "MARGINAL"
        label = "partial-OOS"
    else:
        verdict = "FAIL"
        label = "confirmed-null"

    return PairQualityResult(
        n_triples=int(X.shape[0]),
        n_features=n_features,
        folds=folds,
        overall_rmse_phi=overall_rmse_phi,
        overall_rmse_lr=overall_rmse_lr,
        overall_rmse_window_mean=overall_rmse_wm,
        overall_spearman_phi=overall_spearman_phi,
        overall_spearman_lr=overall_spearman_lr,
        relative_rmse_vs_lr=rel_vs_lr,
        pass_strong=strong,
        pass_app=app_pass,
        marginal=marginal,
        verdict=verdict,
        verdict_label=label,
        top_n=top_n,
        mean_topn_sharpe_phi=float(np.mean(phi_topn)),
        mean_topn_sharpe_lr=float(np.mean(lr_topn)),
        mean_topn_sharpe_oracle=float(np.mean(oracle_topn)),
        mean_topn_sharpe_all_pairs=float(np.mean(all_pairs)),
    )


def result_to_dict(result: PairQualityResult) -> dict:
    return {
        "n_triples": result.n_triples,
        "n_features": result.n_features,
        "overall_rmse_phi": result.overall_rmse_phi,
        "overall_rmse_lr": result.overall_rmse_lr,
        "overall_rmse_window_mean": result.overall_rmse_window_mean,
        "overall_spearman_phi": result.overall_spearman_phi,
        "overall_spearman_lr": result.overall_spearman_lr,
        "relative_rmse_vs_lr": result.relative_rmse_vs_lr,
        "verdict": result.verdict,
        "verdict_label": result.verdict_label,
        "top_n": result.top_n,
        "mean_topn_sharpe_phi": result.mean_topn_sharpe_phi,
        "mean_topn_sharpe_lr": result.mean_topn_sharpe_lr,
        "mean_topn_sharpe_oracle": result.mean_topn_sharpe_oracle,
        "mean_topn_sharpe_all_pairs": result.mean_topn_sharpe_all_pairs,
        "folds": [asdict(f) for f in result.folds],
    }
