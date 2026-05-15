"""Leave-one-window-out Φ-quality evaluation.

For each (app, window) fold:
  1. Hold out all triples with that (app, window).
  2. Standardize state features using train-set stats only.
  3. Train Φ on the remaining triples (cross-app pooled).
  4. Predict held-out Sharpes.
  5. Record both Φ predictions and the three baselines.

Cuts (per app):
  - **App-PASS**: RMSE_Φ / RMSE_per-action-baseline ≤ 0.75 AND Spearman r ≥ +0.20
  - **App-MARGINAL**: RMSE_Φ / RMSE_per-action-baseline ≤ 0.90 AND Spearman r > 0
  - **App-FAIL**: otherwise
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from critic.dataset import Triple
from critic.features import build_state_features, build_action_features
from critic.model import train_phi, predict_phi


@dataclass
class FoldResult:
    app: str
    window_idx: int
    val_start: str
    n_held_out: int
    phi_preds: list[float]
    realized: list[float]
    global_mean_pred: float
    per_app_mean_pred: float
    per_action_mean_pred: list[float]
    action_keys: list[str]


@dataclass
class AppQuality:
    app: str
    n_folds: int
    n_triples_eval: int
    rmse_phi: float
    rmse_global_mean: float
    rmse_per_app_mean: float
    rmse_per_action_mean: float
    spearman_r_phi: float
    spearman_p_phi: float
    relative_rmse_vs_per_action: float
    pass_strong: bool
    pass_app: bool
    marginal_app: bool
    verdict: str


@dataclass
class PhiQualityResult:
    n_total_triples: int
    n_state_features: int
    n_actions: int
    folds: list[FoldResult]
    per_app: list[AppQuality]
    overall_verdict: str
    overall_verdict_label: str


def _rmse(preds: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(y)) ** 2)))


def _per_action_train_mean(
    triples_train: Sequence[Triple], action_keys_eval: Sequence[str]
) -> list[float]:
    """For each held-out action_key, return the training-set mean Sharpe for
    that action; fall back to the global train mean if action is unseen.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    global_sum = 0.0
    global_count = 0
    for t in triples_train:
        sums[t.action_key] = sums.get(t.action_key, 0.0) + t.realized_sharpe
        counts[t.action_key] = counts.get(t.action_key, 0) + 1
        global_sum += t.realized_sharpe
        global_count += 1
    global_mean = global_sum / max(global_count, 1)
    return [sums.get(a, global_mean) / max(counts.get(a, 1), 1) if a in sums else global_mean for a in action_keys_eval]


def _per_app_train_mean(triples_train: Sequence[Triple], app: str) -> float:
    matched = [t.realized_sharpe for t in triples_train if t.app == app]
    if not matched:
        return float(np.mean([t.realized_sharpe for t in triples_train])) if triples_train else 0.0
    return float(np.mean(matched))


def phi_quality_loo(
    triples: Sequence[Triple],
    *,
    hidden: int = 16,
    n_layers: int = 2,
    n_steps: int = 300,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> PhiQualityResult:
    """Run LOO-by-(app, window) over `triples`.

    Returns per-fold predictions + per-app aggregated quality metrics +
    overall day-1 verdict.
    """
    all_actions = sorted({t.action_key for t in triples})

    # Build state features once (point-in-time on val_start; not fold-dependent).
    X_state_all, sf_names = build_state_features(triples)
    X_action_all, _ = build_action_features(triples, all_actions)
    n_state = X_state_all.shape[1]
    n_continuous = 5  # the macro features; everything after is one-hot

    # Distinct (app, window) folds.
    folds_to_run: list[tuple[str, int]] = sorted({(t.app, t.window_idx) for t in triples})

    fold_results: list[FoldResult] = []
    for fold_idx, (app, w) in enumerate(folds_to_run):
        held_mask = np.array([(t.app == app and t.window_idx == w) for t in triples])
        train_mask = ~held_mask
        n_held = int(held_mask.sum())
        n_train = int(train_mask.sum())
        if n_held == 0 or n_train == 0:
            continue

        # Standardize continuous features using train-set stats.
        X_state_train = X_state_all[train_mask]
        mu = X_state_train[:, :n_continuous].mean(axis=0)
        sd = X_state_train[:, :n_continuous].std(axis=0) + 1e-8
        X_state_norm = X_state_all.copy()
        X_state_norm[:, :n_continuous] = (X_state_norm[:, :n_continuous] - mu) / sd

        X_all = np.hstack([X_state_norm, X_action_all])
        y_all = np.array([t.realized_sharpe for t in triples])

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_held = X_all[held_mask]
        y_held = y_all[held_mask]

        held_triples = [t for t, m in zip(triples, held_mask) if m]
        train_triples = [t for t, m in zip(triples, train_mask) if m]

        if verbose:
            print(f"[fold {fold_idx+1}/{len(folds_to_run)}] app={app} w={w}  n_train={n_train} n_held={n_held}")

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

        # Baselines (computed from train set only):
        global_mean = float(np.mean(y_train))
        per_app_mean = _per_app_train_mean(train_triples, app)
        action_keys = [t.action_key for t in held_triples]
        per_action_means = _per_action_train_mean(train_triples, action_keys)

        fold_results.append(
            FoldResult(
                app=app,
                window_idx=w,
                val_start=held_triples[0].val_start if held_triples else "",
                n_held_out=n_held,
                phi_preds=[float(x) for x in phi_preds],
                realized=[float(x) for x in y_held],
                global_mean_pred=global_mean,
                per_app_mean_pred=per_app_mean,
                per_action_mean_pred=[float(x) for x in per_action_means],
                action_keys=action_keys,
            )
        )

    # Per-app aggregation
    per_app: list[AppQuality] = []
    apps = sorted({fr.app for fr in fold_results})
    for app in apps:
        app_folds = [fr for fr in fold_results if fr.app == app]
        phi_all = np.concatenate([np.array(fr.phi_preds) for fr in app_folds])
        realized_all = np.concatenate([np.array(fr.realized) for fr in app_folds])
        global_all = np.concatenate([
            np.full(fr.n_held_out, fr.global_mean_pred) for fr in app_folds
        ])
        per_app_all = np.concatenate([
            np.full(fr.n_held_out, fr.per_app_mean_pred) for fr in app_folds
        ])
        per_action_all = np.concatenate([np.array(fr.per_action_mean_pred) for fr in app_folds])

        rmse_phi = _rmse(phi_all, realized_all)
        rmse_global = _rmse(global_all, realized_all)
        rmse_per_app = _rmse(per_app_all, realized_all)
        rmse_per_action = _rmse(per_action_all, realized_all)

        if len(realized_all) >= 3 and np.std(phi_all) > 1e-8 and np.std(realized_all) > 1e-8:
            r, p = spearmanr(phi_all, realized_all)
            r = float(r) if np.isfinite(r) else 0.0
            p = float(p) if np.isfinite(p) else 1.0
        else:
            r, p = 0.0, 1.0

        rel = rmse_phi / max(rmse_per_action, 1e-8)
        strong = rel <= 0.50 and r >= 0.30
        app_pass = rel <= 0.75 and r >= 0.20
        marginal = (not app_pass) and (rel <= 0.90 and r > 0)

        if strong:
            verdict = "STRONG-PASS"
        elif app_pass:
            verdict = "PASS"
        elif marginal:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        per_app.append(
            AppQuality(
                app=app,
                n_folds=len(app_folds),
                n_triples_eval=int(len(realized_all)),
                rmse_phi=rmse_phi,
                rmse_global_mean=rmse_global,
                rmse_per_app_mean=rmse_per_app,
                rmse_per_action_mean=rmse_per_action,
                spearman_r_phi=r,
                spearman_p_phi=p,
                relative_rmse_vs_per_action=rel,
                pass_strong=strong,
                pass_app=app_pass,
                marginal_app=marginal,
                verdict=verdict,
            )
        )

    # Overall verdict (matches pre-registered cuts)
    n_pass = sum(1 for q in per_app if q.pass_app)
    n_marg = sum(1 for q in per_app if q.marginal_app)
    if n_pass >= 3:
        overall = "STRONG-PASS"
        label = "partial-OOS"
    elif n_pass >= 2 or (n_pass + n_marg) >= 3:
        overall = "PASS"
        label = "partial-OOS"
    elif n_marg >= 3:
        overall = "MARGINAL"
        label = "partial-OOS"
    else:
        overall = "FAIL"
        label = "confirmed-null"

    return PhiQualityResult(
        n_total_triples=len(triples),
        n_state_features=n_state,
        n_actions=len(all_actions),
        folds=fold_results,
        per_app=per_app,
        overall_verdict=overall,
        overall_verdict_label=label,
    )


def result_to_dict(result: PhiQualityResult) -> dict:
    """JSON-serializable view of the eval result."""
    return {
        "n_total_triples": result.n_total_triples,
        "n_state_features": result.n_state_features,
        "n_actions": result.n_actions,
        "overall_verdict": result.overall_verdict,
        "overall_verdict_label": result.overall_verdict_label,
        "per_app": [asdict(q) for q in result.per_app],
        "folds": [asdict(fr) for fr in result.folds],
    }
