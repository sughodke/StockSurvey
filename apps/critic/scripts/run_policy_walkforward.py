"""Policy walk-forward — v0.2 of the critic arc (exploratory).

For each LOO fold over the pair-level pairs data:
  1. Train Φ on the train fold (pair + macro features).
  2. Train two policies against Φ:
     - π_vanilla: just maximize E[p · Φ].
     - π_cql: same plus an anchor toward the empirical top-K inclusion
       rate (50/200 = 0.25 by default).
  3. Deploy each policy on the held-out fold: rank pairs by policy
     score, take top-K, compute mean realized Sharpe.

Compares to:
  - All-pairs baseline (no selection)
  - Ridge LR baseline (the v1 LR predictor analog)
  - Φ-direct (rank by Φ's predicted Sharpe)
  - Within-window oracle (top-K by realized Sharpe — 100% capture)

Pre-registered cuts:
  - PASS:  policy_top-K mean Sharpe > LR + 0.03 AND > Φ-direct + 0.01
  - MARGINAL: policy_top-K > LR + 0.01
  - FAIL:  otherwise

Day-1 of the arc closed `confirmed-null`, so the expected outcome here
is FAIL (policy ≈ Φ ≈ LR). Methodology intentionally documented for
future Φ-quality improvements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from critic.pairs_eval import (
    load_pair_records_npz,
    _val_starts_for_windows,
    _macro_state,
    _linreg_fit_predict,
    _topn_window_sharpe,
)
from critic.model import train_phi, predict_phi
from critic.policy import train_policy, policy_score


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "Output"


@dataclass
class PolicyFold:
    window_idx: int
    val_start: str
    n_held: int
    topk_sharpe_all_pairs: float
    topk_sharpe_lr: float
    topk_sharpe_phi_direct: float
    topk_sharpe_policy_vanilla: float
    topk_sharpe_policy_cql: float
    topk_sharpe_oracle: float
    spearman_policy_vanilla_vs_phi: float  # how closely policy mirrors Φ
    spearman_policy_cql_vs_phi: float


@dataclass
class PolicyResult:
    n_triples: int
    n_features: int
    top_k: int
    cql_weight: float
    empirical_inclusion_rate: float
    folds: list[PolicyFold]
    mean_topk_all_pairs: float
    mean_topk_lr: float
    mean_topk_phi_direct: float
    mean_topk_policy_vanilla: float
    mean_topk_policy_cql: float
    mean_topk_oracle: float
    oracle_headroom: float
    capture_lr_pct: float
    capture_phi_direct_pct: float
    capture_policy_vanilla_pct: float
    capture_policy_cql_pct: float
    pass_strong: bool
    pass_app: bool
    marginal: bool
    verdict: str
    verdict_label: str


def run_policy_walkforward(
    *,
    top_k: int = 50,
    cql_weight: float = 1.0,
    empirical_inclusion_rate: float | None = None,
    hidden: int = 16,
    n_layers: int = 2,
    phi_steps: int = 400,
    policy_steps: int = 300,
    seed: int = 0,
    include_macro: bool = True,
    verbose: bool = False,
) -> PolicyResult:
    """LOO-by-window walk-forward of (Φ-train → π-train → deploy π)."""
    window_idxs, pair_feats, y, feat_names = load_pair_records_npz()
    val_starts_map = _val_starts_for_windows()
    val_starts = [val_starts_map.get(int(w), "") for w in window_idxs]

    # Same feature stack as v0.1 pair+macro variant by default
    if include_macro:
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

    # Median-impute any residual NaNs
    for col in range(X.shape[1]):
        nan_mask = ~np.isfinite(X[:, col])
        if nan_mask.any():
            med = np.nanmedian(X[~nan_mask, col]) if (~nan_mask).any() else 0.0
            X[nan_mask, col] = med

    n_features = X.shape[1]

    # Empirical inclusion rate: top_k / pairs_per_window. Used to anchor π_cql.
    n_per_window = int(np.bincount(window_idxs)[window_idxs[0]])
    if empirical_inclusion_rate is None:
        empirical_inclusion_rate = top_k / max(n_per_window, 1)

    unique_windows = sorted(set(window_idxs.tolist()))
    folds: list[PolicyFold] = []

    for fold_idx, w in enumerate(unique_windows):
        held_mask = window_idxs == w
        train_mask = ~held_mask

        # Standardize on train fold stats only
        mu = X[train_mask].mean(axis=0)
        sd = X[train_mask].std(axis=0) + 1e-8
        X_norm = (X - mu) / sd
        X_train = X_norm[train_mask]
        y_train = y[train_mask]
        X_held = X_norm[held_mask]
        y_held = y[held_mask]

        if verbose:
            print(f"[fold {fold_idx+1}/{len(unique_windows)}] w={w}  n_train={len(y_train)} n_held={len(y_held)}")

        # Stage 1: train Φ on train fold (hyperparams matched to v0.1
        # `pair_phi_quality_loo_npz` so the Φ-direct baseline reproduces
        # the v0.1 deployment numbers exactly).
        phi_res = train_phi(
            X_train, y_train,
            hidden=hidden, n_layers=n_layers,
            n_steps=phi_steps, learning_rate=5e-3,
            weight_decay=1e-3, seed=seed + fold_idx, verbose=False,
        )

        # Stage 2: train two policies against fixed Φ
        pol_vanilla = train_policy(
            X_train, phi_res.params,
            hidden=hidden, n_layers=n_layers,
            n_steps=policy_steps, learning_rate=5e-3,
            cql_weight=0.0, seed=seed + fold_idx + 100,
            verbose=False,
        )
        pol_cql = train_policy(
            X_train, phi_res.params,
            hidden=hidden, n_layers=n_layers,
            n_steps=policy_steps, learning_rate=5e-3,
            cql_weight=cql_weight,
            empirical_inclusion_rate=empirical_inclusion_rate,
            seed=seed + fold_idx + 200,
            verbose=False,
        )

        # Deploy on held-out fold
        phi_preds_held = predict_phi(phi_res.params, X_held)
        lr_preds_held = _linreg_fit_predict(X_train, y_train, X_held)
        policy_vanilla_score = policy_score(pol_vanilla.params, X_held)
        policy_cql_score = policy_score(pol_cql.params, X_held)

        topk_lr = _topn_window_sharpe(lr_preds_held, y_held, n=top_k)
        topk_phi = _topn_window_sharpe(phi_preds_held, y_held, n=top_k)
        topk_pol_v = _topn_window_sharpe(policy_vanilla_score, y_held, n=top_k)
        topk_pol_c = _topn_window_sharpe(policy_cql_score, y_held, n=top_k)
        topk_all = float(np.mean(y_held))
        topk_oracle = _topn_window_sharpe(y_held, y_held, n=top_k)

        # Spearman of policy scores vs Φ predictions (how closely the
        # policy mirrors Φ on this held-out fold)
        if len(y_held) >= 3:
            sp_v = float(spearmanr(policy_vanilla_score, phi_preds_held).statistic)
            sp_c = float(spearmanr(policy_cql_score, phi_preds_held).statistic)
        else:
            sp_v = sp_c = 0.0

        folds.append(PolicyFold(
            window_idx=w,
            val_start=val_starts[np.where(held_mask)[0][0]],
            n_held=int(held_mask.sum()),
            topk_sharpe_all_pairs=topk_all,
            topk_sharpe_lr=topk_lr,
            topk_sharpe_phi_direct=topk_phi,
            topk_sharpe_policy_vanilla=topk_pol_v,
            topk_sharpe_policy_cql=topk_pol_c,
            topk_sharpe_oracle=topk_oracle,
            spearman_policy_vanilla_vs_phi=sp_v,
            spearman_policy_cql_vs_phi=sp_c,
        ))

    # Aggregate across folds
    def m(field):
        return float(np.mean([getattr(f, field) for f in folds]))

    mean_topk_all = m("topk_sharpe_all_pairs")
    mean_topk_lr = m("topk_sharpe_lr")
    mean_topk_phi = m("topk_sharpe_phi_direct")
    mean_topk_pv = m("topk_sharpe_policy_vanilla")
    mean_topk_pc = m("topk_sharpe_policy_cql")
    mean_topk_oracle = m("topk_sharpe_oracle")

    oracle_headroom = max(mean_topk_oracle - mean_topk_all, 1e-9)

    capture_lr_pct = 100 * (mean_topk_lr - mean_topk_all) / oracle_headroom
    capture_phi_pct = 100 * (mean_topk_phi - mean_topk_all) / oracle_headroom
    capture_pv_pct = 100 * (mean_topk_pv - mean_topk_all) / oracle_headroom
    capture_pc_pct = 100 * (mean_topk_pc - mean_topk_all) / oracle_headroom

    # Pre-registered cuts: best policy lifts vs LR + Φ-direct
    best_policy = max(mean_topk_pv, mean_topk_pc)
    delta_vs_lr = best_policy - mean_topk_lr
    delta_vs_phi = best_policy - mean_topk_phi

    strong = delta_vs_lr >= 0.10 and delta_vs_phi >= 0.05
    app_pass = delta_vs_lr >= 0.03 and delta_vs_phi >= 0.01
    marginal = (not app_pass) and (delta_vs_lr >= 0.01)

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

    return PolicyResult(
        n_triples=int(X.shape[0]),
        n_features=n_features,
        top_k=top_k,
        cql_weight=cql_weight,
        empirical_inclusion_rate=empirical_inclusion_rate,
        folds=folds,
        mean_topk_all_pairs=mean_topk_all,
        mean_topk_lr=mean_topk_lr,
        mean_topk_phi_direct=mean_topk_phi,
        mean_topk_policy_vanilla=mean_topk_pv,
        mean_topk_policy_cql=mean_topk_pc,
        mean_topk_oracle=mean_topk_oracle,
        oracle_headroom=oracle_headroom,
        capture_lr_pct=capture_lr_pct,
        capture_phi_direct_pct=capture_phi_pct,
        capture_policy_vanilla_pct=capture_pv_pct,
        capture_policy_cql_pct=capture_pc_pct,
        pass_strong=strong,
        pass_app=app_pass,
        marginal=marginal,
        verdict=verdict,
        verdict_label=label,
    )


def result_to_dict(r: PolicyResult) -> dict:
    return {
        "n_triples": r.n_triples,
        "n_features": r.n_features,
        "top_k": r.top_k,
        "cql_weight": r.cql_weight,
        "empirical_inclusion_rate": r.empirical_inclusion_rate,
        "mean_topk": {
            "all_pairs":      r.mean_topk_all_pairs,
            "lr_baseline":    r.mean_topk_lr,
            "phi_direct":     r.mean_topk_phi_direct,
            "policy_vanilla": r.mean_topk_policy_vanilla,
            "policy_cql":     r.mean_topk_policy_cql,
            "oracle":         r.mean_topk_oracle,
        },
        "oracle_headroom": r.oracle_headroom,
        "capture_pct": {
            "lr_baseline":    r.capture_lr_pct,
            "phi_direct":     r.capture_phi_direct_pct,
            "policy_vanilla": r.capture_policy_vanilla_pct,
            "policy_cql":     r.capture_policy_cql_pct,
        },
        "verdict":       r.verdict,
        "verdict_label": r.verdict_label,
        "folds": [asdict(f) for f in r.folds],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--cql-weight", type=float, default=1.0)
    ap.add_argument("--phi-steps", type=int, default=400)
    ap.add_argument("--policy-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-macro", action="store_true")
    ap.add_argument("--out", type=str, default=str(OUTPUT_DIR / "critic-policy-walkforward-summary.json"))
    args = ap.parse_args()

    print(f"Running policy walk-forward (top_k={args.top_k}, cql_weight={args.cql_weight}, include_macro={not args.no_macro})")
    result = run_policy_walkforward(
        top_k=args.top_k,
        cql_weight=args.cql_weight,
        phi_steps=args.phi_steps,
        policy_steps=args.policy_steps,
        seed=args.seed,
        include_macro=(not args.no_macro),
        verbose=True,
    )

    print()
    print(f"=== Top-{result.top_k} deployment Sharpe (LOO-by-window, mean across 6 folds) ===")
    print(f"  all-pairs baseline:    {result.mean_topk_all_pairs:+.4f}")
    print(f"  LR (v1 analog):        {result.mean_topk_lr:+.4f}   ({result.capture_lr_pct:+.1f}% of oracle headroom)")
    print(f"  Φ direct (v0.1):       {result.mean_topk_phi_direct:+.4f}   ({result.capture_phi_direct_pct:+.1f}%)")
    print(f"  π vanilla (-Φ loss):   {result.mean_topk_policy_vanilla:+.4f}   ({result.capture_policy_vanilla_pct:+.1f}%)")
    print(f"  π CQL (anchor {result.empirical_inclusion_rate:.2f}): {result.mean_topk_policy_cql:+.4f}   ({result.capture_policy_cql_pct:+.1f}%)")
    print(f"  oracle (within-win):   {result.mean_topk_oracle:+.4f}   (100.0%)")
    print()
    print(f"  oracle headroom:       {result.oracle_headroom:+.4f}")
    print()
    print(f"  best policy:           {max(result.mean_topk_policy_vanilla, result.mean_topk_policy_cql):+.4f}")
    print(f"  best policy vs LR:     Δ {max(result.mean_topk_policy_vanilla, result.mean_topk_policy_cql) - result.mean_topk_lr:+.4f}")
    print(f"  best policy vs Φ:      Δ {max(result.mean_topk_policy_vanilla, result.mean_topk_policy_cql) - result.mean_topk_phi_direct:+.4f}")
    print()
    print(f"  VERDICT: {result.verdict} ({result.verdict_label})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_to_dict(result), indent=2))
    print(f"\nsummary written to {out_path}")


if __name__ == "__main__":
    main()
