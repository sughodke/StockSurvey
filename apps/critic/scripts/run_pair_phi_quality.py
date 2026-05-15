"""v0.1 rescue: pair-level Φ for pairs.

Consumes the per-pair feature dump from
`apps/pairs/scripts/run_pair_predictor_walkforward.py` (1200 records:
6 windows × 200 backtested pairs, 7 train-window features + optional
5 macro features at val_start).

Pre-registered cuts (laid out in pairs_eval.py):
  PASS:        RMSE_Φ / RMSE_LR ≤ 0.85 AND overall Spearman r ≥ +0.20
  STRONG-PASS: RMSE_Φ / RMSE_LR ≤ 0.70 AND r ≥ +0.30 AND mean top-N Sharpe (Φ) > (LR) + 0.5
  MARGINAL:    RMSE ratio ≤ 0.95 AND r > 0
  FAIL:        otherwise

Top-N selection Sharpe is computed at top_n=50 (matching pairs v1's
predictor-top-50 deployment baseline).

Usage:
    uv run python apps/critic/scripts/run_pair_phi_quality.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from critic.pairs_eval import pair_phi_quality_loo_npz, result_to_dict


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "Output"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-macro", action="store_true",
                    help="Drop the 5 macro features — pair-only feature set (apples to v1 LR)")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUTPUT_DIR / "critic-pair-phi-quality-summary.json"))
    args = ap.parse_args()

    print(f"Running pair-level Φ LOO (include_macro={not args.no_macro}, top_n={args.top_n})")
    result = pair_phi_quality_loo_npz(
        hidden=args.hidden,
        n_layers=args.n_layers,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        top_n=args.top_n,
        include_macro=(not args.no_macro),
        verbose=True,
    )

    print()
    print("=== Pair-level Φ-quality summary ===")
    print(f"  n_triples: {result.n_triples}")
    print(f"  n_features: {result.n_features}")
    print(f"  RMSE_Φ: {result.overall_rmse_phi:.4f}")
    print(f"  RMSE_LR: {result.overall_rmse_lr:.4f}")
    print(f"  RMSE_window-mean: {result.overall_rmse_window_mean:.4f}")
    print(f"  rel (Φ/LR): {result.relative_rmse_vs_lr:.4f}")
    print(f"  Spearman r (Φ): {result.overall_spearman_phi:+.4f}")
    print(f"  Spearman r (LR): {result.overall_spearman_lr:+.4f}")
    print()
    print(f"  Top-{result.top_n} deployment proxy (mean realized Sharpe across windows):")
    print(f"    Φ:           {result.mean_topn_sharpe_phi:+.4f}")
    print(f"    LR baseline: {result.mean_topn_sharpe_lr:+.4f}")
    print(f"    Oracle:      {result.mean_topn_sharpe_oracle:+.4f}")
    print(f"    All pairs:   {result.mean_topn_sharpe_all_pairs:+.4f}")
    print()
    print(f"  VERDICT: {result.verdict} ({result.verdict_label})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_to_dict(result), indent=2))
    print(f"\nsummary written to {out_path}")


if __name__ == "__main__":
    main()
