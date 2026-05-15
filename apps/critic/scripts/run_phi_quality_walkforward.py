"""Day-1 Φ-quality eval.

Runs leave-one-(app, window)-out CV on the consolidated cross-app
walk-forward triples, emits Output/critic-phi-quality-summary.json.

Usage:
    uv run python apps/critic/scripts/run_phi_quality_walkforward.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from critic.dataset import load_triples
from critic.eval import phi_quality_loo, result_to_dict


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "Output"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-pair-level", action="store_true",
                    help="Include the ~300 pair-level pairs triples")
    ap.add_argument("--drop-oracle-actions", action="store_true",
                    help="Drop oracle arms (clean Φ-quality test, not trivially gameable by action one-hot)")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUTPUT_DIR / "critic-phi-quality-summary.json"))
    args = ap.parse_args()

    triples = load_triples(
        include_pair_level=args.include_pair_level,
        drop_oracle_actions=args.drop_oracle_actions,
    )
    print(
        f"loaded {len(triples)} triples "
        f"(include_pair_level={args.include_pair_level}, "
        f"drop_oracle_actions={args.drop_oracle_actions})"
    )

    by_app = {}
    for t in triples:
        by_app.setdefault(t.app, []).append(t)
    for app, lst in sorted(by_app.items()):
        n_actions = len({t.action_key for t in lst})
        n_windows = len({t.window_idx for t in lst})
        print(f"  {app}: {len(lst):4d} triples ({n_actions} actions × {n_windows} windows)")

    print()
    print("Running LOO-by-(app, window)...")
    result = phi_quality_loo(
        triples,
        hidden=args.hidden,
        n_layers=args.n_layers,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        verbose=True,
    )

    print()
    print(f"=== Per-app Φ-quality ===")
    print(f"{'app':<10}{'n_folds':>10}{'n_eval':>10}{'RMSE_Φ':>12}{'RMSE_pa':>12}{'rel':>8}{'r':>8}{'verdict':>14}")
    for q in result.per_app:
        print(
            f"{q.app:<10}{q.n_folds:>10}{q.n_triples_eval:>10}"
            f"{q.rmse_phi:>12.4f}{q.rmse_per_action_mean:>12.4f}"
            f"{q.relative_rmse_vs_per_action:>8.3f}{q.spearman_r_phi:>8.3f}"
            f"{q.verdict:>14}"
        )

    print()
    print(f"OVERALL VERDICT: {result.overall_verdict} ({result.overall_verdict_label})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_to_dict(result), indent=2))
    print(f"summary written to {out_path}")


if __name__ == "__main__":
    main()
