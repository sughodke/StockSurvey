"""Triple consolidator.

Reads existing walk-forward outputs from `Output/` and produces a unified
list of `(app, window_idx, val_start_date, action_key, realized_sharpe)`
tuples for training Φ(state, action) → predicted-deployment-Sharpe.

The state-feature builder lives in `features.py`; this module concerns
itself only with extracting raw triples + their identifying metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "Output"


@dataclass(frozen=True)
class Triple:
    app: str
    window_idx: int
    val_start: str
    val_end: str
    action_key: str
    realized_sharpe: float
    # Optional pair-level identifier (None for window-level triples).
    pair_id: str | None = None


def _factor_triples(output_dir: Path) -> list[Triple]:
    """Factor walk-forward triples.

    Three action vocabularies stacked:
    - entropy-mixture arms: factor:mixture:a{0, 0.05, 0.1, 0.2, 0.3}
    - fixed-horizon baselines: factor:fixed:h{5, 10, 20, 40, 60}
    - regime-gated arms (if present): factor:regime:*

    All arms share the same six walk-forward windows.
    """
    triples: list[Triple] = []

    sweep = output_dir / "horizon-mixture-sweep-summary.json"
    if sweep.exists():
        data = json.loads(sweep.read_text())
        # Window starts come from the windows npz (mixture has same windows)
        npz = output_dir / "horizon-mixture-windows.npz"
        starts = np.load(npz)["val_start_date"].tolist() if npz.exists() else [""] * 6
        for arm in data["arms"]:
            alpha = arm["entropy_weight"]
            for w_idx, sharpe in enumerate(arm["per_window_endog_sharpe"]):
                triples.append(
                    Triple(
                        app="factor",
                        window_idx=w_idx,
                        val_start=str(starts[w_idx]) if w_idx < len(starts) else "",
                        val_end="",
                        action_key=f"factor:mixture:a{alpha}",
                        realized_sharpe=float(sharpe),
                    )
                )

    # Fixed-horizon baselines live in the per-window npz alongside the mixture.
    npz_path = output_dir / "horizon-mixture-windows.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        starts = d["val_start_date"].tolist()
        for horizon in (5, 10, 20, 40, 60):
            key = f"val_fixed_sharpe_h{horizon}"
            if key not in d.files:
                continue
            vals = d[key]
            for w_idx, sharpe in enumerate(vals):
                triples.append(
                    Triple(
                        app="factor",
                        window_idx=w_idx,
                        val_start=str(starts[w_idx]),
                        val_end="",
                        action_key=f"factor:fixed:h{horizon}",
                        realized_sharpe=float(sharpe),
                    )
                )

    return triples


def _vol_triples(output_dir: Path) -> list[Triple]:
    """Vol v3 regime-gated walk-forward triples.

    Note: the v3.1 composite summary is intentionally NOT consumed here —
    it only has per-arm pooled-across-windows Sharpe (no per-window split),
    which would mean every (window, arm) triple in a given arm shares the
    same label. That zero-variance-within-arm violates LOO-by-window.
    """
    triples: list[Triple] = []

    v3 = output_dir / "vol-walkforward-v3-regime-gated-summary.json"
    if v3.exists():
        data = json.loads(v3.read_text())
        for lookback_str, lookback_data in data["summary_by_lookback"].items():
            for win in lookback_data["per_window"]:
                triples.append(
                    Triple(
                        app="vol",
                        window_idx=win["window_idx"],
                        val_start=win["val_start"],
                        val_end=win["val_end"],
                        action_key=f"vol:v3:lookback-{lookback_str}",
                        realized_sharpe=float(win["fired_alpha_sharpe"]),
                    )
                )

    return triples


def _gate_triples(output_dir: Path) -> list[Triple]:
    """Gate v0 walk-forward triples (unc + gated + oracle-DD + oracle-day)."""
    triples: list[Triple] = []

    path = output_dir / "gate-walkforward-summary.json"
    if not path.exists():
        return triples

    data = json.loads(path.read_text())
    for win in data["per_window"]:
        w_idx = win["window_idx"]
        val_start = win["val_start"]
        val_end = win["val_end"]
        for action_suffix, key in (
            ("unconditional", "unc_sharpe"),
            ("gated", "gated_sharpe"),
            ("oracle-dd", "oracle_dd_sharpe"),
            ("oracle-day", "oracle_day_sharpe"),
        ):
            if key not in win:
                continue
            triples.append(
                Triple(
                    app="gate",
                    window_idx=w_idx,
                    val_start=val_start,
                    val_end=val_end,
                    action_key=f"gate:{action_suffix}",
                    realized_sharpe=float(win[key]),
                )
            )

    return triples


def _pairs_window_triples(output_dir: Path) -> list[Triple]:
    """Pairs window-level triples (all-pairs + predictor + oracle arms)."""
    triples: list[Triple] = []

    # Predictor walk-forward has the most arms (v0 + v1 predictor + oracle).
    pred = output_dir / "pairs-predictor-walkforward-summary.json"
    if not pred.exists():
        return triples
    data = json.loads(pred.read_text())

    # We need val_start / val_end per window; pull from the all-pairs arm's
    # window order, then look them up from the pairs-walkforward-summary v0
    # which has val_start / val_end.
    v0 = output_dir / "pairs-walkforward-summary.json"
    starts: list[str] = []
    ends: list[str] = []
    if v0.exists():
        v0_data = json.loads(v0.read_text())
        for w in v0_data["per_window"]:
            starts.append(w["val_start"])
            ends.append(w["val_end"])

    for arm_name, arm in data["arms"].items():
        for w_idx, sharpe in enumerate(arm["per_window"]):
            triples.append(
                Triple(
                    app="pairs",
                    window_idx=w_idx,
                    val_start=starts[w_idx] if w_idx < len(starts) else "",
                    val_end=ends[w_idx] if w_idx < len(ends) else "",
                    action_key=f"pairs:{arm_name}",
                    realized_sharpe=float(sharpe),
                )
            )

    return triples


def _pairs_pair_level_triples(output_dir: Path) -> list[Triple]:
    """Pair-level triples — the richer pairs dataset (~300 triples).

    Each pair's realized Sharpe in the v0 walk-forward becomes the label;
    the "action" is binary "include in top-50 portfolio" (encoded as
    pair:include / pair:skip depending on whether it would have been
    selected by the per-window argmax rule). Used by `apps/critic` when
    the pair-level feature view is available.
    """
    triples: list[Triple] = []

    path = output_dir / "pairs-walkforward-summary.json"
    if not path.exists():
        return triples

    data = json.loads(path.read_text())
    for win in data["per_window"]:
        w_idx = win["window_idx"]
        val_start = win["val_start"]
        val_end = win["val_end"]
        for pair in win["pairs"]:
            pair_id = f"{pair['a']}:{pair['b']}"
            triples.append(
                Triple(
                    app="pairs",
                    window_idx=w_idx,
                    val_start=val_start,
                    val_end=val_end,
                    action_key="pairs:include",
                    realized_sharpe=float(pair["sharpe"]),
                    pair_id=pair_id,
                )
            )

    return triples


def load_triples(
    output_dir: Path | None = None,
    include_pair_level: bool = False,
    drop_oracle_actions: bool = False,
) -> list[Triple]:
    """Load consolidated triples from the on-disk walk-forward outputs.

    `include_pair_level=True` adds the per-pair triples for pairs (the
    richer ~300-sample dataset). Default is window-level only so that
    cross-app pooling has comparable granularity.

    `drop_oracle_actions=True` drops arms whose action_key contains
    "oracle" — those are hindsight-cheating arms with realized-data
    leakage. Including them in Φ's training data lets the model
    trivially learn the oracle/non-oracle binary via the action one-hot
    (high spearman but no real state-conditional ranking). Drop for the
    honest Φ-quality test.
    """
    output_dir = output_dir if output_dir is not None else OUTPUT_DIR

    triples = []
    triples.extend(_factor_triples(output_dir))
    triples.extend(_vol_triples(output_dir))
    triples.extend(_gate_triples(output_dir))
    triples.extend(_pairs_window_triples(output_dir))
    if include_pair_level:
        triples.extend(_pairs_pair_level_triples(output_dir))

    if drop_oracle_actions:
        triples = [t for t in triples if "oracle" not in t.action_key]

    triples = [t for t in triples if np.isfinite(t.realized_sharpe)]
    return triples


def app_action_vocab(triples: Sequence[Triple]) -> dict[str, list[str]]:
    """Map app → sorted list of distinct action_keys observed in triples."""
    by_app: dict[str, set[str]] = {}
    for t in triples:
        by_app.setdefault(t.app, set()).add(t.action_key)
    return {a: sorted(actions) for a, actions in sorted(by_app.items())}
