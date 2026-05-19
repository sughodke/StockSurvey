"""Stage-0 oracle — does a per-regime rebalance cadence add edge over daily-EW?

Pre-registered kill gate (2026-05-18): if the hindsight per-window
best-cadence Sharpe beats fixed daily-EW by < +0.15 mean, the
dynamic-rebal-per-regime lever is `confirmed-null` — a deployable
"model chooses its own next wake time" selector cannot exceed its own
hindsight oracle, so a sub-threshold oracle closes the lever before any
Stage-1 deployable build.

Reuses the gate v0 EW substrate (stooq_us_long manifest, 6-window
1260/780/780 over 2000-2025) so the daily-EW (k=1) arm reproduces
`Output/gate-walkforward-summary.json` `unc_sharpe` per window — the
validation gate (max abs err < 0.06); without it no swept number is
trustworthy.

Local, ~2 min, no Modal. Reproducibility anchor for the
`findings/dynamic-rebal-cadence-null.md` finding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ss_loaders import load_stooq_matrix

TRADING_DAYS = 252
REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "apps" / "notebook" / "data" / "stooq_us_long" / "manifest.json"
GATE_SUMMARY = REPO / "Output" / "gate-walkforward-summary.json"
OUT = REPO / "Output" / "dynamic-rebal-cadence-oracle.json"
CADENCES = (1, 5, 20, 60)
KILL_GATE = 0.15
REPRO_TOL = 0.06


def ann_sharpe(r: np.ndarray) -> float:
    """ss_portfolio.annualized_sharpe convention: mean/std(ddof=0)*sqrt(252)."""
    r = np.asarray(r, float)
    sd = r.std(ddof=0)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def ew_periodic(R: np.ndarray, active: np.ndarray, k: int) -> np.ndarray:
    """EW reset to 1/N_active every k bars; weights drift with returns between."""
    T, N = R.shape
    w = np.zeros(N)
    port = np.zeros(T)
    for t in range(T):
        if t == 0 or t % k == 0:
            a = active[t]
            w = np.where(a, 1.0 / max(a.sum(), 1), 0.0)
        port[t] = float(w @ R[t])
        w = w * (1.0 + R[t])
        s = w.sum()
        if s > 0:
            w /= s
    return port


def ew_buy_hold(R: np.ndarray, active: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """EW set once at the start of [lo, hi) and held (drift only)."""
    a = active[lo]
    w = np.where(a, 1.0 / max(a.sum(), 1), 0.0)
    out = np.empty(hi - lo)
    for j, t in enumerate(range(lo, hi)):
        out[j] = w @ R[t]
        w = w * (1.0 + R[t])
        s = w.sum()
        if s > 0:
            w /= s
    return out


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    universe = sorted(t["ticker"].upper() for t in man["tickers"])
    prices, _, _, _ = load_stooq_matrix("./StooqData", min_history=150, tickers=universe)
    prices = prices.loc[prices.index >= pd.Timestamp("2000-01-01")]
    P = prices.values.astype(float)
    T, N = P.shape
    ret = np.full((T, N), np.nan)
    ret[1:] = P[1:] / P[:-1] - 1.0
    valid = np.isfinite(P[:-1]) & np.isfinite(P[1:]) & np.isfinite(ret[1:])
    R = np.zeros((T, N))
    R[1:] = np.where(valid, ret[1:], 0.0)
    active = np.isfinite(P)
    idx = prices.index

    series = {k: ew_periodic(R, active, k) for k in CADENCES}
    windows = json.loads(GATE_SUMMARY.read_text())["per_window"]

    per_window, base, oracle, artifact = [], [], [], []
    for w in windows:
        vs, ve = pd.Timestamp(w["val_start"]), pd.Timestamp(w["val_end"])
        m = (idx >= vs) & (idx <= ve)
        lo = int(np.argmax(m))
        hi = lo + int(m.sum())
        sh = {f"k{k}": ann_sharpe(series[k][lo:hi]) for k in CADENCES}
        sh["bh"] = ann_sharpe(ew_buy_hold(R, active, lo, hi))
        co = max(sh.values())
        per_window.append(
            {"window": w["window_idx"], "val_start": w["val_start"][:10],
             "val_end": w["val_end"][:10], "artifact_unc_sharpe": w["unc_sharpe"],
             **sh, "cadence_oracle": co})
        base.append(sh["k1"])
        oracle.append(co)
        artifact.append(w["unc_sharpe"])

    base, oracle, artifact = (np.array(base), np.array(oracle), np.array(artifact))
    repro_err = float(np.abs(base - artifact).max())
    headroom = float(oracle.mean() - base.mean())
    result = {
        "universe": "stooq_us_long",
        "windowing": "6-window 1260/780/780 daily 2000-2025 (gate v0 substrate)",
        "cadences": list(CADENCES) + ["buy_hold"],
        "per_window": per_window,
        "mean_daily_ew_k1": float(base.mean()),
        "mean_cadence_oracle": float(oracle.mean()),
        "cadence_oracle_headroom": headroom,
        "reproduction_max_err_vs_artifact": repro_err,
        "reproduction_pass": bool(repro_err < REPRO_TOL),
        "kill_gate": KILL_GATE,
        "verdict": "confirmed-null" if headroom < KILL_GATE else "PROCEED-to-Stage-1",
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nreconstruction max-err {repro_err:.3f} "
          f"({'PASS' if result['reproduction_pass'] else 'FAIL'}); "
          f"headroom {headroom:+.3f} vs kill gate {KILL_GATE} -> {result['verdict']}")


if __name__ == "__main__":
    main()
