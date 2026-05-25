"""Per-regime universe oracle probe (descriptive, in-sample).

For each ≥21-TD persistent regime at L=252 rolling Sharpe in
`Output/regimes-since-2005.json`, evaluate a curated set of ~20 candidate
universes (equal-weight, rebal_days=20, 10bps commission) restricted to
that regime's date range, and record the in-sample winner.

This is an oracle / future-peeking probe — it answers "what universe COULD
we have picked given perfect regime detection?" It does NOT define a
deployable causal selection rule (that's the meta-allocator brief, which
already showed causal rules fail). Read-only research artifact.

Deliverables:
  - Output/per-regime-universe-oracle.json
  - apps/docs/docs/findings/images/per-regime-universe-oracle.png
  - .research-per-regime-universe-oracle.md (separate)

Frame the writeup explicitly as an upper bound, not a strategy.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / "Output"
STOOQ_DIR = REPO_ROOT / "StooqData/daily/us"


# -----------------------------------------------------------------------------
# Ticker → file path map (manual: stooq's bucket layout is non-canonical)
# -----------------------------------------------------------------------------
ETF_PATHS = {
    "SPY":  STOOQ_DIR / "nyse etfs/2/spy.us.txt",
    "VTI":  STOOQ_DIR / "nyse etfs/2/vti.us.txt",
    "QQQ":  STOOQ_DIR / "nasdaq etfs/qqq.us.txt",
    "IWM":  STOOQ_DIR / "nyse etfs/1/iwm.us.txt",
    "EFA":  STOOQ_DIR / "nyse etfs/1/efa.us.txt",
    "TLT":  STOOQ_DIR / "nasdaq etfs/tlt.us.txt",
    "IEF":  STOOQ_DIR / "nasdaq etfs/ief.us.txt",
    "AGG":  STOOQ_DIR / "nyse etfs/1/agg.us.txt",
    "GLD":  STOOQ_DIR / "nyse etfs/1/gld.us.txt",
    "DBC":  STOOQ_DIR / "nyse etfs/1/dbc.us.txt",
    "XLC":  STOOQ_DIR / "nyse etfs/2/xlc.us.txt",
}
# The 9 SPDR sectors live in the phase4d pickle (XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY).
# We pull those from the pickle (continuous since 2005-02) and the rest from stooq.
SPDR_SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def load_stooq_close(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return pd.Series(df["close"].values, index=df["date"], name=path.stem).astype(float)


def build_close_panel() -> pd.DataFrame:
    import pickle
    with open(OUTPUT / "cfr_phase4d_multiasset_close.pkl", "rb") as f:
        p4d = pickle.load(f)
    cols: dict[str, pd.Series] = {}
    # SPDR sectors + bond + commodity from the canonical phase4d panel
    for c in p4d.columns:
        cols[c] = p4d[c].astype(float)
    # Additions from stooq
    for sym, path in ETF_PATHS.items():
        if sym in cols:
            continue
        if not path.exists():
            print(f"  WARN: missing {sym} at {path}")
            continue
        s = load_stooq_close(path)
        cols[sym] = s
    df = pd.DataFrame(cols)
    df = df.sort_index()
    return df


# -----------------------------------------------------------------------------
# Curated universes — name → list of tickers + family tag
# -----------------------------------------------------------------------------
UNIVERSES: list[tuple[str, list[str], str]] = [
    # (name, tickers, family)
    ("SPY-only",            ["SPY"],                                              "equity"),
    ("VTI-only",            ["VTI"],                                              "equity"),
    ("QQQ-only",            ["QQQ"],                                              "equity-tilt"),
    ("IWM-only",            ["IWM"],                                              "equity-tilt"),
    ("9-SPDR-sectors-EW",   SPDR_SECTORS,                                         "equity"),
    ("TLT-only",            ["TLT"],                                              "bonds"),
    ("IEF-only",            ["IEF"],                                              "bonds"),
    ("AGG-only",            ["AGG"],                                              "bonds"),
    ("GLD-only",            ["GLD"],                                              "commodities"),
    ("DBC-only",            ["DBC"],                                              "commodities"),
    ("canonical-13ETF",     SPDR_SECTORS + ["TLT", "IEF", "GLD", "DBC"],          "multi-asset"),
    ("winner-4ETF",         ["VTI", "TLT", "IEF", "GLD"],                         "multi-asset"),
    ("60-40-SPY-AGG",       ["SPY", "AGG"],                                       "multi-asset"),
    ("golden-butterfly",    ["VTI", "IWM", "TLT", "IEF", "GLD"],                  "multi-asset"),
    ("all-weather",         ["VTI", "TLT", "IEF", "GLD", "DBC"],                  "multi-asset"),
    ("risk-on-SPY-QQQ-EFA", ["SPY", "QQQ", "EFA"],                                "risk-on"),
    ("growth-XLK-XLY-XLC",  ["XLK", "XLY", "XLC"],                                "risk-on"),
    ("riskoff-bonds-gold",  ["TLT", "IEF", "GLD"],                                "risk-off"),
    ("defensives-XLP-XLU-XLV", ["XLP", "XLU", "XLV"],                             "defensive"),
]

FAMILY_COLORS = {
    "equity":       "#1f77b4",
    "equity-tilt":  "#17becf",
    "bonds":        "#2ca02c",
    "commodities":  "#bcbd22",
    "multi-asset":  "#9467bd",
    "risk-on":      "#d62728",
    "risk-off":     "#8c564b",
    "defensive":    "#7f7f7f",
}


# -----------------------------------------------------------------------------
# Portfolio engine — EW, rebal_days=20, 10bps commission
# -----------------------------------------------------------------------------
def passive_ew_daily_returns(prices: pd.DataFrame, rebal_days: int = 20,
                             commission_bps: float = 10.0,
                             min_lookback: int = 21) -> np.ndarray:
    """Equal-weight rebal_days=20 with commission on L1 turnover.

    Self-contained version that doesn't need cfr (the cfr PassiveEW
    interface is fine but takes prices over a fixed N — we want
    per-regime restricted N).
    """
    T, N = prices.shape
    ret = prices.pct_change(fill_method=None).values
    ret = np.where(np.isnan(ret), 0.0, ret)
    # Liquid mask: ticker is active if it has a non-NaN price *at this bar*
    liquid = (~prices.isna().values)
    target = np.zeros((T, N))
    for t in range(T):
        n_active = int(liquid[t].sum())
        if n_active > 0:
            target[t, liquid[t]] = 1.0 / n_active
    if T <= min_lookback:
        return np.zeros(T)
    rebal_idx = np.arange(min_lookback, T, rebal_days, dtype=int)
    if len(rebal_idx) == 0:
        return np.zeros(T)

    w = np.zeros(N)
    daily_ret = np.zeros(T)
    rb_set = set(rebal_idx.tolist())
    comm_rate = commission_bps / 10000.0
    for t in range(T):
        if t in rb_set:
            target_w = target[t]
            l1 = float(np.abs(target_w - w).sum())
            cost = l1 * comm_rate
            daily_ret[t] = -cost  # commission impact on rebal day
            w = target_w.copy()
        # apply this bar's returns
        port_r = float((w * ret[t]).sum())
        daily_ret[t] += port_r
        # drift
        w = w * (1.0 + ret[t])
    return daily_ret


def ann_sharpe(daily: np.ndarray, ppy: float = 252.0) -> float:
    if daily.size < 5:
        return float("nan")
    mu = float(np.mean(daily))
    sd = float(np.std(daily, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return mu / sd * math.sqrt(ppy)


# -----------------------------------------------------------------------------
# Macro classification helpers (mirror count_regimes_since_2005.py)
# -----------------------------------------------------------------------------
NBER_RECESSIONS = [
    (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30")),
    (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
]
FED_TIGHTENING = [
    (pd.Timestamp("2015-12-01"), pd.Timestamp("2018-12-31")),
    (pd.Timestamp("2022-03-01"), pd.Timestamp("2023-07-31")),
]


def regime_overlaps(start: pd.Timestamp, end: pd.Timestamp,
                    intervals) -> bool:
    return any(not (end < lo or start > hi) for lo, hi in intervals)


def vix_class(mean_vix: float) -> str:
    if math.isnan(mean_vix):
        return "unknown"
    if mean_vix < 15:
        return "low"
    if mean_vix < 25:
        return "mid"
    if mean_vix < 40:
        return "high"
    return "extreme"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("Loading regimes...")
    regimes_json = json.loads((OUTPUT / "regimes-since-2005.json").read_text())
    regs = [r for r in regimes_json["252"]["regimes"] if r["length_td"] >= 21]
    print(f"  Loaded {len(regs)} ≥21-TD regimes at L=252")

    print("\nBuilding close panel...")
    close = build_close_panel()
    print(f"  Panel: {close.index[0].date()} → {close.index[-1].date()}, "
          f"{close.shape[0]} bars × {close.shape[1]} symbols")
    print(f"  Symbols: {close.columns.tolist()}")

    # First-valid-date per ticker
    first_valid = {c: close[c].first_valid_index() for c in close.columns}
    for sym, fv in sorted(first_valid.items(), key=lambda kv: kv[1] or pd.Timestamp.max):
        print(f"    {sym:8s} first_valid={fv.date() if fv is not None else 'NONE'}")

    print("\nLoading VIX...")
    try:
        from ss_macro.loaders import load_fred_series
        vix = load_fred_series("VIXCLS")
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        vix = vix.dropna().astype(float)
        print(f"  VIX {vix.index[0].date()}→{vix.index[-1].date()}")
    except Exception as e:
        print(f"  VIX load failed ({e}); falling back to no-VIX classification")
        vix = None

    # ---- Per-regime evaluation ----
    rows = []
    for ri, r in enumerate(regs):
        start = pd.Timestamp(r["start"])
        end = pd.Timestamp(r["end"])
        seg = close.loc[start:end]
        if seg.empty or len(seg) < 5:
            continue
        # Per-universe Sharpe
        per_uni: list[tuple[str, str, float, int]] = []  # (name, family, sharpe, n_active)
        for (uname, tickers, family) in UNIVERSES:
            missing = [t for t in tickers if t not in seg.columns]
            if missing:
                continue
            sub = seg[tickers].copy()
            # require ALL tickers active by regime start (i.e., first_valid <= start)
            all_active = all(first_valid.get(t) is not None and first_valid[t] <= start
                             for t in tickers)
            if not all_active:
                continue
            daily = passive_ew_daily_returns(sub, rebal_days=20, commission_bps=10.0)
            sh = ann_sharpe(daily)
            per_uni.append((uname, family, sh, len(tickers)))
        if not per_uni:
            continue
        per_uni_sorted = sorted(per_uni, key=lambda x: -x[2] if not math.isnan(x[2]) else 1e9)
        best = per_uni_sorted[0]
        runner = per_uni_sorted[1] if len(per_uni_sorted) > 1 else (None, None, float("nan"), 0)
        # VIX class
        if vix is not None:
            v = vix.loc[(vix.index >= start) & (vix.index <= end)]
            mean_vix = float(v.mean()) if len(v) else float("nan")
        else:
            mean_vix = float("nan")
        rec = {
            "idx": ri,
            "start": str(start.date()),
            "end": str(end.date()),
            "length_td": r["length_td"],
            "original_strategy_winner": r["winner"],
            "mean_margin": r["mean_margin"],
            "best_universe": best[0],
            "best_family": best[1],
            "best_sharpe": float(best[2]),
            "runner_up_universe": runner[0],
            "runner_up_sharpe": float(runner[2]) if runner[0] else float("nan"),
            "margin_vs_runner_up": float(best[2] - runner[2]) if runner[0] else float("nan"),
            "mean_vix": mean_vix,
            "vix_class": vix_class(mean_vix),
            "overlaps_recession": regime_overlaps(start, end, NBER_RECESSIONS),
            "overlaps_fed_tightening": regime_overlaps(start, end, FED_TIGHTENING),
            "per_universe": [(n, f, float(s)) for (n, f, s, _) in per_uni_sorted],
        }
        rows.append(rec)

    # ---- Print table ----
    print("\n========== Per-regime winners (in-sample / oracle) ==========")
    hdr = (f"{'start':10s} {'end':10s} {'TD':>4s} {'orig':18s} {'margin':>6s} "
           f"{'best_universe':22s} {'family':12s} {'Sh':>6s} {'runner':22s} {'edge':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['start']:10s} {r['end']:10s} {r['length_td']:>4d} "
              f"{r['original_strategy_winner']:18s} {r['mean_margin']:+6.2f} "
              f"{r['best_universe']:22s} {r['best_family']:12s} "
              f"{r['best_sharpe']:+6.2f} "
              f"{(r['runner_up_universe'] or ''):22s} "
              f"{r['margin_vs_runner_up']:+6.2f}")

    # ---- Family share + cross-tab ----
    print("\n========== Family share (winners) ==========")
    fam_count = Counter(r["best_family"] for r in rows)
    for fam, n in fam_count.most_common():
        print(f"  {fam:14s} {n:>3d} regimes ({100*n/len(rows):.1f}%)")

    print("\n========== Cross-tab: VIX class × winning family ==========")
    crosstab: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        crosstab[(r["vix_class"], r["best_family"])] += 1
    fams = sorted({r["best_family"] for r in rows})
    vix_classes = ["low", "mid", "high", "extreme", "unknown"]
    print(f"  {'vix':10s} " + " ".join(f"{f:>14s}" for f in fams))
    for vc in vix_classes:
        rowstr = " ".join(f"{crosstab.get((vc, f), 0):>14d}" for f in fams)
        if any(crosstab.get((vc, f), 0) > 0 for f in fams):
            print(f"  {vc:10s} {rowstr}")

    print("\n========== Cross-tab: recession × winning family ==========")
    for which, intervals in [("recession", NBER_RECESSIONS),
                              ("fed-tightening", FED_TIGHTENING)]:
        in_pct = Counter(r["best_family"] for r in rows
                         if regime_overlaps(pd.Timestamp(r["start"]),
                                             pd.Timestamp(r["end"]), intervals))
        out_pct = Counter(r["best_family"] for r in rows
                          if not regime_overlaps(pd.Timestamp(r["start"]),
                                                  pd.Timestamp(r["end"]), intervals))
        print(f"  {which}:")
        print(f"    IN  ({sum(in_pct.values())} regimes): " +
              ", ".join(f"{f}={n}" for f, n in in_pct.most_common()))
        print(f"    OUT ({sum(out_pct.values())} regimes): " +
              ", ".join(f"{f}={n}" for f, n in out_pct.most_common()))

    # ---- Entropy of winner distribution per regime class ----
    print("\n========== Entropy of winner-family distribution per VIX class ==========")
    print(f"  (lower = one family dominates; higher = looks like noise; "
          f"max @ {len(fams)} families = {math.log2(len(fams)):.2f} bits)")
    entropies = {}
    for vc in vix_classes:
        cnt = Counter(r["best_family"] for r in rows if r["vix_class"] == vc)
        n = sum(cnt.values())
        if n == 0:
            continue
        H = -sum((c / n) * math.log2(c / n) for c in cnt.values() if c > 0)
        entropies[vc] = (H, n, dict(cnt))
        print(f"  {vc:10s} n={n:>3d}  H={H:.3f} bits   top: " +
              ", ".join(f"{f}({c})" for f, c in cnt.most_common(3)))

    # ---- Write JSON ----
    out_json = OUTPUT / "per-regime-universe-oracle.json"
    out_json.write_text(json.dumps({
        "universes": [{"name": n, "tickers": t, "family": f} for (n, t, f) in UNIVERSES],
        "n_regimes_evaluated": len(rows),
        "rows": rows,
        "family_counts": dict(fam_count),
        "vix_x_family_crosstab": {f"{k[0]}|{k[1]}": v for k, v in crosstab.items()},
        "entropies": {k: {"H_bits": v[0], "n": v[1], "counts": v[2]} for k, v in entropies.items()},
    }, indent=2, default=str))
    print(f"\n→ {out_json}")

    # ---- Chart: Gantt + heatmap ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                        gridspec_kw={"height_ratios": [2, 1]})

        # ---- Gantt (panel 1) ----
        for i, r in enumerate(rows):
            start = pd.Timestamp(r["start"])
            end = pd.Timestamp(r["end"])
            color = FAMILY_COLORS.get(r["best_family"], "#999999")
            ax1.add_patch(Rectangle((start, 0), end - start, 1,
                                      facecolor=color, edgecolor="white", linewidth=0.5))
            # Label only the longest regimes
            if r["length_td"] >= 80:
                ax1.text(start + (end - start) / 2, 0.5,
                          r["best_universe"], ha="center", va="center",
                          fontsize=7, rotation=0, color="white")
        ax1.set_xlim(pd.Timestamp("2005-01-01"), pd.Timestamp("2026-06-01"))
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_title("Per-regime in-sample winning universe (oracle / future-peeking; "
                       "colored by family)", fontsize=11)
        ax1.set_xlabel("date")
        # Recession bands
        for lo, hi in NBER_RECESSIONS:
            ax1.axvspan(lo, hi, alpha=0.15, color="red", zorder=-1)
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=fam) for fam, c in FAMILY_COLORS.items()
                    if fam in fam_count]
        ax1.legend(handles=handles, loc="upper left", ncol=4, fontsize=8,
                    frameon=False, bbox_to_anchor=(0.0, -0.12))

        # ---- Cross-tab heatmap (panel 2) ----
        present_fams = [f for f in fams if f in fam_count]
        present_vcs = [vc for vc in vix_classes
                        if any(crosstab.get((vc, f), 0) > 0 for f in present_fams)]
        mat = np.zeros((len(present_vcs), len(present_fams)))
        for i, vc in enumerate(present_vcs):
            for j, f in enumerate(present_fams):
                mat[i, j] = crosstab.get((vc, f), 0)
        # Row-normalize for the heatmap; annotate raw counts
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat_norm = mat / row_sums
        im = ax2.imshow(mat_norm, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax2.set_xticks(range(len(present_fams)))
        ax2.set_xticklabels(present_fams, rotation=25, ha="right", fontsize=9)
        ax2.set_yticks(range(len(present_vcs)))
        ax2.set_yticklabels([f"{vc} (n={int(mat[i].sum())})"
                              for i, vc in enumerate(present_vcs)], fontsize=9)
        ax2.set_title("VIX class × winning family — row-normalized share "
                       "(annotated with raw counts)", fontsize=11)
        for i in range(len(present_vcs)):
            for j in range(len(present_fams)):
                v = int(mat[i, j])
                if v > 0:
                    ax2.text(j, i, str(v), ha="center", va="center",
                              fontsize=9, color="black" if mat_norm[i, j] < 0.5 else "white")
        plt.colorbar(im, ax=ax2, label="share within VIX class")

        plt.tight_layout()
        out_png = REPO_ROOT / "apps/docs/docs/findings/images/per-regime-universe-oracle.png"
        plt.savefig(out_png, dpi=130, bbox_inches="tight")
        print(f"→ {out_png}")
    except Exception as e:
        print(f"Chart failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
