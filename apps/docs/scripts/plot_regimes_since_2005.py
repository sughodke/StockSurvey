"""Chart the strategy-winning regimes since 2005 at the canonical
L=252 rolling-Sharpe window.

Consumes Output/regimes-since-2005.json (produced by
count_regimes_since_2005.py) and produces a Gantt-style chart of which
strategy was the rolling-Sharpe winner over time, plus a secondary
chart showing the lead margin (winner minus runner-up Sharpe).

Run from repo root:
    uv run python apps/docs/scripts/plot_regimes_since_2005.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
IN = REPO / 'Output' / 'regimes-since-2005.json'
OUT_DIR = REPO / 'apps' / 'docs' / 'docs' / 'findings' / 'images'

# Distinct, color-blind-aware palette for arc labels
ARC_COLORS = {
    'dca':              '#1f77b4',  # blue — canonical 13-ETF DCA
    'dca_winner_4etf':  '#0d3b66',  # dark blue — VTI+TLT+IEF+GLD
    'gate':             '#2ca02c',  # green
    'pairs':            '#9467bd',  # purple
    'relational':       '#ff7f0e',  # orange
    'vol_v3':           '#d62728',  # red — short-vol regime
    'regime_vol_target':'#8c564b',  # brown — regime app vol-target arm
    'regime_dd_gate':   '#e377c2',  # pink — regime app dd-gate arm
}


def _gantt(ax, regimes: list[dict], min_persistence: int = 21) -> None:
    """Gantt chart of regime windows: one horizontal bar per regime
    colored by winner. Only regimes with persistence >= min_persistence
    are drawn (filters the noise flicker)."""
    keep = [r for r in regimes if r['length_td'] >= min_persistence]
    # Stack vertically by arc so each arc gets its own row
    arcs = sorted({r['winner'] for r in keep})
    arc_y = {a: i for i, a in enumerate(arcs)}

    for r in keep:
        start = pd.Timestamp(r['start'])
        end = pd.Timestamp(r['end'])
        y = arc_y[r['winner']]
        width = (end - start).days
        margin = r['mean_margin']
        # Color saturation scales with lead margin (capped at 2.0)
        alpha = 0.35 + min(abs(margin), 2.0) / 2.0 * 0.55
        ax.barh(y, width, left=start, height=0.7,
                color=ARC_COLORS.get(r['winner'], '#999'),
                alpha=alpha, edgecolor='black', linewidth=0.4)

    ax.set_yticks(range(len(arcs)))
    ax.set_yticklabels(arcs, fontsize=10)
    ax.set_xlim(pd.Timestamp('2005-01-01'), pd.Timestamp('2026-01-01'))
    ax.grid(True, axis='x', alpha=0.3)

    # Shade recession + Fed-tightening bands for macro context
    for start, end, label in [
            ('2007-12-01', '2009-06-30', 'GFC'),
            ('2020-02-01', '2020-04-30', 'COVID'),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color='red', alpha=0.10)
    for start, end in [
            ('2015-12-01', '2018-12-31'),
            ('2022-03-01', '2023-07-31'),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color='gray', alpha=0.10)

    ax.set_title(
        f'Strategy-winning regimes since 2005 (L=252 rolling Sharpe; '
        f'≥{min_persistence}TD persistence filter)\n'
        f'Bar saturation ∝ lead margin over runner-up · red bands = '
        f'recessions · gray bands = Fed-tightening cycles',
        fontsize=11)


def _time_share_pie(ax, regimes: list[dict],
                    min_persistence: int = 21) -> None:
    """Time-weighted share of trading days each arc spent as winner."""
    keep = [r for r in regimes if r['length_td'] >= min_persistence]
    if not keep:
        ax.text(0.5, 0.5, 'no regimes', ha='center', va='center')
        return
    by_arc = {}
    for r in keep:
        by_arc.setdefault(r['winner'], 0)
        by_arc[r['winner']] += r['length_td']
    total = sum(by_arc.values())
    labels = sorted(by_arc, key=lambda k: -by_arc[k])
    sizes = [by_arc[a] for a in labels]
    colors = [ARC_COLORS.get(a, '#999') for a in labels]
    wedges, _, autotexts = ax.pie(
        sizes, labels=[f'{a}\n{100*s/total:.1f}%'
                       for a, s in zip(labels, sizes)],
        colors=colors, autopct='', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    )
    ax.set_title(f'Time-weighted winner share\n'
                 f'({sum(sizes)} TD, ≥{min_persistence}TD-persistent regimes)',
                 fontsize=10)


def _lead_margin_hist(ax, regimes: list[dict],
                      min_persistence: int = 21) -> None:
    """Distribution of lead margins (winner minus runner-up Sharpe).
    Shows how 'decisive' each regime's winner was."""
    keep = [r for r in regimes if r['length_td'] >= min_persistence]
    if not keep:
        return
    margins = [r['mean_margin'] for r in keep]
    # Group bars by arc
    by_arc: dict[str, list[float]] = {}
    for r in keep:
        by_arc.setdefault(r['winner'], []).append(r['mean_margin'])
    arcs = sorted(by_arc, key=lambda k: -np.mean(by_arc[k]))

    bins = np.linspace(0, max(margins + [1.0]) * 1.05, 15)
    bottom = np.zeros(len(bins) - 1)
    for arc in arcs:
        counts, _ = np.histogram(by_arc[arc], bins=bins)
        ax.bar(bins[:-1], counts, width=np.diff(bins),
               bottom=bottom, color=ARC_COLORS.get(arc, '#999'),
               alpha=0.85, label=arc, edgecolor='white', linewidth=0.5,
               align='edge')
        bottom += counts
    ax.set_xlabel('Mean lead-margin over runner-up (Sharpe pts)')
    ax.set_ylabel('# regimes')
    ax.set_title(f'Lead-margin distribution (≥{min_persistence}TD-persistent '
                 f'regimes only)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)


def main() -> None:
    if not IN.exists():
        raise FileNotFoundError(f'{IN} not found — '
                                f'run count_regimes_since_2005.py first')
    data = json.loads(IN.read_text())
    # L=252 is the canonical default
    Lkey = '252'
    if Lkey not in data:
        # Convert numeric or pick first key
        Lkey = next(iter(data.keys()))
    regimes = data[Lkey]['regimes']
    print(f'Loaded {len(regimes)} regimes at L={Lkey}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.1],
                          hspace=0.35, wspace=0.25)
    ax_gantt = fig.add_subplot(gs[0, :])
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    _gantt(ax_gantt, regimes, min_persistence=21)
    _time_share_pie(ax_pie, regimes, min_persistence=21)
    _lead_margin_hist(ax_hist, regimes, min_persistence=21)

    out = OUT_DIR / 'regimes-since-2005-L252.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
