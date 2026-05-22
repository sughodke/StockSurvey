"""A2 — continuous VIX-percentile sigmoid meta-gate vs the binary v1b gate.

macro v1b applied a BINARY window-level gate (suspend a pivot-app window if
VIX < 1y rolling median, else take full alpha). The macro-regime diagnostic's
relationship was graduated (Pearson r +0.41 for VIX), so a binary cut discards
magnitude and mis-handles windows near the threshold. A2 replaces the cliff
with a sigmoid on the trailing VIX percentile: exposure = sigmoid((pct-0.5)*slope),
which → the binary gate as slope→∞. Same information, smoother transfer.

Post-processing of the gate/pairs/vol per-window v0 alphas (n=17) — no
retraining. Pre-registered: continuous beats binary (higher mean alpha AND
the relationship holds), else the graduated-gate hypothesis is null at this
(small) sample.

    uv run python apps/gate/scripts/macro_meta_gate_continuous.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ss_macro import load_macro_panel

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'Output'
SOURCES = [
    ('gate',  'gate-walkforward-summary.json',  'alpha_sharpe'),
    ('pairs', 'pairs-walkforward-summary.json', 'val_sharpe'),
    ('vol',   'vol-walkforward-summary.json',   'alpha_sharpe_per_cell'),
]


def _load_windows() -> pd.DataFrame:
    rows = []
    for app, fn, key in SOURCES:
        data = json.loads((OUT / fn).read_text())
        for r in data['per_window']:
            if key in r and 'val_start' in r:
                rows.append({'app': app, 'val_start': pd.Timestamp(r['val_start']),
                             'alpha': float(r[key])})
    return pd.DataFrame(rows)


def main() -> None:
    df = _load_windows().sort_values('val_start').reset_index(drop=True)
    print(f'loaded {len(df)} pivot-app windows ({df.app.value_counts().to_dict()})')

    vix = load_macro_panel()['vix'].dropna().sort_index()
    roll = 252

    def vix_pct(d: pd.Timestamp) -> float:
        w = vix.loc[vix.index <= d].iloc[-roll:]
        if len(w) < 60:
            return 0.5
        return float((w <= w.iloc[-1]).mean())  # percentile of spot in trailing window

    df['vix_pct'] = df['val_start'].apply(vix_pct)
    a = df['alpha'].values
    pct = df['vix_pct'].values

    def report(label, exposure):
        ga = exposure * a
        print(f'  {label:24s} mean alpha {ga.mean():+.4f}  '
              f'pos {int((ga > 0).sum())}/{len(ga)}  '
              f'(avg exposure {np.mean(exposure):.2f})')
        return float(ga.mean()), int((ga > 0).sum())

    print('\narm                        mean-alpha   pos-windows')
    res = {}
    res['ungated'] = report('ungated', np.ones_like(a))
    res['binary_v1b'] = report('binary (pct>=0.5)', (pct >= 0.5).astype(float))
    for slope in (5, 10, 20):
        exp = 1.0 / (1.0 + np.exp(-(pct - 0.5) * slope))
        res[f'sigmoid_slope{slope}'] = report(f'continuous (slope={slope})', exp)

    binary_a = res['binary_v1b'][0]
    best_cont = max(res[f'sigmoid_slope{s}'][0] for s in (5, 10, 20))
    lift = best_cont - binary_a
    verdict = ('PASS — continuous beats binary' if lift > 0.02 else
               'NULL — continuous does not beat binary')
    print(f'\nbest continuous − binary = {lift:+.4f}  → {verdict}')
    print(f'(ungated {res["ungated"][0]:+.4f}, binary {binary_a:+.4f})')

    (OUT / 'macro-meta-gate-continuous.json').write_text(json.dumps({
        'n_windows': len(df), 'results': {k: {'mean_alpha': v[0], 'pos': v[1]}
                                          for k, v in res.items()},
        'best_continuous_minus_binary': lift, 'verdict': verdict,
    }, indent=2))
    print(f'-> {OUT / "macro-meta-gate-continuous.json"}')


if __name__ == '__main__':
    main()
