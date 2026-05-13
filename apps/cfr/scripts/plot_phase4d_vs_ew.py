"""Plot CFR Phase 4d vs passive EW — answer the question 'why does EW win?'

Two panels:
  - top: per-window val Sharpe (the loss metric we use to score each arm)
    for CFR / passive EW / naive uniform mix / trailing-best-greedy
  - bottom: cumulative equity over the concatenated 5 val periods
    (2010-03 → 2025-08) for EW, naive uniform, and the best static
    individual menu mode. This shows that the menu is so flat across
    a multi-asset universe that no single member beats EW — which is
    the structural reason CFR's lift over EW is bounded at +0.056.

Outputs apps/docs/docs/findings/images/cfr-phase4d-vs-ew.png
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfr.baselines import (
    NaiveUniform, PassiveEW, _portfolio_simulate,
)
from cfr.menu import ActionMenu, EqualWeightMode, TopKMode


REPO = Path(__file__).resolve().parents[3]
PHASE4D_JSON = REPO / 'Output' / 'cfr-phase4d.json'
PHASE4D_PKL = REPO / 'Output' / 'cfr_phase4d_multiasset_close.pkl'
OUT_PNG = REPO / 'apps' / 'docs' / 'docs' / 'findings' / 'images' / 'cfr-phase4d-vs-ew.png'


def build_phase4d_menu(top_k: int = 4) -> ActionMenu:
    modes = [
        EqualWeightMode(name='ew'),
        TopKMode(name='mom', score_kind='momentum', score_window=21, top_k=top_k),
        TopKMode(name='rev', score_kind='reversal', score_window=5, top_k=top_k),
        TopKMode(name='lowv', score_kind='low_vol', score_window=21, top_k=top_k),
        TopKMode(name='highv', score_kind='high_vol', score_window=21, top_k=top_k),
        TopKMode(name='mom121', score_kind='mom_12_1', score_window=252,
                 top_k=top_k, min_lookback=252),
        TopKMode(name='lowv252', score_kind='low_vol', score_window=252,
                 top_k=top_k, min_lookback=252),
        TopKMode(name='shtop', score_kind='sharpe_top', score_window=252,
                 top_k=top_k, min_lookback=252),
        TopKMode(name='trend', score_kind='trend_str', score_window=252,
                 top_k=top_k, min_lookback=252),
    ]
    return ActionMenu(modes=modes, gross_levels=(0.0, 0.5, 1.0, 2.0))


def per_action_static_returns(
    val_prices: pd.DataFrame,
    action_weights_val: np.ndarray,
    action_idx: int,
    rebal_days: int = 20,
    min_warmup: int = 21,   # match PassiveEW.min_lookback for apples-to-apples
    commission_bps: float = 10.0,
) -> np.ndarray:
    """Run a single static menu action chronologically over val.

    The action's weight is recomputed at every rebal bar from the
    precomputed `action_weights_val[t, action_idx, :]`, with cash
    when the action is unavailable.
    """
    T, A, N = action_weights_val.shape
    rebal_indices = np.arange(min_warmup, T - rebal_days, rebal_days, dtype=np.int64)
    target = np.zeros((len(rebal_indices), N), dtype=np.float64)
    for k, t in enumerate(rebal_indices):
        target[k] = action_weights_val[int(t), action_idx]
    return _portfolio_simulate(
        val_prices, target, rebal_indices, commission_bps=commission_bps,
    )


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    with open(PHASE4D_JSON) as f:
        res = json.load(f)
    with open(PHASE4D_PKL, 'rb') as f:
        close: pd.DataFrame = pickle.load(f)

    rebal_days = res['config']['rebal_days']
    commission_bps = 10.0
    menu = build_phase4d_menu(top_k=res['config']['top_k'])
    action_weights, _action_avail = menu.precompute(close)

    # --- panel data -------------------------------------------------
    cfr_sh = np.array([w['cfr_sharpe'] for w in res['per_window']])
    ew_sh = np.array([w['passive_ew_sharpe'] for w in res['per_window']])
    naive_sh = np.array([w['naive_uniform_sharpe'] for w in res['per_window']])
    trail_sh = np.array([w['trailing_best_sharpe'] for w in res['per_window']])
    win_lbls = [f"w{w['window_idx']}\n{w['val_start']}\n→{w['val_end']}"
                for w in res['per_window']]

    # --- deterministic baselines + best static mode over val periods ---
    passive = PassiveEW(rebal_days=rebal_days, commission_bps=commission_bps)
    naive = NaiveUniform(rebal_days=rebal_days, commission_bps=commission_bps)

    ew_curves: list[pd.Series] = []
    naive_curves: list[pd.Series] = []
    static_curves: dict[int, list[pd.Series]] = {a: [] for a in range(menu.n_actions)}

    for w in res['per_window']:
        val_slice = close.loc[w['val_start']:w['val_end']]
        # find positional indices for the val slice into `action_weights`
        i0 = close.index.get_loc(val_slice.index[0])
        i1 = close.index.get_loc(val_slice.index[-1]) + 1
        aw_val = action_weights[i0:i1]

        ew_d = passive.daily_returns(val_slice)
        nu_d = naive.daily_returns(val_slice, aw_val)
        ew_curves.append(pd.Series(ew_d, index=val_slice.index))
        naive_curves.append(pd.Series(nu_d, index=val_slice.index))

        for a in range(menu.n_actions):
            d = per_action_static_returns(
                val_slice, aw_val, a, rebal_days=rebal_days,
                commission_bps=commission_bps,
            )
            static_curves[a].append(pd.Series(d, index=val_slice.index))

    ew_full = pd.concat(ew_curves)
    naive_full = pd.concat(naive_curves)

    static_full = {a: pd.concat(static_curves[a]) for a in range(menu.n_actions)}
    # rank static modes by mean per-window Sharpe (matches leaderboard
    # convention; concatenated-span Sharpe biases toward low-gross arms
    # whose vol is also lower).
    per_win_sh: dict[int, list[float]] = {a: [] for a in range(menu.n_actions)}
    for a in range(menu.n_actions):
        for s in static_curves[a]:
            sd = s.std(ddof=0)
            if sd > 0:
                per_win_sh[a].append(float(s.mean() / sd * np.sqrt(252)))
    static_mean_sh = {a: float(np.mean(v)) for a, v in per_win_sh.items() if v}
    # pick best mode at gross=1 only — leverage-invariant Sharpe means
    # gross levels tie; full-gross is the apples-to-apples comparison
    # against EW@g1.
    g1_keys = [a for a in static_mean_sh if menu.action_keys[a].endswith('@g1')]
    best_a = max(g1_keys, key=static_mean_sh.get)
    best_key = menu.action_keys[best_a]
    best_curve = static_full[best_a]
    best_per_win_mean = static_mean_sh[best_a]

    ew_per_win_mean = float(ew_sh.mean())
    print(f'best static mode @g1 (mean per-window Sharpe): {best_key}  '
          f'Sharpe={best_per_win_mean:+.3f}')
    print(f'EW@g1 mean per-window Sharpe: {ew_per_win_mean:+.3f}')
    print(f'CFR Phase 4d mean per-window Sharpe: {float(cfr_sh.mean()):+.3f}')
    print(f'best-static lift over EW: {best_per_win_mean - ew_per_win_mean:+.3f}')
    print(f'CFR lift over EW:         {float(cfr_sh.mean()) - ew_per_win_mean:+.3f}')

    # --- plot -------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [1, 1.4]}
    )

    # top: per-window val Sharpe lines
    x = np.arange(len(cfr_sh))
    ax0.plot(x, cfr_sh, '-o', label=f'CFR Phase 4d  (mean {cfr_sh.mean():+.3f})',
             color='#1f77b4', linewidth=2.2, markersize=8)
    ax0.plot(x, ew_sh, '-s', label=f'Passive EW   (mean {ew_sh.mean():+.3f})',
             color='#2ca02c', linewidth=2.2, markersize=7)
    ax0.plot(x, naive_sh, '-^', label=f'Naive uniform mix  (mean {naive_sh.mean():+.3f})',
             color='#ff7f0e', linewidth=1.6, markersize=6, alpha=0.85)
    ax0.plot(x, trail_sh, '-v', label=f'Trailing-best greedy  (mean {trail_sh.mean():+.3f})',
             color='#888', linewidth=1.2, markersize=5, alpha=0.7)
    ax0.axhline(0, color='k', linewidth=0.4, alpha=0.4)
    ax0.set_xticks(x)
    ax0.set_xticklabels(win_lbls, fontsize=8)
    ax0.set_ylabel('val Sharpe (annualized)')
    ax0.set_title(
        'Phase 4d val Sharpe per window — CFR vs EW vs naive '
        '(13-asset multi-asset universe)\n'
        f'mean alpha vs EW = {(cfr_sh - ew_sh).mean():+.3f}  '
        f'(positive in {int((cfr_sh > ew_sh).sum())}/{len(cfr_sh)} windows)',
        fontsize=11,
    )
    ax0.legend(loc='lower left', fontsize=9)
    ax0.grid(alpha=0.3)

    # bottom: cumulative equity over concatenated val periods
    eq_ew = (1 + ew_full).cumprod()
    eq_naive = (1 + naive_full).cumprod()
    eq_best = (1 + best_curve).cumprod()
    ax1.plot(eq_ew.index, eq_ew.values, label=f'Passive EW',
             color='#2ca02c', linewidth=2.2)
    ax1.plot(eq_naive.index, eq_naive.values,
             label='Naive uniform mix (28 menu actions)',
             color='#ff7f0e', linewidth=1.4, alpha=0.85)
    ax1.plot(eq_best.index, eq_best.values,
             label=(f'Best static menu mode @g1  ({best_key}, '
                    f'mean win-Sharpe {best_per_win_mean:+.3f})'),
             color='#9467bd', linewidth=1.4, alpha=0.85, linestyle='--')

    # mark val window boundaries
    for w in res['per_window']:
        d = pd.Timestamp(w['val_start'])
        if d in eq_ew.index or d > eq_ew.index[0]:
            ax1.axvline(d, color='k', linewidth=0.4, alpha=0.25, linestyle=':')

    ax1.set_yscale('log')
    ax1.set_ylabel('cumulative equity (log scale, $1 start)')
    ax1.set_xlabel('val period (concatenated 5 windows)')
    ax1.set_title(
        'Cumulative val-period equity — multi-asset universe (2010-03 → 2025-08)\n'
        f'Best static menu mode beats EW by only +{best_per_win_mean - ew_per_win_mean:+.3f} '
        f'Sharpe; CFR (a switcher over this menu) achieves '
        f'{float(cfr_sh.mean()) - ew_per_win_mean:+.3f} — same magnitude.',
        fontsize=11,
    )
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3, which='both')

    fig.suptitle(
        'CFR Phase 4d vs passive EW — why does EW win?',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=130)
    print(f'wrote {OUT_PNG.relative_to(REPO)}')


if __name__ == '__main__':
    main()
