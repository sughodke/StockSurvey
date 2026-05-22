"""Regime-scaled DCA — can the aggregate regime signal monetize the live basket?

The repo's two confirmed truths: (1) the only deployable strategy is passive
DCA (DSR deflated-t +2.07, the #1 arc); (2) the only repeatable predictability
is aggregate-level regime (gate forward-drawdown Pearson +0.26, vol/macro
regime gates). This tests whether (2) upgrades (1): scale the DCA basket's
daily exposure by predicted aggregate risk, de-risking before stress, and ask
whether the result clears passive DCA's deflated-t *net of the turnover the
overlay adds*.

Three arms on the 13-ETF Phase-4d basket (base = PassiveEW rebal_days=80, the
DCA stream already on the ladder), all costed at 10 bps on overlay turnover:

  * passive      — exposure ≡ 1 (baseline; reproduces the DCA row).
  * vol-target   — exposure_t = clip(target / trailing_vol_{t-1}); canonical
                   Moreira-Muir vol-management. No training, no look-ahead
                   (Sharpe is scale-invariant, so `target` only sets average
                   leverage). DCA's low turnover is where this can survive cost.
  * dd-gate      — walk-forward OLS (the gate's +0.26-Pearson predictor) on the
                   basket's aggregate features → flat/scaled exposure when
                   predicted forward 20d max-drawdown is in the train top
                   quantile. The literal "monetize the gate signal on DCA" test.

Decision: each arm's deflated-t vs passive's, plus alpha on the matched span.

    uv run python apps/dca/scripts/regime_scaled_dca.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cfr.baselines import PassiveEW
from gate import (
    apply_gate, build_aggregate_features, build_ew_aggregate,
    forward_max_drawdown, predict, train_predictor,
)
from ss_portfolio import standardize_oos

REPO = Path(__file__).resolve().parents[3]
CLOSE_PKL = REPO / 'Output' / 'cfr_phase4d_multiasset_close.pkl'
OUT_DIR = REPO / 'Output'
PPY = 252.0
N_TRIALS = 6  # regime-gating lever has been tried many times — deflate conservatively


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=0)
    return float(r.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def _turnover_cost(exposure: np.ndarray, commission_bps: float) -> np.ndarray:
    """Per-day cost from changing basket exposure (10bps on |Δexposure|)."""
    dexp = np.abs(np.diff(exposure, prepend=exposure[0]))
    return (commission_bps / 1e4) * dexp


def vol_target_exposure(base: np.ndarray, *, win: int = 60, cap: float = 2.0) -> np.ndarray:
    """exposure_t = target / trailing_vol_{t-1}, lagged, clipped, mean≈1."""
    s = pd.Series(base)
    trail = s.rolling(win).std().shift(1)              # uses info through t-1
    raw = 1.0 / trail.replace(0.0, np.nan)
    raw = raw.fillna(0.0).to_numpy()
    # Scale so average (over active days) exposure ≈ 1; Sharpe is invariant to
    # this constant, it only sets leverage. Then clip to [0, cap].
    active = raw[raw > 0]
    scale = 1.0 / active.mean() if active.size else 1.0
    return np.clip(raw * scale, 0.0, cap)


def dd_gate_exposure(
    close: pd.DataFrame, agg, base_dates: pd.DatetimeIndex, *,
    horizon: int = 20, train_w: int = 1260, val_w: int = 780, step: int = 780,
    threshold_q: float = 0.95, mode: str = 'sigmoid',
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward predicted-drawdown gate on the basket aggregate.

    Returns (exposure, val_mask) aligned to `base_dates`. exposure=1 outside
    val windows (no OOS signal there); val_mask marks the OOS test region.
    """
    feat_df = build_aggregate_features(agg)
    target = forward_max_drawdown(agg.ew_log_ret, horizon=horizon)
    mask = (~feat_df.isna().any(axis=1).values) & (~np.isnan(target))
    fdates = feat_df.index[mask]
    feat = feat_df.values[mask]
    targ = target[mask]
    names = list(feat_df.columns)

    exposure = pd.Series(1.0, index=base_dates)
    val_flag = pd.Series(False, index=base_dates)

    n = len(fdates)
    start = 0
    while start + train_w + val_w <= n:
        lo, mid, hi = start, start + train_w, start + train_w + val_w
        pr = train_predictor(feat[lo:mid], targ[lo:mid], names)
        thr = float(np.quantile(predict(pr, feat[lo:mid]), threshold_q))
        val_pred = predict(pr, feat[mid:hi])
        gate = apply_gate(val_pred, thr, mode=mode)        # 1=full, <1=de-risk
        gate_lag = np.concatenate([[1.0], gate[:-1]])      # decide t-1, act t
        vdates = fdates[mid:hi]
        exposure.loc[vdates] = gate_lag
        val_flag.loc[vdates] = True
        start += step
    return exposure.to_numpy(), val_flag.to_numpy()


def main() -> None:
    with open(CLOSE_PKL, 'rb') as f:
        close: pd.DataFrame = pickle.load(f)
    print(f'basket: {close.shape[1]} ETFs, {close.index[0].date()} → '
          f'{close.index[-1].date()}')

    # Base DCA daily stream, aligned to the aggregate's clean dates.
    base_full = np.asarray(PassiveEW(rebal_days=80, commission_bps=10.0)
                           .daily_returns(close), dtype=np.float64)
    base_full = pd.Series(base_full, index=close.index)
    agg = build_ew_aggregate(close, min_active=10)
    base = base_full.reindex(agg.dates).fillna(0.0).to_numpy()
    dates = agg.dates
    cf = 10.0

    # --- arm 1: passive ---
    passive = base.copy()

    # --- arm 2: vol-target ---
    vt_exp = vol_target_exposure(base, win=60, cap=2.0)
    vt = vt_exp * base - _turnover_cost(vt_exp, cf)

    # --- arm 3: predicted-drawdown gate (walk-forward) ---
    dd_exp, val_mask = dd_gate_exposure(close, agg, dates)
    dd = dd_exp * base - _turnover_cost(dd_exp, cf)

    streams = {'passive': passive, 'vol_target': vt, 'dd_gate': dd}

    def _dsr(r, mask=None):
        rr = r[mask] if mask is not None else r
        return standardize_oos(rr, periods_per_year=PPY, n_trials=N_TRIALS)

    print(f'\n{"arm":12s} {"span":7s} {"avg_exp":>7s} {"annSh":>7s} '
          f'{"DSR":>6s} {"defl_t":>7s} {"alpha vs passive":>16s}')
    print('-' * 72)
    results = {}
    # Full-span arms (passive, vol-target have signal ~everywhere).
    for name in ('passive', 'vol_target'):
        mb = _dsr(streams[name])
        avg_exp = {'passive': 1.0,
                   'vol_target': float(vt_exp.mean())}[name]
        alpha = _sharpe(streams[name]) - _sharpe(passive)
        results[name] = {'span': 'full', 'avg_exposure': avg_exp,
                         'ann_sharpe': mb.ann_sharpe, 'dsr': mb.dsr,
                         'deflated_tstat': mb.deflated_tstat, 'alpha': alpha}
        print(f'{name:12s} {"full":7s} {avg_exp:>7.3f} {mb.ann_sharpe:>+7.3f} '
              f'{mb.dsr:>6.3f} {mb.deflated_tstat:>+7.3f} {alpha:>+16.3f}')

    # Matched val-span: all three on the gate's OOS region.
    print('  --- matched on dd-gate val span ---')
    for name in ('passive', 'vol_target', 'dd_gate'):
        mb = _dsr(streams[name], val_mask)
        avg_exp = float((dd_exp if name == 'dd_gate'
                         else (vt_exp if name == 'vol_target'
                               else np.ones_like(base)))[val_mask].mean())
        alpha = _sharpe(streams[name][val_mask]) - _sharpe(passive[val_mask])
        results[f'{name}_val'] = {'span': 'val', 'avg_exposure': avg_exp,
                                  'ann_sharpe': mb.ann_sharpe, 'dsr': mb.dsr,
                                  'deflated_tstat': mb.deflated_tstat, 'alpha': alpha}
        print(f'{name:12s} {"val":7s} {avg_exp:>7.3f} {mb.ann_sharpe:>+7.3f} '
              f'{mb.dsr:>6.3f} {mb.deflated_tstat:>+7.3f} {alpha:>+16.3f}')

    # Dump the deployable full-span streams for the cross-arc ladder.
    np.savez(OUT_DIR / 'regime-scaled-dca-returns.npz',
             passive=passive, vol_target=vt, dd_gate=dd,
             val_mask=val_mask, periods_per_year=np.float64(PPY))
    (OUT_DIR / 'regime-scaled-dca-summary.json').write_text(
        json.dumps(results, indent=2))
    print(f'\n-> {OUT_DIR / "regime-scaled-dca-returns.npz"}')

    pv = results['passive']['deflated_tstat']
    print(f'\nBAR: passive DCA deflated-t = {pv:+.3f}. An arm is a real upgrade '
          f'only if it clears this AND its alpha is positive.')


if __name__ == '__main__':
    main()
