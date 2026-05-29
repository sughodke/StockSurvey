"""Learn an OOS 2-leg blend on (DCA, vol_v3) and beat the deterministic
(DCA + 2x vol_v3) ensemble.

The deterministic recipe is mean-variance-rational: it scales vol_v3 to
match DCA's vol contribution. A learner that fits the same MV problem
on a pre-2024 training slice and applies forward will match or exceed
it whenever the in-sample sigma/alpha mismatch persists into OOS.

Single closed-form learner here: diagonal mean-variance optimization
on the two daily return streams, fit on a train window strictly prior
to the test window. No hyperparameters, no cross-validation: this is
deterministic mathematics applied to data we have.

Reported OOS metrics across several train/test splits to show the
result is not a single-split fluke.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'Output'
sys.path.insert(0, str(REPO / 'apps/cfr/src'))
from cfr.baselines import PassiveEW  # noqa: E402

from ss_portfolio import sharpe_difference_ci  # noqa: E402


def sharpe_ann(r: pd.Series) -> float:
    if r.std() <= 0 or len(r) < 2:
        return float('nan')
    return float(r.mean() / r.std() * np.sqrt(252))


def max_dd(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float(((eq / eq.cummax()) - 1.0).min())


def build_daily_streams() -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    with open(OUT / 'cfr_phase4d_multiasset_close.pkl', 'rb') as f:
        close = pickle.load(f)
    dca = PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(close)
    dca = pd.Series(np.asarray(dca, dtype=np.float64), index=close.index).dropna()

    d = np.load(OUT / 'vol-v3-dolthub-oos-c200-returns.npz', allow_pickle=True)
    vol_dates = pd.to_datetime(np.asarray(d['rebal_dates'], dtype=str))
    vol_alpha = np.asarray(d['full_panel_alpha'], dtype=np.float64)

    vol_daily = pd.Series(0.0, index=dca.index)
    for i in range(len(vol_dates) - 1):
        mask = (vol_daily.index >= vol_dates[i]) & (vol_daily.index < vol_dates[i + 1])
        n = int(mask.sum())
        if n:
            vol_daily.loc[mask] = vol_alpha[i] / n
    mask = vol_daily.index >= vol_dates[-1]
    n = int(mask.sum())
    if n:
        vol_daily.loc[mask] = vol_alpha[-1] / n

    return dca, vol_daily, vol_dates[0]


def fit_mv_blend(r_dca: pd.Series, r_vol: pd.Series) -> tuple[float, float]:
    """Diagonal mean-variance optimal weights on the 2-leg portfolio.
    Returns (w_dca, w_vol) maximizing sharpe of (w_dca * r_dca + w_vol * r_vol).
    Closed form: for unconstrained mean-variance with diagonal cov,
    w_i proportional to mu_i / sigma_i^2.
    Then we anchor w_dca = 1.0 (DCA is the base book; vol is an overlay)
    and let w_vol scale relative to it.
    """
    mu_dca, var_dca = float(r_dca.mean()), float(r_dca.var(ddof=1))
    mu_vol, var_vol = float(r_vol.mean()), float(r_vol.var(ddof=1))
    if mu_dca <= 0 or var_dca <= 0 or var_vol <= 0:
        return 1.0, 2.0
    raw_dca = mu_dca / var_dca
    raw_vol = mu_vol / var_vol
    if raw_dca <= 0:
        return 1.0, 2.0
    w_vol = raw_vol / raw_dca
    return 1.0, float(w_vol)


def fit_sharpe_grad(r_dca: pd.Series, r_vol: pd.Series,
                    lr: float = 0.05, n_steps: int = 2000) -> tuple[float, float]:
    """Gradient-ascent on Sharpe over (w_dca, w_vol).
    Confirms the MV closed form by re-deriving it numerically.
    """
    rd = r_dca.values.astype(np.float64)
    rv = r_vol.values.astype(np.float64)
    n = min(len(rd), len(rv))
    rd, rv = rd[-n:], rv[-n:]
    w = np.array([1.0, 2.0])
    for _ in range(n_steps):
        p = w[0] * rd + w[1] * rv
        mu = p.mean()
        sd = p.std(ddof=1)
        if sd <= 1e-12:
            break
        sh = mu / sd
        d_mu_dw = np.array([rd.mean(), rv.mean()])
        cov = np.cov(np.stack([rd, rv]), ddof=1)
        d_var_dw = 2.0 * cov @ w
        d_sd_dw = d_var_dw / (2.0 * sd)
        d_sh_dw = (d_mu_dw * sd - mu * d_sd_dw) / (sd * sd)
        w = w + lr * d_sh_dw
        w = np.clip(w, 0.0, 10.0)
    return float(w[0]), float(w[1])


def eval_split(dca: pd.Series, vol: pd.Series,
               train_start: str, train_end: str,
               test_start: str, test_end: str | None) -> dict:
    train_dca = dca.loc[train_start:train_end]
    train_vol = vol.loc[train_start:train_end]
    test_dca = dca.loc[test_start:test_end]
    test_vol = vol.loc[test_start:test_end]

    w_mv = fit_mv_blend(train_dca, train_vol)
    w_grad = fit_sharpe_grad(train_dca, train_vol)

    res = {
        'train_range': f'{train_start} -> {train_end}',
        'test_range': f'{test_start} -> {test_end}',
        'train_n': len(train_dca),
        'test_n': len(test_dca),
        'w_mv': w_mv,
        'w_grad': w_grad,
        'train_dca_sharpe': sharpe_ann(train_dca),
        'train_vol_sharpe': sharpe_ann(train_vol),
        'test_dca_sharpe': sharpe_ann(test_dca),
        'test_vol_sharpe': sharpe_ann(test_vol),
    }

    def stream_sharpe(wd, wv):
        return sharpe_ann(wd * test_dca + wv * test_vol)

    res['test_deterministic_2x'] = stream_sharpe(1.0, 2.0)
    res['test_dca_only'] = stream_sharpe(1.0, 0.0)
    res['test_vol_only'] = stream_sharpe(0.0, 1.0)
    res['test_learned_mv'] = stream_sharpe(*w_mv)
    res['test_learned_grad'] = stream_sharpe(*w_grad)

    det_stream = 1.0 * test_dca + 2.0 * test_vol
    learned_stream_mv = w_mv[0] * test_dca + w_mv[1] * test_vol
    learned_stream_grad = w_grad[0] * test_dca + w_grad[1] * test_vol

    ci_mv = sharpe_difference_ci(learned_stream_mv.values, det_stream.values)
    ci_grad = sharpe_difference_ci(learned_stream_grad.values, det_stream.values)
    ann = float(np.sqrt(252))
    res['delta_sr_mv_vs_det'] = float(ci_mv.delta_sr) * ann
    res['ci_mv_vs_det'] = [float(ci_mv.ci_lo) * ann, float(ci_mv.ci_hi) * ann]
    res['delta_sr_grad_vs_det'] = float(ci_grad.delta_sr) * ann
    res['ci_grad_vs_det'] = [float(ci_grad.ci_lo) * ann, float(ci_grad.ci_hi) * ann]
    res['mv_excludes_zero'] = not bool(ci_mv.includes_zero)
    res['grad_excludes_zero'] = not bool(ci_grad.includes_zero)
    res['max_dd_deterministic_2x'] = max_dd(det_stream)
    res['max_dd_learned_mv'] = max_dd(learned_stream_mv)
    res['max_dd_learned_grad'] = max_dd(learned_stream_grad)
    res['max_dd_dca_only'] = max_dd(test_dca)
    return res


def main() -> None:
    dca, vol, vol_start = build_daily_streams()
    print(f'Streams loaded: DCA n={len(dca)}, vol_v3 active from {vol_start.date()}')
    print()

    splits = [
        ('2023-08-02', '2023-12-31', '2024-01-01', None),
        ('2023-08-02', '2024-06-30', '2024-07-01', None),
        ('2023-08-02', '2024-12-31', '2025-01-01', None),
        ('2023-08-02', '2024-03-31', '2024-04-01', None),
    ]

    results = []
    for ts, te, vs, ve in splits:
        r = eval_split(dca, vol, ts, te, vs, ve)
        results.append(r)
        print('-' * 72)
        print(f"TRAIN {r['train_range']} (n={r['train_n']})")
        print(f"  -> w_mv   = ({r['w_mv'][0]:.3f}, {r['w_mv'][1]:.3f})")
        print(f"  -> w_grad = ({r['w_grad'][0]:.3f}, {r['w_grad'][1]:.3f})")
        print(f"TEST  {r['test_range']} (n={r['test_n']})")
        print(f"  DCA only            = {r['test_dca_only']:+7.3f}")
        print(f"  vol_v3 only         = {r['test_vol_only']:+7.3f}")
        print(f"  Deterministic (1,2) = {r['test_deterministic_2x']:+7.3f}")
        print(f"  Learned MV          = {r['test_learned_mv']:+7.3f}  "
              f"dSR vs det = {r['delta_sr_mv_vs_det']:+.3f} "
              f"CI [{r['ci_mv_vs_det'][0]:+.3f}, {r['ci_mv_vs_det'][1]:+.3f}]")
        print(f"  Learned grad-Sharpe = {r['test_learned_grad']:+7.3f}  "
              f"dSR vs det = {r['delta_sr_grad_vs_det']:+.3f} "
              f"CI [{r['ci_grad_vs_det'][0]:+.3f}, {r['ci_grad_vs_det'][1]:+.3f}]")
        print(f"  Max-DD  DCA={r['max_dd_dca_only']*100:+.2f}%  "
              f"det={r['max_dd_deterministic_2x']*100:+.2f}%  "
              f"learned_mv={r['max_dd_learned_mv']*100:+.2f}%  "
              f"learned_grad={r['max_dd_learned_grad']*100:+.2f}%")

    OUT.mkdir(exist_ok=True)
    with open(OUT / 'learned_ensemble_vs_deterministic.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f'Wrote {OUT}/learned_ensemble_vs_deterministic.json')


if __name__ == '__main__':
    main()
