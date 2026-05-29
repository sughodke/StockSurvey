"""Fit the learned 2-leg blend on (DCA daily, vol_v3 daily-aligned).

Two implementations of the same mean-variance optimum:
- `fit_mv_closed_form`: w_vol/w_dca = (mu_vol/var_vol)/(mu_dca/var_dca)
  with w_dca anchored at 1.0 — useful for analytical inspection.
- `fit_grad_sharpe`: gradient ascent on Sharpe over (w_dca, w_vol),
  unconstrained — useful as a numerical confirmation and as the
  deployment-default since it produces a more interpretable scale.

`build_streams` does the data plumbing: loads the DCA basket and the
vol_v3 alpha stream, spreads the per-rebal vol alpha evenly across
each rebal window so both legs are daily-aligned. This matches the
finding's eval methodology exactly.
"""

from __future__ import annotations

import datetime as _dt
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.persist import CHECKPOINT_VERSION, EnsembleCheckpoint


def build_streams(
    dca_close_pkl: str | Path,
    vol_v3_npz: str | Path,
    rebal_days: int = 80,
    commission_bps: float = 10.0,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    """Load DCA daily returns and vol_v3 daily-aligned alpha stream."""
    from cfr.baselines import PassiveEW

    with open(dca_close_pkl, 'rb') as f:
        close = pickle.load(f)
    dca = PassiveEW(
        rebal_days=rebal_days, commission_bps=commission_bps,
    ).daily_returns(close)
    dca = pd.Series(np.asarray(dca, dtype=np.float64), index=close.index).dropna()

    d = np.load(vol_v3_npz, allow_pickle=True)
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


def fit_mv_closed_form(
    r_dca: pd.Series, r_vol: pd.Series,
) -> tuple[float, float]:
    """Diagonal mean-variance optimum, anchored at w_dca=1.0."""
    mu_dca, var_dca = float(r_dca.mean()), float(r_dca.var(ddof=1))
    mu_vol, var_vol = float(r_vol.mean()), float(r_vol.var(ddof=1))
    if mu_dca <= 0 or var_dca <= 0 or var_vol <= 0 or mu_vol <= 0:
        return 1.0, 2.0
    return 1.0, float((mu_vol / var_vol) / (mu_dca / var_dca))


def fit_grad_sharpe(
    r_dca: pd.Series, r_vol: pd.Series,
    init: tuple[float, float] = (1.0, 2.0),
    lr: float = 0.05, n_steps: int = 2000,
    clip: tuple[float, float] = (0.0, 10.0),
) -> tuple[float, float]:
    """Gradient ascent on per-period Sharpe over (w_dca, w_vol)."""
    rd = r_dca.values.astype(np.float64)
    rv = r_vol.values.astype(np.float64)
    n = min(len(rd), len(rv))
    rd, rv = rd[-n:], rv[-n:]
    w = np.array(init, dtype=np.float64)
    cov = np.cov(np.stack([rd, rv]), ddof=1)
    mu_vec = np.array([rd.mean(), rv.mean()])
    for _ in range(n_steps):
        p = w[0] * rd + w[1] * rv
        mu = p.mean()
        sd = p.std(ddof=1)
        if sd <= 1e-12:
            break
        d_sd_dw = (cov @ w) / sd
        d_sh_dw = (mu_vec * sd - mu * d_sd_dw) / (sd * sd)
        w = np.clip(w + lr * d_sh_dw, clip[0], clip[1])
    return float(w[0]), float(w[1])


def annualized_sharpe(r: pd.Series) -> float:
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float('nan')
    return float(r.mean() / sd * np.sqrt(252))


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def train_checkpoint(
    dca_close_pkl: str | Path,
    vol_v3_npz: str | Path,
    train_start: str,
    train_end: str,
    learner: str = 'grad_sharpe',
    dca_checkpoint_path: str = '',
    vol_checkpoint_path: str = '',
    name: str = 'learned-ensemble-v1',
    notes: str = '',
) -> EnsembleCheckpoint:
    """Fit the blend on a strictly-prior window and return a checkpoint."""
    dca, vol, vol_start = build_streams(dca_close_pkl, vol_v3_npz)
    train_dca = dca.loc[train_start:train_end]
    train_vol = vol.loc[train_start:train_end]
    if len(train_dca) < 30:
        raise ValueError(
            f'train window {train_start}->{train_end} too short '
            f'(n={len(train_dca)}); need >= 30 days for stable MV estimates')

    if learner == 'mv_closed_form':
        w_dca, w_vol = fit_mv_closed_form(train_dca, train_vol)
    elif learner == 'grad_sharpe':
        w_dca, w_vol = fit_grad_sharpe(train_dca, train_vol)
    else:
        raise ValueError(f"unknown learner {learner!r}")

    blend = w_dca * train_dca + w_vol * train_vol
    cp = EnsembleCheckpoint(
        version=CHECKPOINT_VERSION,
        name=name,
        w_dca=w_dca,
        w_vol=w_vol,
        learner=learner,
        dca_checkpoint_path=str(dca_checkpoint_path),
        vol_checkpoint_path=str(vol_checkpoint_path),
        train_start=train_start,
        train_end=train_end,
        created_at=_dt.datetime.now(_dt.UTC).isoformat(),
        notes=notes,
        train_sharpe=annualized_sharpe(blend),
        in_sample_max_dd=max_drawdown(blend),
        provenance={
            'dca_close_pkl': str(dca_close_pkl),
            'vol_v3_npz': str(vol_v3_npz),
            'vol_v3_active_from': str(vol_start.date()),
            'n_train_days': int(len(train_dca)),
            'finding': 'findings/learned-ensemble-beats-deterministic.md',
        },
    )
    return cp


__all__ = [
    'build_streams',
    'fit_mv_closed_form',
    'fit_grad_sharpe',
    'annualized_sharpe',
    'max_drawdown',
    'train_checkpoint',
]
